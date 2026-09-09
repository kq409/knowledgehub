import json
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import StreamingResponse
from openai import APIError, NotFoundError
from pydantic import BaseModel, ValidationError
from sqlalchemy.ext.asyncio import AsyncSession

from db import get_session
from routers.conversations import append_turn, ensure_conversation
from schemas import ChatEventType, ChatMessage, ChatRequest, ChatRole
from services.agent.approvals import ApprovalBroker
from services.agent.loop import ResearchAgent
from services.chat_attachments import (
    default_attachment_question,
    file_chat_attachments,
)

router = APIRouter(prefix="/api/chat", tags=["chat"])

SessionDep = Annotated[AsyncSession, Depends(get_session)]


def get_agent(request: Request) -> ResearchAgent:
    agent = getattr(request.app.state, "agent", None)
    if agent is None:
        raise HTTPException(status_code=503, detail="Research agent not ready")
    return agent


AgentDep = Annotated[ResearchAgent, Depends(get_agent)]


class ApprovalDecision(BaseModel):
    request_id: str
    approved: bool


def sse(event_type: ChatEventType, data: dict) -> str:
    payload = json.dumps({"type": event_type.value, **data})
    return f"data: {payload}\n\n"


def _chat_error_message(exc: BaseException) -> str:
    text = str(exc).lower()
    if isinstance(exc, NotFoundError) or any(
        marker in text
        for marker in ("nomic-embed", "embedding", "/v1/embeddings", "embed")
    ):
        return (
            "Embedding request failed. Check EMBEDDING_BASE_URL / EMBEDDING_MODEL "
            "(keep embeddings on Ollama when chat uses a cloud host)."
        )
    if isinstance(exc, APIError):
        status = getattr(exc, "status_code", None)
        status_bit = f" (HTTP {status})" if status else ""
        return f"LLM API error{status_bit}. Check LLM_BASE_URL / LLM_MODEL."
    return "The research agent failed. Check the backend terminal."


def _parse_bool(value: str | bool | None, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return value.strip().lower() not in {"0", "false", "no", "off"}


def _require_source(payload: ChatRequest) -> None:
    if (
        not payload.include_papers
        and not payload.include_voice_notes
        and not payload.include_handwritten_notes
        and not payload.include_documents
    ):
        raise HTTPException(
            status_code=400,
            detail=(
                "Select at least one source: papers, voice notes, handwritten notes, "
                "or documents."
            ),
        )


async def _payload_from_request(
    request: Request,
    session: SessionDep,
) -> ChatRequest:
    content_type = (request.headers.get("content-type") or "").lower()
    if "multipart/form-data" in content_type:
        form = await request.form()
        raw_messages = form.get("messages")
        if not isinstance(raw_messages, str) or not raw_messages.strip():
            raise HTTPException(status_code=400, detail="messages is required")
        try:
            parsed = json.loads(raw_messages)
        except json.JSONDecodeError as exc:
            raise HTTPException(
                status_code=400, detail="messages must be JSON"
            ) from exc
        files = [item for item in form.getlist("files") if isinstance(item, UploadFile)]
        conversation_raw = form.get("conversation_id")
        conversation_id = None
        if isinstance(conversation_raw, str) and conversation_raw.strip():
            try:
                conversation_id = UUID(conversation_raw.strip())
            except ValueError as exc:
                raise HTTPException(
                    status_code=400, detail="conversation_id must be a UUID"
                ) from exc
        messages = [ChatMessage.model_validate(item) for item in parsed]
        if not messages:
            raise HTTPException(status_code=400, detail="Send at least one message")
        prompt = messages[-1].content if messages else ""
        filed = await file_chat_attachments(request.app, session, files, prompt)
        if not messages[-1].content.strip():
            names = [upload.filename or "attachment" for upload in files]
            messages[-1] = ChatMessage(
                role=ChatRole.user,
                content=default_attachment_question(names),
            )
        try:
            return ChatRequest(
                messages=messages,
                include_papers=_parse_bool(form.get("include_papers"), True),
                include_voice_notes=_parse_bool(form.get("include_voice_notes"), True),
                include_handwritten_notes=_parse_bool(
                    form.get("include_handwritten_notes"), True
                ),
                include_documents=_parse_bool(form.get("include_documents"), True),
                conversation_id=conversation_id,
                attachments=[item.attachment for item in filed],
            )
        except ValidationError as exc:
            raise RequestValidationError(exc.errors(include_context=False)) from exc

    body = await request.json()
    try:
        return ChatRequest.model_validate(body)
    except ValidationError as exc:
        raise RequestValidationError(exc.errors(include_context=False)) from exc


@router.post("/approve")
async def approve_tool(request: Request, payload: ApprovalDecision):
    """Resolve a write that the chat stream paused on."""
    broker = getattr(request.app.state, "approvals", None)
    if not isinstance(broker, ApprovalBroker):
        raise HTTPException(
            status_code=400,
            detail="Write-tool approval is not enabled on this server.",
        )
    if not broker.resolve(payload.request_id, payload.approved):
        raise HTTPException(
            status_code=404,
            detail="Unknown or expired approval request.",
        )
    return {"ok": True}


@router.post("")
@router.post("/", include_in_schema=False)
async def chat(
    request: Request,
    session: SessionDep,
    agent: AgentDep,
):
    payload = await _payload_from_request(request, session)
    _require_source(payload)

    question = payload.messages[-1].content
    conversation = await ensure_conversation(
        session, payload.conversation_id, first_question=question
    )
    payload.conversation_id = conversation.id

    async def stream():
        turn: dict = {
            "question": question,
            "steps": [],
            "answer": "",
            "citations": [],
            "artifacts": [],
            "todos": [],
            "subagents": [],
            "compactions": [],
            "approvals": [],
            "model": None,
            "error": None,
            "verdict": None,
            "attachments": [
                item.model_dump(mode="json") for item in payload.attachments
            ],
            "isRunning": False,
        }
        yield sse(
            ChatEventType.conversation,
            {
                "id": str(conversation.id),
                "title": conversation.title,
            },
        )
        for item in payload.attachments:
            yield sse(ChatEventType.attachment, item.model_dump(mode="json"))
        try:
            async for event in agent.run(session, payload):
                _accumulate_turn(turn, event.type, event.data)
                yield sse(event.type, event.data)
        except Exception as exc:
            print(f"❌ Chat agent failed: {exc}")
            turn["error"] = _chat_error_message(exc)
            yield sse(ChatEventType.error, {"message": turn["error"]})
        try:
            await append_turn(session, conversation, turn)
        except Exception as exc:
            print(f"⚠️  Failed to persist conversation turn: {exc}", flush=True)

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


def _step_for_result(steps: list[dict], data: dict) -> int | None:
    """Which announced tool call this result belongs to.

    Parallel calls are announced together and answered together, so two
    `search_library` steps can be open at once. `call_id` pairs them exactly;
    the name scan is the fallback for a result that carries no id.
    """
    call_id = data.get("call_id")
    if call_id:
        for index, step in enumerate(steps):
            if step.get("callId") == call_id:
                return index
    name = data.get("name")
    for index in range(len(steps) - 1, -1, -1):
        step = steps[index]
        if step.get("summary") is None and step.get("name") == name:
            return index
    return None


def _accumulate_turn(turn: dict, event_type: ChatEventType, data: dict) -> None:
    if event_type is ChatEventType.token:
        turn["answer"] = str(turn.get("answer") or "") + str(data.get("text") or "")
    elif event_type is ChatEventType.citations:
        turn["citations"] = data.get("citations") or []
    elif event_type is ChatEventType.artifact:
        turn["artifacts"] = [*(turn.get("artifacts") or []), data]
    elif event_type is ChatEventType.todo:
        turn["todos"] = data.get("items") or []
    elif event_type is ChatEventType.verdict:
        turn["verdict"] = data
    elif event_type is ChatEventType.error:
        turn["error"] = data.get("message")
    elif event_type is ChatEventType.done:
        turn["model"] = data.get("model")
    elif event_type is ChatEventType.tool_call:
        turn["steps"] = [
            *(turn.get("steps") or []),
            {
                "name": data.get("name"),
                "arguments": data.get("arguments") or {},
                "callId": data.get("call_id"),
                "summary": None,
                "permission": None,
                "agentId": data.get("agent_id") or "main",
            },
        ]
    elif event_type is ChatEventType.tool_result:
        steps = list(turn.get("steps") or [])
        index = _step_for_result(steps, data)
        if index is not None:
            steps[index] = {
                **steps[index],
                "summary": data.get("summary"),
                "permission": data.get("permission"),
            }
        turn["steps"] = steps
    elif event_type is ChatEventType.subagent:
        subagents = list(turn.get("subagents") or [])
        existing = next(
            (item for item in subagents if item.get("id") == data.get("id")), None
        )
        if existing is None:
            subagents.append(
                {
                    "id": data.get("id"),
                    "status": data.get("status"),
                    "goal": data.get("goal"),
                    "summary": data.get("summary"),
                    "steps": [],
                }
            )
        else:
            existing["status"] = data.get("status")
            if data.get("summary"):
                existing["summary"] = data.get("summary")
        turn["subagents"] = subagents
    elif event_type is ChatEventType.compact:
        turn["compactions"] = [
            *(turn.get("compactions") or []),
            {
                "mode": data.get("mode"),
                "beforeChars": data.get("before_chars"),
                "afterChars": data.get("after_chars"),
                "agentId": data.get("agent_id") or "main",
            },
        ]
    elif event_type is ChatEventType.approval:
        approvals = list(turn.get("approvals") or [])
        request_id = data.get("request_id")
        existing = next(
            (item for item in approvals if item.get("request_id") == request_id),
            None,
        )
        payload = {
            "request_id": request_id,
            "tool": data.get("tool"),
            "arguments": data.get("arguments") or {},
            "reason": data.get("reason") or "",
            "status": data.get("status") or "pending",
        }
        if existing is None:
            approvals.append(payload)
        else:
            existing.update(payload)
        turn["approvals"] = approvals
