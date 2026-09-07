import json
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from openai import APIError, NotFoundError
from sqlalchemy.ext.asyncio import AsyncSession

from db import get_session
from schemas import ChatEventType, ChatRequest
from services.agent.loop import ResearchAgent

router = APIRouter(prefix="/api/chat", tags=["chat"])

SessionDep = Annotated[AsyncSession, Depends(get_session)]


def get_agent(request: Request) -> ResearchAgent:
    agent = getattr(request.app.state, "agent", None)
    if agent is None:
        raise HTTPException(status_code=503, detail="Research agent not ready")
    return agent


AgentDep = Annotated[ResearchAgent, Depends(get_agent)]


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


@router.post("")
@router.post("/", include_in_schema=False)
async def chat(
    payload: ChatRequest,
    session: SessionDep,
    agent: AgentDep,
):
    if (
        not payload.include_papers
        and not payload.include_voice_notes
        and not payload.include_handwritten_notes
    ):
        raise HTTPException(
            status_code=400,
            detail="Select at least one source: papers, voice notes, or handwritten notes.",
        )

    async def stream():
        try:
            async for event in agent.run(session, payload):
                yield sse(event.type, event.data)
        except Exception as exc:
            print(f"❌ Chat agent failed: {exc}")
            yield sse(ChatEventType.error, {"message": _chat_error_message(exc)})

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
