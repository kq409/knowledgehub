"""The agent loop.

The model decides which tools to call and when to stop. This module only
executes what it asks for, keeps the conversation inside a budget, and reports
what happened as a stream of events.

Two guardrails sit around the model's freedom: a permission policy decides
whether a requested tool may run at all, and an evidence gate reads the draft
answer before it ships. A refused tool and a failed gate both come back as
plain text the model can act on, so neither one ends the turn early.

P3 adds an in-turn todo list, one nested research pass per turn, and automatic
context compaction before each model call.
"""

import time
import uuid
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from pathlib import Path

from openai import OpenAI
from sqlalchemy.ext.asyncio import AsyncSession

from schemas import ChatCitation, ChatEventType, ChatRequest, GateStatus
from services.agent.compact import (
    COMPACT_CHAR_THRESHOLD,
    compact_messages,
    repair_tool_pairing,
)
from services.agent.gate import EvidenceGate, GateVerdict
from services.agent.permissions import (
    DEFAULT_POLICY,
    PermissionPolicy,
    PermissionVerdict,
)
from services.agent.protocol import (
    AgentProtocol,
    ProtocolSwitchedError,
    ToolCall,
    ToolCapabilityCache,
)
from services.agent.subagent import SpawnHolder, run_spawned_subagent
from services.agent.todo import TodoStore
from services.agent.tools import (
    TOOL_HANDLERS,
    CitationRegistry,
    ToolContext,
    ToolError,
    ToolResult,
)
from services.ask_trace import log_chat_trace
from services.compare import CompareService
from services.connect import ConnectService
from services.embeddings import EmbeddingService
from services.extraction import ExtractionService

PROMPT_FILE = Path(__file__).resolve().parent.parent.parent / "chat_agent_prompt.txt"
AGENT_PROMPT = PROMPT_FILE.read_text().strip()
PROMPT_VERSION = "chat-agent-v6"

MAX_ITERATIONS = 8
MAX_TOOL_CALLS_PER_TURN = 3
MAX_TOOL_RESULT_CHARS = 4000
MAX_GATE_RETRIES = 1
MAIN_AGENT_ID = "main"

FINAL_TURN_NUDGE = (
    "You have reached the tool budget for this turn. Answer now using only the "
    "evidence you already retrieved, and say plainly what you could not check."
)
EMPTY_ANSWER = "The model returned an empty answer."


@dataclass(frozen=True)
class AgentEvent:
    type: ChatEventType
    data: dict = field(default_factory=dict)


def _truncate(content: str) -> str:
    if len(content) <= MAX_TOOL_RESULT_CHARS:
        return content
    return (
        content[:MAX_TOOL_RESULT_CHARS].rstrip()
        + "\n\n[truncated: ask for a narrower slice if you need more]"
    )


def refusal_result(call: ToolCall, permission: PermissionVerdict) -> ToolResult:
    """What the model is told when the policy turns a call down."""
    return ToolResult(
        content=(
            f"{call.name} was not run because it {permission.reason}. "
            "Work with the tools you do have."
        ),
        summary=f"{call.name} refused: {permission.reason}",
    )


def budget_skip_result(call: ToolCall, limit: int) -> ToolResult:
    """Stub for a parallel tool_call that exceeded this turn's execute budget.

    Native providers (DeepSeek among them) reject the next request unless every
    `tool_call_id` on the assistant message has a following tool result.
    """
    return ToolResult(
        content=(
            f"{call.name} was not run because this turn already used its "
            f"{limit}-tool budget. Answer with the evidence you already have."
        ),
        summary=f"{call.name} skipped: tool budget",
    )


async def run_tool(ctx: ToolContext, call: ToolCall) -> ToolResult:
    """Run one tool. Failures come back as text for the model, not exceptions.

    A bad argument is something the model can recover from on the next turn, so
    it never breaks the loop. Unknown tool names never reach here: the policy
    turns those away first. `spawn_subagent` is intercepted by the loop before
    this runs.
    """
    handler = TOOL_HANDLERS.get(call.name)
    if handler is None:
        return ToolResult(
            content=f"{call.name} is allowed by policy but has no implementation.",
            summary=f"{call.name} is not implemented",
        )

    try:
        result = await handler(ctx, **call.arguments)
    except ToolError as exc:
        return ToolResult(
            content=f"{call.name} could not run: {exc}",
            summary=f"{call.name} failed: {exc}",
        )
    except TypeError as exc:
        return ToolResult(
            content=f"{call.name} received unexpected arguments: {exc}",
            summary=f"{call.name} got bad arguments",
        )
    return ToolResult(
        content=_truncate(result.content),
        summary=result.summary,
        artifact=result.artifact,
        todo_items=result.todo_items,
    )


class ResearchAgent:
    """Holds the model wiring; one conversation runs through `run`."""

    def __init__(
        self,
        llm_client: OpenAI,
        llm_model: str,
        embeddings: EmbeddingService,
        capabilities: ToolCapabilityCache | None = None,
        gate: EvidenceGate | None = None,
        policy: PermissionPolicy | None = None,
        compare: CompareService | None = None,
        extraction: ExtractionService | None = None,
        connect: ConnectService | None = None,
    ) -> None:
        self.llm_client = llm_client
        self.llm_model = llm_model
        self.embeddings = embeddings
        self.capabilities = capabilities or ToolCapabilityCache()
        self.gate = gate or EvidenceGate()
        self.policy = policy or DEFAULT_POLICY
        self.compare = compare
        self.extraction = extraction
        self.connect = connect
        self.prompt_version = PROMPT_VERSION

    def _protocol(
        self,
        tool_schemas: list[dict] | None = None,
        allowed_tools: set[str] | frozenset[str] | None = None,
    ) -> AgentProtocol:
        return AgentProtocol(
            self.llm_client,
            self.llm_model,
            self.capabilities,
            tool_schemas=tool_schemas,
            allowed_tools=allowed_tools,
        )

    def _context(
        self,
        session: AsyncSession,
        payload: ChatRequest,
        registry: CitationRegistry,
        todo_store: TodoStore,
    ) -> ToolContext:
        return ToolContext(
            session=session,
            embeddings=self.embeddings,
            registry=registry,
            include_papers=payload.include_papers,
            include_voice_notes=payload.include_voice_notes,
            include_handwritten_notes=payload.include_handwritten_notes,
            top_k=payload.top_k,
            compare=self.compare,
            extraction=self.extraction,
            connect=self.connect,
            todo_store=todo_store,
            depth=0,
            subagent_budget=1,
        )

    async def _complete(
        self, protocol: AgentProtocol, messages: list[dict], base_prompt: str
    ):
        """One model turn, rebuilding the system prompt if the channel switches."""
        messages[:] = repair_tool_pairing(messages)
        try:
            return await protocol.complete(messages)
        except ProtocolSwitchedError:
            messages[0] = {
                "role": "system",
                "content": protocol.system_prompt(base_prompt),
            }
            return await protocol.complete(messages)

    async def _maybe_compact(
        self, messages: list[dict]
    ) -> AsyncGenerator[AgentEvent, None]:
        result = await compact_messages(
            messages,
            threshold=COMPACT_CHAR_THRESHOLD,
            llm_client=self.llm_client,
            llm_model=self.llm_model,
        )
        if not result.changed:
            return
        messages[:] = result.messages
        yield AgentEvent(
            type=ChatEventType.compact,
            data={
                "mode": result.mode,
                "before_chars": result.before_chars,
                "after_chars": result.after_chars,
                "agent_id": MAIN_AGENT_ID,
            },
        )

    async def _execute_tool(
        self, ctx: ToolContext, call: ToolCall, protocol: AgentProtocol
    ) -> AsyncGenerator[AgentEvent, None]:
        """Permission-check, run (or spawn), and stream side events."""
        yield AgentEvent(
            type=ChatEventType.tool_call,
            data={
                "name": call.name,
                "arguments": call.arguments,
                "agent_id": MAIN_AGENT_ID,
            },
        )
        permission = self.policy.check(call.name)
        result: ToolResult | None = None

        if permission.allowed and call.name == "spawn_subagent":
            holder = SpawnHolder()
            async for event in run_spawned_subagent(self, ctx, call, holder):
                yield event
            result = holder.result or ToolResult(
                content="spawn_subagent produced no result.",
                summary="spawn_subagent failed",
            )
        elif permission.allowed:
            result = await run_tool(ctx, call)
        else:
            result = refusal_result(call, permission)

        yield AgentEvent(
            type=ChatEventType.tool_result,
            data={
                "name": call.name,
                "summary": result.summary,
                "permission": {
                    "decision": permission.decision.value,
                    "reason": permission.reason,
                },
                "agent_id": MAIN_AGENT_ID,
            },
        )
        if result.todo_items is not None:
            yield AgentEvent(
                type=ChatEventType.todo,
                data={"items": result.todo_items},
            )
        if result.artifact is not None:
            yield AgentEvent(type=ChatEventType.artifact, data=result.artifact)

        # Stash for the caller to append to messages.
        self._last_tool_result = result

    async def run(
        self, session: AsyncSession, payload: ChatRequest
    ) -> AsyncGenerator[AgentEvent, None]:
        protocol = self._protocol()
        registry = CitationRegistry()
        todo_store = TodoStore()
        ctx = self._context(session, payload, registry, todo_store)
        question = payload.messages[-1].content
        base_prompt = AGENT_PROMPT
        request_id = str(uuid.uuid4())
        started = time.perf_counter()
        tools_called: list[str] = []
        prompt_tokens = 0
        completion_tokens = 0
        saw_usage = False

        messages: list[dict] = [
            {"role": "system", "content": protocol.system_prompt(base_prompt)}
        ]
        messages.extend(
            {"role": message.role.value, "content": message.content}
            for message in payload.messages
        )

        answer: str | None = None
        verdict: GateVerdict | None = None
        tool_calls_made = 0
        gate_retries = 0

        for _ in range(MAX_ITERATIONS):
            async for event in self._maybe_compact(messages):
                yield event

            response = await self._complete(protocol, messages, base_prompt)
            messages.append(protocol.assistant_message(response))
            if response.prompt_tokens is not None:
                prompt_tokens += response.prompt_tokens
                saw_usage = True
            if response.completion_tokens is not None:
                completion_tokens += response.completion_tokens
                saw_usage = True

            if response.tool_calls:
                for index, call in enumerate(response.tool_calls):
                    if index >= MAX_TOOL_CALLS_PER_TURN:
                        result = budget_skip_result(call, MAX_TOOL_CALLS_PER_TURN)
                        yield AgentEvent(
                            type=ChatEventType.tool_call,
                            data={
                                "name": call.name,
                                "arguments": call.arguments,
                                "agent_id": MAIN_AGENT_ID,
                            },
                        )
                        yield AgentEvent(
                            type=ChatEventType.tool_result,
                            data={
                                "name": call.name,
                                "summary": result.summary,
                                "permission": {
                                    "decision": "skip",
                                    "reason": (
                                        f"this turn already used its "
                                        f"{MAX_TOOL_CALLS_PER_TURN}-tool budget"
                                    ),
                                },
                                "agent_id": MAIN_AGENT_ID,
                            },
                        )
                        messages.append(
                            protocol.tool_result_message(call, result.content)
                        )
                        continue
                    self._last_tool_result = None
                    async for event in self._execute_tool(ctx, call, protocol):
                        yield event
                    result = self._last_tool_result or ToolResult(
                        content=f"{call.name} produced no result.",
                        summary=f"{call.name} produced no result",
                    )
                    permission = self.policy.check(call.name)
                    if permission.allowed:
                        tool_calls_made += 1
                        tools_called.append(call.name)
                    messages.append(protocol.tool_result_message(call, result.content))
                continue

            if not response.text.strip():
                break

            verdict = await self.gate.check(
                question=question,
                answer=response.text,
                citations=registry.citations(),
                tool_calls_made=tool_calls_made,
            )
            if verdict.ok or gate_retries >= MAX_GATE_RETRIES:
                answer = response.text
                break

            gate_retries += 1
            yield AgentEvent(
                type=ChatEventType.verdict,
                data=verdict.as_schema(GateStatus.retrying).model_dump(mode="json"),
            )
            messages.append({"role": "user", "content": verdict.feedback()})
        else:
            messages.append({"role": "user", "content": FINAL_TURN_NUDGE})
            async for event in self._maybe_compact(messages):
                yield event
            response = await self._complete(protocol, messages, base_prompt)
            answer = response.text
            verdict = await self.gate.check(
                question=question,
                answer=answer,
                citations=registry.citations(),
                tool_calls_made=tool_calls_made,
            )

        if not answer:
            yield AgentEvent(type=ChatEventType.error, data={"message": EMPTY_ANSWER})
        else:
            yield AgentEvent(type=ChatEventType.token, data={"text": answer})

        citations: list[ChatCitation] = registry.citations()
        yield AgentEvent(
            type=ChatEventType.citations,
            data={"citations": [item.model_dump(mode="json") for item in citations]},
        )
        if verdict is not None:
            yield AgentEvent(
                type=ChatEventType.verdict,
                data=verdict.as_schema().model_dump(mode="json"),
            )
        latency_ms = (time.perf_counter() - started) * 1000
        gate_status = None
        if verdict is not None:
            gate_status = verdict.as_schema().status.value
        log_chat_trace(
            question=question,
            tools_called=tools_called,
            citations=[item.model_dump(mode="json") for item in citations],
            answer=answer or "",
            latency_ms=latency_ms,
            gate_status=gate_status,
            request_id=request_id,
            prompt_tokens=prompt_tokens if saw_usage else None,
            completion_tokens=completion_tokens if saw_usage else None,
        )
        yield AgentEvent(
            type=ChatEventType.done,
            data={
                "model": self.llm_model,
                "prompt_version": self.prompt_version,
                "request_id": request_id,
                "latency_ms": round(latency_ms, 1),
                "prompt_tokens": prompt_tokens if saw_usage else None,
                "completion_tokens": completion_tokens if saw_usage else None,
            },
        )
