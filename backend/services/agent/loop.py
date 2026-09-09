"""The agent loop.

The model decides which tools to call and when to stop. This module only
executes what it asks for, keeps the conversation inside a budget, and reports
what happened as a stream of events.

Two guardrails sit around the model's freedom: a hook chain decides whether a
requested tool may run at all (permission is its first hook), and an evidence
gate reads the draft answer before it ships. A refused tool and a failed gate
both come back as plain text the model can act on, so neither one ends the turn
early.

Nothing a single turn produces lives on `ResearchAgent`: the instance is shared
by every request, so per-turn state goes in `RunState` instead.
"""

import asyncio
import time
from collections.abc import AsyncGenerator
from dataclasses import dataclass, replace
from pathlib import Path

from openai import OpenAI
from sqlalchemy.ext.asyncio import AsyncSession

import db
from schemas import ArtifactKind, ChatCitation, ChatEventType, ChatRequest, GateStatus
from services.agent.compact import (
    COMPACT_CHAR_THRESHOLD,
    compact_messages,
    reactive_compact,
    repair_tool_pairing,
)
from services.agent.events import AgentEvent, EventSink
from services.agent.gate import EvidenceGate, GateVerdict, compact_answer_citations
from services.agent.hooks import (
    Approver,
    HookChain,
    PreToolDecision,
    ToolHookContext,
    default_hooks,
)
from services.agent.mcp_client import McpRegistry
from services.agent.memory import (
    EXTRACT_SOURCE_TURN,
    RECALL_LOAD_LIMIT,
    consolidate_memories,
    extract_and_store,
    format_recalled_memories,
    recall_memories,
    search_memories,
)
from services.agent.notifications import (
    INGEST_NOTICES,
    NoticeBoard,
    format_notifications,
)
from services.agent.permissions import DEFAULT_POLICY, PermissionPolicy
from services.agent.planner import plan_tools
from services.agent.protocol import (
    AgentProtocol,
    AgentResponse,
    ProtocolSwitchedError,
    ToolCall,
    ToolCapabilityCache,
)
from services.agent.recovery import (
    LlmFailure,
    backoff_seconds,
    classify_llm_failure,
    max_retries,
)
from services.agent.runstate import RunState, ToolOutcome
from services.agent.skills import skill_catalog
from services.agent.subagent import SpawnHolder, run_spawned_subagent
from services.agent.telemetry import log_event, log_warning
from services.agent.tool_pool import ToolPool, assemble_tool_pool
from services.agent.tool_results import ToolResultStore, truncation_notice
from services.agent.tools import (
    TOOL_HANDLERS,
    ToolContext,
    ToolError,
    ToolResult,
)
from services.ask_policy import QueryKind, classify_query
from services.ask_trace import log_chat_trace
from services.compare import CompareService
from services.connect import ConnectService
from services.embeddings import EmbeddingService
from services.extraction import ExtractionService
from services.web_search import WebSearcher

__all__ = [
    "AgentEvent",
    "MAX_ITERATIONS",
    "MAX_TOOL_CALLS_PER_RESPONSE",
    "MAX_TOOL_CALLS_PER_RUN",
    "MAX_TOOL_RESULT_CHARS",
    "PROMPT_VERSION",
    "ResearchAgent",
    "budget_skip_result",
    "run_tool",
]

PROMPT_FILE = Path(__file__).resolve().parent.parent.parent / "chat_agent_prompt.txt"
AGENT_PROMPT = PROMPT_FILE.read_text().strip()
PROMPT_VERSION = "chat-agent-v11"

MAX_ITERATIONS = 8
# How many of one assistant message's tool calls actually run. A model that
# asks for eight searches at once gets the first three; the rest come back as
# stubs so the provider's tool_call pairing stays valid.
MAX_TOOL_CALLS_PER_RESPONSE = 3
# Ceiling across the whole turn. Without it, MAX_ITERATIONS rounds of three
# calls each is 24 tools on one question.
MAX_TOOL_CALLS_PER_RUN = 12
MAX_TOOL_RESULT_CHARS = 4000
MAX_GATE_RETRIES = 1
# How long the loop waits between draining hook events while a PreToolUse
# hook is still running (an approval pause, typically).
PRE_HOOK_POLL_SECONDS = 0.05
MAIN_AGENT_ID = "main"
# One reactive compaction per model call. A second rejection after summarising
# means the tail alone does not fit, which retrying cannot fix.
MAX_REACTIVE_COMPACTIONS = 1
# Ceiling for the one retry after an empty, truncated reply.
TRUNCATED_RETRY_MAX_TOKENS = 8192

# Tools that may run concurrently, each on its own database session. They are
# read-only and hold no per-turn budget. Everything else runs sequentially on
# the turn's own session: `compare_papers` spends a budget it stores on the
# context, the write tools commit, and `spawn_subagent` streams nested events.
PARALLEL_SAFE_TOOLS = frozenset(
    {
        "search_library",
        "web_search",
        "list_papers",
        "list_notes",
        "list_documents",
        "read_paper",
        "read_note",
        "read_document",
        "list_skills",
        "load_skill",
        "memory_search",
        "fetch_tool_result",
    }
)

# Tools slow enough that the researcher deserves to see them working. These run
# on a polled path so the progress they report reaches the stream while the
# tool is still going; everything else is simply awaited.
PROGRESS_TOOLS = frozenset({"compare_papers"})
PROGRESS_POLL_SECONDS = 0.25

FINAL_TURN_NUDGE = (
    "You have reached the tool budget for this turn. Answer now using only the "
    "evidence you already retrieved, and say plainly what you could not check."
)
EMPTY_ANSWER = "The model returned an empty answer."
EMPTY_PLAN_SUFFIX = (
    "No tools were selected for this message. Answer directly without calling "
    "tools. Do not invent papers, notes, or documents in the library. "
    "Do not announce that you will look something up — answer now."
)

FIELD_WIDE_SUFFIX = (
    "This question asks for field-wide, latest, or SOTA progress. "
    "Open with library coverage. Then call web_search for sources outside "
    "the library (for ArXiv or preprints, put site:arxiv.org in the query) "
    "unless the researcher only asked what this library holds. "
    "If web search is not configured, say so and list the closest library papers. "
    "Web [n] citations are not library papers."
)


def _truncate(content: str, call: ToolCall, store: ToolResultStore | None) -> str:
    """Fit a result into the context budget, leaving a way back to the rest."""
    if len(content) <= MAX_TOOL_RESULT_CHARS:
        return content
    head = content[:MAX_TOOL_RESULT_CHARS].rstrip()
    if store is None:
        # No store (a nested pass, or a unit test): the old dead end.
        return head + "\n\n[truncated: ask for a narrower slice if you need more]"
    store.put(call.id, call.name, content)
    return head + truncation_notice(call.id, len(head), len(content))


def budget_skip_result(
    call: ToolCall, limit: int, *, scope: str = "response"
) -> ToolResult:
    """Stub for a tool call that exceeded a budget before it could run.

    Native providers (DeepSeek among them) reject the next request unless every
    `tool_call_id` on the assistant message has a following tool result, so a
    skipped call still needs one.
    """
    return ToolResult(
        content=(
            f"{call.name} was not run because this {scope} already used its "
            f"{limit}-tool budget. Answer with the evidence you already have."
        ),
        summary=f"{call.name} skipped: tool budget",
    )


def _budget_decision(call: ToolCall, limit: int, *, scope: str) -> PreToolDecision:
    return PreToolDecision(
        result=budget_skip_result(call, limit, scope=scope),
        decision="skip",
        reason=f"this {scope} already used its {limit}-tool budget",
    )


async def run_tool(ctx: ToolContext, call: ToolCall) -> ToolResult:
    """Run one tool. Failures come back as text for the model, not exceptions.

    A bad argument is something the model can recover from on the next turn, so
    it never breaks the loop. Unknown tool names never reach here: the hook
    chain turns those away first. `spawn_subagent` is intercepted by the loop
    before this runs.
    """
    table = ctx.handlers if isinstance(ctx.handlers, dict) else TOOL_HANDLERS
    handler = table.get(call.name)
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
    store = ctx.result_store if isinstance(ctx.result_store, ToolResultStore) else None
    return ToolResult(
        content=_truncate(result.content, call, store),
        summary=result.summary,
        artifact=result.artifact,
        todo_items=result.todo_items,
    )


@dataclass
class _ResultHolder:
    """Filled by `_run_with_progress`, which yields events instead of returning."""

    result: ToolResult | None = None


@dataclass(frozen=True)
class _ToolPlan:
    """One requested call, and whether a hook already turned it down."""

    index: int
    call: ToolCall
    hook_ctx: ToolHookContext
    decision: PreToolDecision | None

    @property
    def runnable(self) -> bool:
        return self.decision is None


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
        web_search: WebSearcher | None = None,
        approver: Approver | None = None,
        hooks: HookChain | None = None,
        notices: NoticeBoard | None = None,
        mcp: McpRegistry | None = None,
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
        self.web_search = web_search
        self.approver = approver
        self.hooks = hooks or default_hooks(self.policy, approver)
        # Shared by every turn on purpose: ingestion finishes on its own
        # schedule, and whichever turn runs next is the one that should say so.
        self.notices = notices if notices is not None else INGEST_NOTICES
        # Outside tools, if this deployment configured any. None means the
        # turn's pool is exactly what this repo ships.
        self.mcp = mcp
        self.prompt_version = PROMPT_VERSION

    def _protocol(
        self,
        tool_schemas: list[dict] | None = None,
        allowed_tools: set[str] | frozenset[str] | None = None,
        pool: ToolPool | None = None,
    ) -> AgentProtocol:
        schemas = tool_schemas
        allowed = allowed_tools
        if pool is not None:
            if schemas is None:
                schemas = pool.schemas
            if allowed is None:
                allowed = pool.names
        return AgentProtocol(
            self.llm_client,
            self.llm_model,
            self.capabilities,
            tool_schemas=schemas,
            allowed_tools=allowed,
        )

    def _answering_protocol(self, tools_enabled: bool, pool: ToolPool) -> AgentProtocol:
        if tools_enabled:
            return self._protocol(pool=pool)
        return self._protocol(tool_schemas=[], allowed_tools=frozenset())

    def _context(
        self,
        session: AsyncSession,
        payload: ChatRequest,
        state: RunState,
    ) -> ToolContext:
        return ToolContext(
            session=session,
            embeddings=self.embeddings,
            registry=state.registry,
            include_papers=payload.include_papers,
            include_voice_notes=payload.include_voice_notes,
            include_handwritten_notes=payload.include_handwritten_notes,
            include_documents=payload.include_documents,
            top_k=payload.top_k,
            compare=self.compare,
            extraction=self.extraction,
            connect=self.connect,
            web_search=self.web_search,
            todo_store=state.todo_store,
            result_store=state.result_store,
            handlers=state.pool.handlers if state.pool is not None else None,
            depth=0,
            subagent_budget=1,
        )

    async def _complete(
        self,
        protocol: AgentProtocol,
        messages: list[dict],
        base_prompt: str,
        max_tokens: int | None = None,
    ):
        """One model turn, rebuilding the system prompt if the channel switches."""
        messages[:] = repair_tool_pairing(messages)
        try:
            return await protocol.complete(messages, max_tokens)
        except ProtocolSwitchedError:
            messages[0] = {
                "role": "system",
                "content": protocol.system_prompt(base_prompt),
            }
            return await protocol.complete(messages, max_tokens)

    async def _complete_with_recovery(
        self,
        protocol: AgentProtocol,
        messages: list[dict],
        base_prompt: str,
        state: RunState,
        question: str,
        sink: EventSink,
    ) -> AgentResponse:
        """One model turn, with the four recoveries from `recovery.py`.

        Notices go on the sink rather than being yielded, because this is a
        plain coroutine the loop awaits for a response. The caller drains the
        sink so the researcher still sees that a retry happened.
        """
        retries = 0
        reactive_compactions = 0
        max_tokens: int | None = None

        def notice(reason: str, detail: str) -> None:
            log_event(
                "llm_recovery",
                run_id=state.run_id,
                reason=reason,
                detail=detail,
            )
            sink.emit(
                AgentEvent(
                    type=ChatEventType.notice,
                    data={
                        "kind": reason,
                        "message": detail,
                        "agent_id": MAIN_AGENT_ID,
                    },
                )
            )

        while True:
            try:
                response = await self._complete(
                    protocol, messages, base_prompt, max_tokens
                )
            except Exception as exc:
                failure = classify_llm_failure(exc)
                if failure is LlmFailure.rate_limited and retries < max_retries():
                    delay = backoff_seconds(retries)
                    retries += 1
                    notice(
                        "rate_limited",
                        f"The model was busy; retrying in {delay:.1f}s "
                        f"(attempt {retries} of {max_retries()}).",
                    )
                    await asyncio.sleep(delay)
                    continue
                if (
                    failure is LlmFailure.context_too_long
                    and reactive_compactions < MAX_REACTIVE_COMPACTIONS
                ):
                    reactive_compactions += 1
                    result = await reactive_compact(
                        messages,
                        active_request=question,
                        llm_client=self.llm_client,
                        llm_model=self.llm_model,
                        store=state.result_store,
                    )
                    messages[:] = result.messages
                    notice(
                        "context_too_long",
                        "The conversation outgrew the context window; "
                        "summarising the earlier history and retrying.",
                    )
                    sink.emit(
                        AgentEvent(
                            type=ChatEventType.compact,
                            data={
                                "mode": result.mode,
                                "before_chars": result.before_chars,
                                "after_chars": result.after_chars,
                                "agent_id": MAIN_AGENT_ID,
                            },
                        )
                    )
                    continue
                log_warning(
                    "llm_call_failed",
                    run_id=state.run_id,
                    failure=failure.value,
                    error=str(exc),
                )
                raise

            # A truncated reply with nothing usable in it is not an answer; the
            # same request with more room usually is.
            if (
                response.truncated
                and not response.text.strip()
                and not response.tool_calls
                and max_tokens is None
            ):
                max_tokens = TRUNCATED_RETRY_MAX_TOKENS
                notice(
                    "output_truncated",
                    "The model ran out of output room before saying anything; "
                    "retrying with a larger ceiling.",
                )
                continue

            return response

    async def _maybe_compact(
        self, messages: list[dict], state: RunState
    ) -> AsyncGenerator[AgentEvent, None]:
        result = await compact_messages(
            messages,
            threshold=COMPACT_CHAR_THRESHOLD,
            llm_client=self.llm_client,
            llm_model=self.llm_model,
            store=state.result_store,
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

    async def _deliver_notifications(
        self, messages: list[dict], state: RunState
    ) -> AsyncGenerator[AgentEvent, None]:
        """Hand the model whatever finished in the background since it last looked.

        Called immediately before each model call rather than once per turn: a
        parse that finishes while the agent is three tool calls deep is exactly
        the case this exists for.
        """
        notices = self.notices.drain()
        content = format_notifications(notices)
        if not content:
            return
        messages.append({"role": "user", "content": content})
        log_event(
            "task_notification",
            run_id=state.run_id,
            count=len(notices),
            kinds=",".join(sorted({notice.kind for notice in notices})),
        )
        yield AgentEvent(
            type=ChatEventType.notice,
            data={
                "kind": "task_notification",
                "message": "; ".join(notice.line() for notice in notices),
                "items": [notice.as_event() for notice in notices],
                "agent_id": MAIN_AGENT_ID,
            },
        )

    async def _run_with_progress(
        self, ctx: ToolContext, plan: _ToolPlan, holder: _ResultHolder
    ) -> AsyncGenerator[AgentEvent, None]:
        """Run a slow tool while forwarding the progress it reports.

        `run_tool` is a coroutine and cannot yield, so the tool pushes progress
        onto a sink and this drains it while the task runs. Only PROGRESS_TOOLS
        take this path: polling is worth it for work measured in tens of
        seconds and pure overhead for a database read.

        The sink goes on the turn's own context rather than a copy, because
        `compare_papers` spends a budget stored there and a copy would throw
        that decrement away.
        """
        sink = EventSink()
        ctx.progress = sink
        task = asyncio.create_task(run_tool(ctx, plan.call))
        try:
            while True:
                done, _ = await asyncio.wait({task}, timeout=PROGRESS_POLL_SECONDS)
                for event in sink.drain():
                    yield event
                if done:
                    break
            holder.result = await task
        finally:
            ctx.progress = None
            if not task.done():
                task.cancel()

    async def _run_in_parallel(
        self, ctx: ToolContext, plans: list[_ToolPlan]
    ) -> dict[int, ToolOutcome]:
        """Run independent read-only tools at once, each on its own session.

        SQLAlchemy's `AsyncSession` is not safe to use from two coroutines, so
        sharing the turn's session here would corrupt it. When the session
        factory is unavailable (unit tests with a stub session) the calls fall
        back to the turn's own context, which is still correct — just serial.
        """
        maker = db.SessionLocal

        async def execute(plan: _ToolPlan) -> tuple[_ToolPlan, ToolResult, float]:
            started = time.perf_counter()
            if maker is None:
                result = await run_tool(ctx, plan.call)
            else:
                async with maker() as session:
                    result = await run_tool(replace(ctx, session=session), plan.call)
            latency_ms = (time.perf_counter() - started) * 1000
            await self.hooks.post(plan.hook_ctx, result, latency_ms)
            return plan, result, latency_ms

        gathered = await asyncio.gather(*(execute(plan) for plan in plans))
        return {
            plan.index: ToolOutcome(
                index=plan.index,
                call_id=plan.call.id,
                name=plan.call.name,
                result=result,
                decision="allow",
                ran=True,
                latency_ms=latency_ms,
            )
            for plan, result, latency_ms in gathered
        }

    async def _plan_calls(
        self,
        calls: list[ToolCall],
        state: RunState,
        depth: int,
        sink: EventSink,
        agent_id: str,
    ) -> AsyncGenerator[AgentEvent | _ToolPlan, None]:
        """Resolve budgets and hooks for one batch, streaming any hook events.

        Yields `_ToolPlan` for each call and `AgentEvent` for anything a hook
        pushed onto the sink (an approval request, for instance) so the caller
        can forward it before the hook's decision is used.
        """
        would_run = state.tool_calls_made
        for index, call in enumerate(calls):
            hook_ctx = ToolHookContext(
                call=call,
                run_id=state.run_id,
                agent_id=agent_id,
                depth=depth,
                sink=sink,
                policy=state.pool.policy if state.pool is not None else None,
            )
            if index >= MAX_TOOL_CALLS_PER_RESPONSE:
                decision = _budget_decision(
                    call, MAX_TOOL_CALLS_PER_RESPONSE, scope="response"
                )
            elif would_run >= MAX_TOOL_CALLS_PER_RUN:
                decision = _budget_decision(call, MAX_TOOL_CALLS_PER_RUN, scope="turn")
            else:
                pending = asyncio.create_task(self.hooks.pre(hook_ctx))
                while not pending.done():
                    for event in sink.drain():
                        yield event
                    if not pending.done():
                        await asyncio.sleep(PRE_HOOK_POLL_SECONDS)
                decision = await pending
                for event in sink.drain():
                    yield event
                if decision is None:
                    would_run += 1
            yield _ToolPlan(
                index=index, call=call, hook_ctx=hook_ctx, decision=decision
            )

    async def _execute_tool_batch(
        self,
        ctx: ToolContext,
        calls: list[ToolCall],
        protocol: AgentProtocol,
        state: RunState,
        messages: list[dict],
    ) -> AsyncGenerator[AgentEvent, None]:
        """Run one assistant message's tool calls and stream their events.

        Independent read-only calls run concurrently, but the events and the
        tool result messages still come out in the order the model asked for
        them: the UI timeline stays readable and every `tool_call_id` keeps its
        matching result.
        """
        for call in calls:
            yield AgentEvent(
                type=ChatEventType.tool_call,
                data={
                    "name": call.name,
                    "arguments": call.arguments,
                    "call_id": call.id,
                    "agent_id": MAIN_AGENT_ID,
                },
            )

        sink = EventSink()
        plans: list[_ToolPlan] = []
        async for item in self._plan_calls(
            calls, state, ctx.depth, sink, MAIN_AGENT_ID
        ):
            if isinstance(item, _ToolPlan):
                plans.append(item)
            else:
                yield item

        outcomes: dict[int, ToolOutcome] = {
            plan.index: ToolOutcome(
                index=plan.index,
                call_id=plan.call.id,
                name=plan.call.name,
                result=plan.decision.result,
                decision=plan.decision.decision,
                reason=plan.decision.reason,
            )
            for plan in plans
            if plan.decision is not None
        }

        runnable = [plan for plan in plans if plan.runnable]
        concurrent = [
            plan for plan in runnable if plan.call.name in PARALLEL_SAFE_TOOLS
        ]
        serial = [
            plan for plan in runnable if plan.call.name not in PARALLEL_SAFE_TOOLS
        ]

        if len(concurrent) > 1:
            outcomes.update(await self._run_in_parallel(ctx, concurrent))
        else:
            # A single call gains nothing from another connection.
            serial = sorted([*concurrent, *serial], key=lambda plan: plan.index)

        for plan in serial:
            started = time.perf_counter()
            if plan.call.name == "spawn_subagent":
                holder = SpawnHolder()
                async for event in run_spawned_subagent(
                    self, ctx, plan.call, holder, run_id=state.run_id
                ):
                    yield event
                result = holder.result or ToolResult(
                    content="spawn_subagent produced no result.",
                    summary="spawn_subagent failed",
                )
            elif plan.call.name in PROGRESS_TOOLS:
                holder = _ResultHolder()
                async for event in self._run_with_progress(ctx, plan, holder):
                    yield event
                result = holder.result or ToolResult(
                    content=f"{plan.call.name} produced no result.",
                    summary=f"{plan.call.name} failed",
                )
            else:
                result = await run_tool(ctx, plan.call)
            latency_ms = (time.perf_counter() - started) * 1000
            await self.hooks.post(plan.hook_ctx, result, latency_ms)
            outcomes[plan.index] = ToolOutcome(
                index=plan.index,
                call_id=plan.call.id,
                name=plan.call.name,
                result=result,
                decision="allow",
                ran=True,
                latency_ms=latency_ms,
            )

        for plan in plans:
            outcome = outcomes[plan.index]
            if outcome.ran:
                state.note_tool(outcome.name)
            yield AgentEvent(
                type=ChatEventType.tool_result,
                data={
                    "name": outcome.name,
                    "summary": outcome.result.summary,
                    "call_id": outcome.call_id,
                    "permission": outcome.permission_payload(),
                    "agent_id": MAIN_AGENT_ID,
                },
            )
            if outcome.result.todo_items is not None:
                yield AgentEvent(
                    type=ChatEventType.todo,
                    data={"items": outcome.result.todo_items},
                )
            if outcome.result.artifact is not None:
                yield AgentEvent(
                    type=ChatEventType.artifact, data=outcome.result.artifact
                )
            messages.append(
                protocol.tool_result_message(plan.call, outcome.result.content)
            )

    async def _maybe_consolidate(self, session: AsyncSession, state: RunState) -> None:
        """Merge duplicate extracted memories once the store grows large."""
        try:
            outcome = await consolidate_memories(
                session,
                llm_client=self.llm_client,
                llm_model=self.llm_model,
                embeddings=self.embeddings,
            )
        except Exception as exc:  # noqa: BLE001 - never fails a delivered answer
            log_warning(
                "memory_consolidate_failed", run_id=state.run_id, error=str(exc)
            )
            return
        if outcome.ran:
            log_event(
                "memory_consolidated",
                run_id=state.run_id,
                before=outcome.before,
                after=outcome.after,
                removed=outcome.removed,
            )
        elif outcome.skipped not in {"", "below"}:
            log_warning(
                "memory_consolidate_skipped",
                run_id=state.run_id,
                reason=outcome.skipped,
                records=outcome.before,
            )

    async def run(
        self, session: AsyncSession, payload: ChatRequest
    ) -> AsyncGenerator[AgentEvent, None]:
        state = RunState()
        pool = assemble_tool_pool(self.mcp, base_rules=self.policy.rules())
        state.pool = pool
        ctx = self._context(session, payload, state)
        question = payload.messages[-1].content
        try:
            recalled_rows = await search_memories(session, limit=RECALL_LOAD_LIMIT)
            selected = await recall_memories(
                session,
                recalled_rows,
                question,
                embeddings=self.embeddings,
            )
            recalled = format_recalled_memories(
                recalled_rows, question, selected=selected
            )
        except Exception as exc:  # noqa: BLE001 - memory must not block a turn
            log_warning("memory_recall_failed", run_id=state.run_id, error=str(exc))
            recalled = ""
        sections = [AGENT_PROMPT]
        catalog = skill_catalog()
        if catalog:
            sections.append(catalog)
        if recalled:
            sections.append(recalled)
        if classify_query(question) is QueryKind.field_wide:
            sections.append(FIELD_WIDE_SUFFIX)
        armed_prompt = "\n\n".join(sections)
        started = time.perf_counter()

        log_event(
            "turn_started",
            run_id=state.run_id,
            model=self.llm_model,
            prompt_version=self.prompt_version,
        )

        plan = await plan_tools(
            self.llm_client,
            self.llm_model,
            question,
            history=payload.messages[:-1],
        )
        tools_enabled = not plan.disables_tools
        base_prompt = armed_prompt
        if plan.disables_tools:
            base_prompt = f"{armed_prompt}\n\n{EMPTY_PLAN_SUFFIX}"
        protocol = self._answering_protocol(tools_enabled, pool)

        messages: list[dict] = [
            {"role": "system", "content": protocol.system_prompt(base_prompt)}
        ]
        messages.extend(
            {"role": message.role.value, "content": message.content}
            for message in payload.messages
        )
        if payload.attachments:
            from services.chat_attachments import (
                FiledAttachment,
                format_attachment_context,
            )

            filed = [
                FiledAttachment(
                    kind=item.kind,
                    attachment=item,
                    preview=item.preview,
                )
                for item in payload.attachments
            ]
            context = format_attachment_context(filed)
            if context:
                messages.append({"role": "user", "content": context})

        answer: str | None = None
        verdict: GateVerdict | None = None

        if plan.calls:
            enabled = self._protocol(pool=pool)
            messages.append(
                enabled.assistant_message(AgentResponse(text="", tool_calls=plan.calls))
            )
            async for event in self._execute_tool_batch(
                ctx, plan.calls, enabled, state, messages
            ):
                yield event

        recovery_sink = EventSink()
        for _ in range(MAX_ITERATIONS):
            async for event in self._deliver_notifications(messages, state):
                yield event
            async for event in self._maybe_compact(messages, state):
                yield event

            response = await self._complete_with_recovery(
                protocol, messages, base_prompt, state, question, recovery_sink
            )
            for event in recovery_sink.drain():
                yield event
            messages.append(protocol.assistant_message(response))
            state.record_usage(response.prompt_tokens, response.completion_tokens)

            if response.tool_calls:
                executing = protocol
                if not tools_enabled:
                    tools_enabled = True
                    protocol = self._answering_protocol(True, pool)
                    base_prompt = armed_prompt
                    messages[0] = {
                        "role": "system",
                        "content": protocol.system_prompt(base_prompt),
                    }
                async for event in self._execute_tool_batch(
                    ctx, response.tool_calls, executing, state, messages
                ):
                    yield event
                continue

            if not response.text.strip():
                break

            verdict = await self.gate.check(
                question=question,
                answer=response.text,
                citations=state.registry.citations(),
                tool_calls_made=state.tool_calls_made,
            )
            if (
                verdict.ok
                or verdict.impossible
                or state.consecutive_gate_blocks >= MAX_GATE_RETRIES
            ):
                answer = response.text
                break

            state.gate_retries += 1
            state.consecutive_gate_blocks += 1
            yield AgentEvent(
                type=ChatEventType.verdict,
                data=verdict.as_schema(GateStatus.retrying).model_dump(mode="json"),
            )
            if not tools_enabled:
                tools_enabled = True
                protocol = self._answering_protocol(True, pool)
                base_prompt = armed_prompt
                messages[0] = {
                    "role": "system",
                    "content": protocol.system_prompt(base_prompt),
                }
            messages.append({"role": "system", "content": verdict.feedback()})
        else:
            async for event in self._deliver_notifications(messages, state):
                yield event
            messages.append({"role": "user", "content": FINAL_TURN_NUDGE})
            async for event in self._maybe_compact(messages, state):
                yield event
            response = await self._complete_with_recovery(
                protocol, messages, base_prompt, state, question, recovery_sink
            )
            for event in recovery_sink.drain():
                yield event
            state.record_usage(response.prompt_tokens, response.completion_tokens)
            answer = response.text
            verdict = await self.gate.check(
                question=question,
                answer=answer,
                citations=state.registry.citations(),
                tool_calls_made=state.tool_calls_made,
            )

        citations: list[ChatCitation] = state.registry.citations()
        if answer:
            answer, citations = compact_answer_citations(answer, citations)
            yield AgentEvent(type=ChatEventType.token, data={"text": answer})
        else:
            yield AgentEvent(type=ChatEventType.error, data={"message": EMPTY_ANSWER})

        yield AgentEvent(
            type=ChatEventType.citations,
            data={"citations": [item.model_dump(mode="json") for item in citations]},
        )
        if verdict is not None:
            yield AgentEvent(
                type=ChatEventType.verdict,
                data=verdict.as_schema().model_dump(mode="json"),
            )
        if answer:
            try:
                extracted = await extract_and_store(
                    session,
                    llm_client=self.llm_client,
                    llm_model=self.llm_model,
                    user_text=question,
                    answer=answer,
                    source_turn=EXTRACT_SOURCE_TURN,
                    embeddings=self.embeddings,
                )
            except Exception as exc:  # noqa: BLE001 - extraction is best effort
                log_warning(
                    "memory_extract_failed", run_id=state.run_id, error=str(exc)
                )
                extracted = []
            if extracted:
                # Only worth checking after a write; the store cannot have
                # crossed the threshold otherwise.
                await self._maybe_consolidate(session, state)
            for row in extracted:
                yield AgentEvent(
                    type=ChatEventType.artifact,
                    data={
                        "kind": ArtifactKind.memory.value,
                        "tool": "memory_write",
                        "data": {
                            "action": "remembered",
                            "key": row.key,
                            "content": row.content,
                            "category": row.category,
                        },
                    },
                )
        latency_ms = (time.perf_counter() - started) * 1000
        gate_status = None
        if verdict is not None:
            gate_status = verdict.as_schema().status.value
        log_chat_trace(
            question=question,
            tools_called=state.tools_called,
            citations=[item.model_dump(mode="json") for item in citations],
            answer=answer or "",
            latency_ms=latency_ms,
            gate_status=gate_status,
            request_id=state.run_id,
            prompt_tokens=state.reported_prompt_tokens,
            completion_tokens=state.reported_completion_tokens,
        )
        log_event(
            "turn_finished",
            run_id=state.run_id,
            latency_ms=round(latency_ms, 1),
            tools=len(state.tools_called),
            citations=len(citations),
            gate_status=gate_status,
            prompt_tokens=state.reported_prompt_tokens,
            completion_tokens=state.reported_completion_tokens,
        )
        yield AgentEvent(
            type=ChatEventType.done,
            data={
                "model": self.llm_model,
                "prompt_version": self.prompt_version,
                "request_id": state.run_id,
                "latency_ms": round(latency_ms, 1),
                "prompt_tokens": state.reported_prompt_tokens,
                "completion_tokens": state.reported_completion_tokens,
            },
        )
