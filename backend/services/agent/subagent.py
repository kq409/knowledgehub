"""Nested research pass with its own messages and citation registry.

The parent agent calls `spawn_subagent`; this module runs a shallow copy of the
main loop (no evidence gate, no further spawning) and streams nested SSE
events under a child `agent_id`.
"""

from __future__ import annotations

import re
import time
import uuid
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from typing import TYPE_CHECKING

from schemas import ChatEventType, SubagentStatus
from services.agent.compact import (
    SUBAGENT_COMPACT_CHAR_THRESHOLD,
    compact_messages,
    repair_tool_pairing,
)
from services.agent.events import AgentEvent, EventSink
from services.agent.hooks import ToolHookContext, default_hooks
from services.agent.permissions import subagent_policy
from services.agent.protocol import ProtocolSwitchedError
from services.agent.tools import (
    SUBAGENT_TOOL_NAMES,
    CitationRegistry,
    ToolContext,
    ToolError,
    ToolResult,
    _as_uuid,
    schemas_for_tools,
)

if TYPE_CHECKING:
    from services.agent.loop import ResearchAgent
    from services.agent.protocol import ToolCall

MAX_SUBAGENT_ITERATIONS = 4
MAX_SUBAGENT_TOOL_CALLS_PER_TURN = 2
SUBAGENT_SUMMARY_CHARS = 1500
SUBAGENT_CALLS_PER_TURN = 1

_CITATION_RE = re.compile(r"\[(\d+)\]")

SUBAGENT_PROMPT = """
You are a nested research pass inside KnowledgeHub. Complete only the goal
below using the library tools you have. Cite with the [n] numbers from your
tool results. When finished, write a short factual summary for the parent
assistant — not a letter to the researcher. Do not claim field-wide SOTA.
""".strip()


@dataclass
class SpawnHolder:
    """Filled by `run_spawned_subagent` so the parent loop can append a result."""

    result: ToolResult | None = None


def _remap_citations(text: str, mapping: dict[int, int]) -> str:
    def replace(match: re.Match[str]) -> str:
        child = int(match.group(1))
        parent = mapping.get(child)
        if parent is None:
            return match.group(0)
        return f"[{parent}]"

    return _CITATION_RE.sub(replace, text)


def _event(event_type: ChatEventType, data: dict, *, agent_id: str) -> AgentEvent:
    payload = {**data, "agent_id": agent_id}
    return AgentEvent(type=event_type, data=payload)


async def run_spawned_subagent(
    parent: ResearchAgent,
    parent_ctx: ToolContext,
    call: ToolCall,
    holder: SpawnHolder,
    *,
    run_id: str = "",
) -> AsyncGenerator[AgentEvent, None]:
    """Stream nested work for one `spawn_subagent` call; set holder.result."""
    from services.agent.loop import (
        MAX_TOOL_RESULT_CHARS,
        budget_skip_result,
        run_tool,
    )

    goal = str(call.arguments.get("goal") or "").strip()
    if not goal:
        holder.result = ToolResult(
            content="spawn_subagent needs a non-empty goal.",
            summary="spawn_subagent failed: empty goal",
        )
        return

    if parent_ctx.depth > 0:
        holder.result = ToolResult(
            content=(
                "Nested agents cannot spawn further agents. "
                "Finish this goal with the tools you have."
            ),
            summary="spawn_subagent refused: nesting depth",
        )
        return

    if parent_ctx.subagent_budget <= 0:
        holder.result = ToolResult(
            content=(
                "This turn already used its subagent budget (one nested pass). "
                "Answer from what you have, or dig in with read_paper yourself."
            ),
            summary="spawn_subagent refused: budget exhausted",
        )
        return

    paper_hint = ""
    raw_paper = call.arguments.get("paper_id")
    if raw_paper is not None and str(raw_paper).strip():
        try:
            paper_id = _as_uuid(raw_paper, field_name="paper_id")
        except ToolError as exc:
            holder.result = ToolResult(
                content=str(exc),
                summary="spawn_subagent failed: bad paper_id",
            )
            return
        paper_hint = (
            f"\n\nFocus on paper_id={paper_id}. Prefer read_paper on that id "
            "after list_papers confirms it exists."
        )

    parent_ctx.subagent_budget -= 1
    agent_id = f"sub_{uuid.uuid4().hex[:8]}"
    child_registry = CitationRegistry()
    child_ctx = ToolContext(
        session=parent_ctx.session,
        embeddings=parent_ctx.embeddings,
        registry=child_registry,
        include_papers=parent_ctx.include_papers,
        include_voice_notes=parent_ctx.include_voice_notes,
        include_handwritten_notes=parent_ctx.include_handwritten_notes,
        top_k=parent_ctx.top_k,
        compare=None,
        extraction=None,
        todo_store=None,
        depth=parent_ctx.depth + 1,
        subagent_budget=0,
    )

    # The nested pass gets its own hook chain: a read-only policy, and the same
    # metrics hook, so its tool calls show up in the logs tagged with agent_id.
    hooks = default_hooks(subagent_policy())
    sink = EventSink()
    schemas = schemas_for_tools(SUBAGENT_TOOL_NAMES)
    protocol = parent._protocol(tool_schemas=schemas, allowed_tools=SUBAGENT_TOOL_NAMES)
    base_prompt = SUBAGENT_PROMPT + f"\n\nGoal: {goal}" + paper_hint
    messages: list[dict] = [
        {"role": "system", "content": protocol.system_prompt(base_prompt)},
        {"role": "user", "content": goal},
    ]

    yield AgentEvent(
        type=ChatEventType.subagent,
        data={
            "id": agent_id,
            "status": SubagentStatus.started.value,
            "goal": goal,
        },
    )

    answer = ""
    try:
        for _ in range(MAX_SUBAGENT_ITERATIONS):
            compact = await compact_messages(
                messages,
                threshold=SUBAGENT_COMPACT_CHAR_THRESHOLD,
                llm_client=parent.llm_client,
                llm_model=parent.llm_model,
            )
            if compact.changed:
                messages = compact.messages
                yield AgentEvent(
                    type=ChatEventType.compact,
                    data={
                        "mode": compact.mode,
                        "before_chars": compact.before_chars,
                        "after_chars": compact.after_chars,
                        "agent_id": agent_id,
                    },
                )

            messages[:] = repair_tool_pairing(messages)
            try:
                response = await protocol.complete(messages)
            except ProtocolSwitchedError:
                messages[0] = {
                    "role": "system",
                    "content": protocol.system_prompt(base_prompt),
                }
                response = await protocol.complete(messages)

            messages.append(protocol.assistant_message(response))

            if response.tool_calls:
                for index, child_call in enumerate(response.tool_calls):
                    skipped = index >= MAX_SUBAGENT_TOOL_CALLS_PER_TURN
                    yield _event(
                        ChatEventType.tool_call,
                        {
                            "name": child_call.name,
                            "arguments": child_call.arguments,
                            "call_id": child_call.id,
                        },
                        agent_id=agent_id,
                    )
                    if skipped:
                        result = budget_skip_result(
                            child_call,
                            MAX_SUBAGENT_TOOL_CALLS_PER_TURN,
                            scope="nested pass",
                        )
                        permission_payload = {
                            "decision": "skip",
                            "reason": (
                                f"this nested pass already used its "
                                f"{MAX_SUBAGENT_TOOL_CALLS_PER_TURN}-tool budget"
                            ),
                        }
                    else:
                        hook_ctx = ToolHookContext(
                            call=child_call,
                            run_id=run_id,
                            agent_id=agent_id,
                            depth=child_ctx.depth,
                            sink=sink,
                        )
                        decision = await hooks.pre(hook_ctx)
                        for pending in sink.drain():
                            yield pending
                        if decision is None:
                            started = time.perf_counter()
                            result = await run_tool(child_ctx, child_call)
                            await hooks.post(
                                hook_ctx,
                                result,
                                (time.perf_counter() - started) * 1000,
                            )
                            permission_payload = {"decision": "allow", "reason": ""}
                        else:
                            result = decision.result
                            permission_payload = {
                                "decision": decision.decision,
                                "reason": decision.reason,
                            }
                    yield _event(
                        ChatEventType.tool_result,
                        {
                            "name": child_call.name,
                            "summary": result.summary,
                            "call_id": child_call.id,
                            "permission": permission_payload,
                        },
                        agent_id=agent_id,
                    )
                    messages.append(
                        protocol.tool_result_message(child_call, result.content)
                    )
                continue

            answer = response.text.strip()
            break
        else:
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "You have reached the nested tool budget. Summarise "
                        "what you found now."
                    ),
                }
            )
            response = await protocol.complete(messages)
            answer = response.text.strip()
    except Exception as exc:
        yield AgentEvent(
            type=ChatEventType.subagent,
            data={
                "id": agent_id,
                "status": SubagentStatus.failed.value,
                "goal": goal,
                "summary": str(exc),
            },
        )
        holder.result = ToolResult(
            content=f"Nested research pass failed: {exc}",
            summary=f"spawn_subagent failed: {exc}",
        )
        return

    if not answer:
        answer = "The nested pass returned no summary."

    mapping = parent_ctx.registry.merge_registry(child_registry)
    remapped = _remap_citations(answer, mapping)
    if len(remapped) > SUBAGENT_SUMMARY_CHARS:
        remapped = (
            remapped[:SUBAGENT_SUMMARY_CHARS].rstrip()
            + "\n\n[truncated nested summary]"
        )

    cite_note = ""
    if mapping:
        pairs = ", ".join(
            f"child [{child}]→[{parent}]" for child, parent in sorted(mapping.items())
        )
        cite_note = (
            f"\n\nCitation numbers above are remapped into your turn's registry "
            f"({pairs}). Cite those parent numbers in your answer."
        )

    content = f"Nested research summary for goal: {goal}\n\n{remapped}{cite_note}"
    if len(content) > MAX_TOOL_RESULT_CHARS:
        content = (
            content[:MAX_TOOL_RESULT_CHARS].rstrip()
            + "\n\n[truncated: ask for a narrower nested goal if you need more]"
        )

    holder.result = ToolResult(
        content=content,
        summary=f"Nested pass: {goal[:80]}" + ("…" if len(goal) > 80 else ""),
    )
    yield AgentEvent(
        type=ChatEventType.subagent,
        data={
            "id": agent_id,
            "status": SubagentStatus.finished.value,
            "goal": goal,
            "summary": remapped[:400],
        },
    )
