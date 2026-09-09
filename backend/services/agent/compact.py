"""Shrink the conversation when tool results pile up.

Two stages run ahead of every model call: cheap truncation of older tool
payloads, then (if still over the budget) one LLM summary of the middle of the
history. The loop streams a `compact` event when anything changed.

A character count can only estimate what a tokenizer will do, so the provider
may still reject a request as too long. `reactive_compact` is the recovery for
that case: it keeps only the newest messages and summarises everything before
them. It is deliberately more aggressive than the proactive path, because by
the time it runs the alternative is losing the turn.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass

from openai import OpenAI

from services.agent.tool_results import ToolResultStore, shortened_notice

COMPACT_CHAR_THRESHOLD = 24000
SUBAGENT_COMPACT_CHAR_THRESHOLD = 12000
KEEP_RECENT_TOOL_RESULTS = 2
TRUNCATED_TOOL_CHARS = 240
SUMMARY_MAX_TOKENS = 600
TEMPERATURE = 0.0

# How many of the newest messages `reactive_compact` keeps verbatim.
REACTIVE_KEEP_RECENT = 5

SUMMARY_PROMPT = (
    "Summarise the following research-assistant conversation middle so a "
    "follow-up model turn can continue without the full tool payloads. Keep "
    "paper titles, note titles, citation numbers [n], and concrete findings. "
    "Drop boilerplate. Reply with plain prose only."
)


@dataclass(frozen=True)
class CompactResult:
    messages: list[dict]
    mode: str | None
    before_chars: int
    after_chars: int

    @property
    def changed(self) -> bool:
        return self.mode is not None


def message_chars(messages: list[dict]) -> int:
    total = 0
    for message in messages:
        content = message.get("content")
        if isinstance(content, str):
            total += len(content)
        elif content is not None:
            total += len(json.dumps(content, ensure_ascii=False))
        tool_calls = message.get("tool_calls")
        if tool_calls:
            total += len(json.dumps(tool_calls, ensure_ascii=False))
        reasoning = message.get("reasoning_content") or message.get("reasoning")
        if isinstance(reasoning, str):
            total += len(reasoning)
    return total


def _is_tool_payload(message: dict) -> bool:
    role = message.get("role")
    if role == "tool":
        return True
    if role == "user" and isinstance(message.get("content"), str):
        return str(message["content"]).startswith("Result of ")
    return False


def _shorten_tool_payload(message: dict, store: ToolResultStore | None) -> dict:
    """Compress an older tool payload, keeping a pointer to the full text.

    A stub with no id is a dead end: the model can see that something was cut
    but not how much, or how to get it. With a store it keeps the full payload
    and the stub names the id to ask for.
    """
    content = message.get("content")
    if not isinstance(content, str) or len(content) <= TRUNCATED_TOOL_CHARS:
        return message
    shortened = content[:TRUNCATED_TOOL_CHARS].rstrip()
    call_id = message.get("tool_call_id")
    if store is not None and isinstance(call_id, str) and call_id:
        store.put(call_id, "tool", content)
        entry = store.get(call_id)
        total = entry.total_chars if entry is not None else len(content)
        return {**message, "content": shortened + shortened_notice(call_id, total)}
    return {**message, "content": shortened + "…"}


def repair_tool_pairing(messages: list[dict]) -> list[dict]:
    """Ensure every assistant `tool_call` has a following tool result.

    OpenAI-compatible providers reject the next request if an assistant
    message with `tool_calls` is not followed by a `role: tool` message for
    each `tool_call_id`. The loop must answer every call; this is the safety
    net when compaction or a budget slice would otherwise leave a gap.
    """
    repaired: list[dict] = []
    index = 0
    while index < len(messages):
        message = messages[index]
        repaired.append(message)
        calls = (
            message.get("tool_calls") if message.get("role") == "assistant" else None
        )
        if not isinstance(calls, list) or not calls:
            index += 1
            continue

        needed: list[object] = []
        for call in calls:
            if isinstance(call, dict) and call.get("id"):
                needed.append(call["id"])

        index += 1
        seen: set[str] = set()
        while index < len(messages) and messages[index].get("role") == "tool":
            tool_message = messages[index]
            call_id = tool_message.get("tool_call_id")
            if call_id in needed and call_id not in seen:
                repaired.append(tool_message)
                seen.add(call_id)
            index += 1

        for call in calls:
            if not isinstance(call, dict):
                continue
            call_id = call.get("id")
            if not call_id or call_id in seen:
                continue
            function = (
                call.get("function") if isinstance(call.get("function"), dict) else {}
            )
            name = function.get("name") or "tool"
            repaired.append(
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": (
                        f"{name} was not run (missing tool result). "
                        "Continue with the evidence you have."
                    ),
                }
            )
    return repaired


def truncate_old_tool_results(
    messages: list[dict], store: ToolResultStore | None = None
) -> list[dict]:
    """Keep recent tool payloads intact; compress older ones to a short stub."""
    tool_indexes = [
        i for i, message in enumerate(messages) if _is_tool_payload(message)
    ]
    if len(tool_indexes) <= KEEP_RECENT_TOOL_RESULTS:
        return list(messages)

    keep = set(tool_indexes[-KEEP_RECENT_TOOL_RESULTS:])
    return [
        (
            _shorten_tool_payload(message, store)
            if i not in keep and _is_tool_payload(message)
            else message
        )
        for i, message in enumerate(messages)
    ]


def _summarize_sync(client: OpenAI, model: str, body: str) -> str:
    from services.llm_chat import effort_from_env, message_text, with_chat_extras

    effort = effort_from_env(
        "COMPACT_REASONING_EFFORT", "LLM_REASONING_EFFORT", default="none"
    )
    completion = client.chat.completions.create(
        **with_chat_extras(
            {
                "model": model,
                "messages": [
                    {"role": "system", "content": SUMMARY_PROMPT},
                    {"role": "user", "content": body},
                ],
                "temperature": TEMPERATURE,
                "max_tokens": SUMMARY_MAX_TOKENS,
                "stream": False,
            },
            effort=effort,
        )
    )
    return message_text(completion.choices[0].message)


def _middle_for_summary(
    messages: list[dict],
) -> tuple[list[dict], list[dict], list[dict]]:
    """Split into [system…], middle, [tail]. Tail keeps the last user question."""
    if not messages:
        return [], [], []

    head_end = 1 if messages[0].get("role") == "system" else 0
    # Keep the final user message (and anything after it) as the live tail.
    tail_start = len(messages)
    for index in range(len(messages) - 1, head_end - 1, -1):
        if messages[index].get("role") == "user" and not _is_tool_payload(
            messages[index]
        ):
            tail_start = index
            break

    head = messages[:head_end]
    middle = messages[head_end:tail_start]
    tail = messages[tail_start:]
    return head, middle, tail


def _format_middle(middle: list[dict]) -> str:
    blocks: list[str] = []
    for message in middle:
        role = message.get("role", "?")
        content = message.get("content")
        if isinstance(content, str) and content.strip():
            blocks.append(f"{role}: {content}")
        elif message.get("tool_calls"):
            names = [
                call.get("function", {}).get("name", "?")
                for call in message["tool_calls"]
            ]
            blocks.append(f"{role}: tool_calls {names}")
    return "\n\n".join(blocks)


async def compact_messages(
    messages: list[dict],
    *,
    threshold: int = COMPACT_CHAR_THRESHOLD,
    llm_client: OpenAI | None = None,
    llm_model: str | None = None,
    store: ToolResultStore | None = None,
) -> CompactResult:
    """Return possibly-shrunk messages and whether/how compaction ran."""
    messages = repair_tool_pairing(list(messages))
    before = message_chars(messages)
    if before <= threshold:
        return CompactResult(
            messages=list(messages),
            mode=None,
            before_chars=before,
            after_chars=before,
        )

    truncated = truncate_old_tool_results(messages, store)
    after_truncate = message_chars(truncated)
    if after_truncate <= threshold:
        return CompactResult(
            messages=truncated,
            mode="truncate",
            before_chars=before,
            after_chars=after_truncate,
        )

    if llm_client is None or not llm_model:
        return CompactResult(
            messages=truncated,
            mode="truncate",
            before_chars=before,
            after_chars=after_truncate,
        )

    head, middle, tail = _middle_for_summary(truncated)
    if not middle:
        return CompactResult(
            messages=truncated,
            mode="truncate",
            before_chars=before,
            after_chars=after_truncate,
        )

    body = _format_middle(middle)
    try:
        summary = await asyncio.to_thread(_summarize_sync, llm_client, llm_model, body)
    except Exception:
        return CompactResult(
            messages=truncated,
            mode="truncate",
            before_chars=before,
            after_chars=after_truncate,
        )

    if not summary:
        return CompactResult(
            messages=truncated,
            mode="truncate",
            before_chars=before,
            after_chars=after_truncate,
        )

    compacted = repair_tool_pairing(
        [
            *head,
            {
                "role": "user",
                "content": f"[compacted context]: {summary}",
            },
            *tail,
        ]
    )
    return CompactResult(
        messages=compacted,
        mode="summarize",
        before_chars=before,
        after_chars=message_chars(compacted),
    )


def _reactive_tail_start(messages: list[dict], keep: int) -> int:
    """Where the verbatim tail may begin without orphaning a tool result.

    A `role: tool` message whose assistant `tool_calls` were summarised away
    makes the next request invalid, so the cut moves back until it lands on
    something that can stand alone.
    """
    tail_start = max(0, len(messages) - keep)
    while tail_start > 0 and messages[tail_start].get("role") == "tool":
        tail_start -= 1
    return tail_start


async def reactive_compact(
    messages: list[dict],
    *,
    active_request: str,
    llm_client: OpenAI | None = None,
    llm_model: str | None = None,
    store: ToolResultStore | None = None,
) -> CompactResult:
    """Recover from a provider rejecting the request as too long.

    `active_request` is passed in explicitly because tool results also use
    `role: user`: after repeated compaction there is no reliable way to find
    the researcher's actual question by scanning the history.
    """
    messages = repair_tool_pairing(list(messages))
    before = message_chars(messages)

    head_end = 1 if messages and messages[0].get("role") == "system" else 0
    tail_start = _reactive_tail_start(messages, REACTIVE_KEEP_RECENT)
    if tail_start <= head_end:
        # Nothing but the system prompt and the tail; the tail itself is the
        # problem and only truncating tool payloads can help.
        truncated = truncate_old_tool_results(messages, store)
        return CompactResult(
            messages=truncated,
            mode="truncate",
            before_chars=before,
            after_chars=message_chars(truncated),
        )

    head = messages[:head_end]
    older = messages[head_end:tail_start]
    tail = messages[tail_start:]

    summary = ""
    if llm_client is not None and llm_model:
        try:
            summary = await asyncio.to_thread(
                _summarize_sync, llm_client, llm_model, _format_middle(older)
            )
        except Exception:
            summary = ""
    if not summary:
        summary = (
            f"{len(older)} earlier message(s) were dropped to fit the context "
            "window. Re-read anything you still need with a tool."
        )

    rebuilt = repair_tool_pairing(
        [
            *head,
            {
                "role": "user",
                "content": (
                    f"[reactive compact] Current researcher request: "
                    f"{active_request}\n\nEarlier conversation: {summary}"
                ),
            },
            *tail,
        ]
    )
    return CompactResult(
        messages=rebuilt,
        mode="reactive",
        before_chars=before,
        after_chars=message_chars(rebuilt),
    )
