"""First-turn tool planner: which tools this message actually needs.

The planner never executes tools and never answers the researcher. It only
returns a list of tool calls (possibly empty). An empty list means the main
loop should answer without library tools.
"""

from __future__ import annotations

import asyncio
import re
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from openai import OpenAI

from services.agent.protocol import ToolCall
from services.agent.telemetry import log_warning
from services.agent.tools import (
    TOOL_HANDLERS,
    TOOL_SCHEMAS,
    infer_library_sources,
    tool_catalog,
)
from services.llm_chat import complete_json_object, effort_from_env

PROMPT_FILE = Path(__file__).resolve().parent.parent.parent / "chat_planner_prompt.txt"
PLANNER_PROMPT = PROMPT_FILE.read_text().strip()

PLANNER_MAX_TOKENS = 800
PLANNER_TEMPERATURE = 0.0
PLANNER_MAX_TOOLS = 3
HISTORY_TURNS = 6
HISTORY_CHARS = 400

_TOOL_NAMES = tuple(sorted(TOOL_HANDLERS))

_ACRONYM = re.compile(r"\b([A-Z]{3,8})\b")
_CLARIFY = re.compile(
    r"^(?:it'?s|that(?:'s| is)|i mean(?:t)?)\s+(.+)$",
    re.IGNORECASE | re.DOTALL,
)
_NAMED_LOOKUP = re.compile(
    r"(?:summar(?:y|ise|ize)|explain|overview|describe|"
    r"what(?:'s| is)|tell me about)\s+(?:of\s+|on\s+|for\s+)?(.+)$",
    re.IGNORECASE | re.DOTALL,
)
_SKIP_ACRONYMS = frozenset(
    {"PDF", "DOI", "URL", "HTTP", "HTTPS", "JSON", "HTML", "LLM"}
)
_INVENTORY = re.compile(
    r"(?:"
    r"\b(?:library|collection)\s+status\b|"
    r"\bstatus\s+of\s+(?:my|the|this)\s+library\b|"
    r"\boverview\b.{0,40}\blibrary\b|"
    r"\blibrary\b.{0,40}\boverview\b|"
    r"\b(?:what(?:'s|s)?|whats)\s+in\s+(?:my|the|this)\s+library\b|"
    r"\bwhat\s+(?:is|are)\s+in\s+(?:my|the|this)\s+library\b|"
    r"\bwhat\s+does\s+(?:my|the|this)\s+library\s+(?:hold|contain|have)\b|"
    r"\b(?:library|collection)\s+(?:contents?|inventory|holdings?|coverage)\b|"
    r"\bhow\s+many\s+(?:papers?|notes?|documents?|items?)\b|"
    r"\bcheck\s+(?:my|the|this)\s+library\b|"
    r"\blist\s+(?:my|the\s+)?(?:papers?|notes?|documents?|library)\b|"
    r"\bwhat\s+(?:papers?|notes?|documents?)\s+(?:do\s+i|are\s+in)\b"
    r")",
    re.IGNORECASE | re.DOTALL,
)
INVENTORY_TOOLS = ("list_papers", "list_notes", "list_documents")


@dataclass(frozen=True)
class ToolPlan:
    """Calls to run before the answer loop. `failed` means skip the plan."""

    calls: list[ToolCall] = field(default_factory=list)
    failed: bool = False

    @property
    def disables_tools(self) -> bool:
        """True when the planner decided no tool result is needed."""
        return not self.failed and not self.calls


def planner_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "tools": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "enum": list(_TOOL_NAMES)},
                        "input": {"type": "object"},
                    },
                    "required": ["name"],
                },
            }
        },
        "required": ["tools"],
    }


def calls_from_payload(
    payload: dict,
    *,
    allowed: set[str] | frozenset[str] | None = None,
    limit: int = PLANNER_MAX_TOOLS,
) -> list[ToolCall]:
    """Turn planner JSON into ToolCalls, dropping unknown names."""
    allowed_names = allowed if allowed is not None else set(TOOL_HANDLERS)
    raw = payload.get("tools")
    if not isinstance(raw, list):
        return []

    calls: list[ToolCall] = []
    for item in raw:
        if len(calls) >= limit:
            break
        if not isinstance(item, dict):
            continue
        name = item.get("name") or item.get("tool")
        if not isinstance(name, str) or name not in allowed_names:
            continue
        arguments = item.get("input")
        if arguments is None:
            arguments = item.get("arguments") or item.get("parameters") or {}
        if not isinstance(arguments, dict):
            arguments = {}
        calls.append(
            ToolCall(
                id=f"plan_{uuid.uuid4().hex[:12]}",
                name=name,
                arguments=arguments,
            )
        )
    return calls


def fallback_search_query(question: str) -> str | None:
    """A library search string when the planner skipped a named lookup."""
    text = " ".join(question.strip().split())
    if not text:
        return None

    clarified = _CLARIFY.match(text)
    if clarified:
        name = re.split(r"[.]", clarified.group(1).strip(), maxsplit=1)[0].strip(" ?!")
        if len(name) >= 3:
            return name

    acronyms = [
        token for token in _ACRONYM.findall(text) if token not in _SKIP_ACRONYMS
    ]
    if acronyms:
        return acronyms[-1]

    if re.search(r"\b(?:my|the|this)\s+library\b", text, re.IGNORECASE):
        return None

    named = _NAMED_LOOKUP.search(text)
    if named:
        name = re.split(r"[.]", named.group(1).strip(), maxsplit=1)[0].strip(" ?!")
        words = name.split()
        if 1 <= len(words) <= 12 and len(name) >= 3 and "library" not in name.lower():
            return name
    return None


def apply_inferred_sources(plan: ToolPlan, question: str) -> ToolPlan:
    """Fill search_library.sources from the question when the plan omitted them."""
    if plan.failed or not plan.calls:
        return plan
    sources = infer_library_sources(question)
    if sources is None:
        return plan
    calls: list[ToolCall] = []
    for call in plan.calls:
        if call.name == "search_library" and "sources" not in call.arguments:
            calls.append(
                ToolCall(
                    id=call.id,
                    name=call.name,
                    arguments={**call.arguments, "sources": sources},
                )
            )
        else:
            calls.append(call)
    return ToolPlan(calls=calls)


def ensure_library_search(plan: ToolPlan, question: str) -> ToolPlan:
    """Search the library when the planner returned nothing for a named work."""
    if plan.failed or plan.calls:
        return plan
    query = fallback_search_query(question)
    if not query:
        return plan
    arguments: dict[str, object] = {"query": query}
    sources = infer_library_sources(question)
    if sources is not None:
        arguments["sources"] = sources
    return ToolPlan(
        calls=[
            ToolCall(
                id=f"plan_{uuid.uuid4().hex[:12]}",
                name="search_library",
                arguments=arguments,
            )
        ]
    )


def is_library_inventory_question(question: str) -> bool:
    """True when the message asks what the library holds, not a named work."""
    return bool(_INVENTORY.search(question))


def ensure_library_inventory(plan: ToolPlan, question: str) -> ToolPlan:
    """List papers, notes, and documents when an empty plan asked for status."""
    if plan.failed or plan.calls:
        return plan
    if not is_library_inventory_question(question):
        return plan
    return ToolPlan(
        calls=[
            ToolCall(
                id=f"plan_{uuid.uuid4().hex[:12]}",
                name=name,
                arguments={},
            )
            for name in INVENTORY_TOOLS
        ]
    )


def _history_block(history: Sequence[Any] | None) -> str:
    if not history:
        return ""
    lines: list[str] = []
    for message in list(history)[-HISTORY_TURNS:]:
        role = getattr(message, "role", None)
        content = getattr(message, "content", None)
        if role is None or not isinstance(content, str):
            continue
        label = role.value if hasattr(role, "value") else str(role)
        text = " ".join(content.split())
        if len(text) > HISTORY_CHARS:
            text = text[:HISTORY_CHARS].rstrip() + "…"
        if text:
            lines.append(f"{label}: {text}")
    if not lines:
        return ""
    return "Earlier in this conversation:\n" + "\n".join(lines) + "\n\n"


async def plan_tools(
    llm_client: OpenAI,
    llm_model: str,
    question: str,
    *,
    catalog: str | None = None,
    history: Sequence[Any] | None = None,
) -> ToolPlan:
    """Ask the model which tools this message needs. Failures skip the plan."""
    listing = catalog if catalog is not None else tool_catalog(TOOL_SCHEMAS)
    earlier = _history_block(history)
    messages = [
        {"role": "system", "content": PLANNER_PROMPT},
        {
            "role": "user",
            "content": f"{earlier}Message:\n{question}\n\nTools:\n{listing}",
        },
    ]
    effort = effort_from_env(
        "PLANNER_REASONING_EFFORT", "JUDGE_REASONING_EFFORT", default="none"
    )
    try:
        payload = await asyncio.to_thread(
            complete_json_object,
            llm_client,
            messages=messages,
            model=llm_model,
            schema=planner_schema(),
            temperature=PLANNER_TEMPERATURE,
            max_tokens=PLANNER_MAX_TOKENS,
            effort=effort,
        )
    except Exception as exc:  # noqa: BLE001 - planner must not take chat down
        log_warning("planner_failed", error=str(exc))
        return ToolPlan(failed=True)

    if not isinstance(payload, dict):
        return ToolPlan(failed=True)
    plan = ensure_library_search(ToolPlan(calls=calls_from_payload(payload)), question)
    plan = ensure_library_inventory(plan, question)
    return apply_inferred_sources(plan, question)
