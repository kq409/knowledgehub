"""First-turn tool planner: which tools this message actually needs.

The planner never executes tools and never answers the researcher. It only
returns a list of tool calls (possibly empty). An empty list means the main
loop should answer without library tools.
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from openai import OpenAI

from services.agent.protocol import ToolCall
from services.agent.telemetry import log_warning
from services.agent.tools import TOOL_HANDLERS, TOOL_SCHEMAS, tool_catalog
from services.llm_chat import complete_json_object, effort_from_env

PROMPT_FILE = Path(__file__).resolve().parent.parent.parent / "chat_planner_prompt.txt"
PLANNER_PROMPT = PROMPT_FILE.read_text().strip()

PLANNER_MAX_TOKENS = 800
PLANNER_TEMPERATURE = 0.0
PLANNER_MAX_TOOLS = 3

_TOOL_NAMES = tuple(sorted(TOOL_HANDLERS))


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


async def plan_tools(
    llm_client: OpenAI,
    llm_model: str,
    question: str,
    *,
    catalog: str | None = None,
) -> ToolPlan:
    """Ask the model which tools this message needs. Failures skip the plan."""
    listing = catalog if catalog is not None else tool_catalog(TOOL_SCHEMAS)
    messages = [
        {"role": "system", "content": PLANNER_PROMPT},
        {
            "role": "user",
            "content": f"Message:\n{question}\n\nTools:\n{listing}",
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
    return ToolPlan(calls=calls_from_payload(payload))
