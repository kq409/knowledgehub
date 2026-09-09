"""State that belongs to one turn, not to the agent.

`ResearchAgent` is built once at startup and shared by every request
(`app.state.agent`). Anything a single turn produces therefore cannot live on
the instance: two chat requests interleave at every `await`, and the second one
would overwrite what the first is about to read. `RunState` is created inside
`ResearchAgent.run` and passed down, so concurrent turns never see each other.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field

from services.agent.todo import TodoStore
from services.agent.tool_pool import ToolPool
from services.agent.tool_results import ToolResultStore
from services.agent.tools import CitationRegistry, ToolResult


@dataclass
class RunState:
    """Everything one `ResearchAgent.run` call accumulates."""

    run_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    registry: CitationRegistry = field(default_factory=CitationRegistry)
    todo_store: TodoStore = field(default_factory=TodoStore)
    result_store: ToolResultStore = field(default_factory=ToolResultStore)
    pool: ToolPool | None = None

    tools_called: list[str] = field(default_factory=list)
    tool_calls_made: int = 0

    prompt_tokens: int = 0
    completion_tokens: int = 0
    saw_usage: bool = False

    gate_retries: int = 0
    consecutive_gate_blocks: int = 0

    def note_tool(self, name: str) -> None:
        """Count a tool that actually ran (refusals and skips do not count)."""
        self.tool_calls_made += 1
        self.tools_called.append(name)

    def record_usage(
        self, prompt_tokens: int | None, completion_tokens: int | None
    ) -> None:
        if prompt_tokens is not None:
            self.prompt_tokens += prompt_tokens
            self.saw_usage = True
        if completion_tokens is not None:
            self.completion_tokens += completion_tokens
            self.saw_usage = True

    @property
    def reported_prompt_tokens(self) -> int | None:
        """None when no provider reported usage, so the UI can hide the row."""
        return self.prompt_tokens if self.saw_usage else None

    @property
    def reported_completion_tokens(self) -> int | None:
        return self.completion_tokens if self.saw_usage else None


@dataclass(frozen=True)
class ToolOutcome:
    """One finished tool call, waiting to be turned into events and messages."""

    index: int
    call_id: str
    name: str
    result: ToolResult
    decision: str
    reason: str = ""
    ran: bool = False
    latency_ms: float = 0.0

    def permission_payload(self) -> dict:
        return {"decision": self.decision, "reason": self.reason}
