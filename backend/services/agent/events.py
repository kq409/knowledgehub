"""The event type the loop streams, and a queue hooks can push onto.

`AgentEvent` lives here rather than in `loop.py` so a hook can build one
without importing the loop that calls it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from schemas import ChatEventType


@dataclass(frozen=True)
class AgentEvent:
    type: ChatEventType
    data: dict = field(default_factory=dict)


class EventSink:
    """Collects events produced outside the generator that yields them.

    A hook cannot `yield` — it is called, not iterated. It pushes here instead
    and the loop drains the sink at the next point where it can emit.
    """

    def __init__(self) -> None:
        self._events: list[AgentEvent] = []

    def emit(self, event: AgentEvent) -> None:
        self._events.append(event)

    def drain(self) -> list[AgentEvent]:
        drained = self._events
        self._events = []
        return drained
