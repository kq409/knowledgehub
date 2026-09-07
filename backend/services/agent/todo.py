"""In-turn todo list the agent keeps for multi-step research work.

The store lives for one `ResearchAgent.run` call. Nothing is written to the
database; the UI learns about updates through SSE `todo` events.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from enum import Enum


class TodoStatus(str, Enum):
    pending = "pending"
    in_progress = "in_progress"
    completed = "completed"
    cancelled = "cancelled"


MAX_TODO_ITEMS = 12


@dataclass
class TodoItem:
    id: str
    content: str
    status: TodoStatus

    def as_dict(self) -> dict:
        return {
            "id": self.id,
            "content": self.content,
            "status": self.status.value,
        }


class TodoStoreError(ValueError):
    """Raised when the model hands the store a list it cannot accept."""


class TodoStore:
    """Whole-list replace, matching the Cursor TodoWrite shape."""

    def __init__(self) -> None:
        self._items: list[TodoItem] = []

    def items(self) -> list[TodoItem]:
        return list(self._items)

    def snapshot(self) -> list[dict]:
        return [item.as_dict() for item in self._items]

    def replace(self, raw_items: object) -> list[TodoItem]:
        if not isinstance(raw_items, list):
            raise TodoStoreError("items must be a list of todo objects")
        if len(raw_items) > MAX_TODO_ITEMS:
            raise TodoStoreError(
                f"at most {MAX_TODO_ITEMS} todo items are allowed, got {len(raw_items)}"
            )

        parsed: list[TodoItem] = []
        in_progress = 0
        for entry in raw_items:
            if not isinstance(entry, dict):
                raise TodoStoreError("each todo item must be an object")
            content = str(entry.get("content") or "").strip()
            if not content:
                raise TodoStoreError("each todo item needs non-empty content")
            status_raw = entry.get("status") or TodoStatus.pending.value
            try:
                status = TodoStatus(str(status_raw))
            except ValueError as exc:
                raise TodoStoreError(
                    f"status must be one of {[s.value for s in TodoStatus]}, "
                    f"got {status_raw!r}"
                ) from exc
            if status is TodoStatus.in_progress:
                in_progress += 1
            item_id = str(entry.get("id") or "").strip() or uuid.uuid4().hex[:8]
            parsed.append(TodoItem(id=item_id, content=content, status=status))

        if in_progress > 1:
            raise TodoStoreError("at most one todo may be in_progress at a time")

        self._items = parsed
        return self.items()

    def format_for_model(self) -> str:
        if not self._items:
            return "Todo list is empty."
        lines = ["Current todo list:"]
        for item in self._items:
            lines.append(f"- [{item.status.value}] ({item.id}) {item.content}")
        return "\n".join(lines)
