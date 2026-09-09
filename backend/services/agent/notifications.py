"""Background work, told to the model while the conversation is still open.

Uploading a paper starts a parse that takes tens of seconds, and the researcher
does not wait for it — they keep typing. Until now the agent had no way to know
that: it searched a library that was one paper short and confidently reported
nothing found, seconds before the paper became searchable.

The fix is s11's `<task_notification>`: work that finishes out of band is
posted here, and the loop injects whatever is waiting immediately before its
next model call. That placement matters. A notification is not a tool result —
inventing a `tool_call_id` for it would break the one-call-one-result pairing
that native providers enforce — and it is not a system rule either, because it
describes a fact rather than an instruction. It goes in as a plain user-role
message, which is what it is: news from the researcher's side of the desk.

The board is in memory and process-wide, which matches what it reports on:
ingestion runs as an `asyncio` task in this same process, so a replica can only
ever have news about its own jobs. `MAX_PENDING` and `NOTICE_TTL_SECONDS` keep
an idle process from accumulating a backlog nobody will read — a parse that
finished two hours ago is not news.
"""

from __future__ import annotations

import time
from collections import deque
from contextlib import suppress
from dataclasses import dataclass, field

# A burst of uploads should not push the rest of the conversation out of the
# context window. Past this many, the oldest news is dropped.
MAX_PENDING = 20
NOTICE_TTL_SECONDS = 3600.0

READY = "ready"
FAILED = "failed"


@dataclass(frozen=True)
class Notice:
    """One finished piece of background work."""

    kind: str
    title: str
    outcome: str = READY
    detail: str = ""
    at: float = field(default_factory=time.monotonic)

    @property
    def dedupe_key(self) -> tuple[str, str, str]:
        return (self.kind, self.title, self.outcome)

    def line(self) -> str:
        if self.outcome == FAILED:
            body = f'{self.kind} "{self.title}" failed to process'
            return f"{body}: {self.detail}" if self.detail else body
        body = f'{self.kind} "{self.title}" finished processing and is searchable now'
        return f"{body} ({self.detail})" if self.detail else body

    def as_event(self) -> dict:
        return {
            "kind": self.kind,
            "title": self.title,
            "outcome": self.outcome,
            "detail": self.detail,
        }


class NoticeBoard:
    """News waiting to be handed to the next model call."""

    def __init__(
        self, max_pending: int = MAX_PENDING, ttl_seconds: float = NOTICE_TTL_SECONDS
    ) -> None:
        self._pending: deque[Notice] = deque(maxlen=max_pending)
        self._ttl = ttl_seconds

    def __len__(self) -> int:
        return len(self._pending)

    def post(
        self, kind: str, title: str, *, outcome: str = READY, detail: str = ""
    ) -> None:
        """Record finished work. Called from a background task, never awaited."""
        notice = Notice(kind=kind, title=title, outcome=outcome, detail=detail)
        # Resuming a queued paper can process the same upload twice; the
        # researcher only needs to hear about it once.
        if any(item.dedupe_key == notice.dedupe_key for item in self._pending):
            return
        self._pending.append(notice)

    def drain(self, *, now: float | None = None) -> list[Notice]:
        """Take everything still fresh. Delivered once, then forgotten."""
        current = time.monotonic() if now is None else now
        fresh = [item for item in self._pending if current - item.at <= self._ttl]
        self._pending.clear()
        return fresh

    def clear(self) -> None:
        self._pending.clear()


# The board every ingestion path posts to. A single-process board is the right
# scope: `schedule_paper_processing` and friends run in this process.
INGEST_NOTICES = NoticeBoard()


def notify_ingested(
    kind: str, title: str, *, outcome: str = READY, detail: str = ""
) -> None:
    """Post to the shared board. Never raises — ingestion is the real work."""
    # A missed notice is a smaller loss than an upload marked failed.
    with suppress(Exception):
        INGEST_NOTICES.post(kind, title, outcome=outcome, detail=detail)


def format_notifications(notices: list[Notice]) -> str:
    """The message the model sees. Empty when there is nothing to say."""
    if not notices:
        return ""
    lines = "\n".join(f"- {notice.line()}" for notice in notices)
    return (
        "<task_notification>\n"
        "Background work finished since the last message. The library has "
        "changed:\n"
        f"{lines}\n"
        "Mention this only if it affects the answer.\n"
        "</task_notification>"
    )
