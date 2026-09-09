"""Full tool output the model can come back for.

Truncation without a way back is lossy in the one direction that matters: the
model reads 4000 characters of a paper, decides the answer is further down, and
has no way to get there except running the same tool again -- which costs
another slot out of a budget that is already tight.

So every result that gets shortened is kept here in full, and the shortened
copy in the conversation says which id to ask for. `fetch_tool_result` reads a
window back out.

The store is deliberately in memory and scoped to one turn. Writing spilled
output to disk (as a CLI harness would) buys nothing here: several replicas of
this app do not share a filesystem, so a path is not a recovery channel the
next request can rely on, and the files would need a reaper. A turn's store
dies with the turn, and `MAX_STORE_CHARS` keeps a single turn from holding
more than a few megabytes.
"""

from __future__ import annotations

from dataclasses import dataclass

# Roughly a few megabytes of text per turn. Reached only by many large reads in
# one conversation; the oldest entries are dropped first.
MAX_STORE_CHARS = 4_000_000
DEFAULT_FETCH_CHARS = 4000
MAX_FETCH_CHARS = 8000


@dataclass(frozen=True)
class StoredResult:
    call_id: str
    tool: str
    content: str

    @property
    def total_chars(self) -> int:
        return len(self.content)


class ToolResultStore:
    """Full tool results for one turn, keyed by `tool_call_id`."""

    def __init__(self, max_chars: int = MAX_STORE_CHARS) -> None:
        self._entries: dict[str, StoredResult] = {}
        self._max_chars = max_chars
        self._total = 0

    def __len__(self) -> int:
        return len(self._entries)

    @property
    def total_chars(self) -> int:
        return self._total

    def put(self, call_id: str, tool: str, content: str) -> None:
        """Keep the most complete version of a result seen so far.

        Both the tool runner and the compaction pass shorten the same payload,
        and compaction sees the already-shortened copy. Without this guard the
        second write would replace the full text with less of it.
        """
        if not call_id or not content:
            return
        existing = self._entries.get(call_id)
        if existing is not None:
            if existing.total_chars >= len(content):
                return
            self._total -= existing.total_chars

        self._entries[call_id] = StoredResult(
            call_id=call_id, tool=tool, content=content
        )
        self._total += len(content)
        self._evict_oldest()

    def _evict_oldest(self) -> None:
        # dicts keep insertion order, so the first key is the oldest result.
        while self._total > self._max_chars and self._entries:
            oldest = next(iter(self._entries))
            dropped = self._entries.pop(oldest)
            self._total -= dropped.total_chars

    def get(self, call_id: str) -> StoredResult | None:
        return self._entries.get(call_id)

    def ids(self) -> list[str]:
        return list(self._entries)

    def window(
        self, call_id: str, offset: int = 0, limit: int = DEFAULT_FETCH_CHARS
    ) -> tuple[StoredResult, str, int] | None:
        """A slice of a stored result, plus where the next slice starts.

        Returns None when the id is unknown. `next_offset` is the end of the
        returned slice, or -1 when the slice reached the end of the content.
        """
        entry = self._entries.get(call_id)
        if entry is None:
            return None
        start = max(0, min(offset, entry.total_chars))
        span = max(1, min(limit, MAX_FETCH_CHARS))
        end = min(entry.total_chars, start + span)
        next_offset = end if end < entry.total_chars else -1
        return entry, entry.content[start:end], next_offset


def truncation_notice(call_id: str, shown: int, total: int) -> str:
    """What replaces the dropped tail, so the model knows how to get it."""
    return (
        f"\n\n[truncated: showed {shown} of {total} characters. "
        f'Call fetch_tool_result(call_id="{call_id}", offset={shown}) '
        "for the next slice.]"
    )


def shortened_notice(call_id: str, total: int) -> str:
    """What replaces an older result that compaction shortened."""
    return (
        f"\n\n[earlier result shortened to save context. "
        f'fetch_tool_result(call_id="{call_id}") still has all {total} '
        "characters.]"
    )
