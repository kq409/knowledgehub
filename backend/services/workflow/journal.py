"""What a workflow already finished, so a retry does not redo it.

The key question is what identifies a step. Not its position: `parallel()`
finishes steps in whatever order the model responds, so "step 3" means
something different on every run and a replay would hand paper A's cells to
paper B. Not its completion order either, for the same reason.

So a step is identified by what it *means*: its kind, its label, and a
fingerprint of the exact inputs (prompt, dimensions, schema). Change the
prompt and the key changes, which is correct -- the old result answered a
different question. Re-run the same workflow unchanged and every key matches,
so nothing is recomputed.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

import db
from models import WorkflowStep

STATUS_DONE = "done"
STATUS_FAILED = "failed"


def fingerprint(*parts: object) -> str:
    """A stable digest of whatever identifies a step's inputs.

    `sort_keys` matters: two dicts that differ only in insertion order are the
    same step, and without it they would be journalled twice.
    """
    payload = json.dumps(parts, sort_keys=True, default=str, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def step_key(kind: str, label: str, *inputs: object) -> str:
    """The journal key for one step. 64 hex chars, matching the column."""
    return fingerprint(kind, label, *inputs)


@dataclass(frozen=True)
class JournalEntry:
    key: str
    kind: str
    label: str
    status: str
    result: dict
    error: str | None
    attempts: int

    @property
    def usable(self) -> bool:
        """Only a finished step may be replayed; a failure must be retried."""
        return self.status == STATUS_DONE


class WorkflowJournal:
    """Reads and writes one run's finished steps.

    Entries are loaded once into memory at the start of a run, so checking
    whether a step is cached costs nothing per step. Writes go straight to the
    database -- a journal that only exists in memory would be no journal at
    all.
    """

    def __init__(self, session: AsyncSession, run_id: str) -> None:
        self.session = session
        self.run_id = run_id
        self._entries: dict[str, JournalEntry] = {}
        self._loaded = False
        self._lock = asyncio.Lock()
        self.replayed = 0
        self.executed = 0

    @asynccontextmanager
    async def _write_session(self) -> AsyncIterator[AsyncSession]:
        """A session to record one entry on.

        Its own connection, for two reasons. Parallel steps finish at the same
        moment and an `AsyncSession` cannot be driven by two coroutines at
        once. And a journal entry committed inside the caller's transaction
        would roll back together with the very failure the journal exists to
        survive.

        Without a session factory (a unit test with a stub) it falls back to
        the caller's session behind a lock, which is correct but couples the
        entry's durability to the caller's transaction.
        """
        maker = db.SessionLocal
        if maker is None:
            async with self._lock:
                yield self.session
            return
        async with maker() as session:
            yield session

    async def load(self) -> None:
        result = await self.session.execute(
            select(WorkflowStep).where(WorkflowStep.run_id == self.run_id)
        )
        self._entries = {
            row.step_key: JournalEntry(
                key=row.step_key,
                kind=row.kind,
                label=row.label,
                status=row.status,
                result=row.result or {},
                error=row.error,
                attempts=row.attempts,
            )
            for row in result.scalars().all()
        }
        self._loaded = True

    @property
    def loaded(self) -> bool:
        return self._loaded

    def cached(self, key: str) -> JournalEntry | None:
        entry = self._entries.get(key)
        return entry if entry is not None and entry.usable else None

    def __len__(self) -> int:
        return len(self._entries)

    async def _upsert(
        self,
        key: str,
        *,
        kind: str,
        label: str,
        status: str,
        result: dict,
        error: str | None,
    ) -> None:
        async with self._write_session() as session:
            existing = await session.execute(
                select(WorkflowStep).where(
                    WorkflowStep.run_id == self.run_id, WorkflowStep.step_key == key
                )
            )
            row = existing.scalar_one_or_none()
            now = datetime.now(UTC)
            if row is None:
                row = WorkflowStep(
                    run_id=self.run_id,
                    step_key=key,
                    kind=kind,
                    label=label[:240],
                    status=status,
                    result=result,
                    error=error,
                    attempts=1,
                    created_at=now,
                    updated_at=now,
                )
                session.add(row)
            else:
                row.status = status
                row.result = result
                row.error = error
                row.attempts = row.attempts + 1
                row.updated_at = now
            await session.commit()
            attempts = row.attempts

        self._entries[key] = JournalEntry(
            key=key,
            kind=kind,
            label=label,
            status=status,
            result=result,
            error=error,
            attempts=attempts,
        )

    async def record(self, key: str, *, kind: str, label: str, result: dict) -> None:
        self.executed += 1
        await self._upsert(
            key,
            kind=kind,
            label=label,
            status=STATUS_DONE,
            result=result,
            error=None,
        )

    async def record_failure(
        self, key: str, *, kind: str, label: str, error: str
    ) -> None:
        """Keep the failure, so a retry can see it was attempted and why."""
        await self._upsert(
            key,
            kind=kind,
            label=label,
            status=STATUS_FAILED,
            result={},
            error=error[:4000],
        )

    def note_replay(self) -> None:
        self.replayed += 1

    async def clear(self) -> None:
        """Forget this run. For a workflow whose inputs are known to be stale."""
        async with self._write_session() as session:
            result = await session.execute(
                select(WorkflowStep).where(WorkflowStep.run_id == self.run_id)
            )
            for row in result.scalars().all():
                await session.delete(row)
            await session.commit()
        self._entries = {}
