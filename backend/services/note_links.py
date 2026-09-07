"""Many-to-many links between notes and papers.

The agent, the Notes UI, and Connect Note to Literature all write through these
helpers. Retrieval reads the same table. Nothing here decides *whether* a note
should be linked — that is the researcher's (or Connect's) call.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from models import Note, NoteLinkSource, NotePaper, NotePaperSkip, Paper

MAX_PAPERS_PER_NOTE = 8
LINK_SOURCE_RESEARCHER = NoteLinkSource.researcher.value
LINK_SOURCE_AI = NoteLinkSource.ai.value


class NoteLinkError(ValueError):
    """Raised when a link cannot be created or replaced."""


def dedupe_paper_ids(paper_ids: list[uuid.UUID]) -> list[uuid.UUID]:
    unique: list[uuid.UUID] = []
    seen: set[uuid.UUID] = set()
    for paper_id in paper_ids:
        if paper_id in seen:
            continue
        seen.add(paper_id)
        unique.append(paper_id)
    return unique


async def list_paper_ids_for_note(
    session: AsyncSession, note_id: uuid.UUID
) -> list[uuid.UUID]:
    result = await session.execute(
        select(NotePaper.paper_id)
        .where(NotePaper.note_id == note_id)
        .order_by(NotePaper.created_at.asc())
    )
    return list(result.scalars().all())


async def list_links_for_note(
    session: AsyncSession, note_id: uuid.UUID
) -> list[NotePaper]:
    result = await session.execute(
        select(NotePaper)
        .where(NotePaper.note_id == note_id)
        .order_by(NotePaper.created_at.asc())
    )
    return list(result.scalars().all())


async def paper_ids_by_note_ids(
    session: AsyncSession, note_ids: list[uuid.UUID]
) -> dict[uuid.UUID, list[uuid.UUID]]:
    mapping: dict[uuid.UUID, list[uuid.UUID]] = {note_id: [] for note_id in note_ids}
    if not note_ids:
        return mapping
    result = await session.execute(
        select(NotePaper.note_id, NotePaper.paper_id)
        .where(NotePaper.note_id.in_(note_ids))
        .order_by(NotePaper.created_at.asc())
    )
    for note_id, paper_id in result.all():
        mapping.setdefault(note_id, []).append(paper_id)
    return mapping


async def links_by_note_ids(
    session: AsyncSession, note_ids: list[uuid.UUID]
) -> dict[uuid.UUID, list[NotePaper]]:
    mapping: dict[uuid.UUID, list[NotePaper]] = {note_id: [] for note_id in note_ids}
    if not note_ids:
        return mapping
    result = await session.execute(
        select(NotePaper)
        .where(NotePaper.note_id.in_(note_ids))
        .order_by(NotePaper.created_at.asc())
    )
    for row in result.scalars().all():
        mapping.setdefault(row.note_id, []).append(row)
    return mapping


async def titles_for_paper_ids(
    session: AsyncSession, paper_ids: list[uuid.UUID]
) -> dict[uuid.UUID, str]:
    if not paper_ids:
        return {}
    result = await session.execute(
        select(Paper.id, Paper.title).where(Paper.id.in_(paper_ids))
    )
    return dict(result.all())


async def linked_titles_for_notes(
    session: AsyncSession, note_ids: list[uuid.UUID]
) -> dict[uuid.UUID, tuple[str, ...]]:
    ids_by_note = await paper_ids_by_note_ids(session, note_ids)
    all_paper_ids = [paper_id for ids in ids_by_note.values() for paper_id in ids]
    titles = await titles_for_paper_ids(session, all_paper_ids)
    return {
        note_id: tuple(titles[paper_id] for paper_id in ids if paper_id in titles)
        for note_id, ids in ids_by_note.items()
    }


async def notes_linked_to_paper(
    session: AsyncSession, paper_id: uuid.UUID
) -> list[Note]:
    result = await session.execute(
        select(Note)
        .join(NotePaper, NotePaper.note_id == Note.id)
        .where(NotePaper.paper_id == paper_id)
        .order_by(Note.created_at.desc())
    )
    return list(result.scalars().all())


async def list_skipped_paper_ids(
    session: AsyncSession, note_id: uuid.UUID
) -> set[uuid.UUID]:
    result = await session.execute(
        select(NotePaperSkip.paper_id).where(NotePaperSkip.note_id == note_id)
    )
    return set(result.scalars().all())


async def skip_note_paper(
    session: AsyncSession, note_id: uuid.UUID, paper_id: uuid.UUID
) -> None:
    existing = await session.get(NotePaperSkip, (note_id, paper_id))
    if existing is not None:
        return
    session.add(
        NotePaperSkip(
            note_id=note_id,
            paper_id=paper_id,
            created_at=datetime.now(UTC),
        )
    )


async def unskip_note_paper(
    session: AsyncSession, note_id: uuid.UUID, paper_id: uuid.UUID
) -> None:
    await session.execute(
        delete(NotePaperSkip).where(
            NotePaperSkip.note_id == note_id, NotePaperSkip.paper_id == paper_id
        )
    )


async def _require_papers(session: AsyncSession, paper_ids: list[uuid.UUID]) -> None:
    if not paper_ids:
        return
    result = await session.execute(select(Paper.id).where(Paper.id.in_(paper_ids)))
    found = set(result.scalars().all())
    missing = [paper_id for paper_id in paper_ids if paper_id not in found]
    if missing:
        raise NoteLinkError(f"Paper not found: {missing[0]}")


async def replace_note_papers(
    session: AsyncSession, note_id: uuid.UUID, paper_ids: list[uuid.UUID]
) -> list[uuid.UUID]:
    """Replace the note's links with `paper_ids`.

    IDs that stay keep their source and reason metadata. Newly checked IDs are
    researcher links. Removing an AI link records a skip so Connect will not
    re-add that pair.
    """
    unique = dedupe_paper_ids(paper_ids)
    if len(unique) > MAX_PAPERS_PER_NOTE:
        raise NoteLinkError(f"A note can link to at most {MAX_PAPERS_PER_NOTE} papers")
    await _require_papers(session, unique)
    existing_rows = await list_links_for_note(session, note_id)
    existing = {row.paper_id: row for row in existing_rows}
    wanted = set(unique)

    for paper_id, row in existing.items():
        if paper_id in wanted:
            continue
        if row.source == LINK_SOURCE_AI:
            await skip_note_paper(session, note_id, paper_id)
        await session.delete(row)

    now = datetime.now(UTC)
    for paper_id in unique:
        if paper_id in existing:
            continue
        await unskip_note_paper(session, note_id, paper_id)
        session.add(
            NotePaper(
                note_id=note_id,
                paper_id=paper_id,
                source=LINK_SOURCE_RESEARCHER,
                created_at=now,
            )
        )
    return unique


async def link_note_paper(
    session: AsyncSession,
    note_id: uuid.UUID,
    paper_id: uuid.UUID,
    *,
    source: str = LINK_SOURCE_RESEARCHER,
    similarity: float | None = None,
    snippet: str | None = None,
    reason: str | None = None,
    reason_mode: str | None = None,
    commit: bool = True,
) -> str:
    """Create one link. Returns 'created' or 'exists'."""
    note = await session.get(Note, note_id)
    if note is None:
        raise NoteLinkError(f"No note with id {note_id}")
    paper = await session.get(Paper, paper_id)
    if paper is None:
        raise NoteLinkError(f"No paper with id {paper_id}")

    existing = await session.get(NotePaper, (note_id, paper_id))
    if existing is not None:
        return "exists"

    current = await list_paper_ids_for_note(session, note_id)
    if len(current) >= MAX_PAPERS_PER_NOTE:
        raise NoteLinkError(f"A note can link to at most {MAX_PAPERS_PER_NOTE} papers")
    if source == LINK_SOURCE_RESEARCHER:
        await unskip_note_paper(session, note_id, paper_id)
    session.add(
        NotePaper(
            note_id=note_id,
            paper_id=paper_id,
            source=source,
            similarity=similarity,
            snippet=snippet,
            reason=reason,
            reason_mode=reason_mode,
            created_at=datetime.now(UTC),
        )
    )
    if commit:
        await session.commit()
    return "created"


async def unlink_note_paper(
    session: AsyncSession, note_id: uuid.UUID, paper_id: uuid.UUID
) -> bool:
    existing = await session.get(NotePaper, (note_id, paper_id))
    if existing is None:
        await session.commit()
        return False
    if existing.source == LINK_SOURCE_AI:
        await skip_note_paper(session, note_id, paper_id)
    await session.delete(existing)
    await session.commit()
    return True
