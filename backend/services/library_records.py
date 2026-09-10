"""Canonical LibraryRecord facade over papers, notes, and documents.

Physical tables stay separate. This module is the catalog appearance: list,
facet, and search with pagination. Ask/Chat retrieval keeps its own small
top_k path in `retrieval.search`, but both call `space_clause`.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime

from sqlalchemy import exists, func, literal, or_, select, union_all
from sqlalchemy.exc import OperationalError, ProgrammingError
from sqlalchemy.ext.asyncio import AsyncSession

from models import (
    AccessionStatus,
    LibraryDocument,
    LibraryDocumentChunk,
    Note,
    NoteChunk,
    Paper,
    PaperChunk,
    Space,
)
from services.accession import accessioned_clause
from services.identity import Identity, bound_space_ids, space_clause

SCHOLARLY_ARTICLE = "scholarly_article"
RESEARCH_NOTE = "research_note"
DOCUMENT = "document"
CONTENT_TYPES = (SCHOLARLY_ARTICLE, RESEARCH_NOTE, DOCUMENT)

DEFAULT_LIMIT = 50
MAX_LIMIT = 100
CATALOG_SCAN_CAP = 500


@dataclass(frozen=True)
class LibraryRecord:
    id: uuid.UUID
    space_id: uuid.UUID
    content_type: str
    title: str
    status: str
    updated_at: datetime
    snippet: str | None = None
    highlight: str | None = None
    revision: int = 1
    checksum: str | None = None
    accession_status: str = AccessionStatus.accessioned.value


@dataclass(frozen=True)
class RecordPage:
    items: list[LibraryRecord]
    total: int
    limit: int
    offset: int


@dataclass(frozen=True)
class SpaceRow:
    id: uuid.UUID
    slug: str
    name: str


def clamp_limit(limit: int | None) -> int:
    if limit is None:
        return DEFAULT_LIMIT
    return max(1, min(int(limit), MAX_LIMIT))


def clamp_offset(offset: int | None) -> int:
    if offset is None:
        return 0
    return max(0, int(offset))


def parse_content_type(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = value.strip().lower()
    aliases = {
        "paper": SCHOLARLY_ARTICLE,
        "scholarly_article": SCHOLARLY_ARTICLE,
        "note": RESEARCH_NOTE,
        "research_note": RESEARCH_NOTE,
        "document": DOCUMENT,
    }
    if cleaned not in aliases:
        raise ValueError(
            "content_type must be scholarly_article, research_note, or document"
        )
    return aliases[cleaned]


def _escape_like(value: str) -> str:
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def _like(query: str) -> str:
    return f"%{_escape_like(query)}%"


def _highlight(text: str, query: str, *, radius: int = 80) -> str:
    haystack = text or ""
    needle = query.strip()
    if not haystack:
        return ""
    if not needle:
        return haystack[:240]
    idx = haystack.lower().find(needle.lower())
    if idx < 0:
        cleaned = " ".join(haystack.split())
        return cleaned[:239] + "…" if len(cleaned) > 240 else cleaned
    start = max(0, idx - radius)
    end = min(len(haystack), idx + len(needle) + radius)
    fragment = haystack[start:end].strip()
    marked = (
        fragment[: idx - start]
        + "«"
        + fragment[idx - start : idx - start + len(needle)]
        + "»"
        + fragment[idx - start + len(needle) :]
    )
    prefix = "…" if start > 0 else ""
    suffix = "…" if end < len(haystack) else ""
    return prefix + marked + suffix


def _record(
    *,
    item_id: uuid.UUID,
    space_id: uuid.UUID,
    content_type: str,
    title: str,
    status: str,
    updated_at: datetime,
    snippet: str | None = None,
    highlight: str | None = None,
    revision: int = 1,
    checksum: str | None = None,
    accession_status: str = AccessionStatus.accessioned.value,
) -> LibraryRecord:
    return LibraryRecord(
        id=item_id,
        space_id=space_id,
        content_type=content_type,
        title=title,
        status=status,
        updated_at=updated_at,
        snippet=snippet,
        highlight=highlight,
        revision=revision,
        checksum=checksum,
        accession_status=accession_status,
    )


async def list_visible_spaces(
    session: AsyncSession, identity: Identity
) -> list[SpaceRow]:
    if not identity.space_ids:
        return []
    result = await session.execute(
        select(Space)
        .where(Space.id.in_(tuple(identity.space_ids)))
        .order_by(Space.name.asc())
    )
    return [
        SpaceRow(id=space.id, slug=space.slug, name=space.name)
        for space in result.scalars().all()
    ]


def _type_selects(
    *,
    spaces: frozenset[uuid.UUID],
    content_type: str | None,
    status: str | None,
    record_ids: list[uuid.UUID] | None,
    accessioned_only: bool = False,
    accession_status: str | None = None,
):
    selects = []
    if content_type in (None, SCHOLARLY_ARTICLE):
        query = select(
            Paper.id.label("id"),
            Paper.space_id.label("space_id"),
            literal(SCHOLARLY_ARTICLE).label("content_type"),
            Paper.title.label("title"),
            Paper.processing_status.label("status"),
            Paper.updated_at.label("updated_at"),
            Paper.revision.label("revision"),
            Paper.sha256.label("checksum"),
            Paper.accession_status.label("accession_status"),
        ).where(space_clause(Paper.space_id, spaces))
        if record_ids is not None:
            query = query.where(Paper.id.in_(record_ids or []))
        if status:
            query = query.where(Paper.processing_status == status)
        if accessioned_only:
            query = query.where(accessioned_clause(Paper.accession_status))
        elif accession_status:
            query = query.where(Paper.accession_status == accession_status)
        selects.append(query)
    if content_type in (None, RESEARCH_NOTE):
        query = select(
            Note.id.label("id"),
            Note.space_id.label("space_id"),
            literal(RESEARCH_NOTE).label("content_type"),
            Note.title.label("title"),
            Note.processing_status.label("status"),
            Note.updated_at.label("updated_at"),
            Note.revision.label("revision"),
            Note.sha256.label("checksum"),
            Note.accession_status.label("accession_status"),
        ).where(space_clause(Note.space_id, spaces))
        if record_ids is not None:
            query = query.where(Note.id.in_(record_ids or []))
        if status:
            query = query.where(Note.processing_status == status)
        if accessioned_only:
            query = query.where(accessioned_clause(Note.accession_status))
        elif accession_status:
            query = query.where(Note.accession_status == accession_status)
        selects.append(query)
    if content_type in (None, DOCUMENT):
        query = select(
            LibraryDocument.id.label("id"),
            LibraryDocument.space_id.label("space_id"),
            literal(DOCUMENT).label("content_type"),
            LibraryDocument.title.label("title"),
            LibraryDocument.processing_status.label("status"),
            LibraryDocument.updated_at.label("updated_at"),
            LibraryDocument.revision.label("revision"),
            LibraryDocument.sha256.label("checksum"),
            LibraryDocument.accession_status.label("accession_status"),
        ).where(space_clause(LibraryDocument.space_id, spaces))
        if record_ids is not None:
            query = query.where(LibraryDocument.id.in_(record_ids or []))
        if status:
            query = query.where(LibraryDocument.processing_status == status)
        if accessioned_only:
            query = query.where(accessioned_clause(LibraryDocument.accession_status))
        elif accession_status:
            query = query.where(LibraryDocument.accession_status == accession_status)
        selects.append(query)
    return selects


async def list_records(
    session: AsyncSession,
    *,
    space_ids: frozenset[uuid.UUID] | None,
    content_type: str | None = None,
    status: str | None = None,
    record_ids: list[uuid.UUID] | None = None,
    accessioned_only: bool = False,
    accession_status: str | None = None,
    limit: int | None = None,
    offset: int | None = None,
) -> RecordPage:
    spaces = bound_space_ids(space_ids)
    page_size = clamp_limit(limit)
    start = clamp_offset(offset)
    if record_ids is not None and not record_ids:
        return RecordPage(items=[], total=0, limit=page_size, offset=start)
    selects = _type_selects(
        spaces=spaces,
        content_type=content_type,
        status=status,
        record_ids=record_ids,
        accessioned_only=accessioned_only,
        accession_status=accession_status,
    )
    if not selects:
        return RecordPage(items=[], total=0, limit=page_size, offset=start)
    combined = union_all(*selects).subquery()
    total = await session.scalar(select(func.count()).select_from(combined)) or 0
    result = await session.execute(
        select(combined)
        .order_by(combined.c.updated_at.desc(), combined.c.title.asc())
        .limit(page_size)
        .offset(start)
    )
    items = [
        _record(
            item_id=row.id,
            space_id=row.space_id,
            content_type=row.content_type,
            title=row.title,
            status=row.status,
            updated_at=row.updated_at,
            revision=int(row.revision or 1),
            checksum=row.checksum,
            accession_status=row.accession_status,
        )
        for row in result.all()
    ]
    return RecordPage(items=items, total=int(total), limit=page_size, offset=start)


async def get_record(
    session: AsyncSession,
    record_id: uuid.UUID,
    *,
    space_ids: frozenset[uuid.UUID] | None,
) -> LibraryRecord | None:
    spaces = bound_space_ids(space_ids)
    paper = await session.get(Paper, record_id)
    if paper is not None and paper.space_id in spaces:
        return _record(
            item_id=paper.id,
            space_id=paper.space_id,
            content_type=SCHOLARLY_ARTICLE,
            title=paper.title,
            status=paper.processing_status,
            updated_at=paper.updated_at,
            revision=int(paper.revision or 1),
            checksum=paper.sha256,
            accession_status=paper.accession_status,
        )
    note = await session.get(Note, record_id)
    if note is not None and note.space_id in spaces:
        return _record(
            item_id=note.id,
            space_id=note.space_id,
            content_type=RESEARCH_NOTE,
            title=note.title,
            status=note.processing_status,
            updated_at=note.updated_at,
            revision=int(note.revision or 1),
            checksum=note.sha256,
            accession_status=note.accession_status,
        )
    document = await session.get(LibraryDocument, record_id)
    if document is not None and document.space_id in spaces:
        return _record(
            item_id=document.id,
            space_id=document.space_id,
            content_type=DOCUMENT,
            title=document.title,
            status=document.processing_status,
            updated_at=document.updated_at,
            revision=int(document.revision or 1),
            checksum=document.sha256,
            accession_status=document.accession_status,
        )
    return None


def _title_rank(title: str, query: str) -> float:
    lowered = title.lower()
    needle = query.lower()
    if lowered == needle:
        return 200.0
    if needle in lowered:
        return 100.0
    return 0.0


async def _search_kind(
    session: AsyncSession,
    query: str,
    *,
    model,
    chunk_model,
    fk_column,
    content_type: str,
    spaces: frozenset[uuid.UUID],
    record_ids: list[uuid.UUID] | None,
    limit: int,
) -> list[LibraryRecord]:
    like = _like(query)
    tsquery = func.plainto_tsquery("simple", query)
    chunk_match = exists(
        select(chunk_model.id).where(
            fk_column == model.id,
            or_(
                chunk_model.tsv.op("@@")(tsquery),
                chunk_model.text.ilike(like, escape="\\"),
            ),
        )
    )
    conditions = [
        space_clause(model.space_id, spaces),
        accessioned_clause(model.accession_status),
        or_(model.title.ilike(like, escape="\\"), chunk_match),
    ]
    if record_ids is not None:
        conditions.append(model.id.in_(record_ids))
    result = await session.execute(select(model).where(*conditions).limit(limit))
    records: list[LibraryRecord] = []
    for row in result.scalars().all():
        chunk_text = await session.scalar(
            select(chunk_model.text)
            .where(
                fk_column == row.id,
                or_(
                    chunk_model.tsv.op("@@")(tsquery),
                    chunk_model.text.ilike(like, escape="\\"),
                ),
            )
            .order_by(chunk_model.chunk_index.asc())
            .limit(1)
        )
        source = (
            row.title
            if query.lower() in (row.title or "").lower()
            else (chunk_text or row.title)
        )
        records.append(
            _record(
                item_id=row.id,
                space_id=row.space_id,
                content_type=content_type,
                title=row.title,
                status=row.processing_status,
                updated_at=row.updated_at,
                snippet=_highlight(chunk_text or row.title, query),
                highlight=_highlight(source, query),
                revision=int(row.revision or 1),
                checksum=row.sha256,
                accession_status=row.accession_status,
            )
        )
    return records


async def search_records(
    session: AsyncSession,
    query: str,
    *,
    space_ids: frozenset[uuid.UUID] | None,
    content_type: str | None = None,
    record_ids: list[uuid.UUID] | None = None,
    limit: int | None = None,
    offset: int | None = None,
) -> RecordPage:
    """Catalog search: pagination and highlights, not RAG top_k=16."""
    spaces = bound_space_ids(space_ids)
    page_size = clamp_limit(limit)
    start = clamp_offset(offset)
    cleaned = (query or "").strip()
    if not cleaned:
        return await list_records(
            session,
            space_ids=spaces,
            content_type=content_type,
            record_ids=record_ids,
            accessioned_only=True,
            limit=page_size,
            offset=start,
        )
    if record_ids is not None and not record_ids:
        return RecordPage(items=[], total=0, limit=page_size, offset=start)

    scan = CATALOG_SCAN_CAP
    found: list[LibraryRecord] = []
    try:
        if content_type in (None, SCHOLARLY_ARTICLE):
            found.extend(
                await _search_kind(
                    session,
                    cleaned,
                    model=Paper,
                    chunk_model=PaperChunk,
                    fk_column=PaperChunk.paper_id,
                    content_type=SCHOLARLY_ARTICLE,
                    spaces=spaces,
                    record_ids=record_ids,
                    limit=scan,
                )
            )
        if content_type in (None, RESEARCH_NOTE):
            found.extend(
                await _search_kind(
                    session,
                    cleaned,
                    model=Note,
                    chunk_model=NoteChunk,
                    fk_column=NoteChunk.note_id,
                    content_type=RESEARCH_NOTE,
                    spaces=spaces,
                    record_ids=record_ids,
                    limit=scan,
                )
            )
        if content_type in (None, DOCUMENT):
            found.extend(
                await _search_kind(
                    session,
                    cleaned,
                    model=LibraryDocument,
                    chunk_model=LibraryDocumentChunk,
                    fk_column=LibraryDocumentChunk.document_id,
                    content_type=DOCUMENT,
                    spaces=spaces,
                    record_ids=record_ids,
                    limit=scan,
                )
            )
    except (ProgrammingError, OperationalError):
        await session.rollback()
        return await list_records(
            session,
            space_ids=spaces,
            content_type=content_type,
            record_ids=record_ids,
            accessioned_only=True,
            limit=page_size,
            offset=start,
        )

    found.sort(
        key=lambda item: (
            _title_rank(item.title, cleaned),
            item.updated_at,
            item.title,
        ),
        reverse=True,
    )
    total = len(found)
    page = found[start : start + page_size]
    return RecordPage(items=page, total=total, limit=page_size, offset=start)
