import os
import uuid
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from models import (
    Note,
    NoteChunk,
    NoteSourceType,
    Paper,
    PaperChunk,
    PaperStatus,
    ProcessingStatus,
)

DEFAULT_TOP_K = 8
MAX_TOP_K = 16
DEFAULT_MIN_SIMILARITY = 0.35
SNIPPET_CHARS = 400
COMPARE_PAPER_TOP_K = 4
COMPARE_NOTE_TOP_K = 2


def min_similarity_from_env() -> float:
    raw = os.getenv("ASK_MIN_SIMILARITY", str(DEFAULT_MIN_SIMILARITY))
    try:
        value = float(raw)
    except ValueError:
        return DEFAULT_MIN_SIMILARITY
    return max(0.0, min(value, 1.0))


def clamp_top_k(top_k: int | None) -> int:
    if top_k is None:
        return DEFAULT_TOP_K
    return max(1, min(top_k, MAX_TOP_K))


def snippet_from(text: str, max_chars: int = SNIPPET_CHARS) -> str:
    cleaned = " ".join(text.split())
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[: max_chars - 1].rstrip() + "..."


def similarity_from_distance(distance: float) -> float:
    return round(1.0 - float(distance), 4)


@dataclass(frozen=True)
class LibraryPaper:
    title: str
    year: int | None


@dataclass(frozen=True)
class LinkedNote:
    id: uuid.UUID
    paper_id: uuid.UUID
    title: str
    source_type: str


@dataclass(frozen=True)
class RetrievalHit:
    source_type: str
    source_id: uuid.UUID
    chunk_id: uuid.UUID
    title: str
    page: int | None
    section: str | None
    text: str
    similarity: float
    year: int | None = None

    def snippet(self, max_chars: int = SNIPPET_CHARS) -> str:
        return snippet_from(self.text, max_chars=max_chars)


def is_insufficient(
    hits: list[RetrievalHit], min_similarity: float = DEFAULT_MIN_SIMILARITY
) -> bool:
    if not hits:
        return True
    return max(hit.similarity for hit in hits) < min_similarity


def filter_by_min_similarity(
    hits: list[RetrievalHit], min_similarity: float = DEFAULT_MIN_SIMILARITY
) -> list[RetrievalHit]:
    return [hit for hit in hits if hit.similarity >= min_similarity]


async def list_ready_papers(session: AsyncSession) -> list[LibraryPaper]:
    result = await session.execute(
        select(Paper.title, Paper.year).where(
            Paper.processing_status == PaperStatus.ready.value
        )
    )
    return [LibraryPaper(title=title, year=year) for title, year in result.all()]


async def search(
    session: AsyncSession,
    query_embedding: list[float],
    *,
    include_papers: bool = True,
    include_voice_notes: bool = True,
    include_handwritten_notes: bool = True,
    top_k: int = DEFAULT_TOP_K,
    paper_ids: list[uuid.UUID] | None = None,
) -> list[RetrievalHit]:
    limit = clamp_top_k(top_k)
    hits: list[RetrievalHit] = []
    if include_papers:
        hits.extend(
            await _search_papers(
                session, query_embedding, limit, paper_ids=paper_ids
            )
        )
    hits.extend(
        await _search_notes(
            session,
            query_embedding,
            include_voice_notes=include_voice_notes,
            include_handwritten_notes=include_handwritten_notes,
            top_k=limit,
            paper_ids=paper_ids,
        )
    )
    hits.sort(key=lambda hit: hit.similarity, reverse=True)
    return hits[:limit]


async def search_for_paper(
    session: AsyncSession,
    query_embedding: list[float],
    paper_id: uuid.UUID,
    *,
    paper_top_k: int = COMPARE_PAPER_TOP_K,
    note_top_k: int = COMPARE_NOTE_TOP_K,
) -> list[RetrievalHit]:
    paper_hits = await _search_papers(
        session,
        query_embedding,
        paper_top_k,
        paper_ids=[paper_id],
    )
    note_hits = await _search_notes(
        session,
        query_embedding,
        include_voice_notes=True,
        include_handwritten_notes=True,
        top_k=note_top_k,
        paper_ids=[paper_id],
    )
    return paper_hits + note_hits


async def list_linked_notes(
    session: AsyncSession, paper_ids: list[uuid.UUID]
) -> list[LinkedNote]:
    if not paper_ids:
        return []
    result = await session.execute(
        select(Note)
        .where(
            Note.paper_id.in_(paper_ids),
            Note.processing_status == ProcessingStatus.ready.value,
        )
        .order_by(Note.created_at.asc())
    )
    notes: list[LinkedNote] = []
    for note in result.scalars().all():
        if note.paper_id is None:
            continue
        notes.append(
            LinkedNote(
                id=note.id,
                paper_id=note.paper_id,
                title=note.title,
                source_type=note.source_type,
            )
        )
    return notes


async def _search_papers(
    session: AsyncSession,
    query_embedding: list[float],
    top_k: int,
    *,
    paper_ids: list[uuid.UUID] | None = None,
) -> list[RetrievalHit]:
    if paper_ids is not None and not paper_ids:
        return []

    distance = PaperChunk.embedding.cosine_distance(query_embedding)
    conditions = [
        PaperChunk.embedding.is_not(None),
        Paper.processing_status == PaperStatus.ready.value,
    ]
    if paper_ids is not None:
        conditions.append(Paper.id.in_(paper_ids))

    result = await session.execute(
        select(PaperChunk, Paper, distance.label("distance"))
        .join(Paper, Paper.id == PaperChunk.paper_id)
        .where(*conditions)
        .order_by(distance)
        .limit(top_k)
    )
    return [
        RetrievalHit(
            source_type="paper",
            source_id=paper.id,
            chunk_id=chunk.id,
            title=paper.title,
            page=chunk.page,
            section=chunk.section,
            text=chunk.text,
            similarity=similarity_from_distance(dist),
            year=paper.year,
        )
        for chunk, paper, dist in result.all()
    ]


async def _search_notes(
    session: AsyncSession,
    query_embedding: list[float],
    *,
    include_voice_notes: bool,
    include_handwritten_notes: bool,
    top_k: int,
    paper_ids: list[uuid.UUID] | None = None,
) -> list[RetrievalHit]:
    if paper_ids is not None and not paper_ids:
        return []

    source_types: list[str] = []
    if include_voice_notes:
        source_types.append(NoteSourceType.voice.value)
    if include_handwritten_notes:
        source_types.append(NoteSourceType.handwritten.value)
    if not source_types:
        return []

    distance = NoteChunk.embedding.cosine_distance(query_embedding)
    conditions = [
        NoteChunk.embedding.is_not(None),
        Note.processing_status == ProcessingStatus.ready.value,
        Note.source_type.in_(source_types),
    ]
    if paper_ids is not None:
        conditions.append(Note.paper_id.in_(paper_ids))

    result = await session.execute(
        select(NoteChunk, Note, distance.label("distance"))
        .join(Note, Note.id == NoteChunk.note_id)
        .where(*conditions)
        .order_by(distance)
        .limit(top_k)
    )
    return [
        RetrievalHit(
            source_type=note.source_type,
            source_id=note.id,
            chunk_id=chunk.id,
            title=note.title,
            page=chunk.page,
            section=chunk.section,
            text=chunk.text,
            similarity=similarity_from_distance(dist),
            year=None,
        )
        for chunk, note, dist in result.all()
    ]
