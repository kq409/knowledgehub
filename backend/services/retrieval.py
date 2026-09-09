import os
import uuid
from dataclasses import dataclass, replace

from sqlalchemy import case, func, literal, or_, select
from sqlalchemy.exc import OperationalError, ProgrammingError
from sqlalchemy.ext.asyncio import AsyncSession

from models import (
    LibraryDocument,
    LibraryDocumentChunk,
    Note,
    NoteChunk,
    NotePaper,
    NoteSourceType,
    Paper,
    PaperChunk,
    PaperStatus,
    ProcessingStatus,
)
from services.note_links import linked_titles_for_notes, paper_ids_by_note_ids

DEFAULT_TOP_K = 8
MAX_TOP_K = 16
DEFAULT_MIN_SIMILARITY = 0.35
SNIPPET_CHARS = 400
COMPARE_PAPER_TOP_K = 4
COMPARE_NOTE_TOP_K = 2
DEFAULT_READ_CHUNKS = 3
MAX_READ_CHUNKS = 6
EXPAND_PAPER_TOP_K = 2
EXPAND_NOTE_TOP_K = 2
MAX_EXPAND_HITS = 8
RRF_K = 60
LEXICAL_POOL_MULTIPLIER = 2


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
class LibraryPaperRow:
    """A paper as the agent sees it when listing the library."""

    id: uuid.UUID
    title: str
    year: int | None
    authors: list[str]
    processing_status: str
    chunk_count: int
    abstract: str | None = None
    summary: str | None = None
    digest: dict | None = None


@dataclass(frozen=True)
class LibraryNoteRow:
    id: uuid.UUID
    title: str
    source_type: str
    review_status: str
    paper_ids: tuple[uuid.UUID, ...]
    linked_titles: tuple[str, ...]
    processing_status: str
    chunk_count: int


@dataclass(frozen=True)
class LibraryDocumentRow:
    id: uuid.UUID
    title: str
    original_filename: str
    processing_status: str
    chunk_count: int


@dataclass(frozen=True)
class DocumentChunk:
    chunk_id: uuid.UUID
    chunk_index: int
    text: str
    page: int | None
    section: str | None


@dataclass(frozen=True)
class PaperChunkPage:
    """A window into a paper's chunks, ordered by chunk_index."""

    chunks: list[DocumentChunk]
    total: int
    start_index: int


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
    via_link: bool = False
    linked_titles: tuple[str, ...] = ()
    rank_score: float | None = None

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


def _escape_like(value: str) -> str:
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def rrf_fuse(
    ranked_lists: list[list[RetrievalHit]], *, k: int = RRF_K
) -> list[RetrievalHit]:
    """Reciprocal rank fusion. Keeps the highest dense similarity per chunk."""
    scores: dict[uuid.UUID, float] = {}
    by_id: dict[uuid.UUID, RetrievalHit] = {}
    for ranked in ranked_lists:
        for rank, hit in enumerate(ranked, start=1):
            scores[hit.chunk_id] = scores.get(hit.chunk_id, 0.0) + 1.0 / (k + rank)
            existing = by_id.get(hit.chunk_id)
            if existing is None or hit.similarity > existing.similarity:
                by_id[hit.chunk_id] = hit
    ordered = sorted(
        by_id.values(),
        key=lambda hit: (scores[hit.chunk_id], hit.similarity),
        reverse=True,
    )
    return [replace(hit, rank_score=round(scores[hit.chunk_id], 6)) for hit in ordered]


def _apply_rerank(
    query_text: str, hits: list[RetrievalHit], keep: int
) -> list[RetrievalHit]:
    from services.rerank import rerank_enabled, rerank_texts

    if not rerank_enabled() or not hits:
        return hits[:keep]
    pool = hits[: max(keep, 32)]
    order = rerank_texts(query_text, [hit.text for hit in pool])
    reranked: list[RetrievalHit] = []
    seen: set[uuid.UUID] = set()
    for index, score in order:
        if index < 0 or index >= len(pool):
            continue
        hit = pool[index]
        if hit.chunk_id in seen:
            continue
        seen.add(hit.chunk_id)
        reranked.append(replace(hit, rank_score=round(float(score), 6)))
        if len(reranked) >= keep:
            break
    if len(reranked) < keep:
        for hit in pool:
            if hit.chunk_id in seen:
                continue
            reranked.append(hit)
            if len(reranked) >= keep:
                break
    return reranked[:keep]


async def list_ready_papers(
    session: AsyncSession, paper_ids: list[uuid.UUID] | None = None
) -> list[LibraryPaper]:
    if paper_ids is not None and not paper_ids:
        return []
    query = select(Paper.title, Paper.year).where(
        Paper.processing_status == PaperStatus.ready.value
    )
    if paper_ids is not None:
        query = query.where(Paper.id.in_(paper_ids))
    result = await session.execute(query)
    return [LibraryPaper(title=title, year=year) for title, year in result.all()]


async def list_library_papers(session: AsyncSession) -> list[LibraryPaperRow]:
    """Every paper in the library, including the ones still processing.

    The agent needs ids and coverage to judge what the library can support,
    so this returns more than the title/year pair `list_ready_papers` gives.
    """
    chunk_count = (
        select(func.count(PaperChunk.id))
        .where(PaperChunk.paper_id == Paper.id)
        .correlate(Paper)
        .scalar_subquery()
    )
    result = await session.execute(
        select(
            Paper.id,
            Paper.title,
            Paper.year,
            Paper.authors,
            Paper.processing_status,
            chunk_count,
            Paper.abstract,
            Paper.summary,
            Paper.digest,
        ).order_by(Paper.created_at.asc())
    )
    return [
        LibraryPaperRow(
            id=paper_id,
            title=title,
            year=year,
            authors=list(authors or []),
            processing_status=status,
            chunk_count=count,
            abstract=abstract,
            summary=summary,
            digest=dict(digest or {}) if digest else {},
        )
        for (
            paper_id,
            title,
            year,
            authors,
            status,
            count,
            abstract,
            summary,
            digest,
        ) in result.all()
    ]


async def list_library_notes(
    session: AsyncSession, *, source_types: list[str] | None = None
) -> list[LibraryNoteRow]:
    chunk_count = (
        select(func.count(NoteChunk.id))
        .where(NoteChunk.note_id == Note.id)
        .correlate(Note)
        .scalar_subquery()
    )
    query = select(
        Note.id,
        Note.title,
        Note.source_type,
        Note.review_status,
        Note.processing_status,
        chunk_count,
    )
    if source_types is not None:
        if not source_types:
            return []
        query = query.where(Note.source_type.in_(source_types))

    result = await session.execute(query.order_by(Note.created_at.asc()))
    rows = result.all()
    note_ids = [note_id for note_id, *_ in rows]
    ids_by_note = await paper_ids_by_note_ids(session, note_ids)
    titles_by_note = await linked_titles_for_notes(session, note_ids)
    return [
        LibraryNoteRow(
            id=note_id,
            title=title,
            source_type=source_type,
            review_status=review_status,
            paper_ids=tuple(ids_by_note.get(note_id, [])),
            linked_titles=titles_by_note.get(note_id, ()),
            processing_status=status,
            chunk_count=count,
        )
        for (
            note_id,
            title,
            source_type,
            review_status,
            status,
            count,
        ) in rows
    ]


async def list_library_documents(session: AsyncSession) -> list[LibraryDocumentRow]:
    chunk_count = (
        select(func.count(LibraryDocumentChunk.id))
        .where(LibraryDocumentChunk.document_id == LibraryDocument.id)
        .correlate(LibraryDocument)
        .scalar_subquery()
    )
    result = await session.execute(
        select(
            LibraryDocument.id,
            LibraryDocument.title,
            LibraryDocument.original_filename,
            LibraryDocument.processing_status,
            chunk_count,
        ).order_by(LibraryDocument.created_at.asc())
    )
    return [
        LibraryDocumentRow(
            id=document_id,
            title=title,
            original_filename=filename,
            processing_status=status,
            chunk_count=count,
        )
        for document_id, title, filename, status, count in result.all()
    ]


def clamp_read_limit(limit: int | None) -> int:
    if limit is None:
        return DEFAULT_READ_CHUNKS
    return max(1, min(limit, MAX_READ_CHUNKS))


async def read_paper_chunks(
    session: AsyncSession,
    paper_id: uuid.UUID,
    *,
    start_index: int = 0,
    limit: int | None = None,
) -> PaperChunkPage:
    """Read a paper in source order instead of by similarity."""
    start = max(0, start_index)
    window = clamp_read_limit(limit)

    total = await session.scalar(
        select(func.count(PaperChunk.id)).where(PaperChunk.paper_id == paper_id)
    )
    result = await session.execute(
        select(PaperChunk)
        .where(
            PaperChunk.paper_id == paper_id,
            PaperChunk.chunk_index >= start,
        )
        .order_by(PaperChunk.chunk_index)
        .limit(window)
    )
    chunks = [
        DocumentChunk(
            chunk_id=chunk.id,
            chunk_index=chunk.chunk_index,
            text=chunk.text,
            page=chunk.page,
            section=chunk.section,
        )
        for chunk in result.scalars().all()
    ]
    return PaperChunkPage(chunks=chunks, total=int(total or 0), start_index=start)


async def read_document_chunks(
    session: AsyncSession,
    document_id: uuid.UUID,
    *,
    start_index: int = 0,
    limit: int | None = None,
) -> PaperChunkPage:
    start = max(0, start_index)
    window = clamp_read_limit(limit)
    total = await session.scalar(
        select(func.count(LibraryDocumentChunk.id)).where(
            LibraryDocumentChunk.document_id == document_id
        )
    )
    result = await session.execute(
        select(LibraryDocumentChunk)
        .where(
            LibraryDocumentChunk.document_id == document_id,
            LibraryDocumentChunk.chunk_index >= start,
        )
        .order_by(LibraryDocumentChunk.chunk_index)
        .limit(window)
    )
    chunks = [
        DocumentChunk(
            chunk_id=chunk.id,
            chunk_index=chunk.chunk_index,
            text=chunk.text,
            page=chunk.page,
            section=chunk.section,
        )
        for chunk in result.scalars().all()
    ]
    return PaperChunkPage(chunks=chunks, total=int(total or 0), start_index=start)


async def search(
    session: AsyncSession,
    query_embedding: list[float],
    *,
    query_text: str | None = None,
    include_papers: bool = True,
    include_voice_notes: bool = True,
    include_handwritten_notes: bool = True,
    include_documents: bool = True,
    top_k: int = DEFAULT_TOP_K,
    paper_ids: list[uuid.UUID] | None = None,
    expand_links: bool = True,
) -> list[RetrievalHit]:
    limit = clamp_top_k(top_k)
    pool = max(limit * LEXICAL_POOL_MULTIPLIER, limit)
    dense: list[RetrievalHit] = []
    if include_papers:
        dense.extend(
            await _search_papers(session, query_embedding, pool, paper_ids=paper_ids)
        )
    if include_documents and paper_ids is None:
        dense.extend(await _search_documents(session, query_embedding, pool))
    dense.extend(
        await _search_notes(
            session,
            query_embedding,
            include_voice_notes=include_voice_notes,
            include_handwritten_notes=include_handwritten_notes,
            top_k=pool,
            paper_ids=paper_ids,
        )
    )
    dense.sort(key=lambda hit: hit.similarity, reverse=True)

    lexical: list[RetrievalHit] = []
    text = (query_text or "").strip()
    if text:
        lexical = await _lexical_search(
            session,
            text,
            include_papers=include_papers,
            include_voice_notes=include_voice_notes,
            include_handwritten_notes=include_handwritten_notes,
            include_documents=include_documents,
            top_k=pool,
            paper_ids=paper_ids,
        )

    if lexical:
        fused = rrf_fuse([dense, lexical])
    else:
        fused = [
            replace(hit, rank_score=round(1.0 / (RRF_K + index), 6))
            for index, hit in enumerate(dense, start=1)
        ]

    hits = _apply_rerank(text, fused, limit) if text else fused[:limit]
    if expand_links:
        hits = await expand_linked_evidence(
            session,
            query_embedding,
            hits,
            include_papers=include_papers,
            include_voice_notes=include_voice_notes,
            include_handwritten_notes=include_handwritten_notes,
        )
    return hits


async def search_for_paper(
    session: AsyncSession,
    query_embedding: list[float],
    paper_id: uuid.UUID,
    *,
    query_text: str | None = None,
    paper_top_k: int = COMPARE_PAPER_TOP_K,
    note_top_k: int = COMPARE_NOTE_TOP_K,
) -> list[RetrievalHit]:
    paper_hits = await _search_papers(
        session,
        query_embedding,
        paper_top_k * LEXICAL_POOL_MULTIPLIER,
        paper_ids=[paper_id],
    )
    note_hits = await _search_notes(
        session,
        query_embedding,
        include_voice_notes=True,
        include_handwritten_notes=True,
        top_k=note_top_k * LEXICAL_POOL_MULTIPLIER,
        paper_ids=[paper_id],
    )
    text = (query_text or "").strip()
    lexical: list[RetrievalHit] = []
    if text:
        lexical = await _lexical_search(
            session,
            text,
            include_papers=True,
            include_voice_notes=True,
            include_handwritten_notes=True,
            top_k=paper_top_k + note_top_k,
            paper_ids=[paper_id],
        )
    dense = paper_hits + note_hits
    dense.sort(key=lambda hit: hit.similarity, reverse=True)
    keep = paper_top_k + note_top_k
    if lexical:
        fused = rrf_fuse([dense, lexical])
        return _apply_rerank(text, fused, keep)
    return dense[:keep]


async def list_linked_notes(
    session: AsyncSession, paper_ids: list[uuid.UUID]
) -> list[LinkedNote]:
    if not paper_ids:
        return []
    result = await session.execute(
        select(Note, NotePaper.paper_id)
        .join(NotePaper, NotePaper.note_id == Note.id)
        .where(
            NotePaper.paper_id.in_(paper_ids),
            Note.processing_status == ProcessingStatus.ready.value,
        )
        .order_by(Note.created_at.asc())
    )
    notes: list[LinkedNote] = []
    for note, paper_id in result.all():
        notes.append(
            LinkedNote(
                id=note.id,
                paper_id=paper_id,
                title=note.title,
                source_type=note.source_type,
            )
        )
    return notes


def _like_pattern(query_text: str) -> str:
    return f"%{_escape_like(query_text)}%"


_LEXICAL_STOP = frozenset(
    {
        "what",
        "does",
        "about",
        "with",
        "from",
        "that",
        "this",
        "have",
        "been",
        "were",
        "which",
        "their",
        "there",
        "would",
        "could",
        "should",
        "into",
        "your",
        "them",
        "then",
        "than",
        "when",
        "where",
        "will",
        "just",
        "also",
        "only",
        "more",
        "some",
        "such",
        "claim",
    }
)


def _lexical_tokens(query_text: str) -> list[str]:
    tokens: list[str] = []
    for raw in query_text.split():
        token = raw.strip("?.,:;\"'()[]").strip()
        if len(token) >= 4 and token.lower() not in _LEXICAL_STOP:
            tokens.append(token)
    return tokens


def _lexical_match_clauses(query_text: str, tsv_column, *text_columns):
    tsquery = func.plainto_tsquery("simple", query_text)
    clauses = [tsv_column.op("@@")(tsquery)]
    for column in text_columns:
        clauses.append(column.ilike(_like_pattern(query_text), escape="\\"))
        for token in _lexical_tokens(query_text):
            clauses.append(column.ilike(_like_pattern(token), escape="\\"))
    return or_(*clauses)


def _title_boost(title_column, query_text: str):
    boost = literal(0.0)
    for token in _lexical_tokens(query_text):
        if len(token) < 6:
            continue
        boost = boost + case(
            (title_column.ilike(_like_pattern(token), escape="\\"), 10.0),
            else_=0.0,
        )
    return boost


async def _lexical_search(
    session: AsyncSession,
    query_text: str,
    *,
    include_papers: bool,
    include_voice_notes: bool,
    include_handwritten_notes: bool,
    include_documents: bool = True,
    top_k: int,
    paper_ids: list[uuid.UUID] | None = None,
) -> list[RetrievalHit]:
    try:
        hits: list[RetrievalHit] = []
        if include_papers:
            hits.extend(
                await _lexical_search_papers(
                    session, query_text, top_k, paper_ids=paper_ids
                )
            )
        if include_documents and paper_ids is None:
            hits.extend(await _lexical_search_documents(session, query_text, top_k))
        hits.extend(
            await _lexical_search_notes(
                session,
                query_text,
                include_voice_notes=include_voice_notes,
                include_handwritten_notes=include_handwritten_notes,
                top_k=top_k,
                paper_ids=paper_ids,
            )
        )
        hits.sort(key=lambda hit: hit.rank_score or 0.0, reverse=True)
        return hits[:top_k]
    except (ProgrammingError, OperationalError):
        await session.rollback()
        return []


async def _lexical_search_papers(
    session: AsyncSession,
    query_text: str,
    top_k: int,
    *,
    paper_ids: list[uuid.UUID] | None = None,
) -> list[RetrievalHit]:
    if paper_ids is not None and not paper_ids:
        return []
    tsquery = func.plainto_tsquery("simple", query_text)
    chunk_tsv = PaperChunk.tsv
    rank = func.ts_rank_cd(chunk_tsv, tsquery) + _title_boost(Paper.title, query_text)
    conditions = [
        Paper.processing_status == PaperStatus.ready.value,
        _lexical_match_clauses(query_text, chunk_tsv, PaperChunk.text, Paper.title),
    ]
    if paper_ids is not None:
        conditions.append(Paper.id.in_(paper_ids))
    result = await session.execute(
        select(PaperChunk, Paper, rank.label("lex_rank"))
        .join(Paper, Paper.id == PaperChunk.paper_id)
        .where(*conditions)
        .order_by(rank.desc(), PaperChunk.chunk_index.asc())
        .limit(top_k)
    )
    hits: list[RetrievalHit] = []
    for chunk, paper, lex_rank in result.all():
        hits.append(
            RetrievalHit(
                source_type="paper",
                source_id=paper.id,
                chunk_id=chunk.id,
                title=paper.title,
                page=chunk.page,
                section=chunk.section,
                text=chunk.text,
                similarity=0.0,
                year=paper.year,
                rank_score=round(float(lex_rank or 0.0), 6),
            )
        )
    return hits


async def _lexical_search_notes(
    session: AsyncSession,
    query_text: str,
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
    tsquery = func.plainto_tsquery("simple", query_text)
    chunk_tsv = NoteChunk.tsv
    rank = func.ts_rank_cd(chunk_tsv, tsquery) + _title_boost(Note.title, query_text)
    conditions = [
        Note.processing_status == ProcessingStatus.ready.value,
        Note.source_type.in_(source_types),
        _lexical_match_clauses(query_text, chunk_tsv, NoteChunk.text, Note.title),
    ]
    if paper_ids is not None:
        conditions.append(
            Note.id.in_(
                select(NotePaper.note_id).where(NotePaper.paper_id.in_(paper_ids))
            )
        )
    result = await session.execute(
        select(NoteChunk, Note, rank.label("lex_rank"))
        .join(Note, Note.id == NoteChunk.note_id)
        .where(*conditions)
        .order_by(rank.desc(), NoteChunk.chunk_index.asc())
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
            similarity=0.0,
            year=None,
            rank_score=round(float(lex_rank or 0.0), 6),
        )
        for chunk, note, lex_rank in result.all()
    ]


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


async def _search_documents(
    session: AsyncSession,
    query_embedding: list[float],
    top_k: int,
) -> list[RetrievalHit]:
    distance = LibraryDocumentChunk.embedding.cosine_distance(query_embedding)
    result = await session.execute(
        select(LibraryDocumentChunk, LibraryDocument, distance.label("distance"))
        .join(LibraryDocument, LibraryDocument.id == LibraryDocumentChunk.document_id)
        .where(
            LibraryDocumentChunk.embedding.is_not(None),
            LibraryDocument.processing_status == ProcessingStatus.ready.value,
        )
        .order_by(distance)
        .limit(top_k)
    )
    return [
        RetrievalHit(
            source_type="document",
            source_id=document.id,
            chunk_id=chunk.id,
            title=document.title,
            page=chunk.page,
            section=chunk.section,
            text=chunk.text,
            similarity=similarity_from_distance(dist),
            year=None,
        )
        for chunk, document, dist in result.all()
    ]


async def _lexical_search_documents(
    session: AsyncSession,
    query_text: str,
    top_k: int,
) -> list[RetrievalHit]:
    tsquery = func.plainto_tsquery("simple", query_text)
    chunk_tsv = LibraryDocumentChunk.tsv
    rank = func.ts_rank_cd(chunk_tsv, tsquery) + _title_boost(
        LibraryDocument.title, query_text
    )
    result = await session.execute(
        select(LibraryDocumentChunk, LibraryDocument, rank.label("lex_rank"))
        .join(LibraryDocument, LibraryDocument.id == LibraryDocumentChunk.document_id)
        .where(
            LibraryDocument.processing_status == ProcessingStatus.ready.value,
            _lexical_match_clauses(
                query_text, chunk_tsv, LibraryDocumentChunk.text, LibraryDocument.title
            ),
        )
        .order_by(rank.desc(), LibraryDocumentChunk.chunk_index.asc())
        .limit(top_k)
    )
    return [
        RetrievalHit(
            source_type="document",
            source_id=document.id,
            chunk_id=chunk.id,
            title=document.title,
            page=chunk.page,
            section=chunk.section,
            text=chunk.text,
            similarity=0.0,
            year=None,
            rank_score=round(float(lex_rank or 0.0), 6),
        )
        for chunk, document, lex_rank in result.all()
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
        conditions.append(
            Note.id.in_(
                select(NotePaper.note_id).where(NotePaper.paper_id.in_(paper_ids))
            )
        )

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


async def _annotate_linked_titles(
    session: AsyncSession, hits: list[RetrievalHit]
) -> list[RetrievalHit]:
    note_ids = [
        hit.source_id for hit in hits if hit.source_type in ("voice", "handwritten")
    ]
    if not note_ids:
        return hits
    titles_by_note = await linked_titles_for_notes(session, note_ids)
    annotated: list[RetrievalHit] = []
    for hit in hits:
        if hit.source_type not in ("voice", "handwritten"):
            annotated.append(hit)
            continue
        annotated.append(
            replace(hit, linked_titles=titles_by_note.get(hit.source_id, ()))
        )
    return annotated


async def expand_linked_evidence(
    session: AsyncSession,
    query_embedding: list[float],
    hits: list[RetrievalHit],
    *,
    include_papers: bool = True,
    include_voice_notes: bool = True,
    include_handwritten_notes: bool = True,
) -> list[RetrievalHit]:
    """Pull paper chunks for note hits and note chunks for paper hits."""
    annotated = await _annotate_linked_titles(session, hits)
    seen = {hit.chunk_id for hit in annotated}
    extras: list[RetrievalHit] = []

    note_ids = [
        hit.source_id
        for hit in annotated
        if hit.source_type in ("voice", "handwritten")
    ]
    paper_ids = [hit.source_id for hit in annotated if hit.source_type == "paper"]

    if include_papers and note_ids:
        ids_by_note = await paper_ids_by_note_ids(session, note_ids)
        linked_paper_ids: list[uuid.UUID] = []
        seen_papers: set[uuid.UUID] = set()
        for ids in ids_by_note.values():
            for paper_id in ids:
                if paper_id not in seen_papers:
                    seen_papers.add(paper_id)
                    linked_paper_ids.append(paper_id)
        for paper_id in linked_paper_ids:
            if len(extras) >= MAX_EXPAND_HITS:
                break
            paper_hits = await _search_papers(
                session,
                query_embedding,
                EXPAND_PAPER_TOP_K,
                paper_ids=[paper_id],
            )
            for hit in paper_hits:
                if hit.chunk_id in seen:
                    continue
                seen.add(hit.chunk_id)
                extras.append(replace(hit, via_link=True))
                if len(extras) >= MAX_EXPAND_HITS:
                    break

    if paper_ids and len(extras) < MAX_EXPAND_HITS:
        remaining = MAX_EXPAND_HITS - len(extras)
        for paper_id in paper_ids:
            if remaining <= 0:
                break
            note_hits = await _search_notes(
                session,
                query_embedding,
                include_voice_notes=include_voice_notes,
                include_handwritten_notes=include_handwritten_notes,
                top_k=EXPAND_NOTE_TOP_K,
                paper_ids=[paper_id],
            )
            for hit in note_hits:
                if hit.chunk_id in seen:
                    continue
                seen.add(hit.chunk_id)
                extras.append(replace(hit, via_link=True))
                remaining -= 1
                if remaining <= 0:
                    break
        extras = await _annotate_linked_titles(session, extras)

    return annotated + extras
