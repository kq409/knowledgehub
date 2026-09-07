import os
import uuid
from datetime import UTC, datetime

import pytest
from dotenv import load_dotenv
from sqlalchemy import text

import db
from models import (
    Note,
    NoteChunk,
    NotePaper,
    NoteSourceType,
    Paper,
    PaperChunk,
    PaperStatus,
    ProcessingStatus,
)
from services.retrieval import list_linked_notes, search, search_for_paper

load_dotenv()

pytestmark = pytest.mark.asyncio

NEAR = [1.0] + [0.0] * 767
FAR = [0.0, 1.0] + [0.0] * 766
QUERY = list(NEAR)


async def _table_exists(session, name: str) -> bool:
    try:
        await session.execute(text(f"SELECT 1 FROM {name} LIMIT 1"))
        return True
    except Exception:
        await session.rollback()
        return False


@pytest.fixture
async def session():
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        pytest.skip("DATABASE_URL is not set")
    db.init_db(database_url)
    if db.SessionLocal is None:
        pytest.skip("Database session factory was not created")
    async with db.SessionLocal() as db_session:
        if not await _table_exists(db_session, "papers"):
            pytest.skip("PostgreSQL papers table is not available")
        yield db_session
    await db.close_db()


async def _insert_paper(session, title: str, embedding: list[float]) -> Paper:
    paper = Paper(
        title=title,
        original_filename=f"{title}.pdf",
        original_file=f"/tmp/{uuid.uuid4()}.pdf",
        processing_status=PaperStatus.ready.value,
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
    )
    session.add(paper)
    await session.flush()
    session.add(
        PaperChunk(
            paper_id=paper.id,
            chunk_index=0,
            text=f"{title} body",
            page=1,
            section="Methods",
            embedding=embedding,
        )
    )
    await session.commit()
    await session.refresh(paper)
    return paper


async def test_paper_ids_filter_excludes_other_papers(session):
    paper_a = await _insert_paper(session, "Filter A", FAR)
    paper_b = await _insert_paper(session, "Filter B", NEAR)
    try:
        hits = await search(
            session,
            QUERY,
            include_voice_notes=False,
            include_handwritten_notes=False,
            paper_ids=[paper_a.id],
        )
        assert hits
        assert all(hit.source_id == paper_a.id for hit in hits)
        assert paper_b.id not in {hit.source_id for hit in hits}
    finally:
        for paper in (paper_a, paper_b):
            await session.delete(paper)
        await session.commit()


async def test_search_for_paper_excludes_unlinked_notes(session):
    if not await _table_exists(session, "notes"):
        pytest.skip("PostgreSQL notes table is not available")

    paper_a = await _insert_paper(session, "Notes A", NEAR)
    paper_b = await _insert_paper(session, "Notes B", NEAR)
    linked = Note(
        source_type=NoteSourceType.voice.value,
        title="Linked to A",
        processing_status=ProcessingStatus.ready.value,
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
    )
    unlinked = Note(
        source_type=NoteSourceType.voice.value,
        title="Unlinked note",
        processing_status=ProcessingStatus.ready.value,
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
    )
    other = Note(
        source_type=NoteSourceType.handwritten.value,
        title="Linked to B",
        processing_status=ProcessingStatus.ready.value,
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
    )
    session.add_all([linked, unlinked, other])
    await session.flush()
    session.add_all(
        [
            NotePaper(note_id=linked.id, paper_id=paper_a.id),
            NotePaper(note_id=other.id, paper_id=paper_b.id),
            NoteChunk(
                note_id=linked.id,
                chunk_index=0,
                text="Commentary on A",
                embedding=NEAR,
            ),
            NoteChunk(
                note_id=unlinked.id,
                chunk_index=0,
                text="General commentary",
                embedding=NEAR,
            ),
            NoteChunk(
                note_id=other.id,
                chunk_index=0,
                text="Commentary on B",
                embedding=NEAR,
            ),
        ]
    )
    await session.commit()
    try:
        hits = await search_for_paper(session, QUERY, paper_a.id)
        note_hits = [hit for hit in hits if hit.source_type != "paper"]
        assert {hit.title for hit in note_hits} == {"Linked to A"}

        linked_notes = await list_linked_notes(session, [paper_a.id])
        assert [note.title for note in linked_notes] == ["Linked to A"]
    finally:
        for note in (linked, unlinked, other):
            await session.delete(note)
        for paper in (paper_a, paper_b):
            await session.delete(paper)
        await session.commit()


async def _insert_note(
    session,
    title: str,
    embedding: list[float],
    *,
    paper: Paper | None = None,
) -> Note:
    note = Note(
        source_type=NoteSourceType.voice.value,
        title=title,
        processing_status=ProcessingStatus.ready.value,
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
    )
    session.add(note)
    await session.flush()
    session.add(
        NoteChunk(
            note_id=note.id,
            chunk_index=0,
            text=f"{title} commentary",
            embedding=embedding,
        )
    )
    if paper is not None:
        session.add(NotePaper(note_id=note.id, paper_id=paper.id))
    await session.commit()
    await session.refresh(note)
    return note


async def test_search_expands_note_hit_with_linked_paper_chunks(session):
    if not await _table_exists(session, "notes"):
        pytest.skip("PostgreSQL notes table is not available")

    paper = await _insert_paper(session, "Linked methods paper", FAR)
    note = await _insert_note(session, "Near note", NEAR, paper=paper)
    try:
        hits = await search(
            session,
            QUERY,
            include_handwritten_notes=False,
            top_k=4,
        )
        paper_hits = [hit for hit in hits if hit.source_type == "paper"]
        ours = [hit for hit in hits if hit.source_id == note.id]
        assert ours
        assert ours[0].linked_titles == ("Linked methods paper",)
        assert any(hit.source_id == paper.id and hit.via_link for hit in paper_hits)
    finally:
        await session.delete(note)
        await session.delete(paper)
        await session.commit()


async def test_search_expands_paper_hit_with_linked_note_chunks(session):
    if not await _table_exists(session, "notes"):
        pytest.skip("PostgreSQL notes table is not available")

    paper = await _insert_paper(session, "Near paper", NEAR)
    note = await _insert_note(session, "Far linked note", FAR, paper=paper)
    try:
        hits = await search(
            session,
            QUERY,
            include_handwritten_notes=False,
            top_k=4,
        )
        note_hits = [hit for hit in hits if hit.source_type != "paper"]
        assert any(hit.source_id == note.id and hit.via_link for hit in note_hits)
    finally:
        await session.delete(note)
        await session.delete(paper)
        await session.commit()


async def test_unlinked_note_does_not_expand_to_other_papers(session):
    if not await _table_exists(session, "notes"):
        pytest.skip("PostgreSQL notes table is not available")

    paper = await _insert_paper(session, "Unrelated paper", FAR)
    note = await _insert_note(session, "Unlinked near note", NEAR)
    try:
        hits = await search(
            session,
            QUERY,
            include_handwritten_notes=False,
            top_k=4,
        )
        ours = [hit for hit in hits if hit.source_id == note.id]
        assert ours
        assert ours[0].linked_titles == ()
        assert all(hit.source_id != paper.id for hit in hits)
    finally:
        await session.delete(note)
        await session.delete(paper)
        await session.commit()


async def test_hybrid_search_finds_acronym_dense_would_miss(session):
    """Lexical match should surface a far embedding when the query is literal."""
    nonce = "ZXQELLA7"
    ella = await _insert_paper_with_text(
        session,
        f"{nonce}: Efficient Lifelong Learning Algorithm",
        FAR,
        f"{nonce} transfers sparse models across tasks.",
    )
    near = await _insert_paper_with_text(
        session,
        "Unrelated neural nets",
        NEAR,
        "Convolutional networks classify images.",
    )
    try:
        dense_only = await search(
            session,
            QUERY,
            include_voice_notes=False,
            include_handwritten_notes=False,
            top_k=8,
            expand_links=False,
        )
        dense_ids = {hit.source_id for hit in dense_only}
        assert near.id in dense_ids

        hybrid = await search(
            session,
            QUERY,
            query_text=nonce,
            include_voice_notes=False,
            include_handwritten_notes=False,
            top_k=8,
            expand_links=False,
        )
        hybrid_ids = {hit.source_id for hit in hybrid}
        assert ella.id in hybrid_ids
    finally:
        for paper in (ella, near):
            await session.delete(paper)
        await session.commit()


async def _insert_paper_with_text(
    session, title: str, embedding: list[float], text: str
) -> Paper:
    paper = Paper(
        title=title,
        original_filename=f"{title}.pdf",
        original_file=f"/tmp/{uuid.uuid4()}.pdf",
        processing_status=PaperStatus.ready.value,
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
    )
    session.add(paper)
    await session.flush()
    session.add(
        PaperChunk(
            paper_id=paper.id,
            chunk_index=0,
            text=text,
            page=1,
            section="Abstract",
            embedding=embedding,
        )
    )
    await session.commit()
    await session.refresh(paper)
    return paper
