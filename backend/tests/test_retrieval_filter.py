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
        paper_id=paper_a.id,
        processing_status=ProcessingStatus.ready.value,
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
    )
    unlinked = Note(
        source_type=NoteSourceType.voice.value,
        title="Unlinked note",
        paper_id=None,
        processing_status=ProcessingStatus.ready.value,
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
    )
    other = Note(
        source_type=NoteSourceType.handwritten.value,
        title="Linked to B",
        paper_id=paper_b.id,
        processing_status=ProcessingStatus.ready.value,
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
    )
    session.add_all([linked, unlinked, other])
    await session.flush()
    session.add_all(
        [
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
