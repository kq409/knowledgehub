import os
import uuid
from datetime import UTC, datetime

import pytest
from dotenv import load_dotenv
from sqlalchemy import text

import db
from models import Paper, PaperChunk, PaperStatus
from services.retrieval import search

load_dotenv()

pytestmark = pytest.mark.asyncio

NEAR = [1.0] + [0.0] * 767
FAR = [0.0, 1.0] + [0.0] * 766


@pytest.fixture
async def session():
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        pytest.skip("DATABASE_URL is not set")
    db.init_db(database_url)
    if db.SessionLocal is None:
        pytest.skip("Database session factory was not created")
    async with db.SessionLocal() as db_session:
        try:
            await db_session.execute(text("SELECT 1 FROM papers LIMIT 1"))
        except Exception:
            await db_session.rollback()
            pytest.skip("PostgreSQL papers table is not available")
        yield db_session
    await db.close_db()


async def _insert(session, title: str, body: str, embedding: list[float]) -> Paper:
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
            text=body,
            page=1,
            section="Abstract",
            embedding=embedding,
        )
    )
    await session.commit()
    await session.refresh(paper)
    return paper


async def test_hybrid_finds_lexical_hit_that_dense_misses(session, monkeypatch):
    monkeypatch.delenv("RERANK_ENABLED", raising=False)
    dense_paper = await _insert(
        session,
        f"Dense cats {uuid.uuid4()}",
        "unrelated filler about cats and kitchens",
        NEAR,
    )
    lexical_paper = await _insert(
        session,
        f"ZXQELLA7 lexical {uuid.uuid4()}",
        "ZXQELLA7: Efficient Lifelong Learning Algorithm body about sparse transfer.",
        FAR,
    )
    fillers = [
        await _insert(
            session,
            f"Near filler {uuid.uuid4()}",
            "unrelated filler about cats and kitchens",
            NEAR,
        )
        for _ in range(5)
    ]
    ids = [dense_paper.id, lexical_paper.id, *[item.id for item in fillers]]
    try:
        dense_only = await search(
            session,
            NEAR,
            query_text="",
            include_voice_notes=False,
            include_handwritten_notes=False,
            top_k=4,
            paper_ids=ids,
        )
        dense_ids = {hit.source_id for hit in dense_only}
        assert dense_paper.id in dense_ids
        assert lexical_paper.id not in dense_ids

        hybrid = await search(
            session,
            NEAR,
            query_text="ZXQELLA7",
            include_voice_notes=False,
            include_handwritten_notes=False,
            top_k=4,
            paper_ids=ids,
        )
        hybrid_ids = {hit.source_id for hit in hybrid}
        assert lexical_paper.id in hybrid_ids
    finally:
        for paper in (dense_paper, lexical_paper, *fillers):
            await session.delete(paper)
        await session.commit()


async def test_hybrid_search_matches_title_initialism(session, monkeypatch):
    """GEM should find Gradient Episodic Memory even when the body never says GEM."""
    monkeypatch.delenv("RERANK_ENABLED", raising=False)
    gem = await _insert(
        session,
        f"Gradient Episodic Memory for Continual Learning {uuid.uuid4()}",
        "A model observes a sequence of tasks and stores examples from each.",
        FAR,
    )
    near = await _insert(
        session,
        f"Unrelated neural nets {uuid.uuid4()}",
        "Convolutional networks classify images.",
        NEAR,
    )
    try:
        hits = await search(
            session,
            NEAR,
            query_text="GEM",
            include_voice_notes=False,
            include_handwritten_notes=False,
            include_documents=False,
            top_k=8,
            paper_ids=[gem.id, near.id],
            expand_links=False,
        )
        assert gem.id in {hit.source_id for hit in hits}
    finally:
        for paper in (gem, near):
            await session.delete(paper)
        await session.commit()
