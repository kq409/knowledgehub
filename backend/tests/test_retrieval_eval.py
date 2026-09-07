import json
import os
import uuid
from datetime import UTC, datetime
from pathlib import Path

import pytest
from dotenv import load_dotenv
from sqlalchemy import text

import db
from models import Paper, PaperChunk, PaperStatus
from schemas import QueryKind
from services.ask_policy import classify_query
from services.retrieval import search

load_dotenv()

pytestmark = pytest.mark.asyncio

GOLD_PATH = Path(__file__).resolve().parent.parent / "eval" / "ask_gold.json"
DIM = 768
NEAR = [1.0] + [0.0] * (DIM - 1)
FAR = [0.0, 1.0] + [0.0] * (DIM - 2)
QUERY = list(NEAR)

ELLA_TITLE = "ZXQELLA7: Efficient Lifelong Learning Algorithm"
CNN_TITLE = "ZXQCNN7 convolutional networks"


def _load_gold() -> list[dict]:
    return json.loads(GOLD_PATH.read_text(encoding="utf-8"))


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
        if not await _table_exists(db_session, "paper_chunks"):
            pytest.skip("PostgreSQL paper_chunks table is not available")
        yield db_session
    await db.close_db()


async def _insert(session, title: str, embedding: list[float], body: str) -> Paper:
    paper = Paper(
        title=title,
        original_filename=f"{title}.pdf",
        original_file=f"/tmp/{uuid.uuid4()}.pdf",
        processing_status=PaperStatus.ready.value,
        year=2006,
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


async def test_gold_queries_hybrid_recall(session):
    ella = await _insert(
        session,
        ELLA_TITLE,
        FAR,
        "ZXQELLA7 transfers sparse models across tasks with efficient lifelong learning.",
    )
    cnn = await _insert(
        session,
        CNN_TITLE,
        NEAR,
        "ZXQCNN7 Convolutional networks classify photographs and other images.",
    )
    try:
        gold = _load_gold()
        assert len(gold) >= 8
        for item in gold:
            kind = item["kind"]
            query = item["query"]
            if kind == "field_wide":
                assert classify_query(query) is QueryKind.field_wide
                continue
            hits = await search(
                session,
                QUERY,
                query_text=query,
                include_voice_notes=False,
                include_handwritten_notes=False,
                top_k=8,
                expand_links=False,
            )
            titles = {hit.title for hit in hits}
            for expected in item.get("must_include_titles") or []:
                assert expected in titles, f"{item['id']} missed {expected}: {titles}"

        if os.getenv("OPIK_API_KEY"):
            from services.ask_trace import log_ask_trace

            log_ask_trace(
                question="gold-eval",
                query_kind="library",
                hits=[],
                decision="OK",
                external_search=False,
                external_status="unavailable",
                web_urls=[],
                citations=[],
                answer=f"gold set {len(gold)} items passed",
                latency_ms=0.0,
            )
    finally:
        await session.delete(ella)
        await session.delete(cnn)
        await session.commit()
