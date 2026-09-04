import asyncio
import json
import os
import uuid
from collections.abc import AsyncGenerator
from datetime import UTC, datetime
from unittest.mock import MagicMock

import pytest
from dotenv import load_dotenv
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from sqlalchemy import text

import db
from models import Paper, PaperChunk, PaperComparison, PaperStatus
from routers.compare import router
from schemas import DEFAULT_COMPARE_DIMENSIONS
from services.compare import CompareService
from services.embeddings import EmbeddingService

load_dotenv()

pytestmark = pytest.mark.asyncio

VEC = [0.1] * 768


def _completion(content: str) -> MagicMock:
    response = MagicMock()
    response.choices = [MagicMock(message=MagicMock(content=content))]
    return response


def _map_payload(label: str) -> str:
    return json.dumps({dim: f"{label} {dim}" for dim in DEFAULT_COMPARE_DIMENSIONS})


async def _table_exists(session, name: str) -> bool:
    try:
        await session.execute(text(f"SELECT 1 FROM {name} LIMIT 1"))
        return True
    except Exception:
        await session.rollback()
        return False


async def _insert_paper(session, title: str, status: str = PaperStatus.ready.value) -> Paper:
    paper = Paper(
        title=title,
        original_filename=f"{title}.pdf",
        original_file=f"/tmp/{uuid.uuid4()}.pdf",
        processing_status=status,
        year=2020,
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
    )
    session.add(paper)
    await session.flush()
    if status == PaperStatus.ready.value:
        session.add(
            PaperChunk(
                paper_id=paper.id,
                chunk_index=0,
                text=f"{title} methods and results.",
                page=1,
                section="Methods",
                embedding=VEC,
            )
        )
    await session.commit()
    await session.refresh(paper)
    return paper


@pytest.fixture
async def api() -> AsyncGenerator[tuple[AsyncClient, MagicMock], None]:
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        pytest.skip("DATABASE_URL is not set")

    try:
        await asyncio.to_thread(db.run_migrations)
    except Exception as exc:
        pytest.skip(f"Could not run migrations: {exc}")
    db.init_db(database_url)
    try:
        if db.SessionLocal is None:
            pytest.skip("Database session factory was not created")
        async with db.SessionLocal() as session:
            if not await _table_exists(session, "papers"):
                pytest.skip("PostgreSQL papers table is not available")
            if not await _table_exists(session, "paper_comparisons"):
                pytest.skip("PostgreSQL paper_comparisons table is not available")
    except Exception:
        await db.close_db()
        pytest.skip("PostgreSQL is not available")

    llm = MagicMock()
    embeddings = MagicMock(spec=EmbeddingService)
    embeddings.embed_query.return_value = VEC
    app = FastAPI()
    app.include_router(router)
    app.state.compare = CompareService(
        llm_client=llm,
        llm_model="test-model",
        embeddings=embeddings,
    )

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac, llm
    await db.close_db()


async def test_compare_rejects_one_paper(api: tuple[AsyncClient, MagicMock]):
    client, _llm = api
    response = await client.post(
        "/api/compare",
        json={
            "paper_ids": [str(uuid.uuid4())],
            "dimensions": ["problem"],
        },
    )
    assert response.status_code == 400
    assert "at least 2" in response.json()["detail"]


async def test_compare_rejects_five_papers(api: tuple[AsyncClient, MagicMock]):
    client, _llm = api
    response = await client.post(
        "/api/compare",
        json={
            "paper_ids": [str(uuid.uuid4()) for _ in range(5)],
            "dimensions": ["problem"],
        },
    )
    assert response.status_code == 400
    assert "at most 4" in response.json()["detail"]


async def test_compare_rejects_non_ready_paper(api: tuple[AsyncClient, MagicMock]):
    client, _llm = api
    assert db.SessionLocal is not None
    async with db.SessionLocal() as session:
        ready = await _insert_paper(session, "Ready Paper")
        pending = await _insert_paper(
            session, "Pending Paper", status=PaperStatus.pending.value
        )
        ready_id, pending_id = ready.id, pending.id
    try:
        response = await client.post(
            "/api/compare",
            json={
                "paper_ids": [str(ready_id), str(pending_id)],
                "dimensions": ["problem"],
            },
        )
        assert response.status_code == 400
        assert "ready" in response.json()["detail"].lower()
    finally:
        async with db.SessionLocal() as session:
            for paper_id in (ready_id, pending_id):
                paper = await session.get(Paper, paper_id)
                if paper is not None:
                    await session.delete(paper)
            await session.commit()


async def test_compare_persist_round_trip(api: tuple[AsyncClient, MagicMock]):
    client, llm = api
    assert db.SessionLocal is not None
    async with db.SessionLocal() as session:
        paper_a = await _insert_paper(session, "Alpha Methods")
        paper_b = await _insert_paper(session, "Beta Methods")
        a_id, b_id = paper_a.id, paper_b.id

    llm.chat.completions.create.side_effect = [
        _completion(_map_payload("A")),
        _completion(_map_payload("B")),
        _completion(
            json.dumps(
                {
                    "agreements": "Both discuss methods.",
                    "disagreements": "Different framing.",
                    "research_gap": "No shared benchmark.",
                }
            )
        ),
    ]

    comparison_id = None
    try:
        created = await client.post(
            "/api/compare",
            json={
                "paper_ids": [str(a_id), str(b_id)],
                "dimensions": list(DEFAULT_COMPARE_DIMENSIONS),
            },
        )
        assert created.status_code == 200, created.text
        body = created.json()
        comparison_id = body["id"]
        assert body["synthesis"]["research_gap"] == "No shared benchmark."
        assert body["papers"][0]["values"]["method"].startswith("A ")
        assert body["papers"][1]["values"]["method"].startswith("B ")
        assert body["model"] == "test-model"
        assert body["citations"]

        fetched = await client.get(f"/api/compare/{comparison_id}")
        assert fetched.status_code == 200
        assert fetched.json()["id"] == comparison_id
        assert fetched.json()["synthesis"]["agreements"] == "Both discuss methods."

        listed = await client.get("/api/compare")
        assert listed.status_code == 200
        assert any(item["id"] == comparison_id for item in listed.json())
    finally:
        async with db.SessionLocal() as session:
            if comparison_id:
                row = await session.get(PaperComparison, uuid.UUID(comparison_id))
                if row is not None:
                    await session.delete(row)
            for paper_id in (a_id, b_id):
                paper = await session.get(Paper, paper_id)
                if paper is not None:
                    await session.delete(paper)
            await session.commit()
