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
from models import (
    EMBEDDING_DIM,
    Note,
    NoteChunk,
    NoteSourceType,
    Paper,
    PaperChunk,
    PaperStatus,
    ProcessingStatus,
    ReviewStatus,
)
from routers.ask import router
from services.ask import AskService

load_dotenv()

pytestmark = pytest.mark.asyncio

DIM = EMBEDDING_DIM


def unit(index: int) -> list[float]:
    vector = [0.0] * DIM
    vector[index] = 1.0
    return vector


def mock_llm(content: str) -> MagicMock:
    client = MagicMock()
    response = MagicMock()
    response.choices = [MagicMock(message=MagicMock(content=content))]
    client.chat.completions.create.return_value = response
    return client


@pytest.fixture
async def client() -> AsyncGenerator[AsyncClient, None]:
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        pytest.skip("DATABASE_URL is not set")

    db.init_db(database_url)
    try:
        if db.SessionLocal is None:
            pytest.skip("Database session factory was not created")
        async with db.SessionLocal() as session:
            await session.execute(text("SELECT 1 FROM paper_chunks LIMIT 1"))
    except Exception:
        await db.close_db()
        pytest.skip("PostgreSQL is not available")

    embeddings = MagicMock()
    embeddings.embed_query.return_value = unit(0)
    app = FastAPI()
    app.include_router(router)
    app.state.ask = AskService(
        llm_client=mock_llm("Hybrid search should help [1]."),
        llm_model="test-model",
        embeddings=embeddings,
        min_similarity=0.35,
    )

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac
    await db.close_db()


async def test_empty_question_is_422(client: AsyncClient):
    response = await client.post("/api/ask", json={"question": "   "})
    assert response.status_code == 422


async def test_no_sources_is_400(client: AsyncClient):
    response = await client.post(
        "/api/ask",
        json={
            "question": "What is hybrid search?",
            "include_papers": False,
            "include_voice_notes": False,
            "include_handwritten_notes": False,
        },
    )
    assert response.status_code == 400


async def test_ask_returns_citations_from_retrieval(client: AsyncClient):
    assert db.SessionLocal is not None
    now = datetime.now(UTC)
    paper = Paper(
        id=uuid.uuid4(),
        title="AskAPI Paper",
        authors=[],
        tags=[],
        original_filename="ask.pdf",
        original_file=f"/tmp/{uuid.uuid4()}.pdf",
        processing_status=PaperStatus.ready.value,
        created_at=now,
        updated_at=now,
    )
    chunk = PaperChunk(
        id=uuid.uuid4(),
        paper_id=paper.id,
        chunk_index=0,
        text="Hybrid retrieval combines dense and sparse search.",
        page=1,
        section="Introduction",
        embedding=unit(0),
        extra={},
        created_at=now,
    )
    async with db.SessionLocal() as session:
        session.add(paper)
        await session.flush()
        session.add(chunk)
        await session.commit()

    try:
        response = await client.post(
            "/api/ask",
            json={"question": "Does hybrid retrieval help?"},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["answer"] == "Hybrid search should help [1]."
        assert body["insufficient_evidence"] is False
        assert body["citations"]
        assert body["citations"][0]["source_id"] == str(paper.id)
        assert body["citations"][0]["source_type"] == "paper"
        assert body["citations"][0]["index"] == 1
        assert "Hybrid retrieval" in body["citations"][0]["snippet"]
        assert body["model"] == "test-model"
    finally:
        async with db.SessionLocal() as session:
            db_paper = await session.get(Paper, paper.id)
            if db_paper is not None:
                await session.delete(db_paper)
                await session.commit()


async def test_low_similarity_sets_insufficient_evidence():
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        pytest.skip("DATABASE_URL is not set")

    db.init_db(database_url)
    if db.SessionLocal is None:
        pytest.skip("Database session factory was not created")

    embeddings = MagicMock()
    embeddings.embed_query.return_value = unit(5)
    app = FastAPI()
    app.include_router(router)
    app.state.ask = AskService(
        llm_client=mock_llm("Tentative answer with low confidence."),
        llm_model="test-model",
        embeddings=embeddings,
        min_similarity=0.9,
    )
    now = datetime.now(UTC)
    note = Note(
        id=uuid.uuid4(),
        source_type=NoteSourceType.voice.value,
        title="AskAPI Weak Note",
        summary="",
        observations=[],
        hypotheses=[],
        questions=[],
        next_steps=[],
        tags=[],
        raw_transcript="",
        cleaned_transcript="unrelated",
        review_status=ReviewStatus.generated.value,
        processing_status=ProcessingStatus.ready.value,
        created_at=now,
        updated_at=now,
    )
    chunk = NoteChunk(
        id=uuid.uuid4(),
        note_id=note.id,
        chunk_index=0,
        text="unrelated leftover thought",
        page=None,
        section="Voice Note",
        embedding=unit(0),
        extra={},
        created_at=now,
    )
    async with db.SessionLocal() as session:
        session.add(note)
        await session.flush()
        session.add(chunk)
        await session.commit()

    transport = ASGITransport(app=app)
    try:
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            response = await ac.post(
                "/api/ask",
                json={
                    "question": "What is the main result?",
                    "include_papers": False,
                    "include_voice_notes": True,
                    "include_handwritten_notes": False,
                },
            )
        assert response.status_code == 200
        body = response.json()
        assert body["insufficient_evidence"] is True
        assert "Tentative" in body["answer"]
    finally:
        async with db.SessionLocal() as session:
            db_note = await session.get(Note, note.id)
            if db_note is not None:
                await session.delete(db_note)
                await session.commit()
        await db.close_db()
