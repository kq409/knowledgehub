import os
from collections.abc import AsyncGenerator

import pytest
from dotenv import load_dotenv
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from sqlalchemy import text

import db
from routers.voice_notes import router

load_dotenv()

pytestmark = pytest.mark.asyncio


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
            await session.execute(text("SELECT 1 FROM notes LIMIT 1"))
    except Exception:
        await db.close_db()
        pytest.skip("PostgreSQL is not available")

    app = FastAPI()
    app.include_router(router)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac
    await db.close_db()


def note_payload(**overrides):
    data = {
        "title": "Hybrid retrieval idea",
        "summary": "Compare dense search with BM25 on my paper library.",
        "observations": ["Dense search misses exact terms"],
        "hypotheses": ["Hybrid retrieval will improve recall"],
        "questions": ["What reranker should I try first?"],
        "next_steps": ["Build a 20-question eval set"],
        "tags": ["rag", "eval"],
        "raw_transcript": "um I think hybrid retrieval might help",
        "cleaned_transcript": "I think hybrid retrieval might help.",
        "review_status": "generated",
        "model": "gemma3:4b",
        "prompt_version": "voice-note-extract-v1",
    }
    data.update(overrides)
    return data


async def test_voice_note_crud_roundtrip(client: AsyncClient):
    created = await client.post("/api/voice-notes", json=note_payload())
    assert created.status_code == 201
    note = created.json()
    note_id = note["id"]
    try:
        assert note["review_status"] == "generated"
        assert note["observations"] == ["Dense search misses exact terms"]

        listed = await client.get("/api/voice-notes")
        assert listed.status_code == 200
        assert any(item["id"] == note_id for item in listed.json())

        fetched = await client.get(f"/api/voice-notes/{note_id}")
        assert fetched.status_code == 200
        assert fetched.json()["title"] == "Hybrid retrieval idea"

        patched = await client.patch(
            f"/api/voice-notes/{note_id}",
            json={"title": "Updated title", "review_status": "draft"},
        )
        assert patched.status_code == 200
        body = patched.json()
        assert body["title"] == "Updated title"
        assert body["review_status"] == "draft"
    finally:
        await client.delete(f"/api/voice-notes/{note_id}")

    missing = await client.get(f"/api/voice-notes/{note_id}")
    assert missing.status_code == 404


async def test_extract_requires_service(client: AsyncClient):
    response = await client.post(
        "/api/voice-notes/extract",
        json={"raw_text": "raw", "cleaned_text": "cleaned"},
    )
    assert response.status_code == 503
