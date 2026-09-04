import asyncio
import os
from collections.abc import AsyncGenerator
from unittest.mock import MagicMock

import pytest
from dotenv import load_dotenv
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from sqlalchemy import text

import db
from routers.notes import router
from services.chunking import ParsedChunk
from services.note_parser import ParsedNotePdf
from services.note_pipeline import NotePipeline

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
        pytest.skip("PostgreSQL notes table is not available")

    app = FastAPI()
    app.include_router(router)

    pipeline = MagicMock(spec=NotePipeline)
    pipeline.parse_pdf.return_value = ParsedNotePdf(
        title="Mock Note Title",
        extracted_text="Page one of my handwritten notes.",
        page_count=1,
        chunks=[
            ParsedChunk(
                text="Page one of my handwritten notes.",
                page=1,
                section="Page 1",
            )
        ],
    )
    pipeline.embed_chunks.return_value = [[0.1] * 768]
    app.state.note_pipeline = pipeline
    app.state.embeddings = MagicMock()

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac
    await db.close_db()


def voice_payload(**overrides):
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


async def wait_until_ready(client: AsyncClient, note_id: str) -> dict:
    body: dict = {}
    for _ in range(20):
        fetched = await client.get(f"/api/notes/{note_id}")
        assert fetched.status_code == 200
        body = fetched.json()
        if body["processing_status"] in {"ready", "failed"}:
            return body
        await asyncio.sleep(0.1)
    return body


async def test_voice_note_create_list_update_delete(client: AsyncClient):
    created = await client.post("/api/notes", json=voice_payload())
    assert created.status_code == 201
    note = created.json()
    note_id = note["id"]
    try:
        assert note["source_type"] == "voice"
        assert note["processing_status"] in {"pending", "processing", "ready"}

        body = await wait_until_ready(client, note_id)
        assert body["processing_status"] == "ready"
        assert body["chunk_count"] >= 1

        listed = await client.get("/api/notes?source_type=voice")
        assert listed.status_code == 200
        assert any(item["id"] == note_id for item in listed.json())

        patched = await client.patch(
            f"/api/notes/{note_id}",
            json={"title": "Updated voice note", "review_status": "draft"},
        )
        assert patched.status_code == 200
        assert patched.json()["title"] == "Updated voice note"
        assert patched.json()["review_status"] == "draft"
        body = await wait_until_ready(client, note_id)
        assert body["processing_status"] == "ready"
    finally:
        deleted = await client.delete(f"/api/notes/{note_id}")
        assert deleted.status_code == 204
        missing = await client.get(f"/api/notes/{note_id}")
        assert missing.status_code == 404


async def test_pdf_note_upload_list_update_delete(client: AsyncClient):
    pdf_bytes = b"%PDF-1.4\n%mock note content\n"
    files = {"file": ("lab-notes.pdf", pdf_bytes, "application/pdf")}
    created = await client.post("/api/notes/upload", files=files)
    assert created.status_code == 202
    note = created.json()
    note_id = note["id"]

    try:
        assert note["source_type"] == "handwritten"
        assert note["original_filename"] == "lab-notes.pdf"

        body = await wait_until_ready(client, note_id)
        assert body["processing_status"] == "ready"
        assert body["title"] == "Mock Note Title"
        assert body["chunk_count"] == 1
        assert body["extracted_text"] == "Page one of my handwritten notes."

        listed = await client.get("/api/notes")
        assert listed.status_code == 200
        assert any(item["id"] == note_id for item in listed.json())

        chunks = await client.get(f"/api/notes/{note_id}/chunks")
        assert chunks.status_code == 200
        assert len(chunks.json()) == 1
        assert chunks.json()[0]["section"] == "Page 1"

        patched = await client.patch(
            f"/api/notes/{note_id}",
            json={"title": "Updated PDF note", "tags": ["methods"]},
        )
        assert patched.status_code == 200
        assert patched.json()["title"] == "Updated PDF note"
        assert patched.json()["tags"] == ["methods"]
    finally:
        deleted = await client.delete(f"/api/notes/{note_id}")
        assert deleted.status_code == 204
        missing = await client.get(f"/api/notes/{note_id}")
        assert missing.status_code == 404


async def test_reject_non_pdf_note(client: AsyncClient):
    files = {"file": ("notes.txt", b"not a pdf", "text/plain")}
    response = await client.post("/api/notes/upload", files=files)
    assert response.status_code == 400
