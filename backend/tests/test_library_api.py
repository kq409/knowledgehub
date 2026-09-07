import asyncio
import os
from collections.abc import AsyncGenerator
from unittest.mock import MagicMock, patch

import pytest
from dotenv import load_dotenv
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from sqlalchemy import text

import db
from routers.library import router as library_router
from routers.notes import router as notes_router
from routers.papers import router as papers_router
from services.chunking import ParsedChunk
from services.note_parser import ParsedNotePdf
from services.note_pipeline import NotePipeline
from services.paper_parser import ParsedPaper
from services.paper_pipeline import PaperPipeline

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
            await session.execute(text("SELECT 1 FROM papers LIMIT 1"))
            await session.execute(text("SELECT 1 FROM notes LIMIT 1"))
    except Exception:
        await db.close_db()
        pytest.skip("PostgreSQL papers/notes tables are not available")

    app = FastAPI()
    app.include_router(library_router)
    app.include_router(papers_router)
    app.include_router(notes_router)

    paper_pipeline = MagicMock(spec=PaperPipeline)
    paper_pipeline.parse_and_embed.return_value = ParsedPaper(
        title="Mock Paper Title",
        abstract="A short abstract.",
        page_count=2,
        chunks=[
            ParsedChunk(text="Chunk one about methods.", page=1, section="Methods"),
        ],
    )
    paper_pipeline.embed_chunks.return_value = [[0.1] * 768]
    app.state.paper_pipeline = paper_pipeline

    note_pipeline = MagicMock(spec=NotePipeline)
    note_pipeline.parse_pdf.return_value = ParsedNotePdf(
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
    note_pipeline.embed_chunks.return_value = [[0.1] * 768]
    app.state.note_pipeline = note_pipeline
    app.state.embeddings = MagicMock()

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac
    await db.close_db()


async def wait_until_ready(client: AsyncClient, kind: str, item_id: str) -> dict:
    path = f"/api/papers/{item_id}" if kind == "paper" else f"/api/notes/{item_id}"
    body: dict = {}
    for _ in range(20):
        fetched = await client.get(path)
        assert fetched.status_code == 200
        body = fetched.json()
        if body["processing_status"] in {"ready", "failed"}:
            return body
        await asyncio.sleep(0.1)
    return body


async def test_library_upload_classifies_as_paper(client: AsyncClient):
    files = {"file": ("sample.pdf", b"%PDF-1.4\n%mock paper\n", "application/pdf")}
    with patch("routers.library.classify_pdf", return_value="paper"):
        created = await client.post("/api/library/upload", files=files)
    assert created.status_code == 202
    payload = created.json()
    assert payload["kind"] == "paper"
    assert payload["note"] is None
    paper = payload["paper"]
    paper_id = paper["id"]

    try:
        assert paper["original_filename"] == "sample.pdf"
        listed = await client.get("/api/papers")
        assert any(item["id"] == paper_id for item in listed.json())
        body = await wait_until_ready(client, "paper", paper_id)
        assert body["processing_status"] == "ready"
        assert body["title"] == "Mock Paper Title"
    finally:
        deleted = await client.delete(f"/api/papers/{paper_id}")
        assert deleted.status_code == 204


async def test_library_upload_classifies_as_note(client: AsyncClient):
    files = {"file": ("lab-notes.pdf", b"%PDF-1.4\n%mock note\n", "application/pdf")}
    with patch("routers.library.classify_pdf", return_value="note"):
        created = await client.post("/api/library/upload", files=files)
    assert created.status_code == 202
    payload = created.json()
    assert payload["kind"] == "note"
    assert payload["paper"] is None
    note = payload["note"]
    note_id = note["id"]

    try:
        assert note["source_type"] == "handwritten"
        assert note["original_filename"] == "lab-notes.pdf"
        listed = await client.get("/api/notes")
        assert any(item["id"] == note_id for item in listed.json())
        body = await wait_until_ready(client, "note", note_id)
        assert body["processing_status"] == "ready"
        assert body["title"] == "Mock Note Title"
    finally:
        deleted = await client.delete(f"/api/notes/{note_id}")
        assert deleted.status_code == 204


async def test_library_upload_rejects_non_pdf(client: AsyncClient):
    files = {"file": ("notes.txt", b"not a pdf", "text/plain")}
    response = await client.post("/api/library/upload", files=files)
    assert response.status_code == 400
