import os
from collections.abc import AsyncGenerator
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock
from uuid import UUID

import pytest
from dotenv import load_dotenv
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from sqlalchemy import text

import db
from models import Note, NotePaper, NoteSourceType, ProcessingStatus
from routers.papers import router
from services.chunking import ParsedChunk
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
    except Exception:
        await db.close_db()
        pytest.skip("PostgreSQL papers table is not available")

    app = FastAPI()
    app.include_router(router)

    pipeline = MagicMock(spec=PaperPipeline)
    pipeline.parse_and_embed.return_value = ParsedPaper(
        title="Mock Paper Title",
        abstract="A short abstract.",
        page_count=2,
        chunks=[
            ParsedChunk(text="Chunk one about methods.", page=1, section="Methods"),
            ParsedChunk(text="Chunk two about results.", page=2, section="Results"),
        ],
    )
    pipeline.embed_chunks.return_value = [[0.1] * 768, [0.2] * 768]
    app.state.paper_pipeline = pipeline

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac
    await db.close_db()


async def test_paper_upload_list_update_delete(client: AsyncClient, tmp_path: Path):
    pdf_bytes = b"%PDF-1.4\n%mock paper content\n"
    files = {"file": ("sample.pdf", pdf_bytes, "application/pdf")}
    created = await client.post("/api/papers", files=files)
    assert created.status_code == 202
    paper = created.json()
    paper_id = paper["id"]

    try:
        assert paper["processing_status"] in {"pending", "processing", "ready"}
        assert paper["original_filename"] == "sample.pdf"

        # Wait briefly for background task with mocked pipeline.
        for _ in range(20):
            fetched = await client.get(f"/api/papers/{paper_id}")
            assert fetched.status_code == 200
            body = fetched.json()
            if body["processing_status"] in {"ready", "failed"}:
                break
            import asyncio

            await asyncio.sleep(0.1)

        fetched = await client.get(f"/api/papers/{paper_id}")
        body = fetched.json()
        assert body["processing_status"] == "ready"
        assert body["title"] == "Mock Paper Title"
        assert body["chunk_count"] == 2

        listed = await client.get("/api/papers")
        assert listed.status_code == 200
        assert any(item["id"] == paper_id for item in listed.json())

        chunks = await client.get(f"/api/papers/{paper_id}/chunks")
        assert chunks.status_code == 200
        assert len(chunks.json()) == 2
        assert chunks.json()[0]["section"] == "Methods"

        patched = await client.patch(
            f"/api/papers/{paper_id}",
            json={
                "title": "Updated Paper",
                "authors": ["Ada Lovelace"],
                "year": 2024,
                "tags": ["vision"],
            },
        )
        assert patched.status_code == 200
        assert patched.json()["title"] == "Updated Paper"
        assert patched.json()["authors"] == ["Ada Lovelace"]
        assert patched.json()["year"] == 2024
    finally:
        deleted = await client.delete(f"/api/papers/{paper_id}")
        assert deleted.status_code == 204
        missing = await client.get(f"/api/papers/{paper_id}")
        assert missing.status_code == 404


async def test_reject_non_pdf(client: AsyncClient):
    files = {"file": ("notes.txt", b"not a pdf", "text/plain")}
    response = await client.post("/api/papers", files=files)
    assert response.status_code == 400


async def test_list_paper_notes(client: AsyncClient):
    created = await client.post(
        "/api/papers",
        files={"file": ("paper.pdf", b"%PDF-1.4\n%mock\n", "application/pdf")},
    )
    if created.status_code not in {201, 202}:
        pytest.skip("Could not upload a paper for the notes listing test")
    paper_id = created.json()["id"]
    note = Note(
        source_type=NoteSourceType.voice.value,
        title="Linked listing note",
        processing_status=ProcessingStatus.ready.value,
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
    )
    assert db.SessionLocal is not None
    async with db.SessionLocal() as session:
        session.add(note)
        await session.flush()
        session.add(NotePaper(note_id=note.id, paper_id=UUID(paper_id)))
        await session.commit()
        note_id = note.id
    try:
        listed = await client.get(f"/api/papers/{paper_id}/notes")
        assert listed.status_code == 200
        titles = [item["title"] for item in listed.json()]
        assert "Linked listing note" in titles
    finally:
        async with db.SessionLocal() as session:
            stored = await session.get(Note, note_id)
            if stored is not None:
                await session.delete(stored)
            await session.commit()
        await client.delete(f"/api/papers/{paper_id}")
