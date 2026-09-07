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
from models import (
    Note,
    NoteLinkSource,
    NotePaper,
    NotePaperSkip,
    NoteSourceType,
    Paper,
    PaperChunk,
    PaperStatus,
    ProcessingStatus,
    ReviewStatus,
)
from routers.notes import router
from services.connect import ConnectService
from services.embeddings import EmbeddingService

load_dotenv()

pytestmark = pytest.mark.asyncio

VEC = [0.1] * 768


def _completion(content: str) -> MagicMock:
    response = MagicMock()
    response.choices = [MagicMock(message=MagicMock(content=content))]
    return response


async def _table_exists(session, name: str) -> bool:
    try:
        await session.execute(text(f"SELECT 1 FROM {name} LIMIT 1"))
        return True
    except Exception:
        await session.rollback()
        return False


async def _insert_paper(session, title: str) -> Paper:
    paper = Paper(
        title=title,
        original_filename=f"{title}.pdf",
        original_file=f"/tmp/{uuid.uuid4()}.pdf",
        processing_status=PaperStatus.ready.value,
        year=2020,
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
    )
    session.add(paper)
    await session.flush()
    session.add(
        PaperChunk(
            paper_id=paper.id,
            chunk_index=0,
            text=f"{title} methods and hybrid retrieval.",
            page=1,
            section="Methods",
            embedding=VEC,
        )
    )
    await session.commit()
    await session.refresh(paper)
    return paper


async def _insert_note(session, title: str = "Hybrid idea") -> Note:
    note = Note(
        source_type=NoteSourceType.voice.value,
        title=title,
        summary="Compare dense search with BM25.",
        observations=["Dense search misses exact terms"],
        hypotheses=["Hybrid retrieval will improve recall"],
        questions=[],
        next_steps=[],
        tags=["rag"],
        raw_transcript="hybrid retrieval",
        cleaned_transcript="I think hybrid retrieval might help.",
        review_status=ReviewStatus.generated.value,
        processing_status=ProcessingStatus.ready.value,
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
    )
    session.add(note)
    await session.commit()
    await session.refresh(note)
    return note


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
            if not await _table_exists(session, "notes"):
                pytest.skip("PostgreSQL notes table is not available")
            if not await _table_exists(session, "note_paper_skips"):
                pytest.skip("PostgreSQL note_paper_skips table is not available")
    except Exception:
        await db.close_db()
        pytest.skip("PostgreSQL is not available")

    llm = MagicMock()
    embeddings = MagicMock(spec=EmbeddingService)
    embeddings.embed_query.return_value = VEC
    app = FastAPI()
    app.include_router(router)
    app.state.connect = ConnectService(
        llm_client=llm,
        llm_model="test-model",
        embeddings=embeddings,
        min_similarity=0.35,
    )

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac, llm
    await db.close_db()


async def test_connect_rejects_unready_note(api: tuple[AsyncClient, MagicMock]):
    client, _llm = api
    assert db.SessionLocal is not None
    async with db.SessionLocal() as session:
        note = Note(
            source_type=NoteSourceType.voice.value,
            title="Pending",
            summary="",
            processing_status=ProcessingStatus.pending.value,
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
        )
        session.add(note)
        await session.commit()
        await session.refresh(note)
        note_id = note.id
    try:
        response = await client.post(f"/api/notes/{note_id}/related", json={})
        assert response.status_code == 409
        assert "not ready" in response.json()["detail"].lower()
    finally:
        async with db.SessionLocal() as session:
            row = await session.get(Note, note_id)
            if row is not None:
                await session.delete(row)
            await session.commit()


async def test_connect_auto_links_and_stores_run(api: tuple[AsyncClient, MagicMock]):
    client, llm = api
    assert db.SessionLocal is not None
    async with db.SessionLocal() as session:
        paper = await _insert_paper(session, "Hybrid Retrieval Survey")
        note = await _insert_note(session)
        paper_id, note_id = paper.id, note.id

    llm.chat.completions.create.return_value = _completion(
        json.dumps(
            {
                "reasons": [
                    {
                        "paper_id": str(paper_id),
                        "reason": "Both discuss hybrid retrieval.",
                    }
                ]
            }
        )
    )

    try:
        created = await client.post(
            f"/api/notes/{note_id}/related",
            json={"reason_mode": "llm"},
        )
        assert created.status_code == 200, created.text
        body = created.json()
        assert body["note_id"] == str(note_id)
        assert body["papers"]
        assert body["papers"][0]["paper_id"] == str(paper_id)
        assert body["papers"][0]["reason"] == "Both discuss hybrid retrieval."
        assert body["papers"][0]["linked"] is True
        assert str(paper_id) in body["linked_paper_ids"]

        fetched = await client.get(f"/api/notes/{note_id}/related")
        assert fetched.status_code == 200
        assert fetched.json()["papers"][0]["reason"] == "Both discuss hybrid retrieval."

        note_body = await client.get(f"/api/notes/{note_id}")
        assert note_body.status_code == 200
        payload = note_body.json()
        assert str(paper_id) in payload["paper_ids"]
        assert payload["paper_links"][0]["source"] == "ai"
        assert payload["related_generated_at"]
    finally:
        async with db.SessionLocal() as session:
            note = await session.get(Note, note_id)
            paper = await session.get(Paper, paper_id)
            if note is not None:
                await session.delete(note)
            if paper is not None:
                await session.delete(paper)
            await session.commit()


async def test_connect_snippet_mode_skips_llm(api: tuple[AsyncClient, MagicMock]):
    client, llm = api
    assert db.SessionLocal is not None
    async with db.SessionLocal() as session:
        paper = await _insert_paper(session, "Snippet Paper")
        note = await _insert_note(session, "Snippet note")
        paper_id, note_id = paper.id, note.id

    try:
        created = await client.post(
            f"/api/notes/{note_id}/related",
            json={"reason_mode": "snippet"},
        )
        assert created.status_code == 200, created.text
        body = created.json()
        assert body["reason_mode"] == "snippet"
        assert body["papers"][0]["reason"]
        llm.chat.completions.create.assert_not_called()
    finally:
        async with db.SessionLocal() as session:
            note = await session.get(Note, note_id)
            paper = await session.get(Paper, paper_id)
            if note is not None:
                await session.delete(note)
            if paper is not None:
                await session.delete(paper)
            await session.commit()


async def test_connect_falls_back_to_snippet_when_llm_fails(
    api: tuple[AsyncClient, MagicMock],
):
    client, llm = api
    assert db.SessionLocal is not None
    async with db.SessionLocal() as session:
        paper = await _insert_paper(session, "Fallback Paper")
        note = await _insert_note(session, "Fallback note")
        paper_id, note_id = paper.id, note.id

    llm.chat.completions.create.return_value = _completion("not json at all")

    try:
        created = await client.post(
            f"/api/notes/{note_id}/related",
            json={"reason_mode": "llm"},
        )
        assert created.status_code == 200, created.text
        body = created.json()
        assert body["papers"][0]["paper_id"] == str(paper_id)
        assert body["papers"][0]["linked"] is True
        assert body["reason_mode"] == "snippet"
    finally:
        async with db.SessionLocal() as session:
            note = await session.get(Note, note_id)
            paper = await session.get(Paper, paper_id)
            if note is not None:
                await session.delete(note)
            if paper is not None:
                await session.delete(paper)
            await session.commit()


async def test_unlinking_ai_paper_skips_it_on_rerun(
    api: tuple[AsyncClient, MagicMock],
):
    client, llm = api
    assert db.SessionLocal is not None
    async with db.SessionLocal() as session:
        paper = await _insert_paper(session, "Skipped Survey")
        note = await _insert_note(session, "Skip me")
        paper_id, note_id = paper.id, note.id

    llm.chat.completions.create.return_value = _completion(
        json.dumps(
            {"reasons": [{"paper_id": str(paper_id), "reason": "Related work."}]}
        )
    )

    try:
        first = await client.post(f"/api/notes/{note_id}/related", json={})
        assert first.status_code == 200, first.text
        assert str(paper_id) in first.json()["linked_paper_ids"]

        cleared = await client.patch(f"/api/notes/{note_id}", json={"paper_ids": []})
        assert cleared.status_code == 200, cleared.text
        assert cleared.json()["paper_ids"] == []

        async with db.SessionLocal() as session:
            skip = await session.get(NotePaperSkip, (note_id, paper_id))
            assert skip is not None

        second = await client.post(f"/api/notes/{note_id}/related", json={})
        assert second.status_code == 200, second.text
        assert str(paper_id) not in second.json()["linked_paper_ids"]
        assert str(paper_id) in second.json()["skipped_paper_ids"]
    finally:
        async with db.SessionLocal() as session:
            note = await session.get(Note, note_id)
            paper = await session.get(Paper, paper_id)
            if note is not None:
                await session.delete(note)
            if paper is not None:
                await session.delete(paper)
            await session.commit()


async def test_researcher_links_are_left_alone(api: tuple[AsyncClient, MagicMock]):
    client, llm = api
    assert db.SessionLocal is not None
    async with db.SessionLocal() as session:
        paper = await _insert_paper(session, "Researcher Linked")
        note = await _insert_note(session, "Keep researcher")
        paper_id, note_id = paper.id, note.id
        session.add(
            NotePaper(
                note_id=note_id,
                paper_id=paper_id,
                source=NoteLinkSource.researcher.value,
                created_at=datetime.now(UTC),
            )
        )
        await session.commit()

    llm.chat.completions.create.return_value = _completion(
        json.dumps(
            {"reasons": [{"paper_id": str(paper_id), "reason": "Still related."}]}
        )
    )

    try:
        created = await client.post(f"/api/notes/{note_id}/related", json={})
        assert created.status_code == 200, created.text
        async with db.SessionLocal() as session:
            link = await session.get(NotePaper, (note_id, paper_id))
            assert link is not None
            assert link.source == NoteLinkSource.researcher.value
    finally:
        async with db.SessionLocal() as session:
            note = await session.get(Note, note_id)
            paper = await session.get(Paper, paper_id)
            if note is not None:
                await session.delete(note)
            if paper is not None:
                await session.delete(paper)
            await session.commit()
