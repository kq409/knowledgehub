import json
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from unittest.mock import MagicMock
from uuid import uuid4

import pytest
from dotenv import load_dotenv
from sqlalchemy import text

import db
from models import DigestStatus, Paper, PaperChunk, PaperStatus
from services.paper_digest import (
    DigestError,
    generate_and_store_digest,
    parse_digest_payload,
    select_digest_source_text,
)

load_dotenv()


@dataclass
class FakeChunk:
    text: str
    section: str | None = None


def test_select_digest_source_prefers_intro_and_conclusion():
    text = select_digest_source_text(
        title="ELLA",
        authors=["Ruvolo"],
        year=2013,
        abstract="A lifelong learning method.",
        chunks=[
            FakeChunk("noise in related work", "Related Work"),
            FakeChunk("we introduce a shared basis", "Introduction"),
            FakeChunk("we conclude transfer works", "Conclusion"),
        ],
        char_budget=800,
    )
    assert "Title: ELLA" in text
    assert "2013" in text
    assert "A lifelong learning method." in text
    intro_at = text.find("we introduce a shared basis")
    related_at = text.find("noise in related work")
    assert intro_at != -1
    assert intro_at < related_at


def test_parse_digest_payload_requires_content():
    summary, digest = parse_digest_payload(
        {
            "summary": "A lifelong learner.",
            "problem": "Catastrophic forgetting",
            "method": "Shared basis",
            "key_results": "Transfer",
            "limitations": "",
        }
    )
    assert summary == "A lifelong learner."
    assert digest["method"] == "Shared basis"
    with pytest.raises(DigestError):
        parse_digest_payload(
            {
                "summary": "",
                "problem": "",
                "method": "",
                "key_results": "",
                "limitations": "",
            }
        )


@pytest.fixture
async def session():
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        pytest.skip("DATABASE_URL is not set")
    db.init_db(database_url)
    try:
        if db.SessionLocal is None:
            pytest.skip("Database session factory was not created")
        async with db.SessionLocal() as probe:
            await probe.execute(text("SELECT 1 FROM papers LIMIT 1"))
            await probe.execute(text("SELECT summary FROM papers LIMIT 1"))
    except Exception:
        await db.close_db()
        pytest.skip("PostgreSQL paper digest columns are not available")

    async with db.SessionLocal() as db_session:
        yield db_session
        await db_session.rollback()
    await db.close_db()


async def _seed_paper(session) -> Paper:
    now = datetime.now(UTC)
    paper = Paper(
        id=uuid4(),
        title="ELLA: Efficient Lifelong Learning",
        authors=["Ruvolo"],
        year=2013,
        abstract="A shared basis for lifelong learning.",
        tags=[],
        original_filename="ella.pdf",
        original_file=f"/tmp/{uuid4()}.pdf",
        processing_status=PaperStatus.ready.value,
        created_at=now,
        updated_at=now,
    )
    session.add(paper)
    await session.flush()
    session.add(
        PaperChunk(
            id=uuid4(),
            paper_id=paper.id,
            chunk_index=0,
            text="ELLA transfers knowledge across tasks.",
            page=1,
            section="Abstract",
            extra={},
            created_at=now,
        )
    )
    await session.commit()
    return paper


def _llm(payload: dict) -> MagicMock:
    client = MagicMock()
    client.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content=json.dumps(payload)))]
    )
    return client


async def test_generate_and_store_digest_persists(session):
    paper = await _seed_paper(session)
    client = _llm(
        {
            "summary": "ELLA learns tasks in sequence with a shared basis.",
            "problem": "Lifelong learning",
            "method": "Shared latent components",
            "key_results": "Transfer across tasks",
            "limitations": "Assumes related tasks",
        }
    )
    stored = await generate_and_store_digest(
        session, paper, llm_client=client, llm_model="test-model"
    )
    assert stored.digest_status == DigestStatus.ready.value
    assert stored.processing_status == PaperStatus.ready.value
    assert "shared basis" in (stored.summary or "")
    assert stored.digest["method"] == "Shared latent components"
    await session.delete(stored)
    await session.commit()


async def test_digest_failure_keeps_paper_ready(session):
    paper = await _seed_paper(session)
    client = MagicMock()
    client.chat.completions.create.side_effect = RuntimeError("llm down")
    stored = await generate_and_store_digest(
        session, paper, llm_client=client, llm_model="test-model"
    )
    assert stored.processing_status == PaperStatus.ready.value
    assert stored.digest_status == DigestStatus.failed.value
    await session.delete(stored)
    await session.commit()
