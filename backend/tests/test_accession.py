import os
from pathlib import Path

import pytest
from dotenv import load_dotenv
from sqlalchemy import text

import db
from eval.discipline_island import (
    RECEIVED_PAPER_ID,
    RECEIVED_TOKEN,
    seed_discipline_island,
)
from models import AccessionStatus, Paper, PaperStatus
from services.accession import EMPTY_ORIGINAL
from services.identity import load_identity
from services.library_ingest import append_revision
from services.library_records import list_records, search_records
from services.paper_parser import ParsedPaper
from services.paper_pipeline import apply_parse_result
from services.retrieval import search
from services.storage import LocalDiskStorage, sha256_hex

load_dotenv()

pytestmark = pytest.mark.asyncio

ZERO = [0.0] * 768


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
        if not await _table_exists(db_session, "papers"):
            pytest.skip("PostgreSQL papers table is not available")
        await seed_discipline_island(db_session)
        yield db_session
    await db.close_db()


async def test_received_paper_is_listed_but_not_searchable(session):
    alice = await load_identity(session, "alice")
    listed = await list_records(session, space_ids=alice.space_ids)
    assert any(item.id == RECEIVED_PAPER_ID for item in listed.items)
    catalog = await search_records(session, RECEIVED_TOKEN, space_ids=alice.space_ids)
    assert all(item.id != RECEIVED_PAPER_ID for item in catalog.items)
    hits = await search(
        session, ZERO, query_text=RECEIVED_TOKEN, space_ids=alice.space_ids
    )
    assert all(hit.source_id != RECEIVED_PAPER_ID for hit in hits)


async def test_empty_parse_rejects_accession(session):
    paper = Paper(
        title="Blank scan",
        authors=[],
        tags=["discipline-test"],
        original_filename="blank.pdf",
        original_file="/tmp/blank.pdf",
        processing_status=PaperStatus.pending.value,
        accession_status=AccessionStatus.received.value,
    )
    session.add(paper)
    await session.commit()
    await session.refresh(paper)
    try:
        await apply_parse_result(
            session, paper, ParsedPaper(title="Blank scan", chunks=[]), []
        )
        stored = await session.get(Paper, paper.id)
        assert stored.processing_status == PaperStatus.failed.value
        assert stored.accession_status == AccessionStatus.rejected.value
        assert stored.processing_error == EMPTY_ORIGINAL
    finally:
        await session.delete(paper)
        await session.commit()


async def test_append_revision_stores_new_key(session, tmp_path: Path, monkeypatch):
    storage = LocalDiskStorage(tmp_path)
    monkeypatch.setattr("services.library_ingest.default_storage", lambda: storage)
    first = storage.put("papers/placeholder/r1.pdf", b"rev-one")
    paper = Paper(
        title="Revisioned",
        authors=[],
        tags=["discipline-test"],
        original_filename="v1.pdf",
        original_file=str(first),
        processing_status=PaperStatus.ready.value,
        revision=1,
        sha256=sha256_hex(b"rev-one"),
        accession_status=AccessionStatus.accessioned.value,
    )
    session.add(paper)
    await session.commit()
    await session.refresh(paper)
    try:
        next_rev = await append_revision(session, paper, b"rev-two", "v2.pdf")
        assert next_rev == 2
        assert paper.revision == 2
        assert paper.sha256 == sha256_hex(b"rev-two")
        assert paper.accession_status == AccessionStatus.received.value
        assert paper.processing_status == PaperStatus.pending.value
        assert Path(paper.original_file).read_bytes() == b"rev-two"
        assert storage.get("papers/placeholder/r1.pdf") == b"rev-one"
    finally:
        await session.delete(paper)
        await session.commit()
