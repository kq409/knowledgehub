import os
from collections.abc import AsyncGenerator

import pytest
from dotenv import load_dotenv
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from sqlalchemy import text

import db
from eval.library_island import (
    ALICE_BUDGET_NUMBER,
    ALICE_BUDGET_PAPER_ID,
    BOB_ACCEPTANCE_NUMBER,
    BOB_ACCEPTANCE_PAPER_ID,
    BOB_INJECTION_DOC_ID,
    FINANCE_SPACE_ID,
    INJECTION_TOKEN,
    seed_library_island,
)
from models import DEFAULT_SPACE_ID
from routers.records import router as records_router
from services.identity import load_identity
from services.library_records import list_records, search_records
from services.retrieval import search

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
        await seed_library_island(db_session)
        yield db_session
    await db.close_db()


@pytest.fixture
async def client(session) -> AsyncGenerator[AsyncClient, None]:
    app = FastAPI()
    app.include_router(records_router)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


async def test_alice_records_hide_bob_paper(session):
    alice = await load_identity(session, "alice")
    page = await list_records(session, space_ids=alice.space_ids)
    ids = {item.id for item in page.items}
    assert ALICE_BUDGET_PAPER_ID in ids
    assert BOB_ACCEPTANCE_PAPER_ID not in ids
    assert BOB_INJECTION_DOC_ID not in ids


async def test_catalog_search_hits_doc_number(session):
    alice = await load_identity(session, "alice")
    page = await search_records(session, ALICE_BUDGET_NUMBER, space_ids=alice.space_ids)
    assert any(item.id == ALICE_BUDGET_PAPER_ID for item in page.items)
    bob_miss = await search_records(
        session, BOB_ACCEPTANCE_NUMBER, space_ids=alice.space_ids
    )
    assert all(item.id != BOB_ACCEPTANCE_PAPER_ID for item in bob_miss.items)


async def test_catalog_pagination_and_facet(session):
    alice = await load_identity(session, "alice")
    first = await list_records(
        session,
        space_ids=alice.space_ids,
        content_type="scholarly_article",
        limit=1,
        offset=0,
    )
    assert first.limit == 1
    assert first.total >= 1
    finance = await list_records(session, space_ids=frozenset({FINANCE_SPACE_ID}))
    ids = {item.id for item in finance.items}
    assert ids == {ALICE_BUDGET_PAPER_ID}


async def test_rag_record_ids_exclude_other_titles(session):
    alice = await load_identity(session, "alice")
    hits = await search(
        session,
        ZERO,
        query_text="ZXQELLA7",
        space_ids=alice.space_ids | {DEFAULT_SPACE_ID},
        record_ids=[ALICE_BUDGET_PAPER_ID],
    )
    assert all(hit.source_id == ALICE_BUDGET_PAPER_ID for hit in hits)


async def test_api_search_respects_user_header(client: AsyncClient):
    alice = await client.post(
        "/api/search",
        json={"query": INJECTION_TOKEN},
        headers={"X-User-Id": "alice"},
    )
    assert alice.status_code == 200
    assert all(
        item["id"] != str(BOB_INJECTION_DOC_ID) for item in alice.json()["items"]
    )
    bob = await client.post(
        "/api/search",
        json={"query": INJECTION_TOKEN},
        headers={"X-User-Id": "bob"},
    )
    assert bob.status_code == 200
    assert any(item["id"] == str(BOB_INJECTION_DOC_ID) for item in bob.json()["items"])


async def test_api_spaces_are_membership_scoped(client: AsyncClient):
    alice = await client.get("/api/spaces", headers={"X-User-Id": "alice"})
    slugs = {item["slug"] for item in alice.json()}
    assert "finance" in slugs
    assert "engineering" not in slugs
    unknown = await client.get("/api/spaces", headers={"X-User-Id": "mallory"})
    assert unknown.json() == []
