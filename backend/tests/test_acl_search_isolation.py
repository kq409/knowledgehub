import os
from collections.abc import AsyncGenerator

import pytest
from dotenv import load_dotenv
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

import db
from eval.library_island import (
    BOB_ACCEPTANCE_NUMBER,
    BOB_ACCEPTANCE_PAPER_ID,
    BOB_ACCEPTANCE_TITLE,
    INJECTION_TOKEN,
    cleanup_library_island,
    seed_library_island,
)
from models import Paper
from services.agent.permissions import DEFAULT_POLICY, Decision
from services.agent.tools import (
    CitationRegistry,
    ToolContext,
    ToolError,
    read_paper,
    search_library,
)
from services.identity import HIDDEN_PAPER, load_identity
from services.retrieval import search

load_dotenv()

pytestmark = pytest.mark.asyncio


@pytest.fixture
async def session() -> AsyncGenerator[AsyncSession, None]:
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        pytest.skip("DATABASE_URL is not set")
    db.init_db(database_url)
    try:
        if db.SessionLocal is None:
            pytest.skip("Database session factory was not created")
        async with db.SessionLocal() as probe:
            await probe.execute(text("SELECT 1 FROM spaces LIMIT 1"))
    except Exception:
        await db.close_db()
        pytest.skip("PostgreSQL spaces table is not available")

    async with db.SessionLocal() as db_session:
        await seed_library_island(db_session)
        yield db_session
        await cleanup_library_island(db_session)
    await db.close_db()


async def test_alice_search_does_not_return_bob_paper(session: AsyncSession):
    alice = await load_identity(session, "alice")
    embedding = [0.0] * 768
    hits = await search(
        session,
        embedding,
        query_text=BOB_ACCEPTANCE_NUMBER,
        space_ids=alice.space_ids,
    )
    leaked = {hit.source_id for hit in hits}
    assert BOB_ACCEPTANCE_PAPER_ID not in leaked
    assert all(BOB_ACCEPTANCE_NUMBER not in (hit.title or "") for hit in hits)


async def test_bob_search_returns_own_paper(session: AsyncSession):
    bob = await load_identity(session, "bob")
    embedding = [0.0] * 768
    hits = await search(
        session,
        embedding,
        query_text=BOB_ACCEPTANCE_NUMBER,
        space_ids=bob.space_ids,
    )
    found = {hit.source_id for hit in hits}
    assert BOB_ACCEPTANCE_PAPER_ID in found


async def test_alice_read_paper_does_not_leak_title(session: AsyncSession):
    alice = await load_identity(session, "alice")
    ctx = ToolContext(
        session=session,
        embeddings=object(),  # type: ignore[arg-type]
        registry=CitationRegistry(),
        user_id=alice.user_id,
        space_ids=alice.space_ids,
    )
    with pytest.raises(ToolError) as raised:
        await read_paper(ctx, paper_id=str(BOB_ACCEPTANCE_PAPER_ID))
    message = str(raised.value)
    assert message == HIDDEN_PAPER
    assert BOB_ACCEPTANCE_TITLE not in message
    assert BOB_ACCEPTANCE_NUMBER not in message


async def test_alice_search_library_tool_hides_bob_paper(session: AsyncSession):
    alice = await load_identity(session, "alice")

    class Embeddings:
        def embed_query(self, text: str) -> list[float]:
            return [0.0] * 768

    ctx = ToolContext(
        session=session,
        embeddings=Embeddings(),  # type: ignore[arg-type]
        registry=CitationRegistry(),
        user_id=alice.user_id,
        space_ids=alice.space_ids,
    )
    result = await search_library(ctx, query=BOB_ACCEPTANCE_NUMBER)
    assert BOB_ACCEPTANCE_TITLE not in result.content
    assert str(BOB_ACCEPTANCE_PAPER_ID) not in result.content


async def test_unknown_user_sees_no_papers(session: AsyncSession):
    stranger = await load_identity(session, "eve")
    assert stranger.space_ids == frozenset()
    embedding = [0.0] * 768
    hits = await search(
        session,
        embedding,
        query_text=BOB_ACCEPTANCE_NUMBER,
        space_ids=stranger.space_ids,
    )
    assert hits == []


async def test_injection_string_does_not_bypass_memory_delete_policy():
    verdict = DEFAULT_POLICY.check("memory_delete")
    assert verdict.decision is Decision.ask
    assert INJECTION_TOKEN


async def test_bob_paper_exists_in_db(session: AsyncSession):
    paper = await session.get(Paper, BOB_ACCEPTANCE_PAPER_ID)
    assert paper is not None
    assert paper.title == BOB_ACCEPTANCE_TITLE
