import os

import pytest
from dotenv import load_dotenv
from sqlalchemy import text

import db
from eval.fixtures import KeywordEmbeddingService, cleanup_eval_fixtures, seed_library
from mcp_server import invoke_readonly_tool
from services.agent.tools import SUBAGENT_TOOL_NAMES, ToolError

load_dotenv()

pytestmark = pytest.mark.asyncio


@pytest.fixture
async def session():
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        pytest.skip("DATABASE_URL is not set")
    db.init_db(database_url)
    if db.SessionLocal is None:
        pytest.skip("Database session factory was not created")
    async with db.SessionLocal() as db_session:
        try:
            await db_session.execute(text("SELECT 1 FROM papers LIMIT 1"))
        except Exception:
            await db_session.rollback()
            pytest.skip("PostgreSQL papers table is not available")
        yield db_session
    await db.close_db()


async def test_mcp_list_papers_includes_seed_title(session):
    embeddings = KeywordEmbeddingService()
    await seed_library(session)
    try:
        listed = await invoke_readonly_tool(
            "list_papers", session=session, embeddings=embeddings
        )
        assert "ZXQELLA7" in listed
        found = await invoke_readonly_tool(
            "search_library",
            {"query": "ZXQELLA7"},
            session=session,
            embeddings=embeddings,
        )
        assert "ZXQELLA7" in found
    finally:
        await cleanup_eval_fixtures(session)


async def test_mcp_rejects_write_tools(session):
    embeddings = KeywordEmbeddingService()
    with pytest.raises(ToolError, match="not exposed"):
        await invoke_readonly_tool(
            "memory_write",
            {"key": "x", "content": "y"},
            session=session,
            embeddings=embeddings,
        )
    assert "memory_write" not in SUBAGENT_TOOL_NAMES
