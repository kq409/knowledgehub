import os

import pytest
from dotenv import load_dotenv
from sqlalchemy import text

import db
from eval.fixtures import KeywordEmbeddingService, cleanup_eval_fixtures, seed_library
from eval.graders import retrieval_titles
from services.retrieval import search

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


async def test_seeded_lexical_query_retrieves_ella(session):
    embeddings = KeywordEmbeddingService()
    seed = await seed_library(session)
    try:
        hits = await search(
            session,
            embeddings.embed_query("ZXQELLA7"),
            query_text="ZXQELLA7",
            paper_ids=seed.paper_ids,
        )
        titles = [hit.title for hit in hits]
        grade = retrieval_titles(
            titles,
            must_include=["ZXQELLA7: Efficient Lifelong Learning Algorithm"],
            k=8,
        )
        assert grade.passed, grade.detail
    finally:
        await cleanup_eval_fixtures(session)
