import os

import pytest
from dotenv import load_dotenv
from httpx import ASGITransport, AsyncClient
from sqlalchemy import text

import db
from app import app
from services.agent.memory import upsert_memory

load_dotenv()


@pytest.fixture
async def client():
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        pytest.skip("DATABASE_URL is not set")
    db.init_db(database_url)
    try:
        if db.SessionLocal is None:
            pytest.skip("Database session factory was not created")
        async with db.SessionLocal() as probe:
            await probe.execute(text("SELECT 1 FROM agent_memories LIMIT 1"))
    except Exception:
        await db.close_db()
        pytest.skip("PostgreSQL or agent_memories table is not available")

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac

    async with db.SessionLocal() as session:
        await session.execute(text("DELETE FROM agent_memories"))
        await session.commit()
    await db.close_db()


async def test_list_and_delete_memories(client):
    async with db.SessionLocal() as session:
        await upsert_memory(
            session,
            key="focus:demo",
            content="Demo focus area",
            category="focus",
        )

    listed = await client.get("/api/memories")
    assert listed.status_code == 200
    payload = listed.json()
    assert any(item["key"] == "focus:demo" for item in payload)

    deleted = await client.delete("/api/memories/focus:demo")
    assert deleted.status_code == 204

    missing = await client.delete("/api/memories/focus:demo")
    assert missing.status_code == 404

    empty = await client.get("/api/memories")
    assert empty.json() == []
