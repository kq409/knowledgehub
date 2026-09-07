import json
import os
from unittest.mock import MagicMock

import pytest
from dotenv import load_dotenv
from sqlalchemy import text

import db
from schemas import ChatEventType, ChatMessage, ChatRequest, ChatRole
from services.agent.loop import PROMPT_VERSION, ResearchAgent
from services.agent.memory import (
    MemoryError,
    delete_memory,
    search_memories,
    upsert_memory,
)
from services.agent.protocol import ToolCapabilityCache
from services.agent.tools import ToolContext, memory_delete, memory_search, memory_write

load_dotenv()

MODEL = "test-model"


def reply(content: str) -> MagicMock:
    return MagicMock(
        choices=[MagicMock(message=MagicMock(content=content, tool_calls=None))]
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
            await probe.execute(text("SELECT 1 FROM agent_memories LIMIT 1"))
    except Exception:
        await db.close_db()
        pytest.skip("PostgreSQL or agent_memories table is not available")

    async with db.SessionLocal() as session:
        yield session
        await session.execute(text("DELETE FROM agent_memories"))
        await session.commit()
    await db.close_db()


async def test_upsert_search_and_delete(session):
    row = await upsert_memory(
        session,
        key="preferred_compare_dimensions",
        content="method, dataset",
        category="preference",
    )
    assert row.key == "preferred_compare_dimensions"

    hits = await search_memories(session, query="method")
    assert any(item.key == "preferred_compare_dimensions" for item in hits)

    updated = await upsert_memory(
        session,
        key="preferred_compare_dimensions",
        content="method only",
        category="preference",
    )
    assert updated.content == "method only"

    assert await delete_memory(session, key="preferred_compare_dimensions")
    assert await search_memories(session, query="preferred") == []


async def test_rejects_bad_category(session):
    with pytest.raises(MemoryError, match="category"):
        await upsert_memory(session, key="x", content="y", category="not-a-category")


async def test_memory_tools_emit_artifacts(session):
    ctx = ToolContext(session=session, embeddings=MagicMock())
    written = await memory_write(
        ctx,
        key="focus:robot-learning",
        content="Lifelong learning papers",
        category="focus",
    )
    assert written.artifact["kind"] == "memory"
    assert written.artifact["data"]["action"] == "remembered"

    found = await memory_search(ctx, query="Lifelong")
    assert "focus:robot-learning" in found.content

    deleted = await memory_delete(ctx, key="focus:robot-learning")
    assert deleted.artifact["data"]["action"] == "forgot"


async def test_agent_memory_write_emits_artifact_event(session):
    client = MagicMock()
    client.chat.completions.create.side_effect = [
        reply(
            json.dumps(
                {
                    "tool": "memory_write",
                    "input": {
                        "key": "preferred_compare_dimensions",
                        "content": "method, dataset",
                        "category": "preference",
                    },
                }
            )
        ),
        reply("I will remember that."),
    ]
    capabilities = ToolCapabilityCache()
    capabilities.set(MODEL, False)
    agent = ResearchAgent(
        llm_client=client,
        llm_model=MODEL,
        embeddings=MagicMock(),
        capabilities=capabilities,
    )
    events = [
        event
        async for event in agent.run(
            session,
            ChatRequest(
                messages=[
                    ChatMessage(
                        role=ChatRole.user,
                        content="Remember my compare dims",
                    )
                ]
            ),
        )
    ]
    artifacts = [e for e in events if e.type is ChatEventType.artifact]
    assert any(a.data.get("kind") == "memory" for a in artifacts)
    assert events[-1].data["prompt_version"] == PROMPT_VERSION
