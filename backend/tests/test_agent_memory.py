import json
import os
from dataclasses import dataclass
from unittest.mock import MagicMock

import pytest
from dotenv import load_dotenv
from sqlalchemy import text

import db
from schemas import ChatEventType, ChatMessage, ChatRequest, ChatRole
from services.agent.loop import PROMPT_VERSION, ResearchAgent
from services.agent.memory import (
    SMALL_STORE,
    MemoryError,
    catalog_line,
    delete_memory,
    extract_and_store,
    format_recalled_memories,
    search_memories,
    select_relevant_memories,
    should_consider_extract,
    should_store_memory,
    upsert_memory,
)
from services.agent.protocol import ToolCapabilityCache
from services.agent.tools import ToolContext, memory_delete, memory_search, memory_write

load_dotenv()

MODEL = "test-model"


@dataclass
class FakeMemory:
    key: str
    content: str
    category: str = "other"


def reply(content: str) -> MagicMock:
    return MagicMock(
        choices=[MagicMock(message=MagicMock(content=content, tool_calls=None))]
    )


def text_protocol_agent(client: MagicMock) -> ResearchAgent:
    capabilities = ToolCapabilityCache()
    capabilities.set(MODEL, False)
    return ResearchAgent(
        llm_client=client,
        llm_model=MODEL,
        embeddings=MagicMock(),
        capabilities=capabilities,
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


def test_select_relevant_memories_ranks_keyword_hits():
    preference = FakeMemory(
        key="preferred_compare_dimensions",
        content="method, dataset",
        category="preference",
    )
    hypothesis = FakeMemory(
        key="hypothesis:scaling",
        content="Scaling laws may not hold for robot learning",
        category="hypothesis",
    )
    hits = select_relevant_memories(
        [preference, hypothesis],
        "Compare papers using my preferred dimensions",
    )
    assert [item.key for item in hits] == ["preferred_compare_dimensions"]


def test_format_recalled_memories_injects_all_in_small_store():
    rows = [
        FakeMemory("preferred_compare_dimensions", "method, dataset", "preference"),
        FakeMemory("focus:robots", "Lifelong robot learning", "focus"),
    ]
    text = format_recalled_memories(rows, "anything unrelated")
    assert "preferred_compare_dimensions" in text
    assert "focus:robots" in text
    assert "Lifelong robot learning" in text
    assert "Memory catalog:" not in text
    assert "takes priority" in text


def test_format_recalled_memories_large_store_selects_hits():
    rows = [
        FakeMemory(f"noise-{index}", f"unrelated fact {index}", "other")
        for index in range(SMALL_STORE + 1)
    ]
    rows.append(
        FakeMemory(
            "preferred_compare_dimensions",
            "Always use method and dataset",
            "preference",
        )
    )
    text = format_recalled_memories(rows, "What compare dimensions do I prefer?")
    assert "Memory catalog:" in text
    assert "preferred_compare_dimensions" in text
    assert "Always use method and dataset" in text
    assert "unrelated fact 0" not in text.split("Relevant memory records:")[-1]


def test_should_consider_extract_skips_library_questions():
    assert (
        should_consider_extract("What does the ELLA paper say about robots?") is False
    )
    assert should_consider_extract("Remember I prefer method and dataset") is True
    assert should_consider_extract("From now on focus on lifelong learning") is True


def test_should_store_memory_rejects_temporary_and_dupes():
    existing = [FakeMemory("preferred_compare_dimensions", "method, dataset")]
    valid = {
        "key": "focus:robots",
        "content": "Lifelong robot learning",
        "category": "focus",
        "scope": "persistent",
    }
    assert should_store_memory(valid, existing) is True
    assert should_store_memory({**valid, "scope": "current_task"}, existing) is False
    assert (
        should_store_memory(
            {**valid, "content": "Only for this session, skip extra files"},
            existing,
        )
        is False
    )
    assert (
        should_store_memory(
            {**valid, "content": "See finding [1] in the paper"},
            existing,
        )
        is False
    )
    assert (
        should_store_memory(
            {
                "key": "other-key",
                "content": "method, dataset",
                "category": "preference",
                "scope": "persistent",
            },
            existing,
        )
        is False
    )


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


async def test_upsert_skips_duplicate_content_under_new_key(session):
    first = await upsert_memory(
        session,
        key="focus:robots",
        content="Lifelong robot learning",
        category="focus",
    )
    skipped = await upsert_memory(
        session,
        key="research-focus",
        content="lifelong  robot   learning",
        category="focus",
    )
    assert skipped.key == first.key
    rows = await search_memories(session, limit=50)
    assert [row.key for row in rows] == ["focus:robots"]


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
    assert "Lifelong learning papers" in found.content

    deleted = await memory_delete(ctx, key="focus:robot-learning")
    assert deleted.artifact["data"]["action"] == "forgot"


async def test_memory_search_empty_query_returns_catalog_only(session):
    long_content = "Always compare on method, dataset, and evaluation " * 8
    await upsert_memory(
        session,
        key="preferred_compare_dimensions",
        content=long_content,
        category="preference",
    )
    ctx = ToolContext(session=session, embeddings=MagicMock())
    catalog = await memory_search(ctx, query="")
    assert "catalog" in catalog.content.lower()
    assert (
        catalog_line(
            FakeMemory("preferred_compare_dimensions", long_content, "preference")
        )
        in catalog.content
    )
    assert long_content not in catalog.content

    listed = await memory_search(ctx)
    assert long_content not in listed.content


async def test_extract_and_store_skips_library_questions(session):
    client = MagicMock()
    stored = await extract_and_store(
        session,
        llm_client=client,
        llm_model=MODEL,
        user_text="What papers are about robot learning?",
        answer="The library has one 2006 paper.",
    )
    assert stored == []
    client.chat.completions.create.assert_not_called()


async def test_extract_and_store_persists_admitted_memory(session):
    client = MagicMock()
    client.chat.completions.create.return_value = reply(
        json.dumps(
            {
                "memories": [
                    {
                        "key": "preferred_compare_dimensions",
                        "category": "preference",
                        "content": "method, dataset",
                        "scope": "persistent",
                    },
                    {
                        "key": "temp-skip",
                        "category": "workflow",
                        "content": "Do not create files in this session",
                        "scope": "current_task",
                    },
                ]
            }
        )
    )
    stored = await extract_and_store(
        session,
        llm_client=client,
        llm_model=MODEL,
        user_text="Remember I prefer method and dataset when comparing.",
        answer="I will keep that in mind.",
    )
    assert [row.key for row in stored] == ["preferred_compare_dimensions"]
    hits = await search_memories(session, query="method")
    assert any(item.key == "preferred_compare_dimensions" for item in hits)


async def test_agent_recalls_memory_without_memory_search(session):
    await upsert_memory(
        session,
        key="preferred_compare_dimensions",
        content="method, dataset",
        category="preference",
    )
    client = MagicMock()
    client.chat.completions.create.side_effect = [
        reply("Your library holds papers from 2006."),
        reply("Your library holds papers from 2006."),
    ]
    agent = text_protocol_agent(client)
    events = [
        event
        async for event in agent.run(
            session,
            ChatRequest(
                messages=[
                    ChatMessage(
                        role=ChatRole.user,
                        content="Compare the papers in the library.",
                    )
                ]
            ),
        )
    ]
    first_messages = client.chat.completions.create.call_args_list[0].kwargs["messages"]
    system = first_messages[0]["content"]
    assert "preferred_compare_dimensions" in system
    assert "method, dataset" in system
    assert "current request wins" in system.lower() or "takes priority" in system
    tools = [e.data.get("name") for e in events if e.type is ChatEventType.tool_call]
    assert "memory_search" not in tools
    assert events[-1].data["prompt_version"] == PROMPT_VERSION


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
        reply(json.dumps({"memories": []})),
    ]
    agent = text_protocol_agent(client)
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
