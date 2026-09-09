"""Vector recall, recency decay, and consolidation with rollback."""

import json
import os
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock

import pytest
from dotenv import load_dotenv
from sqlalchemy import select, text

import db
from models import EMBEDDING_DIM, AgentMemory
from services.agent.memory import (
    CONSOLIDATE_THRESHOLD,
    EXTRACT_SOURCE_TURN,
    RECALL_HALF_LIFE_DAYS,
    TOOL_SOURCE_PREFIX,
    consolidate_memories,
    recall_memories,
    search_memories,
    select_relevant_memories,
    upsert_memory,
)
from services.agent.tools import ToolContext, memory_write

load_dotenv()

MODEL = "test-model"


def unit(index: int) -> list[float]:
    vector = [0.0] * EMBEDDING_DIM
    vector[index % EMBEDDING_DIM] = 1.0
    return vector


def embedder(vectors: dict[str, list[float]], default: list[float]) -> MagicMock:
    """An embedding service that maps a marker in the text to a fixed vector."""
    service = MagicMock()

    def embed(payload: str) -> list[float]:
        for marker, vector in vectors.items():
            if marker in payload:
                return vector
        return default

    service.embed_document.side_effect = embed
    service.embed_query.side_effect = embed
    return service


@dataclass
class Stamped:
    key: str
    content: str
    category: str = "preference"
    last_recalled_at: datetime | None = None


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
            await probe.execute(text("SELECT embedding FROM agent_memories LIMIT 1"))
    except Exception:
        await db.close_db()
        pytest.skip("PostgreSQL or the memory embedding column is not available")

    async with db.SessionLocal() as session:
        await session.execute(text("DELETE FROM agent_memories"))
        await session.commit()
        yield session
        await session.execute(text("DELETE FROM agent_memories"))
        await session.commit()
    await db.close_db()


def test_a_stale_memory_loses_a_tie_to_a_fresh_one():
    """Both match the query; only recency separates them."""
    now = datetime.now(UTC)
    fresh = Stamped(key="fresh_pref", content="ablation tables", last_recalled_at=now)
    stale = Stamped(
        key="stale_pref",
        content="ablation tables",
        last_recalled_at=now - timedelta(days=RECALL_HALF_LIFE_DAYS * 6),
    )

    ranked = select_relevant_memories([stale, fresh], "ablation tables", 2)

    assert [row.key for row in ranked] == ["fresh_pref", "stale_pref"]


def test_recency_never_outranks_a_better_match():
    """A decayed but relevant memory still beats a fresh irrelevant one."""
    now = datetime.now(UTC)
    relevant = Stamped(
        key="old_but_right",
        content="always report ablation tables and dataset splits",
        last_recalled_at=now - timedelta(days=RECALL_HALF_LIFE_DAYS * 8),
    )
    irrelevant = Stamped(
        key="new_but_wrong", content="prefers dark mode", last_recalled_at=now
    )

    ranked = select_relevant_memories(
        [irrelevant, relevant], "ablation tables and dataset splits", 1
    )

    assert [row.key for row in ranked] == ["old_but_right"]


async def test_vector_recall_finds_a_memory_no_keyword_shares(session):
    """The keyword path cannot connect 'search quality' to 'retrieval'."""
    embeddings = embedder(
        {"retrieval": unit(3), "dark mode": unit(200)}, default=unit(3)
    )
    await upsert_memory(
        session,
        key="retrieval_focus",
        content="I work on retrieval for scientific papers",
        category="focus",
        embeddings=embeddings,
    )
    await upsert_memory(
        session,
        key="ui_pref",
        content="dark mode please",
        category="preference",
        embeddings=embeddings,
    )

    question = "am I making progress on finding relevant literature?"

    rows = await search_memories(session)
    selected = await recall_memories(session, rows, question, embeddings=embeddings)

    assert [row.key for row in selected] == ["retrieval_focus"]
    # The point of the change: substring counting shares nothing with this
    # phrasing, so the keyword path returns nothing at all.
    assert select_relevant_memories(rows, question) == []


async def test_recall_stamps_what_it_used(session):
    embeddings = embedder({}, default=unit(4))
    await upsert_memory(
        session,
        key="compare_dims",
        content="always compare on method and dataset",
        category="preference",
        embeddings=embeddings,
    )

    rows = await search_memories(session)
    assert rows[0].last_recalled_at is None

    await recall_memories(session, rows, "compare on method", embeddings=embeddings)

    refreshed = await search_memories(session)
    assert refreshed[0].last_recalled_at is not None


async def test_a_memory_written_without_embeddings_is_still_recallable(session):
    """The embedding host being down must not lose the write."""
    broken = MagicMock()
    broken.embed_document.side_effect = RuntimeError("embeddings offline")
    broken.embed_query.side_effect = RuntimeError("embeddings offline")

    row = await upsert_memory(
        session,
        key="ablation_pref",
        content="always include ablation tables",
        category="preference",
        embeddings=broken,
    )
    assert row.embedding is None

    rows = await search_memories(session)
    selected = await recall_memories(
        session, rows, "include ablation tables", embeddings=broken
    )

    assert [item.key for item in selected] == ["ablation_pref"]


async def test_a_later_write_backfills_a_missing_embedding(session):
    broken = MagicMock()
    broken.embed_document.side_effect = RuntimeError("offline")
    await upsert_memory(
        session,
        key="focus",
        content="retrieval",
        category="focus",
        embeddings=broken,
    )

    working = embedder({}, default=unit(7))
    row = await upsert_memory(
        session,
        key="focus",
        content="retrieval and reranking",
        category="focus",
        embeddings=working,
    )

    assert row.embedding is not None


async def test_memory_write_namespaces_its_source_note(session):
    """A model must not be able to make its own write look extractor-written."""
    ctx = ToolContext(session=session, embeddings=embedder({}, default=unit(9)))

    await memory_write(
        ctx,
        key="model_pref",
        content="prefers tables",
        category="preference",
        source_turn=EXTRACT_SOURCE_TURN,
    )

    rows = await search_memories(session)
    assert rows[0].source_turn.startswith(TOOL_SOURCE_PREFIX)
    assert rows[0].source_turn != EXTRACT_SOURCE_TURN


async def seed_extracted(session, count: int, *, embeddings=None) -> None:
    for index in range(count):
        await upsert_memory(
            session,
            key=f"extracted_{index}",
            content=f"the researcher prefers detail {index}",
            category="preference",
            source_turn=EXTRACT_SOURCE_TURN,
            embeddings=embeddings,
        )


async def test_consolidation_waits_until_the_store_is_large(session):
    await seed_extracted(session, 3)

    outcome = await consolidate_memories(
        session, llm_client=MagicMock(), llm_model=MODEL
    )

    assert not outcome.ran
    assert outcome.skipped == "below"
    assert len(await search_memories(session, limit=50)) == 3


async def test_consolidation_merges_extracted_records(session):
    await seed_extracted(session, CONSOLIDATE_THRESHOLD)
    merged = {
        "memories": [
            {
                "key": "prefers_detail",
                "category": "preference",
                "content": "the researcher prefers detailed reporting",
            },
            {
                "key": "prefers_tables",
                "category": "preference",
                "content": "the researcher prefers tables",
            },
            *[
                {
                    "key": f"extracted_{index}",
                    "category": "preference",
                    "content": f"kept {index}",
                }
                for index in range(6)
            ],
        ]
    }
    client = MagicMock()
    client.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content=json.dumps(merged)))]
    )

    outcome = await consolidate_memories(
        session,
        llm_client=client,
        llm_model=MODEL,
        embeddings=embedder({}, default=unit(11)),
    )

    assert outcome.ran
    assert outcome.before == CONSOLIDATE_THRESHOLD
    assert outcome.after == 8
    keys = {row.key for row in await search_memories(session, limit=50)}
    assert "prefers_detail" in keys
    assert "extracted_20" not in keys


async def test_consolidation_never_touches_what_the_researcher_wrote(session):
    """Design rule: AI must not silently rewrite human-authored content."""
    await seed_extracted(session, CONSOLIDATE_THRESHOLD)
    await upsert_memory(
        session,
        key="hand_written",
        content="never drop this",
        category="workflow",
        source_turn=f"{TOOL_SOURCE_PREFIX}asked in chat",
    )
    await upsert_memory(
        session, key="settings_written", content="nor this", category="workflow"
    )

    merged = {
        "memories": [
            {"key": "merged_all", "category": "preference", "content": "one record"},
            *[
                {
                    "key": f"extracted_{index}",
                    "category": "preference",
                    "content": f"kept {index}",
                }
                for index in range(6)
            ],
        ]
    }
    client = MagicMock()
    client.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content=json.dumps(merged)))]
    )

    outcome = await consolidate_memories(session, llm_client=client, llm_model=MODEL)

    assert outcome.ran
    keys = {row.key for row in await search_memories(session, limit=50)}
    assert {"hand_written", "settings_written"} <= keys


async def test_a_suspiciously_empty_merge_is_rejected(session):
    """An empty list would delete the whole extracted store."""
    await seed_extracted(session, CONSOLIDATE_THRESHOLD)
    client = MagicMock()
    client.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content=json.dumps({"memories": []})))]
    )

    outcome = await consolidate_memories(session, llm_client=client, llm_model=MODEL)

    assert not outcome.ran
    assert outcome.skipped == "rejected"
    assert len(await search_memories(session, limit=50)) == CONSOLIDATE_THRESHOLD


async def test_a_merge_that_collapses_almost_everything_is_rejected(session):
    """One record out of twenty-four is a truncated generation, not a judgement."""
    await seed_extracted(session, CONSOLIDATE_THRESHOLD)
    client = MagicMock()
    client.chat.completions.create.return_value = MagicMock(
        choices=[
            MagicMock(
                message=MagicMock(
                    content=json.dumps(
                        {
                            "memories": [
                                {
                                    "key": "only_one",
                                    "category": "preference",
                                    "content": "everything",
                                }
                            ]
                        }
                    )
                )
            )
        ]
    )

    outcome = await consolidate_memories(session, llm_client=client, llm_model=MODEL)

    assert not outcome.ran
    assert outcome.skipped == "rejected"
    assert len(await search_memories(session, limit=50)) == CONSOLIDATE_THRESHOLD


async def test_a_failed_llm_call_leaves_the_store_alone(session):
    await seed_extracted(session, CONSOLIDATE_THRESHOLD)
    client = MagicMock()
    client.chat.completions.create.side_effect = RuntimeError("model down")

    outcome = await consolidate_memories(session, llm_client=client, llm_model=MODEL)

    assert not outcome.ran
    assert outcome.skipped == "llm_failed"
    assert len(await search_memories(session, limit=50)) == CONSOLIDATE_THRESHOLD


async def test_a_failed_write_rolls_the_whole_merge_back(session):
    """A half-merged store is worse than an un-merged one."""
    await seed_extracted(session, CONSOLIDATE_THRESHOLD)
    merged = {
        "memories": [
            # 121 characters: past the key column's limit, so the insert fails
            # after the deletes have already been flushed.
            {"key": "k" * 121, "category": "preference", "content": "too long"},
            *[
                {
                    "key": f"extracted_{index}",
                    "category": "preference",
                    "content": f"kept {index}",
                }
                for index in range(8)
            ],
        ]
    }
    client = MagicMock()
    client.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content=json.dumps(merged)))]
    )

    outcome = await consolidate_memories(session, llm_client=client, llm_model=MODEL)

    assert not outcome.ran
    survivors = await session.execute(select(AgentMemory))
    keys = {row.key for row in survivors.scalars().all()}
    assert len(keys) == CONSOLIDATE_THRESHOLD
    assert "extracted_0" in keys
