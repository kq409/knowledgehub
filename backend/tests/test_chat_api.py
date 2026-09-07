import json
import os
import uuid
from collections.abc import AsyncGenerator
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest
from dotenv import load_dotenv
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from sqlalchemy import text

import db
from models import EMBEDDING_DIM, Paper, PaperChunk, PaperStatus
from routers.chat import router
from schemas import ComparePaperResult, CompareResponse, CompareSynthesis
from services.agent.loop import ResearchAgent
from services.agent.protocol import ToolCapabilityCache

load_dotenv()

DIM = EMBEDDING_DIM
MODEL = "test-model"


def unit(index: int) -> list[float]:
    vector = [0.0] * DIM
    vector[index] = 1.0
    return vector


def scripted_llm(*replies: str) -> MagicMock:
    client = MagicMock()
    client.chat.completions.create.side_effect = [
        MagicMock(choices=[MagicMock(message=MagicMock(content=item, tool_calls=None))])
        for item in replies
    ]
    return client


def build_app(client: MagicMock, compare: MagicMock | None = None) -> FastAPI:
    capabilities = ToolCapabilityCache()
    capabilities.set(MODEL, False)
    embeddings = MagicMock()
    embeddings.embed_query.return_value = unit(0)

    app = FastAPI()
    app.include_router(router)
    app.state.agent = ResearchAgent(
        llm_client=client,
        llm_model=MODEL,
        embeddings=embeddings,
        capabilities=capabilities,
        compare=compare,
    )
    return app


def parse_sse(body: str) -> list[dict]:
    events: list[dict] = []
    for frame in body.split("\n\n"):
        for line in frame.split("\n"):
            if line.startswith("data: "):
                events.append(json.loads(line[6:]))
    return events


@pytest.fixture
async def seeded_paper() -> AsyncGenerator[Paper, None]:
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        pytest.skip("DATABASE_URL is not set")

    db.init_db(database_url)
    try:
        if db.SessionLocal is None:
            pytest.skip("Database session factory was not created")
        async with db.SessionLocal() as probe:
            await probe.execute(text("SELECT 1 FROM paper_chunks LIMIT 1"))
    except Exception:
        await db.close_db()
        pytest.skip("PostgreSQL is not available")

    now = datetime.now(UTC)
    paper = Paper(
        id=uuid.uuid4(),
        title="ChatAPI Paper",
        authors=[],
        year=2019,
        tags=[],
        original_filename="chat.pdf",
        original_file=f"/tmp/{uuid.uuid4()}.pdf",
        processing_status=PaperStatus.ready.value,
        created_at=now,
        updated_at=now,
    )
    chunk = PaperChunk(
        id=uuid.uuid4(),
        paper_id=paper.id,
        chunk_index=0,
        text="Hybrid retrieval combines dense and sparse search.",
        page=1,
        section="Introduction",
        embedding=unit(0),
        extra={},
        created_at=now,
    )
    async with db.SessionLocal() as session:
        session.add(paper)
        await session.flush()
        session.add(chunk)
        await session.commit()

    try:
        yield paper
    finally:
        async with db.SessionLocal() as session:
            stored = await session.get(Paper, paper.id)
            if stored is not None:
                await session.delete(stored)
                await session.commit()
        await db.close_db()


async def post(app: FastAPI, payload: dict):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        return await ac.post("/api/chat", json=payload)


async def test_streams_tool_steps_answer_and_citations(seeded_paper: Paper):
    app = build_app(
        scripted_llm(
            '{"tool": "search_library", "input": {"query": "hybrid retrieval"}}',
            "Hybrid retrieval combines dense and sparse search [1].",
        )
    )

    response = await post(
        app, {"messages": [{"role": "user", "content": "Does hybrid retrieval help?"}]}
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    events = parse_sse(response.text)
    assert [event["type"] for event in events] == [
        "tool_call",
        "tool_result",
        "token",
        "citations",
        "verdict",
        "done",
    ]
    assert events[0]["name"] == "search_library"
    assert "result(s)" in events[1]["summary"]
    assert events[1]["permission"]["decision"] == "allow"
    assert events[2]["text"].startswith("Hybrid retrieval")
    assert events[3]["citations"][0]["source_id"] == str(seeded_paper.id)
    assert events[4]["status"] == "supported"
    assert events[4]["checked_by"] == "deterministic"
    assert events[5]["model"] == MODEL


async def test_an_unsupported_answer_reaches_the_client(seeded_paper: Paper):
    app = build_app(
        scripted_llm(
            '{"tool": "list_papers", "input": {}}',
            "Hybrid retrieval is settled science [1].",
            "Hybrid retrieval is settled science [1].",
        )
    )

    response = await post(
        app, {"messages": [{"role": "user", "content": "Is hybrid retrieval settled?"}]}
    )

    events = parse_sse(response.text)
    verdicts = [event for event in events if event["type"] == "verdict"]
    assert [verdict["status"] for verdict in verdicts] == ["retrying", "unsupported"]
    assert verdicts[1]["problems"][0]["kind"] == "fabricated_citation"
    assert "[1]" in verdicts[1]["reason"]


async def test_a_comparison_artifact_survives_the_sse_stream(seeded_paper: Paper):
    other_id = uuid.uuid4()
    response = CompareResponse(
        id=uuid.uuid4(),
        paper_ids=[seeded_paper.id, other_id],
        dimensions=["method"],
        papers=[
            ComparePaperResult(
                paper_id=seeded_paper.id,
                title=seeded_paper.title,
                year=2019,
                values={"method": "Hybrid retrieval"},
            ),
            ComparePaperResult(
                paper_id=other_id,
                title="Dense Only",
                year=2021,
                values={"method": "Dense retrieval"},
            ),
        ],
        synthesis=CompareSynthesis(agreements="Both retrieve before generating."),
        citations=[],
        model=MODEL,
        prompt_version="compare-papers-v1",
        created_at=datetime.now(UTC),
    )
    compare = MagicMock()
    compare.compare = AsyncMock(return_value=response)
    app = build_app(
        scripted_llm(
            '{"tool": "compare_papers", "input": {"paper_ids": ["'
            + str(seeded_paper.id)
            + '", "'
            + str(other_id)
            + '"]}}',
            "They differ on how retrieval is done.",
            "They differ on how retrieval is done.",
        ),
        compare=compare,
    )

    result = await post(
        app, {"messages": [{"role": "user", "content": "Compare these two"}]}
    )

    events = parse_sse(result.text)
    artifacts = [event for event in events if event["type"] == "artifact"]
    assert len(artifacts) == 1
    assert artifacts[0]["kind"] == "comparison"
    assert artifacts[0]["tool"] == "compare_papers"
    assert [paper["title"] for paper in artifacts[0]["data"]["papers"]] == [
        seeded_paper.title,
        "Dense Only",
    ]


async def test_a_workspace_artifact_survives_the_sse_stream(seeded_paper: Paper):
    app = build_app(
        scripted_llm(
            '{"tool": "present_workspace", "input": {"module": "voice_notes"}}',
            "The recorder is open — go ahead when you are ready.",
            "The recorder is open — go ahead when you are ready.",
        )
    )

    result = await post(
        app, {"messages": [{"role": "user", "content": "I want to jot down a thought"}]}
    )

    events = parse_sse(result.text)
    artifacts = [event for event in events if event["type"] == "artifact"]
    assert len(artifacts) == 1
    assert artifacts[0]["kind"] == "workspace"
    assert artifacts[0]["tool"] == "present_workspace"
    assert artifacts[0]["data"]["module"] == "voice_notes"


async def test_rejects_an_empty_conversation(seeded_paper: Paper):
    app = build_app(scripted_llm("unused"))
    response = await post(app, {"messages": []})
    assert response.status_code == 422


async def test_rejects_a_blank_question(seeded_paper: Paper):
    app = build_app(scripted_llm("unused"))
    response = await post(app, {"messages": [{"role": "user", "content": "   "}]})
    assert response.status_code == 422


async def test_rejects_a_conversation_ending_on_the_assistant(seeded_paper: Paper):
    app = build_app(scripted_llm("unused"))
    response = await post(
        app, {"messages": [{"role": "assistant", "content": "Hello"}]}
    )
    assert response.status_code == 422


async def test_rejects_a_request_with_no_sources(seeded_paper: Paper):
    app = build_app(scripted_llm("unused"))
    response = await post(
        app,
        {
            "messages": [{"role": "user", "content": "Anything?"}],
            "include_papers": False,
            "include_voice_notes": False,
            "include_handwritten_notes": False,
        },
    )
    assert response.status_code == 400


async def test_missing_agent_is_503(seeded_paper: Paper):
    app = FastAPI()
    app.include_router(router)

    response = await post(app, {"messages": [{"role": "user", "content": "Hi"}]})

    assert response.status_code == 503


async def test_model_failure_streams_a_generic_error(seeded_paper: Paper):
    client = MagicMock()
    client.chat.completions.create.side_effect = RuntimeError("boom: secret host")
    app = build_app(client)

    response = await post(app, {"messages": [{"role": "user", "content": "Hi"}]})

    assert response.status_code == 200
    events = parse_sse(response.text)
    assert events[-1]["type"] == "error"
    assert "boom" not in events[-1]["message"]
