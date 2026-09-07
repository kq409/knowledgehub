import json
import os
import uuid
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from dotenv import load_dotenv
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

import db
from models import EMBEDDING_DIM, Paper, PaperChunk, PaperStatus
from schemas import (
    ChatEventType,
    ChatMessage,
    ChatRequest,
    ChatRole,
    CitationSourceType,
    CompareCitation,
    ComparePaperResult,
    CompareResponse,
    CompareSynthesis,
    GateProblemKind,
    GateStatus,
)
from services.agent.loop import (
    MAX_ITERATIONS,
    MAX_TOOL_CALLS_PER_TURN,
    MAX_TOOL_RESULT_CHARS,
    AgentEvent,
    ResearchAgent,
    run_tool,
)
from services.agent.permissions import Decision, PermissionPolicy, PermissionRule
from services.agent.protocol import ToolCall, ToolCapabilityCache
from services.agent.tools import (
    TOOL_HANDLERS,
    CitationRegistry,
    ToolContext,
    ToolResult,
)

load_dotenv()

DIM = EMBEDDING_DIM
MODEL = "test-model"
SOTA_QUESTION = "What is the SOTA robot learning algorithm so far in 2026?"


def unit(index: int) -> list[float]:
    vector = [0.0] * DIM
    vector[index] = 1.0
    return vector


def reply(content: str) -> MagicMock:
    return MagicMock(
        choices=[MagicMock(message=MagicMock(content=content, tool_calls=None))]
    )


def scripted_llm(*replies: str) -> MagicMock:
    """A model that answers with the given texts, in order."""
    client = MagicMock()
    client.chat.completions.create.side_effect = [reply(item) for item in replies]
    return client


def text_protocol_agent(
    client: MagicMock,
    policy: PermissionPolicy | None = None,
    compare: MagicMock | None = None,
) -> ResearchAgent:
    """Pin the agent to the JSON text channel, as gemma3 in the devcontainer."""
    capabilities = ToolCapabilityCache()
    capabilities.set(MODEL, False)
    embeddings = MagicMock()
    embeddings.embed_query.return_value = unit(5)
    return ResearchAgent(
        llm_client=client,
        llm_model=MODEL,
        embeddings=embeddings,
        capabilities=capabilities,
        policy=policy,
        compare=compare,
    )


def native_protocol_agent(client: MagicMock) -> ResearchAgent:
    """Pin the agent to OpenAI-style function calling, as DeepSeek does."""
    capabilities = ToolCapabilityCache()
    capabilities.set(MODEL, True)
    embeddings = MagicMock()
    embeddings.embed_query.return_value = unit(5)
    return ResearchAgent(
        llm_client=client,
        llm_model=MODEL,
        embeddings=embeddings,
        capabilities=capabilities,
    )


def native_parallel_completion(*calls: tuple[str, dict]) -> MagicMock:
    tool_calls = []
    for index, (name, arguments) in enumerate(calls, start=1):
        function = MagicMock()
        function.name = name
        function.arguments = json.dumps(arguments)
        tool_calls.append(MagicMock(id=f"call_{index}", function=function))
    message = MagicMock(content=None, tool_calls=tool_calls)
    return MagicMock(choices=[MagicMock(message=message)])


def ask(question: str) -> ChatRequest:
    return ChatRequest(messages=[ChatMessage(role=ChatRole.user, content=question)])


def events_of(events: list[AgentEvent], event_type: ChatEventType) -> list[dict]:
    return [event.data for event in events if event.type is event_type]


@dataclass
class Library:
    session: AsyncSession
    paper: Paper


@pytest.fixture
async def library() -> AsyncGenerator[Library, None]:
    """A deliberately thin library: one 2006 paper."""
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
        title="ELLA: Efficient Lifelong Learning",
        authors=["Ruvolo"],
        year=2006,
        tags=[],
        original_filename="ella.pdf",
        original_file=f"/tmp/{uuid.uuid4()}.pdf",
        processing_status=PaperStatus.ready.value,
        created_at=now,
        updated_at=now,
    )
    chunk = PaperChunk(
        id=uuid.uuid4(),
        paper_id=paper.id,
        chunk_index=0,
        text="ELLA transfers knowledge across tasks efficiently. ELLA-FIXTURE-2006",
        page=1,
        section="Abstract",
        embedding=unit(5),
        extra={},
        created_at=now,
    )
    async with db.SessionLocal() as session:
        session.add(paper)
        await session.flush()
        session.add(chunk)
        await session.commit()

    try:
        async with db.SessionLocal() as session:
            yield Library(session=session, paper=paper)
    finally:
        async with db.SessionLocal() as session:
            stored = await session.get(Paper, paper.id)
            if stored is not None:
                await session.delete(stored)
                await session.commit()
        await db.close_db()


async def test_native_parallel_tool_calls_all_get_results(library: Library):
    """DeepSeek 400s if an assistant tool_calls block is missing any tool result."""
    extra_calls = MAX_TOOL_CALLS_PER_TURN + 1
    parallel = native_parallel_completion(
        *[("list_papers", {}) for _ in range(extra_calls)]
    )
    client = MagicMock()
    client.chat.completions.create.side_effect = [
        parallel,
        reply("Your library holds 1 paper from 2006."),
    ]
    agent = native_protocol_agent(client)

    events = [
        event
        async for event in agent.run(library.session, ask("What is in the library?"))
    ]

    second_messages = client.chat.completions.create.call_args_list[1].kwargs[
        "messages"
    ]
    assistant = next(
        message for message in second_messages if message.get("tool_calls")
    )
    call_ids = [call["id"] for call in assistant["tool_calls"]]
    tool_messages = [
        message for message in second_messages if message.get("role") == "tool"
    ]
    assert call_ids == [f"call_{i}" for i in range(1, extra_calls + 1)]
    assert [message["tool_call_id"] for message in tool_messages] == call_ids
    assert any("tool budget" in message["content"] for message in tool_messages)
    assert [data["name"] for data in events_of(events, ChatEventType.tool_call)] == [
        "list_papers"
    ] * extra_calls
    skipped = [
        data
        for data in events_of(events, ChatEventType.tool_result)
        if "skipped" in data["summary"]
    ]
    assert len(skipped) == 1
    assert skipped[0]["permission"]["decision"] == "skip"


async def test_agent_runs_a_tool_then_answers(library: Library):
    client = scripted_llm(
        '{"tool": "search_library", "input": {"query": "ELLA-FIXTURE-2006"}}',
        "ELLA reports transfer across tasks [1].",
    )
    agent = text_protocol_agent(client)

    events = [event async for event in agent.run(library.session, ask("Transfer?"))]

    assert [data["name"] for data in events_of(events, ChatEventType.tool_call)] == [
        "search_library"
    ]
    assert events_of(events, ChatEventType.token)[0]["text"].startswith("ELLA reports")
    citations = events_of(events, ChatEventType.citations)[0]["citations"]
    assert citations[0]["index"] == 1
    assert citations[0]["source_id"] == str(library.paper.id)
    assert events[-1].type is ChatEventType.done
    assert events[-1].data["model"] == MODEL


async def test_tool_results_carry_citation_numbers_back_to_the_model(
    library: Library,
):
    client = scripted_llm(
        '{"tool": "search_library", "input": {"query": "transfer"}}',
        "Answer [1].",
    )
    agent = text_protocol_agent(client)

    [event async for event in agent.run(library.session, ask("Transfer?"))]

    final_messages = client.chat.completions.create.call_args_list[-1].kwargs[
        "messages"
    ]
    tool_result = final_messages[-1]["content"]
    assert "[1]" in tool_result
    assert "ELLA" in tool_result


async def test_field_wide_question_reaches_the_model_with_real_coverage(
    library: Library,
):
    """The SOTA gate is the model's judgment now, not a regex in front of it.

    `/api/ask` refuses to call the LLM for this question. Here the model runs,
    sees that the library holds one 2006 paper, and decides for itself.
    """
    client = scripted_llm(
        '{"tool": "list_papers", "input": {}}',
        "Your library holds 1 paper from 2006, which cannot settle a 2026 SOTA claim.",
    )
    agent = text_protocol_agent(client)

    events = [event async for event in agent.run(library.session, ask(SOTA_QUESTION))]

    assert client.chat.completions.create.called
    assert [data["name"] for data in events_of(events, ChatEventType.tool_call)] == [
        "list_papers"
    ]

    inventory = client.chat.completions.create.call_args_list[-1].kwargs["messages"][-1]
    assert "2006" in inventory["content"]
    assert str(library.paper.id) in inventory["content"]
    assert events_of(events, ChatEventType.token)[0]["text"].startswith("Your library")


async def test_iteration_cap_forces_a_final_answer(library: Library):
    calls = {"count": 0}

    def create(**_):
        calls["count"] += 1
        if calls["count"] > MAX_ITERATIONS:
            return reply("Answering with what I already have.")
        return reply('{"tool": "list_papers", "input": {}}')

    client = MagicMock()
    client.chat.completions.create.side_effect = create
    agent = text_protocol_agent(client)

    events = [event async for event in agent.run(library.session, ask("Keep going"))]

    assert calls["count"] == MAX_ITERATIONS + 1
    assert len(events_of(events, ChatEventType.tool_call)) == MAX_ITERATIONS
    assert events_of(events, ChatEventType.token)[0]["text"] == (
        "Answering with what I already have."
    )


async def test_empty_answer_becomes_an_error_event(library: Library):
    agent = text_protocol_agent(scripted_llm(""))

    events = [event async for event in agent.run(library.session, ask("Anything?"))]

    assert events_of(events, ChatEventType.error)
    assert not events_of(events, ChatEventType.token)
    assert events[-1].type is ChatEventType.done


async def test_conversation_history_is_replayed_to_the_model(library: Library):
    # Two replies because answering without a tool call sends it back once.
    client = scripted_llm("Still one paper.", "Still one paper.")
    agent = text_protocol_agent(client)
    payload = ChatRequest(
        messages=[
            ChatMessage(role=ChatRole.user, content="What do I have?"),
            ChatMessage(role=ChatRole.assistant, content="One paper from 2006."),
            ChatMessage(role=ChatRole.user, content="And now?"),
        ]
    )

    [event async for event in agent.run(library.session, payload)]

    messages = client.chat.completions.create.call_args_list[0].kwargs["messages"]
    assert messages[0]["role"] == "system"
    assert [message["content"] for message in messages[1:]] == [
        "What do I have?",
        "One paper from 2006.",
        "And now?",
    ]


async def test_a_refused_tool_is_reported_and_never_runs(library: Library):
    client = scripted_llm(
        '{"tool": "search_library", "input": {"query": "transfer"}}',
        "I could not search your library.",
        "I could not search your library.",
    )
    agent = text_protocol_agent(client, policy=PermissionPolicy([]))

    events = [event async for event in agent.run(library.session, ask("Transfer?"))]

    result = events_of(events, ChatEventType.tool_result)[0]
    assert result["permission"]["decision"] == "deny"
    assert "search_library" in result["permission"]["reason"]
    assert "refused" in result["summary"]
    assert events_of(events, ChatEventType.citations)[0]["citations"] == []


async def test_a_refused_tool_does_not_count_as_gathering_evidence(library: Library):
    policy = PermissionPolicy(
        [PermissionRule(tool="list_papers", decision=Decision.ask)]
    )
    client = scripted_llm(
        '{"tool": "list_papers", "input": {}}',
        "Your library is empty.",
        "I could not check your library.",
    )
    agent = text_protocol_agent(client, policy=policy)

    events = [event async for event in agent.run(library.session, ask("What is here?"))]

    refused = events_of(events, ChatEventType.tool_result)[0]
    assert refused["permission"]["decision"] == "ask"
    retrying = events_of(events, ChatEventType.verdict)[0]
    assert retrying["status"] == GateStatus.retrying.value
    assert retrying["problems"][0]["kind"] == GateProblemKind.no_evidence_gathered.value


async def test_a_fabricated_citation_is_sent_back_exactly_once(library: Library):
    """The smoke-test failure: an answer citing [1] when nothing was registered.

    `list_papers` returns titles, not evidence, so the model has no [1] to give.
    """
    client = scripted_llm(
        '{"tool": "list_papers", "input": {}}',
        "ELLA is the 2026 SOTA robot learning algorithm [1].",
        '{"tool": "search_library", "input": {"query": "ELLA"}}',
        "ELLA transfers knowledge across tasks [1].",
    )
    agent = text_protocol_agent(client)

    events = [event async for event in agent.run(library.session, ask(SOTA_QUESTION))]

    verdicts = events_of(events, ChatEventType.verdict)
    assert [verdict["status"] for verdict in verdicts] == [
        GateStatus.retrying.value,
        GateStatus.supported.value,
    ]
    assert verdicts[0]["problems"][0]["kind"] == (
        GateProblemKind.fabricated_citation.value
    )
    assert events_of(events, ChatEventType.token)[0]["text"] == (
        "ELLA transfers knowledge across tasks [1]."
    )


async def test_the_retry_carries_the_specific_complaint(library: Library):
    client = scripted_llm(
        '{"tool": "list_papers", "input": {}}',
        "ELLA is the 2026 SOTA robot learning algorithm [1].",
        '{"tool": "search_library", "input": {"query": "ELLA"}}',
        "ELLA transfers knowledge across tasks [1].",
    )
    agent = text_protocol_agent(client)

    [event async for event in agent.run(library.session, ask(SOTA_QUESTION))]

    complaint = client.chat.completions.create.call_args_list[2].kwargs["messages"][-1]
    assert complaint["role"] == "user"
    assert "[1]" in complaint["content"]
    assert "none yet" in complaint["content"]


async def test_an_answer_that_stays_unsupported_still_ships_with_the_reason(
    library: Library,
):
    client = scripted_llm(
        '{"tool": "list_papers", "input": {}}',
        "ELLA is the 2026 SOTA [1].",
        "ELLA is still the 2026 SOTA [1].",
    )
    agent = text_protocol_agent(client)

    events = [event async for event in agent.run(library.session, ask(SOTA_QUESTION))]

    verdicts = events_of(events, ChatEventType.verdict)
    assert [verdict["status"] for verdict in verdicts] == [
        GateStatus.retrying.value,
        GateStatus.unsupported.value,
    ]
    assert "[1]" in verdicts[1]["reason"]
    assert events_of(events, ChatEventType.token)[0]["text"] == (
        "ELLA is still the 2026 SOTA [1]."
    )


async def test_a_well_cited_answer_needs_no_retry(library: Library):
    client = scripted_llm(
        '{"tool": "search_library", "input": {"query": "transfer"}}',
        "ELLA transfers knowledge across tasks [1].",
    )
    agent = text_protocol_agent(client)

    events = [event async for event in agent.run(library.session, ask("Transfer?"))]

    verdicts = events_of(events, ChatEventType.verdict)
    assert [verdict["status"] for verdict in verdicts] == [GateStatus.supported.value]
    assert client.chat.completions.create.call_count == 2


async def test_a_comparison_reaches_the_ui_as_an_artifact(library: Library):
    """The model gets prose it can cite; the researcher gets the real table."""
    other_id = uuid.uuid4()
    response = CompareResponse(
        id=uuid.uuid4(),
        paper_ids=[library.paper.id, other_id],
        dimensions=["method"],
        papers=[
            ComparePaperResult(
                paper_id=library.paper.id,
                title=library.paper.title,
                year=2006,
                values={"method": "Shared latent basis"},
            ),
            ComparePaperResult(
                paper_id=other_id,
                title="Progressive Nets",
                year=2016,
                values={"method": "Frozen columns"},
            ),
        ],
        synthesis=CompareSynthesis(agreements="Both reuse past tasks."),
        citations=[
            CompareCitation(
                index=1,
                source_type=CitationSourceType.paper,
                source_id=library.paper.id,
                chunk_id=uuid.uuid4(),
                paper_id=library.paper.id,
                title=library.paper.title,
                page=1,
                section="Method",
                year=2006,
                snippet="ELLA maintains a shared basis.",
                similarity=0.8,
            )
        ],
        model=MODEL,
        prompt_version="compare-papers-v1",
        created_at=datetime.now(UTC),
    )
    compare = MagicMock()
    compare.compare = AsyncMock(return_value=response)

    client = scripted_llm(
        '{"tool": "compare_papers", "input": {"paper_ids": ["'
        + str(library.paper.id)
        + '", "'
        + str(other_id)
        + '"]}}',
        "Both reuse past tasks, but differ on how [1].",
    )
    agent = text_protocol_agent(client, compare=compare)

    events = [event async for event in agent.run(library.session, ask("Compare them"))]

    types = [event.type for event in events]
    assert (
        types.index(ChatEventType.artifact)
        == types.index(ChatEventType.tool_result) + 1
    )
    artifact = events_of(events, ChatEventType.artifact)[0]
    assert artifact["kind"] == "comparison"
    assert artifact["tool"] == "compare_papers"

    turn_citations = events_of(events, ChatEventType.citations)[0]["citations"]
    assert [item["index"] for item in artifact["data"]["citations"]] == [
        item["index"] for item in turn_citations
    ]
    assert events_of(events, ChatEventType.verdict)[0]["status"] == (
        GateStatus.supported.value
    )


async def test_run_tool_keeps_the_artifact_when_it_truncates_the_content():
    ctx = ToolContext(
        session=MagicMock(), embeddings=MagicMock(), registry=CitationRegistry()
    )
    long_result = ToolResult(
        content="x" * (MAX_TOOL_RESULT_CHARS + 500),
        summary="a long tool result",
        artifact={"kind": "comparison", "tool": "compare_papers", "data": {}},
    )
    call = ToolCall(id="call_1", name="stub_tool", arguments={})

    with patch.dict(TOOL_HANDLERS, {"stub_tool": AsyncMock(return_value=long_result)}):
        result = await run_tool(ctx, call)

    assert len(result.content) < len(long_result.content)
    assert result.artifact == long_result.artifact


async def test_tool_failure_comes_back_as_a_result_not_a_crash():
    ctx = ToolContext(
        session=MagicMock(), embeddings=MagicMock(), registry=CitationRegistry()
    )
    call = ToolCall(id="call_1", name="read_paper", arguments={"paper_id": "ELLA"})

    result = await run_tool(ctx, call)

    assert "read_paper could not run" in result.content
    assert "must be a UUID" in result.content


async def test_unexpected_arguments_come_back_as_a_result():
    ctx = ToolContext(
        session=MagicMock(), embeddings=MagicMock(), registry=CitationRegistry()
    )
    call = ToolCall(id="call_1", name="list_papers", arguments={"self": "oops"})

    result = await run_tool(ctx, call)

    assert "unexpected arguments" in result.content
