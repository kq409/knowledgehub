import asyncio
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
    MAX_TOOL_CALLS_PER_RESPONSE,
    MAX_TOOL_RESULT_CHARS,
    PROMPT_VERSION,
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


def reply(content: str, *, reasoning: str | None = None) -> MagicMock:
    message_kwargs: dict = {"content": content, "tool_calls": None}
    if reasoning is not None:
        message_kwargs["reasoning_content"] = reasoning
    return MagicMock(choices=[MagicMock(message=MagicMock(**message_kwargs))])


def scripted_llm(*replies: str) -> MagicMock:
    """A model that answers with the given texts, in order."""
    client = MagicMock()
    client.chat.completions.create.side_effect = [reply(item) for item in replies]
    return client


def planned(*calls: tuple[str, dict] | str) -> str:
    """JSON the tool planner is expected to return."""
    tools = []
    for item in calls:
        if isinstance(item, str):
            tools.append({"name": item, "input": {}})
        else:
            name, arguments = item
            tools.append({"name": name, "input": arguments})
    return json.dumps({"tools": tools})


def text_protocol_agent(
    client: MagicMock,
    policy: PermissionPolicy | None = None,
    compare: MagicMock | None = None,
    web_search: object | None = None,
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
        web_search=web_search,
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
    extra_calls = MAX_TOOL_CALLS_PER_RESPONSE + 1
    parallel = native_parallel_completion(
        *[("list_papers", {}) for _ in range(extra_calls)]
    )
    client = MagicMock()
    client.chat.completions.create.side_effect = [
        reply(planned("list_papers")),
        parallel,
        reply("Your library holds 1 paper from 2006."),
    ]
    agent = native_protocol_agent(client)

    events = [
        event
        async for event in agent.run(library.session, ask("What is in the library?"))
    ]

    answer_messages = client.chat.completions.create.call_args_list[2].kwargs[
        "messages"
    ]
    parallel_assistant = [
        message for message in answer_messages if message.get("tool_calls")
    ][-1]
    call_ids = [call["id"] for call in parallel_assistant["tool_calls"]]
    tool_messages = [
        message for message in answer_messages if message.get("role") == "tool"
    ]
    parallel_results = tool_messages[-extra_calls:]
    assert call_ids == [f"call_{i}" for i in range(1, extra_calls + 1)]
    assert [message["tool_call_id"] for message in parallel_results] == call_ids
    assert any("tool budget" in message["content"] for message in parallel_results)
    assert [data["name"] for data in events_of(events, ChatEventType.tool_call)] == [
        "list_papers"
    ] * (extra_calls + 1)
    skipped = [
        data
        for data in events_of(events, ChatEventType.tool_result)
        if "skipped" in data["summary"]
    ]
    assert len(skipped) == 1
    assert skipped[0]["permission"]["decision"] == "skip"


async def test_concurrent_turns_do_not_swap_tool_results(library: Library):
    """One `ResearchAgent` serves every request, so per-turn state cannot live on it.

    The two turns are forced to sit inside tool execution at the same moment,
    which is where an instance attribute would let the second turn overwrite
    what the first is about to read.
    """
    if db.SessionLocal is None:
        pytest.skip("Database session factory was not created")

    both_inside = asyncio.Barrier(2)

    async def tagged_list(ctx: ToolContext, **_: object) -> ToolResult:
        # top_k is the only per-request value the handler can see, so each turn
        # sends a different one and reads its own back.
        await both_inside.wait()
        return ToolResult(
            content=f"library for turn {ctx.top_k}",
            summary=f"listed for turn {ctx.top_k}",
        )

    def create(**kwargs):
        blob = json.dumps(kwargs.get("messages") or [])
        if "response_format" in kwargs:
            return reply(planned("list_papers"))
        return reply("ALPHA" if "ALPHA" in blob else "BRAVO")

    client = MagicMock()
    client.chat.completions.create.side_effect = create
    agent = text_protocol_agent(client)

    async def turn(question: str, top_k: int) -> list[AgentEvent]:
        payload = ChatRequest(
            messages=[ChatMessage(role=ChatRole.user, content=question)],
            top_k=top_k,
        )
        async with db.SessionLocal() as session:
            return [event async for event in agent.run(session, payload)]

    with patch.dict(TOOL_HANDLERS, {"list_papers": tagged_list}):
        alpha, bravo = await asyncio.gather(
            turn("ALPHA question", 1), turn("BRAVO question", 2)
        )

    assert events_of(alpha, ChatEventType.tool_result)[0]["summary"] == (
        "listed for turn 1"
    )
    assert events_of(bravo, ChatEventType.tool_result)[0]["summary"] == (
        "listed for turn 2"
    )
    assert events_of(alpha, ChatEventType.token)[0]["text"] == "ALPHA"
    assert events_of(bravo, ChatEventType.token)[0]["text"] == "BRAVO"
    assert alpha[-1].data["request_id"] != bravo[-1].data["request_id"]


async def test_parallel_read_tools_run_concurrently(library: Library):
    """Three independent reads should overlap, not queue behind each other."""
    if db.SessionLocal is None:
        pytest.skip("Database session factory was not created")

    all_three = asyncio.Barrier(3)

    async def gated_read(ctx: ToolContext, **_: object) -> ToolResult:
        # Only reachable if all three calls are in flight at once; a serial
        # runner would deadlock here and time out.
        await asyncio.wait_for(all_three.wait(), timeout=5)
        return ToolResult(content="ok", summary="read ok")

    client = MagicMock()
    client.chat.completions.create.side_effect = [
        reply(planned("list_papers", "list_notes", "list_documents")),
        reply("Your library holds 1 paper."),
    ]
    agent = native_protocol_agent(client)

    handlers = {
        "list_papers": gated_read,
        "list_notes": gated_read,
        "list_documents": gated_read,
    }
    async with db.SessionLocal() as session:
        with patch.dict(TOOL_HANDLERS, handlers):
            events = [event async for event in agent.run(session, ask("Overview?"))]

    results = events_of(events, ChatEventType.tool_result)
    assert [data["name"] for data in results] == [
        "list_papers",
        "list_notes",
        "list_documents",
    ]
    assert all(data["permission"]["decision"] == "allow" for data in results)


async def test_dsml_leaked_into_content_runs_as_tool_calls(library: Library):
    """DeepSeek V4 sometimes writes DSML into content instead of tool_calls.

    An empty planner result turns tools off for the answering turn; the leak
    must still run, or the preamble is shipped as the whole answer.
    """
    dsml = (
        "I'll pull up your library contents to give you an overview.\n\n"
        "<｜｜DSML｜｜tool_calls>\n"
        '<｜｜DSML｜｜invoke name="list_papers"/>\n'
        '<｜｜DSML｜｜invoke name="list_notes"/>\n'
        '<｜｜DSML｜｜invoke name="list_documents"/>\n'
        "</｜｜DSML｜｜tool_calls>"
    )
    client = MagicMock()
    client.chat.completions.create.side_effect = [
        reply(planned()),
        reply(dsml, reasoning="List papers, notes, and documents."),
        reply("Your library holds 1 paper from 2006."),
    ]
    agent = native_protocol_agent(client)

    events = [
        event
        async for event in agent.run(
            library.session, ask("give me an overview on my Library")
        )
    ]

    assert [data["name"] for data in events_of(events, ChatEventType.tool_call)] == [
        "list_papers",
        "list_notes",
        "list_documents",
    ]
    answer = events_of(events, ChatEventType.token)[0]["text"]
    assert "DSML" not in answer
    assert "list_papers" not in answer
    assert "I'll pull up" not in answer
    assert "Your library holds 1 paper" in answer
    follow_up = client.chat.completions.create.call_args_list[2].kwargs["messages"]
    echoed = [
        message
        for message in follow_up
        if message.get("role") == "assistant" and message.get("tool_calls")
    ]
    assert echoed[-1]["reasoning_content"] == "List papers, notes, and documents."


async def test_agent_runs_a_tool_then_answers(library: Library):
    client = scripted_llm(
        planned(("search_library", {"query": "ELLA-FIXTURE-2006"})),
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
        planned(("search_library", {"query": "transfer"})),
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
    """Chat still runs for SOTA questions; it is not blocked by Ask's abstain.

    The model may call web_search. This test scripts list_papers then a
    coverage-limited answer, which remains valid.
    """
    client = scripted_llm(
        planned("list_papers"),
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


async def test_field_wide_question_uses_web_search(library: Library):
    from services.web_search import ExternalHit

    class _ReadyWeb:
        def available(self) -> bool:
            return True

        def search(self, query: str) -> list[ExternalHit]:
            return [
                ExternalHit(
                    url="https://arxiv.org/abs/2401.00001",
                    title="Survey of robot learning 2026",
                    snippet="Recent work surveys SOTA methods.",
                )
            ]

    client = scripted_llm(
        planned(("web_search", {"query": "SOTA robot learning 2026"})),
        "The library has one 2006 paper. Web sources survey recent methods [1].",
    )
    agent = text_protocol_agent(client, web_search=_ReadyWeb())
    events = [event async for event in agent.run(library.session, ask(SOTA_QUESTION))]

    system = client.chat.completions.create.call_args_list[1].kwargs["messages"][0][
        "content"
    ]
    assert "field-wide" in system.lower() or "SOTA" in system
    assert [data["name"] for data in events_of(events, ChatEventType.tool_call)] == [
        "web_search"
    ]
    citations = events_of(events, ChatEventType.citations)[0]["citations"]
    assert any(item.get("source_type") == "web" for item in citations)
    assert any(
        item.get("url") == "https://arxiv.org/abs/2401.00001" for item in citations
    )
    assert events[-1].data["prompt_version"] == PROMPT_VERSION


async def test_iteration_cap_forces_a_final_answer(library: Library):
    calls = {"count": 0}

    def create(**kwargs):
        calls["count"] += 1
        if "response_format" in kwargs:
            return reply(planned("list_papers"))
        if calls["count"] > MAX_ITERATIONS + 1:
            return reply("Answering with what I already have.")
        return reply('{"tool": "list_papers", "input": {}}')

    client = MagicMock()
    client.chat.completions.create.side_effect = create
    agent = text_protocol_agent(client)

    events = [event async for event in agent.run(library.session, ask("Keep going"))]

    assert calls["count"] == MAX_ITERATIONS + 2
    assert len(events_of(events, ChatEventType.tool_call)) == MAX_ITERATIONS + 1
    assert events_of(events, ChatEventType.token)[0]["text"] == (
        "Answering with what I already have."
    )


async def test_empty_answer_becomes_an_error_event(library: Library):
    agent = text_protocol_agent(scripted_llm(planned(), ""))

    events = [event async for event in agent.run(library.session, ask("Anything?"))]

    assert events_of(events, ChatEventType.error)
    assert not events_of(events, ChatEventType.token)
    assert events[-1].type is ChatEventType.done


async def test_conversation_history_is_replayed_to_the_model(library: Library):
    # Planner first, then one answer (zero tools is no longer a gate retry).
    client = scripted_llm(planned(), "Still one paper.")
    agent = text_protocol_agent(client)
    payload = ChatRequest(
        messages=[
            ChatMessage(role=ChatRole.user, content="What do I have?"),
            ChatMessage(role=ChatRole.assistant, content="One paper from 2006."),
            ChatMessage(role=ChatRole.user, content="And now?"),
        ]
    )

    [event async for event in agent.run(library.session, payload)]

    messages = client.chat.completions.create.call_args_list[1].kwargs["messages"]
    assert messages[0]["role"] == "system"
    assert [message["content"] for message in messages[1:]] == [
        "What do I have?",
        "One paper from 2006.",
        "And now?",
    ]


async def test_a_refused_tool_is_reported_and_never_runs(library: Library):
    client = scripted_llm(
        planned(("search_library", {"query": "transfer"})),
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
        planned("list_papers"),
        "I could not check your library.",
    )
    agent = text_protocol_agent(client, policy=policy)

    events = [event async for event in agent.run(library.session, ask("What is here?"))]

    refused = events_of(events, ChatEventType.tool_result)[0]
    assert refused["permission"]["decision"] == "ask"
    assert events_of(events, ChatEventType.verdict)[0]["status"] == (
        GateStatus.supported.value
    )


async def test_a_fabricated_citation_is_sent_back_exactly_once(library: Library):
    """The smoke-test failure: an answer citing [99] when nothing was registered.

    `list_papers` now registers inventory as [1], [2], … so a fake index must
    be a number the list tool did not hand out.
    """
    client = scripted_llm(
        planned("list_papers"),
        "ELLA is the 2026 SOTA robot learning algorithm [99].",
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
        planned("list_papers"),
        "ELLA is the 2026 SOTA robot learning algorithm [99].",
        '{"tool": "search_library", "input": {"query": "ELLA"}}',
        "ELLA transfers knowledge across tasks [1].",
    )
    agent = text_protocol_agent(client)

    [event async for event in agent.run(library.session, ask(SOTA_QUESTION))]

    complaint = client.chat.completions.create.call_args_list[2].kwargs["messages"][-1]
    assert complaint["role"] == "system"
    assert "[99]" in complaint["content"]
    assert "none yet" not in complaint["content"]
    assert "not the researcher" in complaint["content"]
    assert "Do not thank the user" in complaint["content"]


async def test_an_answer_that_stays_unsupported_still_ships_with_the_reason(
    library: Library,
):
    client = scripted_llm(
        planned("list_papers"),
        "ELLA is the 2026 SOTA [99].",
        "ELLA is still the 2026 SOTA [99].",
    )
    agent = text_protocol_agent(client)

    events = [event async for event in agent.run(library.session, ask(SOTA_QUESTION))]

    verdicts = events_of(events, ChatEventType.verdict)
    assert [verdict["status"] for verdict in verdicts] == [
        GateStatus.retrying.value,
        GateStatus.unsupported.value,
    ]
    assert "[99]" in verdicts[1]["reason"]
    assert events_of(events, ChatEventType.token)[0]["text"] == (
        "ELLA is still the 2026 SOTA [99]."
    )


async def test_a_well_cited_answer_needs_no_retry(library: Library):
    client = scripted_llm(
        planned(("search_library", {"query": "transfer"})),
        "ELLA transfers knowledge across tasks [1].",
    )
    agent = text_protocol_agent(client)

    events = [event async for event in agent.run(library.session, ask("Transfer?"))]

    verdicts = events_of(events, ChatEventType.verdict)
    assert [verdict["status"] for verdict in verdicts] == [GateStatus.supported.value]
    assert client.chat.completions.create.call_count == 2


async def test_library_overview_from_list_tools_needs_no_retry(library: Library):
    client = scripted_llm(
        planned("list_papers", "list_notes", "list_documents"),
        "The library holds 1 paper from 2006 (ELLA). The document library is empty.",
    )
    agent = text_protocol_agent(client)

    events = [
        event
        async for event in agent.run(
            library.session, ask("give me an overview on my Library")
        )
    ]

    verdicts = events_of(events, ChatEventType.verdict)
    assert [verdict["status"] for verdict in verdicts] == [GateStatus.supported.value]
    assert client.chat.completions.create.call_count == 2
    assert events_of(events, ChatEventType.token)[0]["text"].startswith(
        "The library holds 1 paper"
    )


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
        planned(
            (
                "compare_papers",
                {"paper_ids": [str(library.paper.id), str(other_id)]},
            )
        ),
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


async def test_empty_plan_answers_without_tools(library: Library):
    client = scripted_llm(
        planned(),
        "Hello — what would you like to look up in your library?",
    )
    agent = text_protocol_agent(client)

    events = [event async for event in agent.run(library.session, ask("Hello"))]

    assert events_of(events, ChatEventType.tool_call) == []
    text = events_of(events, ChatEventType.token)[0]["text"]
    assert text.startswith("Hello")
    assert "ELLA" not in text
    assert events_of(events, ChatEventType.verdict)[0]["status"] == (
        GateStatus.supported.value
    )
    answer_system = client.chat.completions.create.call_args_list[1].kwargs["messages"][
        0
    ]["content"]
    assert "No tools were selected" in answer_system
    assert "How to call a tool" not in answer_system


async def test_empty_plan_still_searches_a_named_acronym(library: Library):
    client = scripted_llm(
        planned(),
        "No GEM paper showed up in this test library.",
    )
    agent = text_protocol_agent(client)

    events = [
        event
        async for event in agent.run(
            library.session,
            ask("give me a short summary of GEM. less than 50 words."),
        )
    ]

    assert [data["name"] for data in events_of(events, ChatEventType.tool_call)] == [
        "search_library"
    ]
    assert "GEM" in events_of(events, ChatEventType.tool_call)[0]["arguments"]["query"]


async def test_empty_plan_retry_enables_tools(library: Library):
    client = scripted_llm(
        planned(),
        "ELLA transfers across tasks [1].",
        '{"tool": "search_library", "input": {"query": "ELLA"}}',
        "ELLA transfers knowledge across tasks [1].",
    )
    agent = text_protocol_agent(client)

    events = [event async for event in agent.run(library.session, ask("Hello"))]

    verdicts = events_of(events, ChatEventType.verdict)
    assert verdicts[0]["status"] == GateStatus.retrying.value
    retry_system = client.chat.completions.create.call_args_list[2].kwargs["messages"][
        0
    ]["content"]
    assert "search_library" in retry_system
    assert [data["name"] for data in events_of(events, ChatEventType.tool_call)] == [
        "search_library"
    ]
