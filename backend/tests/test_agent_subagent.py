import json
import uuid
from unittest.mock import MagicMock

from models import EMBEDDING_DIM
from schemas import (
    ChatCitation,
    ChatEventType,
    ChatMessage,
    ChatRequest,
    ChatRole,
    CitationSourceType,
)
from services.agent.loop import PROMPT_VERSION, ResearchAgent
from services.agent.protocol import ToolCapabilityCache
from services.agent.tools import ToolResult

MODEL = "test-model"


def unit(index: int) -> list[float]:
    vector = [0.0] * EMBEDDING_DIM
    vector[index % EMBEDDING_DIM] = 1.0
    return vector


def reply(content: str) -> MagicMock:
    return MagicMock(
        choices=[MagicMock(message=MagicMock(content=content, tool_calls=None))]
    )


def text_protocol_agent(client: MagicMock) -> ResearchAgent:
    capabilities = ToolCapabilityCache()
    capabilities.set(MODEL, False)
    embeddings = MagicMock()
    embeddings.embed_query.return_value = unit(0)
    return ResearchAgent(
        llm_client=client,
        llm_model=MODEL,
        embeddings=embeddings,
        capabilities=capabilities,
    )


def ask(question: str) -> ChatRequest:
    return ChatRequest(messages=[ChatMessage(role=ChatRole.user, content=question)])


async def test_todo_write_emits_todo_sse_event():
    client = MagicMock()
    client.chat.completions.create.side_effect = [
        reply(
            json.dumps(
                {
                    "tools": [
                        {
                            "name": "todo_write",
                            "input": {
                                "items": [
                                    {
                                        "id": "1",
                                        "content": "List the library",
                                        "status": "in_progress",
                                    }
                                ]
                            },
                        }
                    ]
                }
            )
        ),
        reply("I will list the library next."),
    ]
    agent = text_protocol_agent(client)
    events = [event async for event in agent.run(MagicMock(), ask("Plan a survey"))]

    todo_events = [e for e in events if e.type is ChatEventType.todo]
    assert len(todo_events) == 1
    assert todo_events[0].data["items"][0]["content"] == "List the library"
    assert todo_events[0].data["items"][0]["status"] == "in_progress"
    assert events[-1].data["prompt_version"] == PROMPT_VERSION


async def test_spawn_subagent_streams_nested_events_and_merges_citations():
    child_chunk = uuid.uuid4()
    paper_id = uuid.uuid4()

    client = MagicMock()
    client.chat.completions.create.side_effect = [
        reply(
            json.dumps(
                {
                    "tools": [
                        {
                            "name": "spawn_subagent",
                            "input": {"goal": "Dig into ELLA method"},
                        }
                    ]
                }
            )
        ),
        reply(json.dumps({"tool": "list_papers", "input": {}})),
        reply("ELLA uses expectation maximisation [1]."),
        reply("Based on the nested pass, ELLA uses EM [1]."),
    ]
    agent = text_protocol_agent(client)

    async def fake_run_tool(ctx, call):
        if call.name == "list_papers":
            index = ctx.registry._add(
                child_chunk,
                lambda i: ChatCitation(
                    index=i,
                    source_type=CitationSourceType.paper,
                    source_id=paper_id,
                    chunk_id=child_chunk,
                    title="ELLA",
                    page=1,
                    section="Method",
                    year=2006,
                    snippet="EM details",
                    similarity=None,
                ),
            )
            return ToolResult(
                content=f"[{index}] Paper — ELLA\nEM details",
                summary="Listed papers",
            )
        return ToolResult(content="unexpected", summary="unexpected")

    import services.agent.loop as loop_mod

    original = loop_mod.run_tool
    loop_mod.run_tool = fake_run_tool
    try:
        events = [
            event async for event in agent.run(MagicMock(), ask("Explain ELLA method"))
        ]
    finally:
        loop_mod.run_tool = original

    sub_events = [e for e in events if e.type is ChatEventType.subagent]
    assert any(e.data["status"] == "started" for e in sub_events)
    assert any(e.data["status"] == "finished" for e in sub_events)

    nested_calls = [
        e
        for e in events
        if e.type is ChatEventType.tool_call
        and e.data.get("agent_id", "main") != "main"
    ]
    assert any(e.data["name"] == "list_papers" for e in nested_calls)

    main_spawn = [
        e
        for e in events
        if e.type is ChatEventType.tool_call
        and e.data.get("name") == "spawn_subagent"
        and e.data.get("agent_id") == "main"
    ]
    assert len(main_spawn) == 1

    citations = next(e for e in events if e.type is ChatEventType.citations)
    assert len(citations.data["citations"]) >= 1
    assert citations.data["citations"][0]["title"] == "ELLA"


async def test_second_spawn_in_same_turn_is_refused():
    client = MagicMock()
    client.chat.completions.create.side_effect = [
        reply(
            json.dumps(
                {"tools": [{"name": "spawn_subagent", "input": {"goal": "First dig"}}]}
            )
        ),
        reply("Nested summary one."),
        reply(json.dumps({"tool": "spawn_subagent", "input": {"goal": "Second dig"}})),
        reply("I could not spawn again."),
    ]
    agent = text_protocol_agent(client)
    events = [event async for event in agent.run(MagicMock(), ask("Two digs"))]

    results = [e for e in events if e.type is ChatEventType.tool_result]
    assert any("budget" in (e.data.get("summary") or "") for e in results)
