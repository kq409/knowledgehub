from unittest.mock import MagicMock

from services.agent.planner import (
    PLANNER_PROMPT,
    ToolPlan,
    calls_from_payload,
    ensure_library_search,
    fallback_search_query,
    plan_tools,
    planner_schema,
)
from services.agent.tools import TOOL_HANDLERS


def test_planner_schema_only_allows_known_tools():
    schema = planner_schema()
    names = schema["properties"]["tools"]["items"]["properties"]["name"]["enum"]
    assert set(names) == set(TOOL_HANDLERS)
    assert "intent" not in schema["properties"]
    assert "kind" not in schema["properties"]


def test_calls_from_payload_drops_unknown_names_and_caps():
    payload = {
        "tools": [
            {"name": "search_library", "input": {"query": "ELLA"}},
            {"name": "not_a_tool", "input": {}},
            {"name": "list_papers", "input": {}},
            {"name": "list_notes", "input": {}},
            {"name": "list_documents", "input": {}},
        ]
    }
    calls = calls_from_payload(payload, limit=3)
    assert [call.name for call in calls] == [
        "search_library",
        "list_papers",
        "list_notes",
    ]
    assert calls[0].arguments == {"query": "ELLA"}


def test_empty_payload_is_an_empty_plan():
    assert calls_from_payload({"tools": []}) == []
    assert calls_from_payload({}) == []
    assert ToolPlan(calls=[]).disables_tools is True
    assert ToolPlan(failed=True).disables_tools is False


def test_planner_prompt_has_no_intent_labels():
    assert "intent" not in PLANNER_PROMPT.lower()
    assert "greeting" not in PLANNER_PROMPT.lower()
    assert "chitchat" not in PLANNER_PROMPT.lower()


def test_planner_prompt_looks_up_short_names():
    lowered = PLANNER_PROMPT.lower()
    assert "acronym" in lowered
    assert "search_library" in lowered
    assert "initialism" in lowered


def test_planner_prompt_prefers_list_tools_for_overview():
    lowered = PLANNER_PROMPT.lower()
    assert "overview" in lowered
    assert "read_paper" in lowered
    assert "stored summary" in lowered


def test_fallback_search_query_extracts_named_works():
    assert (
        fallback_search_query("give me a short summary of GEM. less than 50 words.")
        == "GEM"
    )
    assert fallback_search_query("It's Gradient Episodic Memory") == (
        "Gradient Episodic Memory"
    )
    assert fallback_search_query("Hello") is None
    assert fallback_search_query("give me an overview on my Library") is None


def test_ensure_library_search_fills_an_empty_plan():
    filled = ensure_library_search(ToolPlan(calls=[]), "Summarize GEM")
    assert [call.name for call in filled.calls] == ["search_library"]
    assert filled.calls[0].arguments == {"query": "GEM"}
    assert ensure_library_search(ToolPlan(calls=[]), "Hello").calls == []


async def test_plan_tools_reads_json_from_the_model():
    client = MagicMock()
    client.chat.completions.create.return_value = MagicMock(
        choices=[
            MagicMock(
                message=MagicMock(
                    content='{"tools": [{"name": "search_library", "input": {"query": "x"}}]}'
                )
            )
        ],
        usage=None,
    )
    plan = await plan_tools(client, "test-model", "What does ELLA claim?")
    assert not plan.failed
    assert [call.name for call in plan.calls] == ["search_library"]
    assert plan.calls[0].arguments == {"query": "x"}
    messages = client.chat.completions.create.call_args.kwargs["messages"]
    assert messages[0]["content"] == PLANNER_PROMPT
    assert "What does ELLA claim?" in messages[1]["content"]


async def test_plan_tools_searches_when_the_model_skips_an_acronym():
    client = MagicMock()
    client.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content='{"tools": []}'))],
        usage=None,
    )
    plan = await plan_tools(
        client, "test-model", "give me a short summary of GEM. less than 50 words."
    )
    assert [call.name for call in plan.calls] == ["search_library"]
    assert plan.calls[0].arguments["query"] == "GEM"


async def test_plan_tools_fails_open_when_the_model_errors():
    client = MagicMock()
    client.chat.completions.create.side_effect = RuntimeError("boom")
    plan = await plan_tools(client, "test-model", "Hello")
    assert plan.failed
    assert plan.calls == []
    assert plan.disables_tools is False
