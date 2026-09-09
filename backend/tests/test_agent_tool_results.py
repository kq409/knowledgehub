"""Truncation leaves a way back to the rest."""

from unittest.mock import AsyncMock, MagicMock, patch

from services.agent.compact import TRUNCATED_TOOL_CHARS, truncate_old_tool_results
from services.agent.loop import MAX_TOOL_RESULT_CHARS, run_tool
from services.agent.protocol import ToolCall
from services.agent.tool_results import MAX_FETCH_CHARS, ToolResultStore
from services.agent.tools import (
    TOOL_HANDLERS,
    CitationRegistry,
    ToolContext,
    ToolResult,
)


def context(store: ToolResultStore | None) -> ToolContext:
    return ToolContext(
        session=MagicMock(),
        embeddings=MagicMock(),
        registry=CitationRegistry(),
        result_store=store,
    )


def test_the_store_keeps_the_most_complete_version_of_a_result():
    """The tool runner and compaction both write; the shorter write must lose."""
    store = ToolResultStore()
    store.put("call_1", "read_paper", "x" * 5000)
    store.put("call_1", "tool", "x" * 240)

    entry = store.get("call_1")
    assert entry is not None
    assert entry.total_chars == 5000
    assert entry.tool == "read_paper"


def test_the_store_drops_the_oldest_entries_once_it_is_full():
    store = ToolResultStore(max_chars=1000)
    for index in range(5):
        store.put(f"call_{index}", "read_paper", "x" * 400)

    assert store.total_chars <= 1000
    assert store.get("call_0") is None
    assert store.get("call_4") is not None


def test_a_window_reports_where_the_next_one_starts():
    store = ToolResultStore()
    store.put("call_1", "read_paper", "abcdefghij")

    entry, text, next_offset = store.window("call_1", offset=0, limit=4)
    assert (entry.tool, text, next_offset) == ("read_paper", "abcd", 4)

    _, text, next_offset = store.window("call_1", offset=8, limit=4)
    assert (text, next_offset) == ("ij", -1)

    assert store.window("missing") is None


async def test_a_truncated_result_names_the_id_to_fetch():
    store = ToolResultStore()
    long_result = ToolResult(
        content="y" * (MAX_TOOL_RESULT_CHARS + 6000),
        summary="a long read",
    )
    call = ToolCall(id="call_abc", name="read_paper", arguments={})

    with patch.dict(TOOL_HANDLERS, {"read_paper": AsyncMock(return_value=long_result)}):
        result = await run_tool(context(store), call)

    assert "fetch_tool_result" in result.content
    assert 'call_id="call_abc"' in result.content
    assert str(MAX_TOOL_RESULT_CHARS + 6000) in result.content
    stored = store.get("call_abc")
    assert stored is not None
    assert stored.total_chars == MAX_TOOL_RESULT_CHARS + 6000


async def test_fetch_tool_result_reads_the_part_that_was_cut():
    store = ToolResultStore()
    body = "HEAD" + "z" * MAX_TOOL_RESULT_CHARS + "THE-BURIED-FINDING"
    long_result = ToolResult(content=body, summary="a long read")
    call = ToolCall(id="call_abc", name="read_paper", arguments={})
    ctx = context(store)

    with patch.dict(TOOL_HANDLERS, {"read_paper": AsyncMock(return_value=long_result)}):
        first = await run_tool(ctx, call)
    assert "THE-BURIED-FINDING" not in first.content

    fetched = await run_tool(
        ctx,
        ToolCall(
            id="call_fetch",
            name="fetch_tool_result",
            arguments={"call_id": "call_abc", "offset": MAX_TOOL_RESULT_CHARS},
        ),
    )

    assert "THE-BURIED-FINDING" in fetched.content
    assert "end of the stored result" in fetched.content


async def test_fetch_tool_result_explains_an_unknown_id():
    store = ToolResultStore()
    store.put("call_known", "read_paper", "x" * 100)

    result = await run_tool(
        context(store),
        ToolCall(
            id="call_1", name="fetch_tool_result", arguments={"call_id": "call_gone"}
        ),
    )

    assert "No stored result" in result.content
    assert "call_known" in result.content


async def test_fetch_tool_result_rejects_a_negative_offset():
    store = ToolResultStore()
    store.put("call_1", "read_paper", "x" * 100)

    result = await run_tool(
        context(store),
        ToolCall(
            id="call_2",
            name="fetch_tool_result",
            arguments={"call_id": "call_1", "offset": -5},
        ),
    )

    assert "offset must be at least 0" in result.content


async def test_fetch_tool_result_caps_how_much_it_returns():
    store = ToolResultStore()
    store.put("call_1", "read_paper", "x" * 100_000)

    result = await run_tool(
        context(store),
        ToolCall(
            id="call_2",
            name="fetch_tool_result",
            arguments={"call_id": "call_1", "limit": 999_999},
        ),
    )

    # The handler caps the window, then run_tool trims to the context budget.
    assert len(result.content) <= MAX_FETCH_CHARS + 500


async def test_fetch_tool_result_needs_a_store():
    result = await run_tool(
        context(None),
        ToolCall(id="call_1", name="fetch_tool_result", arguments={"call_id": "x"}),
    )

    assert "not available in this context" in result.content


def test_compaction_stubs_point_at_the_full_payload():
    store = ToolResultStore()
    messages = [
        {"role": "system", "content": "system"},
        *[
            item
            for index in range(4)
            for item in (
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": f"call_{index}",
                            "type": "function",
                            "function": {"name": "read_paper", "arguments": "{}"},
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": f"call_{index}",
                    "content": f"PAYLOAD-{index} " + "w" * 3000,
                },
            )
        ],
    ]

    shortened = truncate_old_tool_results(messages, store)

    stub = next(
        message for message in shortened if message.get("tool_call_id") == "call_0"
    )
    assert 'fetch_tool_result(call_id="call_0")' in stub["content"]
    assert len(stub["content"]) < 3000
    # The full payload survives even though the conversation no longer has it.
    entry = store.get("call_0")
    assert entry is not None
    assert entry.content.startswith("PAYLOAD-0")
    assert entry.total_chars > TRUNCATED_TOOL_CHARS


def test_compaction_without_a_store_still_shortens():
    """Nested passes have no store; they must not crash, only lose the tail."""
    messages = [
        {"role": "tool", "tool_call_id": f"call_{index}", "content": "w" * 3000}
        for index in range(4)
    ]

    shortened = truncate_old_tool_results(messages)

    assert shortened[0]["content"].endswith("…")
