import json

from services.agent.compact import (
    compact_messages,
    message_chars,
    repair_tool_pairing,
    truncate_old_tool_results,
)


def _tool_result(name: str, body: str) -> dict:
    return {"role": "user", "content": f"Result of {name}:\n{body}"}


async def test_below_threshold_is_a_no_op():
    messages = [
        {"role": "system", "content": "You are a helper."},
        {"role": "user", "content": "What does ELLA say?"},
    ]
    result = await compact_messages(messages, threshold=24_000)
    assert not result.changed
    assert result.messages == messages


def test_truncate_shortens_older_tool_payloads():
    big = "x" * 2000
    messages = [
        {"role": "system", "content": "sys"},
        _tool_result("search_library", big),
        _tool_result("read_paper", big),
        _tool_result("read_paper", big),
        {"role": "user", "content": "Answer now"},
    ]
    truncated = truncate_old_tool_results(messages)
    assert len(truncated[1]["content"]) < len(messages[1]["content"])
    assert len(truncated[2]["content"]) == len(messages[2]["content"])
    assert len(truncated[3]["content"]) == len(messages[3]["content"])


async def test_truncate_mode_when_over_threshold():
    big = "x" * 2000
    messages = [
        {"role": "system", "content": "sys"},
        _tool_result("search_library", big),
        _tool_result("read_paper", big),
        _tool_result("read_paper", big),
        {"role": "user", "content": "Answer now"},
    ]
    result = await compact_messages(messages, threshold=5000)
    assert result.changed
    assert result.mode == "truncate"
    assert result.after_chars < result.before_chars


async def test_summarize_when_still_over_budget():
    big = "evidence " * 4000
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "old question"},
        _tool_result("search_library", big),
        _tool_result("read_paper", big),
        _tool_result("read_paper", big),
        {"role": "user", "content": "final question"},
    ]

    class FakeClient:
        class chat:  # noqa: N801 — mirrors OpenAI client.chat.completions
            class completions:  # noqa: N801
                @staticmethod
                def create(**kwargs):
                    return type(
                        "C",
                        (),
                        {
                            "choices": [
                                type(
                                    "Ch",
                                    (),
                                    {
                                        "message": type(
                                            "M",
                                            (),
                                            {"content": "Short middle summary."},
                                        )()
                                    },
                                )()
                            ]
                        },
                    )()

    result = await compact_messages(
        messages,
        threshold=100,
        llm_client=FakeClient(),
        llm_model="test-model",
    )
    assert result.mode == "summarize"
    assert any(
        "[compacted context]" in str(message.get("content", ""))
        for message in result.messages
    )
    assert result.messages[-1]["content"] == "final question"


def test_repair_tool_pairing_stubs_missing_results():
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "What papers?"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "list_papers", "arguments": "{}"},
                },
                {
                    "id": "call_2",
                    "type": "function",
                    "function": {"name": "list_notes", "arguments": "{}"},
                },
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "1 paper"},
    ]
    repaired = repair_tool_pairing(messages)
    tool_ids = [
        message["tool_call_id"] for message in repaired if message.get("role") == "tool"
    ]
    assert tool_ids == ["call_1", "call_2"]
    stub = next(
        message for message in repaired if message.get("tool_call_id") == "call_2"
    )
    assert "list_notes" in stub["content"]


def test_repair_tool_pairing_is_idempotent_when_complete():
    messages = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "list_papers", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "ok"},
    ]
    assert repair_tool_pairing(messages) == messages


def test_message_chars_counts_tool_calls():
    messages = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "1",
                    "type": "function",
                    "function": {"name": "list_papers", "arguments": "{}"},
                }
            ],
        }
    ]
    assert message_chars(messages) == len(
        json.dumps(messages[0]["tool_calls"], ensure_ascii=False)
    )


def test_message_chars_counts_reasoning_content():
    messages = [
        {
            "role": "assistant",
            "content": "ok",
            "reasoning_content": "thinking",
        }
    ]
    assert message_chars(messages) == len("ok") + len("thinking")
