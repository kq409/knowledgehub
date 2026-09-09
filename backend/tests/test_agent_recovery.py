"""The model call fails; the turn survives."""

import json
from unittest.mock import MagicMock, patch

import pytest

from schemas import ChatEventType, ChatMessage, ChatRequest, ChatRole
from services.agent.compact import reactive_compact
from services.agent.loop import ResearchAgent
from services.agent.protocol import ToolCapabilityCache
from services.agent.recovery import (
    LlmFailure,
    backoff_seconds,
    classify_llm_failure,
    max_retries,
)

MODEL = "test-model"


def reply(content: str, *, finish_reason: str = "stop") -> MagicMock:
    message = MagicMock(content=content, tool_calls=None)
    choice = MagicMock(message=message, finish_reason=finish_reason)
    return MagicMock(choices=[choice])


def agent(client: MagicMock) -> ResearchAgent:
    capabilities = ToolCapabilityCache()
    capabilities.set(MODEL, False)
    return ResearchAgent(
        llm_client=client,
        llm_model=MODEL,
        embeddings=MagicMock(),
        capabilities=capabilities,
    )


def ask(question: str = "What is in my library?") -> ChatRequest:
    return ChatRequest(messages=[ChatMessage(role=ChatRole.user, content=question)])


def events_of(events: list, event_type: ChatEventType) -> list[dict]:
    return [event.data for event in events if event.type is event_type]


class ProviderError(Exception):
    """A provider error carrying an HTTP status, as the OpenAI SDK does."""

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


def test_a_429_is_classified_as_retryable():
    assert (
        classify_llm_failure(ProviderError("slow down", 429)) is LlmFailure.rate_limited
    )
    assert (
        classify_llm_failure(ProviderError("overloaded", 529))
        is LlmFailure.rate_limited
    )
    assert (
        classify_llm_failure(ProviderError("Rate limit reached"))
        is LlmFailure.rate_limited
    )


def test_a_context_length_error_is_not_treated_as_a_rate_limit():
    """Some providers report it as a 400 and others fold it into a 500."""
    too_long = ProviderError("This model's maximum context length is 8192 tokens", 400)
    assert classify_llm_failure(too_long) is LlmFailure.context_too_long
    folded = ProviderError("prompt_too_long", 500)
    assert classify_llm_failure(folded) is LlmFailure.context_too_long


def test_an_ordinary_error_is_fatal():
    assert classify_llm_failure(ProviderError("no such model", 404)) is LlmFailure.fatal
    assert classify_llm_failure(ValueError("bad json")) is LlmFailure.fatal


def test_backoff_grows_then_caps():
    delays = [backoff_seconds(attempt) for attempt in range(6)]
    assert delays == sorted(delays)
    assert delays[-1] <= 8.0
    assert max_retries() >= 0


async def test_a_rate_limited_call_is_retried_and_the_turn_still_answers():
    calls = {"count": 0}

    def create(**kwargs):
        calls["count"] += 1
        if "response_format" in kwargs:
            return reply(json.dumps({"tools": []}))
        if calls["count"] == 2:
            raise ProviderError("Too Many Requests", 429)
        return reply("Your library holds one paper.")

    client = MagicMock()
    client.chat.completions.create.side_effect = create
    session = MagicMock()

    with (
        patch("services.agent.loop.backoff_seconds", return_value=0),
        patch("services.agent.loop.search_memories", side_effect=Exception("no db")),
        patch("services.agent.loop.extract_and_store", return_value=[]),
    ):
        events = [event async for event in agent(client).run(session, ask())]

    notices = events_of(events, ChatEventType.notice)
    assert [item["kind"] for item in notices] == ["rate_limited"]
    assert events_of(events, ChatEventType.token)[0]["text"] == (
        "Your library holds one paper."
    )


async def test_a_fatal_error_still_reaches_the_caller():
    def create(**kwargs):
        if "response_format" in kwargs:
            return reply(json.dumps({"tools": []}))
        raise ProviderError("model not found", 404)

    client = MagicMock()
    client.chat.completions.create.side_effect = create
    session = MagicMock()

    with (
        patch("services.agent.loop.search_memories", side_effect=Exception("no db")),
        patch("services.agent.loop.extract_and_store", return_value=[]),
        pytest.raises(ProviderError),
    ):
        [event async for event in agent(client).run(session, ask())]


async def test_a_context_length_rejection_compacts_then_retries():
    calls = {"count": 0}

    def create(**kwargs):
        calls["count"] += 1
        if "response_format" in kwargs:
            return reply(json.dumps({"tools": []}))
        if calls["count"] == 2:
            raise ProviderError(
                "prompt_too_long: reduce the length of the messages", 400
            )
        return reply("Answering from a shorter history.")

    client = MagicMock()
    client.chat.completions.create.side_effect = create
    session = MagicMock()

    # A long thread is the only shape reactive compaction can act on: with two
    # messages there is no earlier history to summarise.
    history = ChatRequest(
        messages=[
            ChatMessage(
                role=ChatRole.user if index % 2 == 0 else ChatRole.assistant,
                content=f"turn {index} " + "detail " * 40,
            )
            for index in range(9)
        ]
    )

    with (
        patch("services.agent.loop.search_memories", side_effect=Exception("no db")),
        patch("services.agent.loop.extract_and_store", return_value=[]),
    ):
        events = [event async for event in agent(client).run(session, history)]

    assert [item["kind"] for item in events_of(events, ChatEventType.notice)] == [
        "context_too_long"
    ]
    assert "reactive" in [
        item["mode"] for item in events_of(events, ChatEventType.compact)
    ]
    assert events_of(events, ChatEventType.token)[0]["text"] == (
        "Answering from a shorter history."
    )


async def test_an_empty_truncated_reply_is_retried_with_more_room():
    calls = {"count": 0}
    seen_ceilings: list[int | None] = []

    def create(**kwargs):
        calls["count"] += 1
        if "response_format" in kwargs:
            return reply(json.dumps({"tools": []}))
        seen_ceilings.append(kwargs.get("max_tokens"))
        if calls["count"] == 2:
            return reply("", finish_reason="length")
        return reply("Here is the answer.")

    client = MagicMock()
    client.chat.completions.create.side_effect = create
    session = MagicMock()

    with (
        patch("services.agent.loop.search_memories", side_effect=Exception("no db")),
        patch("services.agent.loop.extract_and_store", return_value=[]),
    ):
        events = [event async for event in agent(client).run(session, ask())]

    assert [item["kind"] for item in events_of(events, ChatEventType.notice)] == [
        "output_truncated"
    ]
    assert seen_ceilings[1] > seen_ceilings[0]
    assert events_of(events, ChatEventType.token)[0]["text"] == "Here is the answer."


async def test_reactive_compact_never_orphans_a_tool_result():
    """The cut point must keep every assistant tool_call with its result."""
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "the original question"},
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
                            "function": {"name": "search_library", "arguments": "{}"},
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": f"call_{index}",
                    "content": "x" * 400,
                },
            )
        ],
    ]

    result = await reactive_compact(messages, active_request="the original question")

    assert result.mode == "reactive"
    assert result.after_chars < result.before_chars
    assert "the original question" in result.messages[1]["content"]

    open_calls: list[str] = []
    for message in result.messages:
        if message.get("role") == "assistant" and message.get("tool_calls"):
            open_calls.extend(call["id"] for call in message["tool_calls"])
        elif message.get("role") == "tool":
            assert message["tool_call_id"] in open_calls
