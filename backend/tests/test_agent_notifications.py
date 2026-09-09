"""Background work reaches the model mid-conversation."""

import json
import time
from unittest.mock import MagicMock, patch

from schemas import ChatEventType, ChatMessage, ChatRequest, ChatRole
from services.agent.loop import ResearchAgent
from services.agent.notifications import (
    FAILED,
    NoticeBoard,
    format_notifications,
    notify_ingested,
)
from services.agent.protocol import ToolCapabilityCache

MODEL = "test-model"


def reply(content: str) -> MagicMock:
    message = MagicMock(content=content, tool_calls=None)
    return MagicMock(choices=[MagicMock(message=message, finish_reason="stop")])


def ask(question: str = "What is in my library?") -> ChatRequest:
    return ChatRequest(messages=[ChatMessage(role=ChatRole.user, content=question)])


def agent(client: MagicMock, board: NoticeBoard) -> ResearchAgent:
    capabilities = ToolCapabilityCache()
    capabilities.set(MODEL, False)
    return ResearchAgent(
        llm_client=client,
        llm_model=MODEL,
        embeddings=MagicMock(),
        capabilities=capabilities,
        notices=board,
    )


def test_news_is_handed_over_once():
    board = NoticeBoard()
    board.post("paper", "ELLA", detail="12 chunks")

    assert len(board.drain()) == 1
    assert board.drain() == []


def test_the_same_upload_processed_twice_is_announced_once():
    """`resume_pending_papers` can reprocess a paper the researcher already saw."""
    board = NoticeBoard()
    board.post("paper", "ELLA", detail="12 chunks")
    board.post("paper", "ELLA", detail="12 chunks")

    assert len(board.drain()) == 1


def test_a_failure_is_distinct_news_from_a_success():
    board = NoticeBoard()
    board.post("paper", "ELLA")
    board.post("paper", "ELLA", outcome=FAILED, detail="GROBID is down")

    assert len(board.drain()) == 2


def test_stale_news_is_dropped_rather_than_delivered():
    """A parse that finished two hours ago is not worth a paragraph of context."""
    board = NoticeBoard(ttl_seconds=60)
    board.post("paper", "ELLA")

    assert board.drain(now=time.monotonic() + 61) == []


def test_a_burst_of_uploads_cannot_grow_without_bound():
    board = NoticeBoard(max_pending=3)
    for index in range(10):
        board.post("paper", f"paper-{index}")

    remaining = board.drain()
    assert len(remaining) == 3
    assert [notice.title for notice in remaining] == [
        "paper-7",
        "paper-8",
        "paper-9",
    ]


def test_the_injected_text_names_what_changed():
    board = NoticeBoard()
    board.post("paper", "ELLA", detail="12 chunks")
    board.post("document", "grant.docx", outcome=FAILED, detail="unsupported")

    text = format_notifications(board.drain())

    assert text.startswith("<task_notification>")
    assert text.endswith("</task_notification>")
    assert 'paper "ELLA" finished processing and is searchable now (12 chunks)' in text
    assert 'document "grant.docx" failed to process: unsupported' in text


def test_nothing_waiting_means_nothing_injected():
    assert format_notifications([]) == ""


async def test_a_notice_posted_mid_turn_reaches_the_next_model_call():
    """A parse that lands while the agent is working still gets through.

    This is the case the whole mechanism exists for: injecting once per turn
    would leave the model searching a library it has stale knowledge of.
    """
    board = NoticeBoard()
    calls = {"count": 0}

    def create(**kwargs):
        calls["count"] += 1
        if "response_format" in kwargs:
            return reply(json.dumps({"tools": []}))
        if calls["count"] == 2:
            # The upload finishes while this first tool round is running.
            board.post("paper", "ELLA", detail="12 chunks")
            return reply(json.dumps({"tool": "list_skills", "input": {}}))
        return reply("Your library holds one paper.")

    client = MagicMock()
    client.chat.completions.create.side_effect = create

    with (
        patch("services.agent.loop.search_memories", side_effect=Exception("no db")),
        patch("services.agent.loop.extract_and_store", return_value=[]),
    ):
        events = [event async for event in agent(client, board).run(MagicMock(), ask())]

    answering = client.chat.completions.create.call_args_list[-1].kwargs["messages"]
    injected = [
        message
        for message in answering
        if "<task_notification>" in (message.get("content") or "")
    ]
    assert len(injected) == 1
    assert [event.type for event in events].count(ChatEventType.notice) == 1


async def test_news_waiting_before_a_turn_is_injected_as_a_user_message():
    board = NoticeBoard()
    board.post("paper", "ELLA", detail="12 chunks")

    def create(**kwargs):
        if "response_format" in kwargs:
            return reply(json.dumps({"tools": []}))
        return reply("Your library holds one paper.")

    client = MagicMock()
    client.chat.completions.create.side_effect = create

    with (
        patch("services.agent.loop.search_memories", side_effect=Exception("no db")),
        patch("services.agent.loop.extract_and_store", return_value=[]),
    ):
        events = [event async for event in agent(client, board).run(MagicMock(), ask())]

    answering = client.chat.completions.create.call_args_list[-1].kwargs["messages"]
    injected = [
        message
        for message in answering
        if "<task_notification>" in (message.get("content") or "")
    ]
    assert len(injected) == 1
    assert injected[0]["role"] == "user"
    assert 'paper "ELLA"' in injected[0]["content"]

    notices = [event.data for event in events if event.type is ChatEventType.notice]
    assert [item["kind"] for item in notices] == ["task_notification"]
    assert notices[0]["items"][0]["title"] == "ELLA"
    # Delivered once: a second turn must not repeat it.
    assert board.drain() == []


def test_a_broken_board_never_fails_an_upload():
    with patch(
        "services.agent.notifications.INGEST_NOTICES.post",
        side_effect=RuntimeError("board is gone"),
    ):
        notify_ingested("paper", "ELLA")
