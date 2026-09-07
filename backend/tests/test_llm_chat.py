from openai import APIError, NotFoundError

from routers.ask import ask_http_error
from services.ask import AskError
from services.llm_chat import (
    effort_from_env,
    max_tokens_from_env,
    message_text,
    reasoning_extra_body,
    with_chat_extras,
)


def test_default_reasoning_effort_is_none(monkeypatch):
    monkeypatch.delenv("LLM_REASONING_EFFORT", raising=False)
    assert reasoning_extra_body() == {"reasoning_effort": "none"}


def test_blank_reasoning_effort_omits_the_field(monkeypatch):
    monkeypatch.setenv("LLM_REASONING_EFFORT", "  ")
    assert reasoning_extra_body() == {}


def test_reasoning_extra_body_accepts_explicit_effort(monkeypatch):
    monkeypatch.setenv("LLM_REASONING_EFFORT", "high")
    assert reasoning_extra_body("none") == {"reasoning_effort": "none"}
    assert reasoning_extra_body("") == {}


def test_effort_from_env_priority(monkeypatch):
    monkeypatch.delenv("ASK_REASONING_EFFORT", raising=False)
    monkeypatch.delenv("LLM_REASONING_EFFORT", raising=False)
    assert effort_from_env("ASK_REASONING_EFFORT", "LLM_REASONING_EFFORT") == "none"

    monkeypatch.setenv("LLM_REASONING_EFFORT", "high")
    assert effort_from_env("ASK_REASONING_EFFORT", "LLM_REASONING_EFFORT") == "high"

    monkeypatch.setenv("ASK_REASONING_EFFORT", "low")
    assert effort_from_env("ASK_REASONING_EFFORT", "LLM_REASONING_EFFORT") == "low"


def test_judge_effort_does_not_fall_back_to_global(monkeypatch):
    monkeypatch.setenv("LLM_REASONING_EFFORT", "high")
    monkeypatch.delenv("JUDGE_REASONING_EFFORT", raising=False)
    assert effort_from_env("JUDGE_REASONING_EFFORT", default="none") == "none"


def test_with_chat_extras_merges_without_clobbering(monkeypatch):
    monkeypatch.setenv("LLM_REASONING_EFFORT", "none")
    kwargs = with_chat_extras(
        {
            "model": "qwen3:4b",
            "extra_body": {"keep": True},
            "max_tokens": 100,
        }
    )
    assert kwargs["extra_body"]["keep"] is True
    assert kwargs["extra_body"]["reasoning_effort"] == "none"
    assert kwargs["max_tokens"] == 100


def test_with_chat_extras_path_effort_overrides_env(monkeypatch):
    monkeypatch.setenv("LLM_REASONING_EFFORT", "high")
    kwargs = with_chat_extras({"model": "x"}, effort="none")
    assert kwargs["extra_body"]["reasoning_effort"] == "none"


def test_max_tokens_from_env(monkeypatch):
    monkeypatch.delenv("ASK_MAX_TOKENS", raising=False)
    assert max_tokens_from_env("ASK_MAX_TOKENS", 4096) == 4096
    monkeypatch.setenv("ASK_MAX_TOKENS", "2048")
    assert max_tokens_from_env("ASK_MAX_TOKENS", 4096) == 2048
    monkeypatch.setenv("ASK_MAX_TOKENS", "nope")
    assert max_tokens_from_env("ASK_MAX_TOKENS", 4096) == 4096


def test_message_text_strips_empty():
    class Msg:
        content = "  hello  "

    assert message_text(Msg()) == "hello"
    assert message_text(type("M", (), {"content": None})()) == ""


def test_ask_http_error_empty_answer_mentions_effort():
    err = ask_http_error(AskError("empty"))
    assert err.status_code == 502
    assert "ASK_REASONING_EFFORT" in err.detail
    assert "ASK_MAX_TOKENS" in err.detail


def test_ask_http_error_not_found_mentions_embedding():
    import httpx

    response = httpx.Response(
        404, request=httpx.Request("POST", "https://example.com/v1/embeddings")
    )
    exc = NotFoundError(
        message="model nomic-embed-text not found",
        response=response,
        body={"error": {"message": "model nomic-embed-text not found"}},
    )
    err = ask_http_error(exc)
    assert err.status_code == 502
    assert "EMBEDDING" in err.detail


def test_ask_http_error_api_includes_status():
    import httpx

    request = httpx.Request("POST", "https://example.com/v1/chat/completions")
    exc = APIError(message="rate limited", request=request, body=None)
    exc.status_code = 429
    err = ask_http_error(exc)
    assert err.status_code == 502
    assert "429" in err.detail
    assert "LLM_BASE_URL" in err.detail
