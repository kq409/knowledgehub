import json

from services.ask_trace import estimate_cost, log_ask_trace, log_chat_trace, traces_dir


def test_estimate_cost_uses_env(monkeypatch):
    monkeypatch.setenv("USD_PER_1K_INPUT", "1")
    monkeypatch.setenv("USD_PER_1K_OUTPUT", "2")
    assert estimate_cost(1000, 500) == 2.0


def test_ask_trace_writes_jsonl(tmp_path, monkeypatch):
    monkeypatch.setattr("services.ask_trace.TRACE_DIR", tmp_path)
    monkeypatch.delenv("OPIK_API_KEY", raising=False)
    log_ask_trace(
        question="What does ZXQELLA7 claim?",
        query_kind="library",
        hits=[{"title": "ZXQELLA7"}],
        decision="OK",
        external_search=False,
        external_status="unavailable",
        web_urls=[],
        citations=[{"index": 1, "title": "ZXQELLA7"}],
        answer="It transfers [1].",
        latency_ms=12.3,
        request_id="req-1",
        prompt_tokens=10,
        completion_tokens=4,
    )
    lines = (tmp_path / "ask.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["request_id"] == "req-1"
    assert payload["question"].startswith("What does")
    assert traces_dir() == tmp_path


def test_chat_trace_writes_jsonl(tmp_path, monkeypatch):
    monkeypatch.setattr("services.ask_trace.TRACE_DIR", tmp_path)
    monkeypatch.delenv("OPIK_API_KEY", raising=False)
    log_chat_trace(
        question="Cite ZXQELLA7",
        tools_called=["search_library"],
        citations=[{"index": 1, "title": "ZXQELLA7"}],
        answer="Transfer [1].",
        latency_ms=40.0,
        gate_status="supported",
        request_id="chat-1",
        prompt_tokens=20,
        completion_tokens=6,
    )
    payload = json.loads(
        (tmp_path / "chat.jsonl").read_text(encoding="utf-8").strip().splitlines()[0]
    )
    assert payload["request_id"] == "chat-1"
    assert payload["tools"] == ["search_library"]
    assert payload["gate_status"] == "supported"
    assert estimate_cost(20, 6) is not None


def test_opik_trace_does_not_end_immediately(tmp_path, monkeypatch):
    monkeypatch.setattr("services.ask_trace.TRACE_DIR", tmp_path)
    monkeypatch.setenv("OPIK_API_KEY", "test-key")
    monkeypatch.setattr("services.ask_trace._opik_client", None)

    class FakeTrace:
        def __init__(self) -> None:
            self.ended = False

        def end(self) -> None:
            self.ended = True

    class FakeOpik:
        def __init__(self, **_: object) -> None:
            self.traces: list[dict] = []
            self.last_trace = FakeTrace()

        def trace(self, **kwargs: object) -> FakeTrace:
            self.traces.append(kwargs)
            return self.last_trace

    fake = FakeOpik()
    monkeypatch.setattr("services.ask_trace._get_opik_client", lambda: fake)

    log_chat_trace(
        question="Hello",
        tools_called=[],
        citations=[],
        answer="Hello — what would you like to look up?",
        latency_ms=5.0,
        gate_status="supported",
        request_id="chat-hello",
    )

    assert len(fake.traces) == 1
    assert fake.traces[0]["name"] == "chat"
    assert fake.traces[0]["output"]["answer"].startswith("Hello")
    assert fake.last_trace.ended is False
