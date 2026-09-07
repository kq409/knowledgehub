from unittest.mock import MagicMock

from services.extraction import parse_json_object
from services.llm_chat import complete_json_object, usage_tokens


def test_usage_tokens_reads_prompt_and_completion():
    response = MagicMock()
    response.usage.prompt_tokens = 11
    response.usage.completion_tokens = 7
    assert usage_tokens(response) == (11, 7)


def test_usage_tokens_missing_usage():
    response = MagicMock(usage=None)
    assert usage_tokens(response) == (None, None)


def test_complete_json_falls_back_when_response_format_rejected():
    client = MagicMock()

    def create(**kwargs):
        if "response_format" in kwargs:
            raise RuntimeError("response_format not supported")
        message = MagicMock()
        message.content = '{"title": "Idea"}'
        choice = MagicMock(message=message)
        return MagicMock(choices=[choice], usage=None)

    client.chat.completions.create.side_effect = create
    payload = complete_json_object(
        client,
        messages=[{"role": "user", "content": "x"}],
        model="m",
        schema={"type": "object"},
        temperature=0,
        max_tokens=256,
        effort="none",
    )
    assert payload == {"title": "Idea"}
    assert client.chat.completions.create.call_count == 2


def test_parse_json_object_still_used_on_schema_success():
    client = MagicMock()
    message = MagicMock()
    message.content = '{"ok": true}'
    client.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=message)], usage=None
    )
    payload = complete_json_object(
        client,
        messages=[{"role": "user", "content": "x"}],
        model="m",
        schema={"type": "object"},
        temperature=0,
        max_tokens=256,
        effort="none",
    )
    assert payload == {"ok": True}
    assert parse_json_object('{"ok": true}') == {"ok": True}
