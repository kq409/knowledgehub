import json
from unittest.mock import MagicMock

import pytest

from services.agent.protocol import (
    AgentProtocol,
    ProtocolSwitchedError,
    ToolCall,
    ToolCapabilityCache,
    parse_text_tool_call,
    tools_unsupported,
)

OLLAMA_ERROR = (
    "registry.ollama.ai/library/gemma3:4b does not support tools (status code: 400)"
)


def text_completion(content: str) -> MagicMock:
    return MagicMock(
        choices=[MagicMock(message=MagicMock(content=content, tool_calls=None))]
    )


def native_completion(name: str, arguments: dict) -> MagicMock:
    function = MagicMock()
    function.name = name
    function.arguments = json.dumps(arguments)
    call = MagicMock(id="call_abc", function=function)
    message = MagicMock(content=None, tool_calls=[call])
    return MagicMock(choices=[MagicMock(message=message)])


def test_parses_json_tool_call():
    call = parse_text_tool_call(
        '{"tool": "search_library", "input": {"query": "ella"}}'
    )
    assert call is not None
    assert call.name == "search_library"
    assert call.arguments == {"query": "ella"}


def test_parses_fenced_tool_call():
    call = parse_text_tool_call('```json\n{"tool": "list_papers", "input": {}}\n```')
    assert call is not None
    assert call.name == "list_papers"


def test_prose_is_not_a_tool_call():
    assert parse_text_tool_call("The library holds 2 papers {see above}.") is None


def test_unknown_tool_name_is_not_a_tool_call():
    assert parse_text_tool_call('{"tool": "delete_everything", "input": {}}') is None


def test_recognizes_ollama_tools_error():
    assert tools_unsupported(Exception(OLLAMA_ERROR)) is True
    assert tools_unsupported(Exception("connection refused")) is False


async def test_native_channel_returns_tool_calls():
    client = MagicMock()
    client.chat.completions.create.return_value = native_completion("list_papers", {})
    protocol = AgentProtocol(client, "gpt-test")

    response = await protocol.complete([{"role": "user", "content": "hi"}])

    assert [call.name for call in response.tool_calls] == ["list_papers"]
    assert protocol.capabilities.get("gpt-test") is True
    assert "tools" in client.chat.completions.create.call_args.kwargs


async def test_provider_without_tools_switches_and_caches():
    client = MagicMock()

    def create(**kwargs):
        if "tools" in kwargs:
            raise RuntimeError(OLLAMA_ERROR)
        return text_completion('{"tool": "list_papers", "input": {}}')

    client.chat.completions.create.side_effect = create
    capabilities = ToolCapabilityCache()
    protocol = AgentProtocol(client, "gemma3:4b", capabilities)

    with pytest.raises(ProtocolSwitchedError):
        await protocol.complete([{"role": "user", "content": "hi"}])
    assert capabilities.get("gemma3:4b") is False

    response = await protocol.complete([{"role": "user", "content": "hi"}])
    assert [call.name for call in response.tool_calls] == ["list_papers"]

    # A second protocol on the same cache never probes the native channel again.
    again = AgentProtocol(client, "gemma3:4b", capabilities)
    assert again.uses_native_tools() is False
    await again.complete([{"role": "user", "content": "hi"}])
    assert all(
        "tools" not in call.kwargs
        for call in client.chat.completions.create.call_args_list[1:]
    )


async def test_unrelated_error_is_not_swallowed():
    client = MagicMock()
    client.chat.completions.create.side_effect = RuntimeError("connection refused")
    protocol = AgentProtocol(client, "gpt-test")

    with pytest.raises(RuntimeError, match="connection refused"):
        await protocol.complete([{"role": "user", "content": "hi"}])
    assert protocol.capabilities.get("gpt-test") is None


async def test_native_channel_recovers_when_provider_ignores_tools():
    client = MagicMock()
    client.chat.completions.create.return_value = text_completion(
        '{"tool": "list_papers", "input": {}}'
    )
    protocol = AgentProtocol(client, "gpt-test")

    response = await protocol.complete([{"role": "user", "content": "hi"}])

    assert [call.name for call in response.tool_calls] == ["list_papers"]


async def test_text_channel_answer_is_plain_text():
    client = MagicMock()
    client.chat.completions.create.return_value = text_completion(
        "Your library holds 1 paper from 2006."
    )
    capabilities = ToolCapabilityCache()
    capabilities.set("gemma3:4b", False)
    protocol = AgentProtocol(client, "gemma3:4b", capabilities)

    response = await protocol.complete([{"role": "user", "content": "hi"}])

    assert response.tool_calls == []
    assert response.text == "Your library holds 1 paper from 2006."


def test_text_channel_prompt_lists_the_tools():
    capabilities = ToolCapabilityCache()
    capabilities.set("gemma3:4b", False)
    protocol = AgentProtocol(MagicMock(), "gemma3:4b", capabilities)

    prompt = protocol.system_prompt("BASE")

    assert "BASE" in prompt
    assert "search_library" in prompt
    assert "read_paper" in prompt


def test_native_prompt_stays_untouched():
    protocol = AgentProtocol(MagicMock(), "gpt-test")
    assert protocol.system_prompt("BASE") == "BASE"


def test_tool_results_use_the_channel_message_shape():
    call = ToolCall(id="call_1", name="list_papers", arguments={})

    native = AgentProtocol(MagicMock(), "gpt-test")
    assert native.tool_result_message(call, "two papers") == {
        "role": "tool",
        "tool_call_id": "call_1",
        "content": "two papers",
    }

    capabilities = ToolCapabilityCache()
    capabilities.set("gemma3:4b", False)
    text = AgentProtocol(MagicMock(), "gemma3:4b", capabilities)
    message = text.tool_result_message(call, "two papers")
    assert message["role"] == "user"
    assert "two papers" in message["content"]
