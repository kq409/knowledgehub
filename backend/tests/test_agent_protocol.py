import json
from unittest.mock import MagicMock

import pytest

from services.agent.protocol import (
    AgentProtocol,
    AgentResponse,
    ProtocolSwitchedError,
    ToolCall,
    ToolCapabilityCache,
    pad_assistant_reasoning,
    parse_text_tool_call,
    parse_text_tool_calls,
    strip_dsml_markup,
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


async def test_native_complete_omits_tools_when_schemas_are_empty():
    client = MagicMock()
    client.chat.completions.create.return_value = text_completion("Hello!")
    capabilities = ToolCapabilityCache()
    capabilities.set("m", True)
    protocol = AgentProtocol(
        client, "m", capabilities, tool_schemas=[], allowed_tools=frozenset()
    )

    response = await protocol.complete([{"role": "user", "content": "Hi"}])

    assert response.text == "Hello!"
    assert "tools" not in client.chat.completions.create.call_args.kwargs
    assert protocol.system_prompt("base") == "base"


def test_text_channel_without_schemas_skips_the_catalog():
    capabilities = ToolCapabilityCache()
    capabilities.set("gemma3:4b", False)
    protocol = AgentProtocol(
        MagicMock(),
        "gemma3:4b",
        capabilities,
        tool_schemas=[],
        allowed_tools=frozenset(),
    )

    assert protocol.system_prompt("BASE") == "BASE"


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


LIBRARY_OVERVIEW_DSML = """
I'll pull up your library contents to give you an overview.

<｜｜DSML｜｜tool_calls>
<｜｜DSML｜｜invoke name="list_papers"/>
<｜｜DSML｜｜invoke name="list_notes"/>
<｜｜DSML｜｜invoke name="list_documents"/>
</｜｜DSML｜｜tool_calls>
""".strip()


def test_parses_degraded_dsml_parallel_invokes():
    calls = parse_text_tool_calls(LIBRARY_OVERVIEW_DSML)
    assert [call.name for call in calls] == [
        "list_papers",
        "list_notes",
        "list_documents",
    ]
    assert all(call.arguments == {} for call in calls)


def test_parses_canonical_dsml_with_parameters():
    text = (
        "<｜DSML｜tool_calls>\n"
        '<｜DSML｜invoke name="search_library">\n'
        '<｜DSML｜parameter name="query" string="true">hybrid retrieval</｜DSML｜parameter>\n'
        '<｜DSML｜parameter name="top_k" string="false">5</｜DSML｜parameter>\n'
        "</｜DSML｜invoke>\n"
        "</｜DSML｜tool_calls>"
    )
    call = parse_text_tool_call(text)
    assert call is not None
    assert call.name == "search_library"
    assert call.arguments == {"query": "hybrid retrieval", "top_k": 5}


def test_parses_ascii_double_pipe_dsml_and_unwraps_arguments():
    text = (
        "<||DSML||tool_calls>"
        '<||DSML||invoke name="search_library">'
        '<||DSML||parameter name="arguments" string="false">'
        '{"query": "ella"}'
        "</||DSML||parameter>"
        "</||DSML||invoke>"
        "</||DSML||tool_calls>"
    )
    call = parse_text_tool_call(text)
    assert call is not None
    assert call.name == "search_library"
    assert call.arguments == {"query": "ella"}


def test_parses_orphan_dsml_invoke_without_wrapper():
    call = parse_text_tool_call('<｜DSML｜invoke name="list_papers"/>')
    assert call is not None
    assert call.name == "list_papers"


def test_unknown_dsml_invoke_is_not_a_tool_call():
    assert parse_text_tool_call('<｜DSML｜invoke name="delete_everything"/>') is None


def test_strip_dsml_keeps_preamble():
    assert (
        strip_dsml_markup(LIBRARY_OVERVIEW_DSML)
        == "I'll pull up your library contents to give you an overview."
    )


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


async def test_native_channel_recovers_dsml_leaked_into_content():
    client = MagicMock()
    client.chat.completions.create.return_value = text_completion(LIBRARY_OVERVIEW_DSML)
    protocol = AgentProtocol(client, "gpt-test")

    response = await protocol.complete([{"role": "user", "content": "hi"}])

    assert [call.name for call in response.tool_calls] == [
        "list_papers",
        "list_notes",
        "list_documents",
    ]
    assert response.text == ""


async def test_native_channel_recovers_dsml_leaked_into_reasoning():
    message = MagicMock(
        content="",
        reasoning_content='<｜DSML｜invoke name="list_papers"/>',
        tool_calls=None,
    )
    client = MagicMock()
    client.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=message)]
    )
    protocol = AgentProtocol(client, "gpt-test")

    response = await protocol.complete([{"role": "user", "content": "hi"}])

    assert [call.name for call in response.tool_calls] == ["list_papers"]
    assert response.text == ""


async def test_dsml_is_recovered_even_when_tools_were_not_advertised():
    client = MagicMock()
    client.chat.completions.create.return_value = text_completion(LIBRARY_OVERVIEW_DSML)
    protocol = AgentProtocol(
        client, "gpt-test", tool_schemas=[], allowed_tools=frozenset()
    )

    response = await protocol.complete([{"role": "user", "content": "hi"}])

    assert [call.name for call in response.tool_calls] == [
        "list_papers",
        "list_notes",
        "list_documents",
    ]
    assert response.text == ""


async def test_dsml_respects_a_nonempty_allowed_subset():
    client = MagicMock()
    client.chat.completions.create.return_value = text_completion(LIBRARY_OVERVIEW_DSML)
    protocol = AgentProtocol(
        client,
        "gpt-test",
        allowed_tools=frozenset({"search_library"}),
    )

    response = await protocol.complete([{"role": "user", "content": "hi"}])

    assert response.tool_calls == []
    assert "DSML" not in response.text
    assert (
        response.text == "I'll pull up your library contents to give you an overview."
    )


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


def test_assistant_message_echoes_reasoning_content():
    protocol = AgentProtocol(MagicMock(), "gpt-test")
    call = ToolCall(id="call_1", name="list_papers", arguments={})
    message = protocol.assistant_message(
        AgentResponse(
            text="",
            tool_calls=[call],
            reasoning_content="Need to list the library first.",
        )
    )

    assert message["reasoning_content"] == "Need to list the library first."
    assert message["tool_calls"][0]["function"]["name"] == "list_papers"


def test_native_assistant_message_always_has_reasoning_content():
    protocol = AgentProtocol(MagicMock(), "gpt-test")
    message = protocol.assistant_message(AgentResponse(text="Hello."))
    assert message["reasoning_content"] == ""


def test_pad_assistant_reasoning_fills_gaps_and_keeps_existing():
    messages = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
        {
            "role": "assistant",
            "content": "",
            "reasoning_content": "thought",
            "tool_calls": [],
        },
    ]
    padded = pad_assistant_reasoning(messages)
    assert padded[1]["reasoning_content"] == ""
    assert padded[2]["reasoning_content"] == "thought"
    assert padded[0] == messages[0]


async def test_native_tool_request_pads_history_reasoning():
    client = MagicMock()
    client.chat.completions.create.return_value = text_completion("ok")
    protocol = AgentProtocol(client, "gpt-test")

    await protocol.complete(
        [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "earlier answer"},
            {"role": "user", "content": "again"},
        ]
    )

    sent = client.chat.completions.create.call_args.kwargs["messages"]
    assistants = [message for message in sent if message["role"] == "assistant"]
    assert assistants
    assert all(message.get("reasoning_content") == "" for message in assistants)


async def test_native_channel_captures_reasoning_from_completion():
    message = MagicMock(
        content=LIBRARY_OVERVIEW_DSML,
        reasoning_content="List papers, notes, and documents.",
        tool_calls=None,
    )
    client = MagicMock()
    client.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=message)]
    )
    protocol = AgentProtocol(client, "gpt-test")

    response = await protocol.complete([{"role": "user", "content": "hi"}])

    assert response.reasoning_content == "List papers, notes, and documents."
    assert protocol.assistant_message(response)["reasoning_content"] == (
        "List papers, notes, and documents."
    )
