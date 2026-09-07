"""Two ways to ask a model for a tool call, behind one interface.

Providers that implement OpenAI function calling get the native channel.
Everything else (the devcontainer default gemma3:4b among them) gets a text
channel where the model writes a JSON object and the harness parses it.
The loop cannot tell the difference.
"""

import asyncio
import json
import re
import uuid
from dataclasses import dataclass, field

from openai import OpenAI

from services.agent.tools import TOOL_HANDLERS, TOOL_SCHEMAS, tool_catalog
from services.extraction import parse_json_object

MAX_TOKENS_DEFAULT = 4096
TEMPERATURE = 0.2

_TOOLS_UNSUPPORTED = re.compile(
    r"does not support tools"
    r"|tools? (?:is |are )?not supported"
    r"|support for tools"
    r"|function[ _]calling.*not supported"
    r"|unsupported.*\btools?\b"
    r"|unrecognized.*\btools?\b"
    r"|unknown (?:parameter|field).*\btools?\b"
    r"|no such parameter.*\btools?\b",
    re.IGNORECASE,
)

TEXT_PROTOCOL_INSTRUCTIONS = """
## How to call a tool

You cannot call tools natively. To use one, reply with a single JSON object and
nothing else:

{"tool": "search_library", "input": {"query": "hybrid retrieval"}}

Rules for tool calls:
- One tool per reply. No prose before or after the JSON.
- "input" holds the arguments; use {} when the tool takes none.
- You will receive the tool's output, then you may call another tool.

When you are ready to answer the researcher, reply with your answer as plain
text and no JSON object.

## Tools

{catalog}
""".strip()


class ProtocolSwitchedError(Exception):
    """The provider has no native tool calling, so the text channel takes over.

    Raised instead of retrying inline because the text channel needs a
    different system prompt, which only the caller owns.
    """


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: dict


@dataclass(frozen=True)
class AgentResponse:
    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)


def tools_unsupported(error: Exception) -> bool:
    """Did this call fail because the provider has no function calling?"""
    return bool(_TOOLS_UNSUPPORTED.search(str(error)))


def parse_text_tool_call(
    text: str, allowed_tools: set[str] | frozenset[str] | None = None
) -> ToolCall | None:
    """Read a `{"tool": ..., "input": {...}}` object out of a plain reply.

    Returns None for ordinary prose, so a final answer that happens to contain
    braces is never mistaken for a tool call.
    """
    if not text or "{" not in text:
        return None
    try:
        payload = parse_json_object(text)
    except (json.JSONDecodeError, ValueError):
        return None

    name = payload.get("tool") or payload.get("name")
    allowed = allowed_tools if allowed_tools is not None else set(TOOL_HANDLERS)
    if not isinstance(name, str) or name not in allowed:
        return None

    arguments = payload.get("input")
    if arguments is None:
        arguments = payload.get("arguments") or payload.get("parameters") or {}
    if not isinstance(arguments, dict):
        return None

    return ToolCall(id=f"call_{uuid.uuid4().hex[:12]}", name=name, arguments=arguments)


def _decode_arguments(raw: object) -> dict:
    if isinstance(raw, dict):
        return raw
    if not raw:
        return {}
    try:
        decoded = json.loads(str(raw))
    except json.JSONDecodeError:
        return {}
    return decoded if isinstance(decoded, dict) else {}


class ToolCapabilityCache:
    """Remembers which models rejected native tools, so we probe once."""

    def __init__(self) -> None:
        self._supported: dict[str, bool] = {}

    def get(self, model: str) -> bool | None:
        return self._supported.get(model)

    def set(self, model: str, supported: bool) -> None:
        self._supported[model] = supported


class AgentProtocol:
    """Runs one model turn and normalizes whatever comes back."""

    def __init__(
        self,
        llm_client: OpenAI,
        llm_model: str,
        capabilities: ToolCapabilityCache | None = None,
        tool_schemas: list[dict] | None = None,
        allowed_tools: set[str] | frozenset[str] | None = None,
    ) -> None:
        self.llm_client = llm_client
        self.llm_model = llm_model
        self.capabilities = capabilities or ToolCapabilityCache()
        self.tool_schemas = tool_schemas if tool_schemas is not None else TOOL_SCHEMAS
        self.allowed_tools: set[str] | frozenset[str] = (
            allowed_tools if allowed_tools is not None else frozenset(TOOL_HANDLERS)
        )

    def uses_native_tools(self) -> bool:
        return self.capabilities.get(self.llm_model) is not False

    def system_prompt(self, base_prompt: str) -> str:
        """The text channel needs the tool catalog spelled out in the prompt."""
        if self.uses_native_tools():
            return base_prompt
        catalog = tool_catalog(self.tool_schemas)
        instructions = TEXT_PROTOCOL_INSTRUCTIONS.replace("{catalog}", catalog)
        return f"{base_prompt}\n\n{instructions}"

    def assistant_message(self, response: AgentResponse) -> dict:
        if not response.tool_calls or not self.uses_native_tools():
            return {"role": "assistant", "content": response.text}
        return {
            "role": "assistant",
            "content": response.text or None,
            "tool_calls": [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.name,
                        "arguments": json.dumps(call.arguments),
                    },
                }
                for call in response.tool_calls
            ],
        }

    def tool_result_message(self, call: ToolCall, content: str) -> dict:
        if self.uses_native_tools():
            return {
                "role": "tool",
                "tool_call_id": call.id,
                "content": content,
            }
        return {
            "role": "user",
            "content": f"Result of {call.name}:\n{content}",
        }

    def _call_native(self, messages: list[dict]):
        from services.llm_chat import (
            effort_from_env,
            max_tokens_from_env,
            with_chat_extras,
        )

        effort = effort_from_env(
            "CHAT_REASONING_EFFORT", "LLM_REASONING_EFFORT", default="none"
        )
        max_tokens = max_tokens_from_env("CHAT_MAX_TOKENS", MAX_TOKENS_DEFAULT)
        return self.llm_client.chat.completions.create(
            **with_chat_extras(
                {
                    "model": self.llm_model,
                    "messages": messages,
                    "tools": self.tool_schemas,
                    "tool_choice": "auto",
                    "temperature": TEMPERATURE,
                    "max_tokens": max_tokens,
                    "stream": False,
                },
                effort=effort,
            )
        )

    def _call_text(self, messages: list[dict]):
        from services.llm_chat import (
            effort_from_env,
            max_tokens_from_env,
            with_chat_extras,
        )

        effort = effort_from_env(
            "CHAT_REASONING_EFFORT", "LLM_REASONING_EFFORT", default="none"
        )
        max_tokens = max_tokens_from_env("CHAT_MAX_TOKENS", MAX_TOKENS_DEFAULT)
        return self.llm_client.chat.completions.create(
            **with_chat_extras(
                {
                    "model": self.llm_model,
                    "messages": messages,
                    "temperature": TEMPERATURE,
                    "max_tokens": max_tokens,
                    "stream": False,
                },
                effort=effort,
            )
        )

    def _complete_sync(self, messages: list[dict]) -> AgentResponse:
        if self.uses_native_tools():
            try:
                completion = self._call_native(messages)
            except Exception as exc:
                if not tools_unsupported(exc):
                    raise
                print(
                    f"ℹ️  {self.llm_model} has no native tool calling; "
                    "switching to the JSON text protocol."
                )
                self.capabilities.set(self.llm_model, False)
                raise ProtocolSwitchedError(str(exc)) from exc
            self.capabilities.set(self.llm_model, True)
            return self._from_native(completion)

        completion = self._call_text(messages)
        return self._from_text(completion)

    def _from_native(self, completion) -> AgentResponse:
        message = completion.choices[0].message
        text = (getattr(message, "content", None) or "").strip()
        raw_calls = getattr(message, "tool_calls", None) or []

        calls = [
            ToolCall(
                id=getattr(call, "id", None) or f"call_{uuid.uuid4().hex[:12]}",
                name=call.function.name,
                arguments=_decode_arguments(call.function.arguments),
            )
            for call in raw_calls
            if getattr(call, "function", None) is not None
        ]
        if calls:
            return AgentResponse(text=text, tool_calls=calls)

        # Some providers accept `tools` and then quietly ignore them, answering
        # with the JSON object in plain text instead.
        fallback = parse_text_tool_call(text, self.allowed_tools)
        if fallback is not None:
            return AgentResponse(text="", tool_calls=[fallback])
        return AgentResponse(text=text)

    def _from_text(self, completion) -> AgentResponse:
        text = (completion.choices[0].message.content or "").strip()
        call = parse_text_tool_call(text, self.allowed_tools)
        if call is not None:
            return AgentResponse(text="", tool_calls=[call])
        return AgentResponse(text=text)

    async def complete(self, messages: list[dict]) -> AgentResponse:
        return await asyncio.to_thread(self._complete_sync, list(messages))
