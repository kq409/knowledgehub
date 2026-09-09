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

from services.agent.telemetry import log_event
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
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    reasoning_content: str | None = None
    finish_reason: str | None = None

    @property
    def truncated(self) -> bool:
        """The provider stopped because it ran out of output room."""
        return self.finish_reason == "length"


def tools_unsupported(error: Exception) -> bool:
    """Did this call fail because the provider has no function calling?"""
    return bool(_TOOLS_UNSUPPORTED.search(str(error)))


# DeepSeek V3.2/V4 emit tool calls as DSML. Official APIs usually parse that
# into `message.tool_calls`; when they miss, the markup lands in `content`
# (canonical `<｜DSML｜tool_calls>`, or a degraded `||` / `｜｜` variant).
DSML_TOKEN = "｜DSML｜"
_DSML_PREFIX = re.compile(
    r"<\s*(/?)\s*(?:\|{1,2}|｜{1,2})\s*DSML\s*(?:\|{1,2}|｜{1,2})\s*"
)
_INVOKE_RE = re.compile(
    rf"<{re.escape(DSML_TOKEN)}invoke\s+name=(?P<q>[\"'])(?P<name>[^\"']+)(?P=q)\s*"
    rf"(?:/>|>(?P<body>.*?)</{re.escape(DSML_TOKEN)}invoke>)",
    re.DOTALL,
)
_PARAM_RE = re.compile(
    rf"<{re.escape(DSML_TOKEN)}parameter\s+name=(?P<q>[\"'])(?P<name>[^\"']+)(?P=q)"
    rf"(?:\s+string=(?P<sq>[\"'])(?P<string>true|false)(?P=sq))?\s*>"
    rf"(?P<value>.*?)</{re.escape(DSML_TOKEN)}parameter>",
    re.DOTALL | re.IGNORECASE,
)
_DSML_BLOCK_RE = re.compile(
    rf"<{re.escape(DSML_TOKEN)}(?:tool_calls|function_calls)>.*?"
    rf"</{re.escape(DSML_TOKEN)}(?:tool_calls|function_calls)>",
    re.DOTALL,
)
_DSML_INVOKE_BLOCK_RE = re.compile(
    rf"<{re.escape(DSML_TOKEN)}invoke\b[^>]*/>|"
    rf"<{re.escape(DSML_TOKEN)}invoke\b.*?</{re.escape(DSML_TOKEN)}invoke>",
    re.DOTALL,
)
_DSML_WRAPPER_RE = re.compile(
    rf"</?{re.escape(DSML_TOKEN)}(?:tool_calls|function_calls)>"
)


def _normalize_dsml(text: str) -> str:
    """Map `<||DSML||tag>` / `<｜｜DSML｜｜tag>` back to `<｜DSML｜tag>`."""
    return _DSML_PREFIX.sub(lambda match: f"<{match.group(1)}{DSML_TOKEN}", text)


def contains_dsml(text: str) -> bool:
    return bool(text) and DSML_TOKEN in _normalize_dsml(text)


def strip_dsml_markup(text: str) -> str:
    """Drop leaked DSML so leftover prose can ship as the answer."""
    if not contains_dsml(text):
        return text
    stripped = _DSML_BLOCK_RE.sub("", _normalize_dsml(text))
    stripped = _DSML_INVOKE_BLOCK_RE.sub("", stripped)
    stripped = _DSML_WRAPPER_RE.sub("", stripped)
    return re.sub(r"\n{3,}", "\n\n", stripped).strip()


def _coerce_parameter(raw: str, is_string: bool | None) -> object:
    cleaned = raw.strip()
    if is_string is True or not cleaned:
        return cleaned
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return cleaned


def _unwrap_arguments(arguments: dict) -> dict:
    if set(arguments) != {"arguments"}:
        return arguments
    inner = arguments["arguments"]
    if isinstance(inner, dict):
        return inner
    if isinstance(inner, str):
        try:
            decoded = json.loads(inner)
        except json.JSONDecodeError:
            return arguments
        if isinstance(decoded, dict):
            return decoded
    return arguments


def _parse_dsml_arguments(body: str | None) -> dict:
    if not body:
        return {}
    arguments: dict = {}
    for match in _PARAM_RE.finditer(body):
        flag = match.group("string")
        is_string = None if flag is None else flag.lower() == "true"
        arguments[match.group("name")] = _coerce_parameter(
            match.group("value"), is_string
        )
    return _unwrap_arguments(arguments)


def _parse_dsml_tool_calls(
    text: str, allowed: set[str] | frozenset[str]
) -> list[ToolCall]:
    if not contains_dsml(text):
        return []
    calls: list[ToolCall] = []
    for match in _INVOKE_RE.finditer(_normalize_dsml(text)):
        name = match.group("name")
        if name not in allowed:
            continue
        calls.append(
            ToolCall(
                id=f"call_{uuid.uuid4().hex[:12]}",
                name=name,
                arguments=_parse_dsml_arguments(match.group("body")),
            )
        )
    return calls


def _parse_json_tool_call(
    text: str, allowed: set[str] | frozenset[str]
) -> ToolCall | None:
    if "{" not in text:
        return None
    try:
        payload = parse_json_object(text)
    except (json.JSONDecodeError, ValueError):
        return None

    name = payload.get("tool") or payload.get("name")
    if not isinstance(name, str) or name not in allowed:
        return None

    arguments = payload.get("input")
    if arguments is None:
        arguments = payload.get("arguments") or payload.get("parameters") or {}
    if not isinstance(arguments, dict):
        return None

    return ToolCall(id=f"call_{uuid.uuid4().hex[:12]}", name=name, arguments=arguments)


def parse_text_tool_calls(
    text: str, allowed_tools: set[str] | frozenset[str] | None = None
) -> list[ToolCall]:
    """Read tool calls out of a plain reply (JSON protocol or leaked DSML)."""
    if not text:
        return []
    allowed = allowed_tools if allowed_tools is not None else set(TOOL_HANDLERS)
    dsml_calls = _parse_dsml_tool_calls(text, allowed)
    if dsml_calls:
        return dsml_calls
    json_call = _parse_json_tool_call(text, allowed)
    return [json_call] if json_call else []


def parse_text_tool_call(
    text: str, allowed_tools: set[str] | frozenset[str] | None = None
) -> ToolCall | None:
    """Read a `{"tool": ..., "input": {...}}` object out of a plain reply.

    Returns None for ordinary prose, so a final answer that happens to contain
    braces is never mistaken for a tool call. Also recovers DeepSeek DSML that
    leaked into `message.content` instead of `tool_calls`.
    """
    calls = parse_text_tool_calls(text, allowed_tools)
    return calls[0] if calls else None


def _message_field(message: object, *names: str) -> str:
    """String attributes only — MagicMock otherwise invents truthy stand-ins."""
    for name in names:
        value = getattr(message, name, None)
        if isinstance(value, str):
            return value.strip()
    return ""


def _reasoning_content(message: object) -> str | None:
    """Chain-of-thought to echo on the next request (DeepSeek thinking mode)."""
    for name in ("reasoning_content", "reasoning"):
        value = getattr(message, name, None)
        if isinstance(value, str) and value.strip():
            return value
    extra = getattr(message, "model_extra", None)
    if isinstance(extra, dict):
        value = extra.get("reasoning_content") or extra.get("reasoning")
        if isinstance(value, str) and value.strip():
            return value
    dump = getattr(message, "model_dump", None)
    if callable(dump):
        try:
            payload = dump()
        except TypeError:
            payload = None
        if isinstance(payload, dict):
            value = payload.get("reasoning_content")
            if isinstance(value, str) and value.strip():
                return value
    return None


def pad_assistant_reasoning(messages: list[dict]) -> list[dict]:
    """DeepSeek thinking + tools 400s unless every assistant has this field.

    Planner-injected tool calls, replayed chat history, and recovered DSML
    turns often have no reasoning. An empty string satisfies the API.
    """
    padded: list[dict] = []
    for message in messages:
        if message.get("role") != "assistant":
            padded.append(message)
            continue
        if isinstance(message.get("reasoning_content"), str):
            padded.append(message)
            continue
        padded.append({**message, "reasoning_content": ""})
    return padded


def _fallback_tool_calls(
    text: str,
    reasoning: str,
    allowed: set[str] | frozenset[str],
) -> list[ToolCall]:
    return parse_text_tool_calls(text, allowed) or parse_text_tool_calls(
        reasoning, allowed
    )


def _visible_text(text: str, *, recovered_calls: bool) -> str:
    """Plain text that may ship as the answer.

    Recovered JSON/DSML calls are not an answer — "I'll pull up the library"
    is the model announcing a tool call. Native `tool_calls` keep any prose
    (the loop continues). Leftover DSML markup is always stripped.
    """
    if recovered_calls:
        return ""
    return strip_dsml_markup(text) if contains_dsml(text) else text


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

    def _recovery_tools(self) -> set[str] | frozenset[str]:
        """Names we honor when the model writes a call into plain text.

        An empty `allowed_tools` means this turn did not advertise tools (the
        planner thought none were needed). A leaked DSML/JSON invoke is still
        a call — recover against the full catalog. A non-empty subset
        (subagent) stays restricted.
        """
        if self.allowed_tools:
            return self.allowed_tools
        return frozenset(TOOL_HANDLERS)

    def system_prompt(self, base_prompt: str) -> str:
        """The text channel needs the tool catalog spelled out in the prompt."""
        if self.uses_native_tools() or not self.tool_schemas:
            return base_prompt
        catalog = tool_catalog(self.tool_schemas)
        instructions = TEXT_PROTOCOL_INSTRUCTIONS.replace("{catalog}", catalog)
        return f"{base_prompt}\n\n{instructions}"

    def assistant_message(self, response: AgentResponse) -> dict:
        if not response.tool_calls or not self.uses_native_tools():
            message: dict = {"role": "assistant", "content": response.text}
        else:
            message = {
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
        # DeepSeek thinking + tools: the field must be present even when empty.
        if self.uses_native_tools():
            message["reasoning_content"] = response.reasoning_content or ""
        return message

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

    def _call_native(self, messages: list[dict], max_tokens: int | None = None):
        from services.llm_chat import (
            effort_from_env,
            max_tokens_from_env,
            with_chat_extras,
        )

        effort = effort_from_env(
            "CHAT_REASONING_EFFORT", "LLM_REASONING_EFFORT", default="none"
        )
        max_tokens = max_tokens or max_tokens_from_env(
            "CHAT_MAX_TOKENS", MAX_TOKENS_DEFAULT
        )
        payload: dict = {
            "model": self.llm_model,
            "messages": messages,
            "temperature": TEMPERATURE,
            "max_tokens": max_tokens,
            "stream": False,
        }
        if self.tool_schemas:
            payload["tools"] = self.tool_schemas
            payload["tool_choice"] = "auto"
            payload["messages"] = pad_assistant_reasoning(messages)
        return self.llm_client.chat.completions.create(
            **with_chat_extras(payload, effort=effort)
        )

    def _call_text(self, messages: list[dict], max_tokens: int | None = None):
        from services.llm_chat import (
            effort_from_env,
            max_tokens_from_env,
            with_chat_extras,
        )

        effort = effort_from_env(
            "CHAT_REASONING_EFFORT", "LLM_REASONING_EFFORT", default="none"
        )
        max_tokens = max_tokens or max_tokens_from_env(
            "CHAT_MAX_TOKENS", MAX_TOKENS_DEFAULT
        )
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

    def _complete_sync(
        self, messages: list[dict], max_tokens: int | None = None
    ) -> AgentResponse:
        if self.uses_native_tools():
            try:
                completion = self._call_native(messages, max_tokens)
            except Exception as exc:
                if not tools_unsupported(exc):
                    raise
                log_event(
                    "native_tools_unsupported",
                    model=self.llm_model,
                    detail="switching to the JSON text protocol",
                )
                self.capabilities.set(self.llm_model, False)
                raise ProtocolSwitchedError(str(exc)) from exc
            self.capabilities.set(self.llm_model, True)
            return self._from_native(completion)

        completion = self._call_text(messages, max_tokens)
        return self._from_text(completion)

    def _native_tool_calls(self, message) -> list[ToolCall]:
        raw_calls = getattr(message, "tool_calls", None) or []
        return [
            ToolCall(
                id=getattr(call, "id", None) or f"call_{uuid.uuid4().hex[:12]}",
                name=call.function.name,
                arguments=_decode_arguments(call.function.arguments),
            )
            for call in raw_calls
            if getattr(call, "function", None) is not None
        ]

    def _from_message(
        self, completion, *, native_calls: list[ToolCall]
    ) -> AgentResponse:
        from services.llm_chat import usage_tokens

        choice = completion.choices[0]
        message = choice.message
        text = _message_field(message, "content")
        reasoning = _reasoning_content(message)
        prompt, completion_tokens = usage_tokens(completion)
        finish_reason = getattr(choice, "finish_reason", None)

        # Providers that accept `tools` sometimes still write the call as JSON
        # or DeepSeek DSML in `content` / `reasoning` instead of `tool_calls`.
        calls = native_calls or _fallback_tool_calls(
            text, reasoning or "", self._recovery_tools()
        )
        return AgentResponse(
            text=_visible_text(text, recovered_calls=bool(calls) and not native_calls),
            tool_calls=calls,
            prompt_tokens=prompt,
            completion_tokens=completion_tokens,
            reasoning_content=reasoning,
            # A MagicMock invents a truthy stand-in, so only real strings count.
            finish_reason=finish_reason if isinstance(finish_reason, str) else None,
        )

    def _from_native(self, completion) -> AgentResponse:
        message = completion.choices[0].message
        return self._from_message(
            completion, native_calls=self._native_tool_calls(message)
        )

    def _from_text(self, completion) -> AgentResponse:
        return self._from_message(completion, native_calls=[])

    async def complete(
        self, messages: list[dict], max_tokens: int | None = None
    ) -> AgentResponse:
        return await asyncio.to_thread(self._complete_sync, list(messages), max_tokens)
