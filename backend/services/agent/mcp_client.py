"""Tools from outside this codebase, on the host's terms.

The repo already speaks MCP in one direction: `mcp_server.py` exposes the
library to Cursor. This is the other direction — the agent consuming tools a
research group already runs (an institutional search index, a lab notebook, a
compliance archive). Without it every integration means editing `tools.py`,
which is exactly the coupling MCP exists to remove.

Three rules shape what follows, and all three come from s14.

*Names are the host's.* A server calls its tool `search`; so does this
library. Both are exposed as `mcp__{server}__{tool}`, normalised to the
`[A-Za-z0-9_-]{1,64}` window that OpenAI-compatible providers accept.
Normalising can make two distinct remote names collide, so collisions are
detected and the loser is dropped rather than silently shadowing another
server's tool.

*A server's self-description is not authorisation.* `readOnlyHint` is a claim
by the same party asking to be trusted. The host decides: a tool is allowed
only if it is named in this deployment's config, and everything else arrives
as `ask`, which the approval flow either escalates or refuses.

*The model never picks what to connect to.* Server commands and URLs come from
`mcp_servers.json` or `MCP_SERVERS`, never from a tool argument — otherwise
"call this tool" becomes "run this process".

A server that is slow, broken, or absent must not cost the researcher a turn,
so connection and discovery failures are logged and the server is skipped.
"""

from __future__ import annotations

import json
import os
import re
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from services.agent.telemetry import log_event, log_warning

# Providers reject a function name outside this window; DeepSeek and OpenAI
# both cap it at 64 characters.
MAX_TOOL_NAME_CHARS = 64
NAMESPACE_PREFIX = "mcp"
SEPARATOR = "__"
_ILLEGAL = re.compile(r"[^A-Za-z0-9_-]+")

DEFAULT_CONFIG_FILENAME = "mcp_servers.json"
DEFAULT_TIMEOUT_SECONDS = 30.0
# A tool result is context. The same ceiling the built-in tools live under.
MAX_RESULT_CHARS = 8000


class McpConfigError(Exception):
    """The deployment's MCP configuration cannot be read."""


def normalize_segment(value: str) -> str:
    """One name segment, safe for a provider's function-name field."""
    cleaned = _ILLEGAL.sub("_", value.strip()).strip("_")
    return cleaned or "unnamed"


def namespaced(server: str, tool: str) -> str:
    """`mcp__server__tool`, trimmed to fit rather than rejected.

    Truncation is what makes collision detection load-bearing: two long remote
    names can survive normalisation intact and still collide once cut.
    """
    name = f"{NAMESPACE_PREFIX}{SEPARATOR}{normalize_segment(server)}"
    name = f"{name}{SEPARATOR}{normalize_segment(tool)}"
    return name[:MAX_TOOL_NAME_CHARS]


@dataclass(frozen=True)
class McpServerConfig:
    """One server this deployment is willing to talk to."""

    name: str
    command: str = ""
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    url: str = ""
    # Host-owned allow-list. Remote tool names, not namespaced ones: the
    # config is written against the server's own documentation.
    allow: frozenset[str] = frozenset()
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    enabled: bool = True

    @property
    def transport(self) -> str:
        return "http" if self.url else "stdio"

    def permits(self, remote_tool: str) -> bool:
        return remote_tool in self.allow


def _as_str_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]


def parse_server_configs(payload: object) -> list[McpServerConfig]:
    """Read the `mcpServers` shape Claude Desktop and Cursor already use."""
    if isinstance(payload, dict) and "mcpServers" in payload:
        entries = payload["mcpServers"]
    else:
        entries = payload
    if not isinstance(entries, dict):
        raise McpConfigError("expected an object of server name -> config")

    configs: list[McpServerConfig] = []
    for name, entry in entries.items():
        if not isinstance(entry, dict):
            raise McpConfigError(f"server {name!r} is not an object")
        command = str(entry.get("command") or "")
        url = str(entry.get("url") or "")
        if not command and not url:
            raise McpConfigError(f"server {name!r} needs a command or a url")
        env = entry.get("env")
        configs.append(
            McpServerConfig(
                name=str(name),
                command=command,
                args=_as_str_list(entry.get("args")),
                env=(
                    {str(k): str(v) for k, v in env.items()}
                    if isinstance(env, dict)
                    else {}
                ),
                url=url,
                allow=frozenset(_as_str_list(entry.get("allow"))),
                timeout_seconds=float(
                    entry.get("timeout_seconds") or DEFAULT_TIMEOUT_SECONDS
                ),
                enabled=bool(entry.get("enabled", True)),
            )
        )
    return configs


def load_server_configs(
    *, inline: str | None = None, path: str | Path | None = None
) -> list[McpServerConfig]:
    """Where the deployment says its servers are. Absent config is not an error."""
    raw = inline if inline is not None else os.getenv("MCP_SERVERS")
    if raw and raw.strip():
        try:
            return parse_server_configs(json.loads(raw))
        except json.JSONDecodeError as exc:
            raise McpConfigError(f"MCP_SERVERS is not valid JSON: {exc}") from exc

    candidate = path or os.getenv("MCP_SERVERS_FILE")
    config_path = (
        Path(candidate)
        if candidate
        else Path(__file__).resolve().parent.parent.parent / DEFAULT_CONFIG_FILENAME
    )
    if not config_path.exists():
        return []
    try:
        return parse_server_configs(json.loads(config_path.read_text()))
    except json.JSONDecodeError as exc:
        raise McpConfigError(f"{config_path} is not valid JSON: {exc}") from exc


@dataclass(frozen=True)
class McpTool:
    """A remote tool, as this host will expose it to the model."""

    server: str
    remote_name: str
    name: str
    description: str
    input_schema: dict
    allowed: bool

    def schema(self) -> dict:
        """The provider-facing function schema.

        The server's description is quoted, and where it came from is stated
        rather than implied: the model should weigh a third-party tool
        differently from one this repo ships.
        """
        parameters = self.input_schema or {"type": "object", "properties": {}}
        described = self.description.strip() or f"{self.remote_name} on {self.server}"
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": f"[{self.server} via MCP] {described}",
                "parameters": parameters,
            },
        }


def _tool_description(tool: object) -> str:
    value = getattr(tool, "description", None)
    return value if isinstance(value, str) else ""


def _tool_schema(tool: object) -> dict:
    value = getattr(tool, "inputSchema", None)
    return value if isinstance(value, dict) else {}


def content_to_text(result: object) -> str:
    """Flatten an MCP tool result into something a chat message can hold."""
    blocks = getattr(result, "content", None) or []
    parts: list[str] = []
    for block in blocks:
        text = getattr(block, "text", None)
        if isinstance(text, str) and text.strip():
            parts.append(text)
            continue
        kind = getattr(block, "type", None) or type(block).__name__
        # Images and embedded resources cannot go in a tool result message;
        # naming them is more useful than dropping them silently.
        parts.append(f"[{kind} content omitted]")
    if not parts:
        structured = getattr(result, "structuredContent", None)
        if structured:
            parts.append(json.dumps(structured, ensure_ascii=False, default=str))
    joined = "\n\n".join(parts).strip()
    if len(joined) > MAX_RESULT_CHARS:
        return (
            joined[:MAX_RESULT_CHARS].rstrip()
            + f"\n\n[truncated at {MAX_RESULT_CHARS} characters]"
        )
    return joined


class McpRegistry:
    """Every connected server's tools, under host-chosen names.

    Connections are opened once at startup and reused: an stdio server means
    spawning a process, which is not something to do inside a chat turn.
    `tools()` is therefore cheap enough for the loop to call every turn, which
    is the point — a server that dropped out between turns stops being offered.
    """

    def __init__(self, configs: list[McpServerConfig] | None = None) -> None:
        self._configs = list(configs or [])
        self._stack = AsyncExitStack()
        self._sessions: dict[str, Any] = {}
        self._tools: dict[str, McpTool] = {}
        self._connected = False

    @property
    def connected(self) -> bool:
        return self._connected

    def tools(self) -> list[McpTool]:
        return list(self._tools.values())

    def get(self, name: str) -> McpTool | None:
        return self._tools.get(name)

    def config(self, server: str) -> McpServerConfig | None:
        for config in self._configs:
            if config.name == server:
                return config
        return None

    async def connect(self) -> None:
        """Open every configured server. One failure never stops the rest."""
        if self._connected:
            return
        self._connected = True
        for config in self._configs:
            if not config.enabled:
                continue
            try:
                await self._connect_one(config)
            except Exception as exc:  # noqa: BLE001 - a server is not the app
                log_warning(
                    "mcp_connect_failed",
                    server=config.name,
                    transport=config.transport,
                    error=str(exc),
                )

    async def _open_session(self, config: McpServerConfig):
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        if config.url:
            from mcp.client.streamable_http import streamablehttp_client

            streams = await self._stack.enter_async_context(
                streamablehttp_client(config.url)
            )
            read, write = streams[0], streams[1]
        else:
            params = StdioServerParameters(
                command=config.command,
                args=config.args,
                env={**os.environ, **config.env} if config.env else None,
            )
            read, write = await self._stack.enter_async_context(stdio_client(params))
        session = await self._stack.enter_async_context(ClientSession(read, write))
        await session.initialize()
        return session

    async def _connect_one(self, config: McpServerConfig) -> None:
        session = await self._open_session(config)
        self._sessions[config.name] = session
        listing = await session.list_tools()
        self.register(config, getattr(listing, "tools", []) or [])

    def register(self, config: McpServerConfig, tools: list[object]) -> int:
        """Namespace a server's tools and take the ones that fit.

        Split out from `_connect_one` because it is the part with rules in it:
        collision handling and the allow-list. A test can exercise it without
        a subprocess.
        """
        taken = 0
        for tool in tools:
            remote = getattr(tool, "name", None)
            if not isinstance(remote, str) or not remote:
                continue
            name = namespaced(config.name, remote)
            clash = self._tools.get(name)
            if clash is not None:
                # Dropping is the safe direction: shadowing would route the
                # model's call to a different server than the name says.
                log_warning(
                    "mcp_tool_name_collision",
                    server=config.name,
                    tool=remote,
                    namespaced=name,
                    kept_from=clash.server,
                    kept=clash.remote_name,
                )
                continue
            self._tools[name] = McpTool(
                server=config.name,
                remote_name=remote,
                name=name,
                description=_tool_description(tool),
                input_schema=_tool_schema(tool),
                allowed=config.permits(remote),
            )
            taken += 1
        log_event(
            "mcp_server_ready",
            server=config.name,
            transport=config.transport,
            tools=taken,
            allowed=sum(
                1
                for item in self._tools.values()
                if item.server == config.name and item.allowed
            ),
        )
        return taken

    async def call(self, name: str, arguments: dict) -> str:
        """Run a remote tool. Errors come back as text, as tool errors do."""
        tool = self._tools.get(name)
        if tool is None:
            return f"{name} is not a connected MCP tool."
        session = self._sessions.get(tool.server)
        if session is None:
            return f"{tool.server} is not connected right now."
        try:
            result = await session.call_tool(tool.remote_name, arguments)
        except Exception as exc:  # noqa: BLE001 - the model can work around it
            log_warning(
                "mcp_call_failed",
                server=tool.server,
                tool=tool.remote_name,
                error=str(exc),
            )
            return f"{name} failed: {exc}"
        if getattr(result, "isError", False):
            return f"{name} reported an error: {content_to_text(result)}"
        return content_to_text(result) or f"{name} returned nothing."

    async def close(self) -> None:
        await self._stack.aclose()
        self._sessions.clear()
        self._tools.clear()
        self._connected = False
