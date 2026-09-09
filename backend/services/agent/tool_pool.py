"""The tools one turn may use, assembled when the turn starts.

`TOOL_HANDLERS` and `SHIPPED_RULES` are module-level and fixed, which is right
for tools that live in this repo — they cannot appear or vanish at runtime.
MCP tools can: a server drops out, a deployment adds one, an operator widens
the allow-list. Freezing that set at import time would mean offering the model
a tool whose server died an hour ago.

So the catalog is assembled per turn from two sources — the built-ins, and
whatever the registry currently has connected — and everything downstream (the
schemas advertised to the provider, the handler dispatch, the permission
policy) reads from the same assembled pool. One source of truth per turn, so a
tool cannot be advertised without a handler or run without a rule.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from services.agent.mcp_client import McpRegistry, McpTool
from services.agent.permissions import (
    SHIPPED_RULES,
    Decision,
    PermissionPolicy,
    PermissionRule,
)
from services.agent.telemetry import log_warning
from services.agent.tools import TOOL_HANDLERS, TOOL_SCHEMAS, ToolContext, ToolResult

Handler = Callable[..., Awaitable[ToolResult]]

UNCONFIGURED_REASON = (
    "is an external MCP tool this deployment has not put on its allow-list"
)


@dataclass(frozen=True)
class ToolPool:
    """One turn's catalog: what may be offered, run, and allowed."""

    schemas: list[dict]
    handlers: dict[str, Handler]
    policy: PermissionPolicy
    mcp_names: frozenset[str] = frozenset()

    @property
    def names(self) -> frozenset[str]:
        return frozenset(self.handlers)

    def handler(self, name: str) -> Handler | None:
        return self.handlers.get(name)


def _mcp_handler(registry: McpRegistry, tool: McpTool) -> Handler:
    """Adapt a remote tool to the same signature the built-ins have."""

    async def handler(ctx: ToolContext, **kwargs: object) -> ToolResult:
        content = await registry.call(tool.name, dict(kwargs))
        return ToolResult(
            content=content,
            summary=f"{tool.remote_name} on {tool.server}",
        )

    handler.__qualname__ = f"mcp_handler[{tool.name}]"
    return handler


def _mcp_rule(tool: McpTool) -> PermissionRule:
    """The host's decision, not the server's.

    A server may advertise `readOnlyHint`; that is a claim by the party asking
    to be trusted, so it is not consulted. A tool runs only if this
    deployment's config named it, and everything else needs a human.
    """
    if tool.allowed:
        return PermissionRule(
            tool=tool.name,
            decision=Decision.allow,
            reason=f"{tool.server} is configured to allow {tool.remote_name}",
        )
    return PermissionRule(
        tool=tool.name,
        decision=Decision.ask,
        reason=f"{tool.name!r} {UNCONFIGURED_REASON}",
    )


def assemble_tool_pool(
    registry: McpRegistry | None = None,
    *,
    base_schemas: list[dict] | None = None,
    base_handlers: dict[str, Handler] | None = None,
    base_rules: list[PermissionRule] | None = None,
) -> ToolPool:
    """Build the catalog for one turn."""
    schemas = list(base_schemas if base_schemas is not None else TOOL_SCHEMAS)
    handlers = dict(base_handlers if base_handlers is not None else TOOL_HANDLERS)
    rules = list(base_rules if base_rules is not None else SHIPPED_RULES)

    if registry is None:
        return ToolPool(
            schemas=schemas, handlers=handlers, policy=PermissionPolicy(rules)
        )

    mcp_names: set[str] = set()
    for tool in registry.tools():
        if tool.name in handlers:
            # Cannot happen with the `mcp__` prefix, but a shadowed built-in
            # would be the worst possible failure to debug.
            log_warning(
                "mcp_tool_shadows_builtin",
                tool=tool.name,
                server=tool.server,
            )
            continue
        handlers[tool.name] = _mcp_handler(registry, tool)
        schemas.append(tool.schema())
        rules.append(_mcp_rule(tool))
        mcp_names.add(tool.name)

    return ToolPool(
        schemas=schemas,
        handlers=handlers,
        policy=PermissionPolicy(rules),
        mcp_names=frozenset(mcp_names),
    )
