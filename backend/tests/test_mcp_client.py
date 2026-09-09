from dataclasses import dataclass

from services.agent.mcp_client import (
    McpRegistry,
    McpServerConfig,
    namespaced,
    normalize_segment,
    parse_server_configs,
)
from services.agent.permissions import Decision
from services.agent.tool_pool import assemble_tool_pool
from services.agent.tools import TOOL_HANDLERS


@dataclass
class FakeTool:
    name: str
    description: str = ""
    inputSchema: dict | None = None  # noqa: N815 — matches MCP SDK attribute


def test_namespacing_and_truncation():
    assert namespaced("docs", "search") == "mcp__docs__search"
    long_name = namespaced("a-very-long-server-name", "a-very-long-tool-name-too")
    assert len(long_name) <= 64
    assert long_name.startswith("mcp__")


def test_normalising_illegal_characters():
    assert normalize_segment("docs.one/get.version") == "docs_one_get_version"


def test_parse_cursor_shaped_config():
    configs = parse_server_configs(
        {
            "mcpServers": {
                "docs": {
                    "command": "uv",
                    "args": ["run", "docs"],
                    "allow": ["search"],
                }
            }
        }
    )
    assert configs[0].name == "docs"
    assert configs[0].permits("search")
    assert not configs[0].permits("delete")


def test_unconfigured_mcp_tools_ask_and_allowed_ones_run():
    registry = McpRegistry()
    config = McpServerConfig(name="docs", command="x", allow=frozenset({"search"}))
    registry.register(config, [FakeTool("search"), FakeTool("delete")])
    pool = assemble_tool_pool(registry)

    assert pool.policy.check("mcp__docs__search").decision is Decision.allow
    assert pool.policy.check("mcp__docs__delete").decision is Decision.ask
    assert "mcp__docs__search" in pool.handlers
    assert "mcp__docs__search" not in TOOL_HANDLERS


def test_a_collision_after_normalising_keeps_the_first_tool():
    registry = McpRegistry()
    first = McpServerConfig(name="docs.one", command="x", allow=frozenset({"get"}))
    second = McpServerConfig(name="docs_one", command="x", allow=frozenset({"get"}))
    # Both become mcp__docs_one__get after normalisation.
    taken_first = registry.register(first, [FakeTool("get")])
    taken_second = registry.register(second, [FakeTool("get")])
    assert taken_first == 1
    assert taken_second == 0
    kept = registry.get(namespaced("docs.one", "get"))
    assert kept is not None
    assert kept.server == "docs.one"
