import asyncio

import pytest

from services.agent.approvals import ApprovalBroker
from services.agent.events import EventSink
from services.agent.hooks import ToolHookContext, default_hooks
from services.agent.permissions import (
    DEFAULT_POLICY,
    WRITE_TOOLS,
    Decision,
    PermissionPolicy,
    PermissionRule,
    approval_enabled,
)
from services.agent.protocol import ToolCall
from services.agent.tools import TOOL_HANDLERS


def test_every_shipped_tool_has_a_rule():
    """A new tool has to be written into the policy, not just registered."""
    assert sorted(TOOL_HANDLERS) == DEFAULT_POLICY.known_tools()
    for name in TOOL_HANDLERS:
        assert DEFAULT_POLICY.check(name).decision in {Decision.allow, Decision.ask}


def test_write_tools_ask():
    for name in WRITE_TOOLS:
        verdict = DEFAULT_POLICY.check(name)
        assert verdict.decision is Decision.ask
        assert not verdict.allowed


def test_read_tools_still_run_without_asking():
    assert DEFAULT_POLICY.check("search_library").allowed
    assert DEFAULT_POLICY.check("fetch_tool_result").allowed


@pytest.mark.parametrize("off", ["off", "0", "false"])
def test_approval_mode_can_be_switched_off(monkeypatch, off):
    monkeypatch.setenv("ASK_APPROVAL_MODE", off)
    assert approval_enabled() is False


async def test_the_broker_resolves_an_outstanding_request():
    broker = ApprovalBroker(timeout_seconds=5)
    sink = EventSink()
    ctx = ToolHookContext(
        call=ToolCall(id="c1", name="memory_write", arguments={"key": "k"}),
        run_id="run",
        agent_id="main",
        depth=0,
        sink=sink,
    )
    verdict = DEFAULT_POLICY.check("memory_write")

    async def decide() -> None:
        for _ in range(50):
            await asyncio.sleep(0.01)
            if broker:
                request_id = next(iter(broker._pending))
                assert broker.resolve(request_id, True)
                return
        raise AssertionError("broker never registered the request")

    approved, _ = await asyncio.gather(broker.request(ctx, verdict), decide())
    assert approved is True
    statuses = [event.data["status"] for event in sink.drain()]
    assert "pending" in statuses
    assert "approved" in statuses


async def test_a_timeout_is_a_denial():
    broker = ApprovalBroker(timeout_seconds=0.05)
    sink = EventSink()
    ctx = ToolHookContext(
        call=ToolCall(id="c1", name="memory_write", arguments={}),
        run_id="run",
        agent_id="main",
        depth=0,
        sink=sink,
    )
    approved = await broker.request(ctx, DEFAULT_POLICY.check("memory_write"))
    assert approved is False
    assert any(event.data["status"] == "timeout" for event in sink.drain())


async def test_hooks_allow_writes_when_approval_is_off(monkeypatch):
    monkeypatch.setenv("ASK_APPROVAL_MODE", "off")
    chain = default_hooks(DEFAULT_POLICY)
    sink = EventSink()
    ctx = ToolHookContext(
        call=ToolCall(id="c1", name="memory_write", arguments={"key": "k"}),
        run_id="run",
        agent_id="main",
        depth=0,
        sink=sink,
    )
    assert await chain.pre(ctx) is None


async def test_hooks_refuse_a_custom_ask_when_approval_is_off(monkeypatch):
    monkeypatch.setenv("ASK_APPROVAL_MODE", "off")
    policy = PermissionPolicy(
        [PermissionRule(tool="list_papers", decision=Decision.ask)]
    )
    chain = default_hooks(policy)
    ctx = ToolHookContext(
        call=ToolCall(id="c1", name="list_papers", arguments={}),
        run_id="run",
        agent_id="main",
        depth=0,
        sink=EventSink(),
    )
    decision = await chain.pre(ctx)
    assert decision is not None
    assert decision.decision == Decision.ask.value


async def test_hooks_refuse_ask_without_an_approver(monkeypatch):
    monkeypatch.setenv("ASK_APPROVAL_MODE", "on")
    chain = default_hooks(DEFAULT_POLICY, approver=None)
    ctx = ToolHookContext(
        call=ToolCall(id="c1", name="memory_write", arguments={"key": "k"}),
        run_id="run",
        agent_id="main",
        depth=0,
        sink=EventSink(),
    )
    decision = await chain.pre(ctx)
    assert decision is not None
    assert decision.decision == Decision.ask.value
