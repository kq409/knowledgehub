"""Extension points around tool execution.

The loop used to check permissions inline, which meant every new cross-cutting
concern -- metrics, redaction, rate limiting, human approval -- had to be
edited into the middle of the execution line. Here each concern is a function
registered on a chain instead:

    PreToolUse   runs before the handler. Returning a decision intercepts the
                 call; returning None passes it to the next hook, and an
                 unintercepted call runs.
    PostToolUse  runs after the handler with the result and how long it took.
                 It cannot change the result, only observe it.

Permission is just the first `PreToolUse` hook. A hook that raises is logged
and skipped: an observability concern must never take a turn down, and a
`PreToolUse` that fails must not silently become an allow, so it is treated as
"no opinion" only after the failure is recorded.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass

from services.agent.events import EventSink
from services.agent.permissions import (
    WRITE_TOOLS,
    Decision,
    PermissionPolicy,
    PermissionVerdict,
    approval_enabled,
)
from services.agent.protocol import ToolCall
from services.agent.telemetry import log_event, log_warning
from services.agent.tools import ToolResult

APPROVAL_TIMEOUT_REASON = "was not approved in time, so it did not run"


@dataclass(frozen=True)
class ToolHookContext:
    """What a hook knows about the call it is inspecting."""

    call: ToolCall
    run_id: str
    agent_id: str
    depth: int
    sink: EventSink
    # The turn's policy, when it differs from the one the chain was built
    # with. MCP tools only exist while their server is connected, so the rule
    # set is assembled per turn (see `tool_pool.assemble_tool_pool`).
    policy: PermissionPolicy | None = None

    @property
    def name(self) -> str:
        return self.call.name


@dataclass(frozen=True)
class PreToolDecision:
    """A hook's interception. `result` is what the model is told instead."""

    result: ToolResult
    decision: str
    reason: str = ""


PreToolUse = Callable[[ToolHookContext], Awaitable[PreToolDecision | None]]
PostToolUse = Callable[[ToolHookContext, ToolResult, float], Awaitable[None]]

# Set by the host once an approval transport exists (see routers/chat.py). It
# answers whether a tool the policy marked `ask` may run for this turn.
Approver = Callable[[ToolHookContext, PermissionVerdict], Awaitable[bool]]


def refusal_result(call: ToolCall, reason: str) -> ToolResult:
    """What the model is told when a hook turns a call down."""
    return ToolResult(
        content=(
            f"{call.name} was not run because it {reason}. "
            "Work with the tools you do have."
        ),
        summary=f"{call.name} refused: {reason}",
    )


def permission_hook(
    policy: PermissionPolicy, approver: Approver | None = None
) -> PreToolUse:
    """Deny unknown and denied tools; route `ask` through the approver.

    Without an approver an `ask` tool is refused, because a chat turn that
    cannot pause for a human must not let an unapproved write through.
    """

    async def hook(ctx: ToolHookContext) -> PreToolDecision | None:
        verdict = (ctx.policy or policy).check(ctx.name)
        if verdict.allowed:
            return None

        if verdict.decision is Decision.ask:
            # Off restores the previous always-allow behaviour for the write
            # tools this product ships. A custom `ask` rule (tests, an
            # unconfigured MCP tool) still needs a human or it is refused.
            if not approval_enabled() and ctx.name in WRITE_TOOLS:
                return None
            if approver is not None:
                if await approver(ctx, verdict):
                    return None
                return PreToolDecision(
                    result=refusal_result(ctx.call, "the researcher declined it"),
                    decision=Decision.deny.value,
                    reason="the researcher declined it",
                )

        return PreToolDecision(
            result=refusal_result(ctx.call, verdict.reason),
            decision=verdict.decision.value,
            reason=verdict.reason,
        )

    return hook


async def metrics_hook(
    ctx: ToolHookContext, result: ToolResult, latency_ms: float
) -> None:
    """One structured line per finished tool call."""
    log_event(
        "tool_finished",
        run_id=ctx.run_id,
        agent_id=ctx.agent_id,
        depth=ctx.depth,
        tool=ctx.name,
        latency_ms=round(latency_ms, 1),
        result_chars=len(result.content),
        has_artifact=result.artifact is not None,
    )


class HookChain:
    """The hooks that wrap one agent's tool calls."""

    def __init__(
        self,
        pre: Sequence[PreToolUse] = (),
        post: Sequence[PostToolUse] = (),
    ) -> None:
        self._pre = list(pre)
        self._post = list(post)

    def add_pre(self, hook: PreToolUse) -> None:
        self._pre.append(hook)

    def add_post(self, hook: PostToolUse) -> None:
        self._post.append(hook)

    async def pre(self, ctx: ToolHookContext) -> PreToolDecision | None:
        """The first hook with an opinion wins."""
        for hook in self._pre:
            try:
                decision = await hook(ctx)
            except Exception as exc:  # noqa: BLE001 - a hook cannot end the turn
                log_warning(
                    "pre_tool_hook_failed",
                    run_id=ctx.run_id,
                    tool=ctx.name,
                    hook=getattr(hook, "__qualname__", repr(hook)),
                    error=str(exc),
                )
                continue
            if decision is not None:
                log_event(
                    "tool_intercepted",
                    run_id=ctx.run_id,
                    agent_id=ctx.agent_id,
                    tool=ctx.name,
                    decision=decision.decision,
                    reason=decision.reason,
                )
                return decision
        return None

    async def post(
        self, ctx: ToolHookContext, result: ToolResult, latency_ms: float
    ) -> None:
        for hook in self._post:
            try:
                await hook(ctx, result, latency_ms)
            except Exception as exc:  # noqa: BLE001 - observability is not the turn
                log_warning(
                    "post_tool_hook_failed",
                    run_id=ctx.run_id,
                    tool=ctx.name,
                    hook=getattr(hook, "__qualname__", repr(hook)),
                    error=str(exc),
                )


def default_hooks(
    policy: PermissionPolicy, approver: Approver | None = None
) -> HookChain:
    """The chain every agent scope ships with: permission, then metrics."""
    return HookChain(
        pre=[permission_hook(policy, approver)],
        post=[metrics_hook],
    )
