"""Pause a write until the researcher says it may run.

The chat channel is a single SSE stream, so a tool that needs a human cannot
return until that human has answered. This broker is the matching half of the
stream: the PreToolUse hook emits an `approval` event, then waits on a Future
this process owns. `POST /api/chat/approve` resolves that Future.

The wait is process-local on purpose. Two replicas do not share these Futures,
so a request that started on this process has to be approved on this process.
That matches how this app already runs (one uvicorn, in-memory ingest
notifications). A multi-replica deployment would move the wait onto a shared
store; until then the README states the boundary rather than pretending.
"""

from __future__ import annotations

import asyncio
import os
import uuid

from schemas import ChatEventType
from services.agent.events import AgentEvent
from services.agent.hooks import ToolHookContext
from services.agent.permissions import PermissionVerdict
from services.agent.telemetry import log_event, log_warning

DEFAULT_TIMEOUT_SECONDS = 120.0


def approval_timeout_seconds() -> float:
    raw = os.getenv("ASK_APPROVAL_TIMEOUT", str(DEFAULT_TIMEOUT_SECONDS))
    try:
        return max(1.0, float(raw))
    except ValueError:
        return DEFAULT_TIMEOUT_SECONDS


class ApprovalBroker:
    """One in-process table of outstanding approval requests.

    Callable so it can be passed straight in as the hook chain's `Approver`.
    """

    def __init__(self, timeout_seconds: float | None = None) -> None:
        self._pending: dict[str, asyncio.Future[bool]] = {}
        self.timeout_seconds = (
            timeout_seconds
            if timeout_seconds is not None
            else approval_timeout_seconds()
        )

    def __len__(self) -> int:
        return len(self._pending)

    def resolve(self, request_id: str, approved: bool) -> bool:
        """Record the researcher's decision. False when the request is gone."""
        future = self._pending.get(request_id)
        if future is None or future.done():
            return False
        future.set_result(bool(approved))
        return True

    async def __call__(self, ctx: ToolHookContext, verdict: PermissionVerdict) -> bool:
        return await self.request(ctx, verdict)

    async def request(self, ctx: ToolHookContext, verdict: PermissionVerdict) -> bool:
        """Emit an approval event, then wait for `resolve` or a timeout."""
        request_id = str(uuid.uuid4())
        loop = asyncio.get_running_loop()
        future: asyncio.Future[bool] = loop.create_future()
        self._pending[request_id] = future

        def emit(status: str) -> None:
            ctx.sink.emit(
                AgentEvent(
                    type=ChatEventType.approval,
                    data={
                        "request_id": request_id,
                        "tool": ctx.name,
                        "arguments": ctx.call.arguments,
                        "reason": verdict.reason,
                        "status": status,
                        "agent_id": ctx.agent_id,
                    },
                )
            )

        emit("pending")
        log_event(
            "approval_requested",
            run_id=ctx.run_id,
            tool=ctx.name,
            request_id=request_id,
        )
        try:
            approved = await asyncio.wait_for(
                asyncio.shield(future), self.timeout_seconds
            )
        except TimeoutError:
            log_warning(
                "approval_timed_out",
                run_id=ctx.run_id,
                tool=ctx.name,
                request_id=request_id,
            )
            emit("timeout")
            return False
        finally:
            self._pending.pop(request_id, None)

        emit("approved" if approved else "denied")
        log_event(
            "approval_resolved",
            run_id=ctx.run_id,
            tool=ctx.name,
            request_id=request_id,
            approved=approved,
        )
        return approved
