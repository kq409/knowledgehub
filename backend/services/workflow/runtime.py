"""Primitives for workflows the harness controls, not the model.

A comparison is not an agent decision -- it is always: retrieve evidence for
each paper, extract cells for each paper, synthesise once. When the control
flow is known up front, spending model calls to rediscover it is waste, and
the failure modes get worse: there is no way to resume, and one bad response
in the middle loses everything before it.

Three primitives cover it:

    step()      one unit of work, journalled and validated
    parallel()  independent steps at once, one failure does not cancel the rest
    pipeline()  ordered steps, each free to read what came before

`phase()` groups them for progress reporting. Nothing here decides *what* to
do; the caller writes that as ordinary Python.

Validation is not optional. s16's point is that a structured response from a
model is exactly as untrustworthy as one from a tool: a step declares what
shape it expects, a mismatch is retried once, and a second mismatch fails that
step alone.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from services.agent.telemetry import log_event, log_warning
from services.workflow.journal import WorkflowJournal, step_key

DEFAULT_PARALLELISM = 4
DEFAULT_VALIDATION_RETRIES = 1


class WorkflowError(Exception):
    """A step failed in a way the workflow cannot work around."""


class StepValidationError(Exception):
    """A step produced something the declared shape rejects."""


# Returns the step's result as a JSON-serialisable dict, so it can be
# journalled and replayed without the caller's Python objects.
StepRunner = Callable[[], Awaitable[dict]]
Validator = Callable[[dict], None]
ProgressSink = Callable[["Progress"], None]


@dataclass(frozen=True)
class Progress:
    """One thing worth telling the researcher while they wait."""

    phase: str
    label: str
    done: int
    total: int
    cached: bool = False

    def as_event(self) -> dict:
        return {
            "phase": self.phase,
            "label": self.label,
            "done": self.done,
            "total": self.total,
            "cached": self.cached,
        }


@dataclass(frozen=True)
class Step:
    """A unit of work, plus everything needed to identify and check it."""

    kind: str
    label: str
    run: StepRunner
    # Whatever makes this step's inputs unique: the prompt, the dimensions, the
    # paper id. Never a position or a counter.
    inputs: tuple[Any, ...] = ()
    validate: Validator | None = None
    retries: int = DEFAULT_VALIDATION_RETRIES

    @property
    def key(self) -> str:
        return step_key(self.kind, self.label, *self.inputs)


@dataclass
class StepOutcome:
    step: Step
    result: dict | None = None
    error: str | None = None
    cached: bool = False

    @property
    def ok(self) -> bool:
        return self.error is None

    def require(self) -> dict:
        if self.error is not None or self.result is None:
            raise WorkflowError(f"{self.step.label}: {self.error or 'no result'}")
        return self.result


def require_keys(*names: str) -> Validator:
    """A validator for the common case: a dict that must carry these keys."""

    def validate(payload: dict) -> None:
        if not isinstance(payload, dict):
            raise StepValidationError(
                f"expected an object, got {type(payload).__name__}"
            )
        missing = [name for name in names if name not in payload]
        if missing:
            raise StepValidationError(f"missing key(s): {', '.join(missing)}")

    return validate


@dataclass
class WorkflowRun:
    """One resumable execution. Re-running with the same `run_id` resumes it."""

    journal: WorkflowJournal
    on_progress: ProgressSink | None = None
    parallelism: int = DEFAULT_PARALLELISM
    _phase: str = field(default="", init=False)

    async def start(self) -> None:
        if not self.journal.loaded:
            await self.journal.load()
        log_event(
            "workflow_started",
            run_id=self.journal.run_id,
            journalled_steps=len(self.journal),
        )

    def phase(self, name: str) -> None:
        """Name what is happening now. Only affects progress reporting."""
        self._phase = name

    def report(self, label: str, done: int, total: int, *, cached: bool) -> None:
        """Tell the caller where the workflow is. Never raises."""
        if self.on_progress is None:
            return
        try:
            self.on_progress(
                Progress(
                    phase=self._phase,
                    label=label,
                    done=done,
                    total=total,
                    cached=cached,
                )
            )
        except Exception as exc:  # noqa: BLE001 - progress is not the work
            log_warning(
                "workflow_progress_failed",
                run_id=self.journal.run_id,
                error=str(exc),
            )

    async def step(self, step: Step, *, done: int = 1, total: int = 1) -> StepOutcome:
        """Run one step, or replay it from the journal, and report it."""
        outcome = await self.execute(step)
        self.report(step.label, done, total, cached=outcome.cached)
        return outcome

    async def execute(self, step: Step) -> StepOutcome:
        """Run one step without reporting.

        `parallel` reports for itself: it only knows a step's position once the
        step has finished, so it cannot let the step report on its own behalf.
        """
        key = step.key
        cached = self.journal.cached(key)
        if cached is not None:
            self.journal.note_replay()
            return StepOutcome(step=step, result=cached.result, cached=True)

        last_error: str | None = None
        for attempt in range(max(1, step.retries + 1)):
            try:
                result = await step.run()
                if step.validate is not None:
                    step.validate(result)
            except StepValidationError as exc:
                last_error = f"invalid output: {exc}"
                log_warning(
                    "workflow_step_invalid",
                    run_id=self.journal.run_id,
                    kind=step.kind,
                    label=step.label,
                    attempt=attempt + 1,
                    error=str(exc),
                )
                continue
            except Exception as exc:  # noqa: BLE001 - one step, not the workflow
                last_error = str(exc)
                log_warning(
                    "workflow_step_failed",
                    run_id=self.journal.run_id,
                    kind=step.kind,
                    label=step.label,
                    attempt=attempt + 1,
                    error=str(exc),
                )
                break

            await self.journal.record(
                key, kind=step.kind, label=step.label, result=result
            )
            return StepOutcome(step=step, result=result)

        await self.journal.record_failure(
            key, kind=step.kind, label=step.label, error=last_error or "unknown"
        )
        return StepOutcome(step=step, error=last_error or "unknown")

    async def parallel(self, steps: Sequence[Step]) -> list[StepOutcome]:
        """Run independent steps at once, in the order they were given back.

        One step failing must not cancel its siblings: their results are worth
        journalling even if the workflow ends up incomplete, because the retry
        will then only need to redo the one that failed.
        """
        if not steps:
            return []
        total = len(steps)
        gate = asyncio.Semaphore(max(1, self.parallelism))
        finished = 0
        lock = asyncio.Lock()

        async def guarded(step: Step) -> StepOutcome:
            nonlocal finished
            async with gate:
                outcome = await self.execute(step)
            # The position is assigned after the step returns, so it counts
            # completions rather than pretending to know the order.
            async with lock:
                finished += 1
                position = finished
            self.report(step.label, position, total, cached=outcome.cached)
            return outcome

        return list(await asyncio.gather(*(guarded(step) for step in steps)))

    async def pipeline(self, steps: Sequence[Step]) -> list[StepOutcome]:
        """Ordered steps. Stops at the first failure, keeping what succeeded."""
        outcomes: list[StepOutcome] = []
        total = len(steps)
        for index, step in enumerate(steps, start=1):
            outcome = await self.step(step, done=index, total=total)
            outcomes.append(outcome)
            if not outcome.ok:
                break
        return outcomes

    def summary(self) -> dict:
        return {
            "run_id": self.journal.run_id,
            "replayed": self.journal.replayed,
            "executed": self.journal.executed,
        }

    async def finish(self) -> None:
        log_event("workflow_finished", **self.summary())
