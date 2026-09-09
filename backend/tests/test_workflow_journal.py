"""A workflow that fails halfway must not make you pay for the first half twice."""

import asyncio
import os

import pytest
from dotenv import load_dotenv
from sqlalchemy import text

import db
from services.workflow import (
    Step,
    StepValidationError,
    WorkflowJournal,
    WorkflowRun,
    require_keys,
    step_key,
)
from services.workflow.runtime import Progress

load_dotenv()


@pytest.fixture
async def session():
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        pytest.skip("DATABASE_URL is not set")
    db.init_db(database_url)
    try:
        if db.SessionLocal is None:
            pytest.skip("Database session factory was not created")
        async with db.SessionLocal() as probe:
            await probe.execute(text("SELECT 1 FROM workflow_steps LIMIT 1"))
    except Exception:
        await db.close_db()
        pytest.skip("PostgreSQL or the workflow_steps table is not available")

    async with db.SessionLocal() as session:
        yield session
        await session.execute(text("DELETE FROM workflow_steps"))
        await session.commit()
    await db.close_db()


def counting_step(label: str, calls: list[str], *, fail: bool = False) -> Step:
    async def run() -> dict:
        calls.append(label)
        if fail:
            raise RuntimeError(f"{label} exploded")
        return {"value": label}

    return Step(
        kind="unit",
        label=label,
        run=run,
        inputs=(label,),
        validate=require_keys("value"),
        retries=0,
    )


def test_the_key_describes_what_a_step_means_not_where_it_sits():
    """Parallel steps finish in an unpredictable order, so position is unusable."""
    first = step_key("compare_map", "ELLA", "paper-1", ["method"])
    same = step_key("compare_map", "ELLA", "paper-1", ["method"])
    different_paper = step_key("compare_map", "ELLA", "paper-2", ["method"])
    different_dims = step_key("compare_map", "ELLA", "paper-1", ["dataset"])

    assert first == same
    assert first != different_paper
    assert first != different_dims
    assert len(first) == 64


def test_the_key_ignores_dict_ordering():
    from services.workflow import fingerprint

    assert fingerprint({"a": 1, "b": 2}) == fingerprint({"b": 2, "a": 1})


async def test_a_resumed_run_only_recomputes_what_failed(session):
    """The point of the journal: three good calls survive the fourth failing."""
    calls: list[str] = []
    run_id = "test-resume"

    first = WorkflowRun(journal=WorkflowJournal(session, run_id))
    await first.start()
    outcomes = await first.pipeline(
        [
            counting_step("a", calls),
            counting_step("b", calls),
            counting_step("c", calls),
            counting_step("d", calls, fail=True),
        ]
    )

    assert calls == ["a", "b", "c", "d"]
    assert [outcome.ok for outcome in outcomes] == [True, True, True, False]

    calls.clear()
    second = WorkflowRun(journal=WorkflowJournal(session, run_id))
    await second.start()
    retried = await second.pipeline(
        [
            counting_step("a", calls),
            counting_step("b", calls),
            counting_step("c", calls),
            counting_step("d", calls),
        ]
    )

    assert calls == ["d"], "a, b and c should have been replayed from the journal"
    assert all(outcome.ok for outcome in retried)
    assert second.summary()["replayed"] == 3
    assert second.summary()["executed"] == 1


async def test_a_different_run_id_shares_nothing(session):
    calls: list[str] = []
    for run_id in ("run-one", "run-two"):
        workflow = WorkflowRun(journal=WorkflowJournal(session, run_id))
        await workflow.start()
        await workflow.step(counting_step("a", calls))

    assert calls == ["a", "a"]


async def test_a_changed_input_invalidates_the_cached_step(session):
    """A new prompt means the journalled answer answered a different question."""
    ran: list[str] = []

    def step_with(marker: str) -> Step:
        async def run() -> dict:
            ran.append(marker)
            return {"value": marker}

        return Step(kind="unit", label="cells", run=run, inputs=(marker,))

    workflow = WorkflowRun(journal=WorkflowJournal(session, "test-inputs"))
    await workflow.start()
    await workflow.step(step_with("v1"))
    await workflow.step(step_with("v1"))
    await workflow.step(step_with("v2"))

    assert ran == ["v1", "v2"]


async def test_parallel_steps_run_at_once_and_come_back_in_order(session):
    gate = asyncio.Barrier(3)

    def gated(label: str) -> Step:
        async def run() -> dict:
            await asyncio.wait_for(gate.wait(), timeout=5)
            return {"value": label}

        return Step(kind="unit", label=label, run=run, inputs=(label,))

    workflow = WorkflowRun(journal=WorkflowJournal(session, "test-parallel"))
    await workflow.start()

    outcomes = await workflow.parallel([gated("x"), gated("y"), gated("z")])

    assert [outcome.result["value"] for outcome in outcomes] == ["x", "y", "z"]


async def test_one_failing_step_does_not_cancel_its_siblings(session):
    """Their results are worth journalling; the retry then costs only the failure."""
    calls: list[str] = []
    workflow = WorkflowRun(journal=WorkflowJournal(session, "test-sibling"))
    await workflow.start()

    outcomes = await workflow.parallel(
        [
            counting_step("ok-1", calls),
            counting_step("boom", calls, fail=True),
            counting_step("ok-2", calls),
        ]
    )

    assert [outcome.ok for outcome in outcomes] == [True, False, True]
    assert sorted(calls) == ["boom", "ok-1", "ok-2"]

    calls.clear()
    retry = WorkflowRun(journal=WorkflowJournal(session, "test-sibling"))
    await retry.start()
    await retry.parallel(
        [
            counting_step("ok-1", calls),
            counting_step("boom", calls),
            counting_step("ok-2", calls),
        ]
    )
    assert calls == ["boom"]


async def test_an_invalid_shape_is_retried_once_then_fails_that_step(session):
    """A structured response from a model is as untrustworthy as one from a tool."""
    attempts = {"count": 0}

    async def run() -> dict:
        attempts["count"] += 1
        return {"wrong_key": "nope"}

    step = Step(
        kind="unit",
        label="shaped",
        run=run,
        inputs=("shaped",),
        validate=require_keys("value"),
        retries=1,
    )
    workflow = WorkflowRun(journal=WorkflowJournal(session, "test-shape"))
    await workflow.start()

    outcome = await workflow.step(step)

    assert attempts["count"] == 2
    assert not outcome.ok
    assert "missing key(s): value" in outcome.error


async def test_a_failed_step_is_not_replayed_as_a_success(session):
    workflow = WorkflowRun(journal=WorkflowJournal(session, "test-failed-entry"))
    await workflow.start()
    calls: list[str] = []
    await workflow.step(counting_step("a", calls, fail=True))

    reopened = WorkflowJournal(session, "test-failed-entry")
    await reopened.load()

    assert len(reopened) == 1
    assert reopened.cached(counting_step("a", calls).key) is None


async def test_progress_reports_completions_and_cache_hits(session):
    seen: list[Progress] = []
    calls: list[str] = []
    run_id = "test-progress"

    first = WorkflowRun(
        journal=WorkflowJournal(session, run_id), on_progress=seen.append
    )
    await first.start()
    first.phase("per-paper")
    await first.parallel([counting_step("a", calls), counting_step("b", calls)])

    assert [item.phase for item in seen] == ["per-paper", "per-paper"]
    assert sorted(item.done for item in seen) == [1, 2]
    assert all(item.total == 2 for item in seen)
    assert not any(item.cached for item in seen)

    seen.clear()
    second = WorkflowRun(
        journal=WorkflowJournal(session, run_id), on_progress=seen.append
    )
    await second.start()
    second.phase("per-paper")
    await second.parallel([counting_step("a", calls), counting_step("b", calls)])

    assert all(item.cached for item in seen)


async def test_a_broken_progress_sink_does_not_break_the_workflow(session):
    def explode(progress: Progress) -> None:
        raise RuntimeError("the UI is gone")

    workflow = WorkflowRun(
        journal=WorkflowJournal(session, "test-bad-sink"), on_progress=explode
    )
    await workflow.start()
    calls: list[str] = []

    outcome = await workflow.step(counting_step("a", calls))

    assert outcome.ok


async def test_require_keys_rejects_a_non_object():
    with pytest.raises(StepValidationError, match="expected an object"):
        require_keys("value")(["not", "a", "dict"])
