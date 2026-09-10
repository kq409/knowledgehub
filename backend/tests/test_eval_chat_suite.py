import json
from pathlib import Path

import pytest

from eval.graders import must_include_facts, score_task, tool_used
from eval.harness import (
    _summarize,
    expand_suite_names,
    grade_reference,
    load_suite,
    regression_gate_failed,
)
from eval.metrics import pass_at_k, pass_hat_k


def test_suite_aliases_split_ask_and_chat():
    assert expand_suite_names("ask") == ["ask-regression", "ask-quality"]
    assert expand_suite_names("chat") == ["chat-regression", "chat-quality"]
    assert "ask-regression" in expand_suite_names("all")
    assert "acl-regression" in expand_suite_names("all")
    assert "discipline-regression" in expand_suite_names("all")


def test_chat_suites_have_references_and_split_kinds():
    regression = load_suite("chat-regression")
    quality = load_suite("chat-quality")
    assert regression["trials"] == 1
    assert quality["trials"] == 3
    assert len(regression["tasks"]) >= 6
    assert len(quality["tasks"]) >= 6
    ids = {task["id"] for task in regression["tasks"]} | {
        task["id"] for task in quality["tasks"]
    }
    assert "grounded-ella-transfer" in ids
    assert "abstain-sota" in ids
    assert "greeting-no-inventory" in ids
    assert "should-not-web-search-ella" in ids
    assert "should-web-search-sota" in ids
    assert "refuse-missing-paper" in ids
    for task in regression["tasks"]:
        assert task["kind"] == "regression"
        assert task.get("reference"), task["id"]
        assert all(
            spec.get("type") != "tool_used" or spec.get("required") is False
            for spec in task["graders"]
            if spec.get("type") == "tool_used"
        )


def test_chat_quality_uses_facts_not_required_tool_paths():
    suite = load_suite("chat-quality")
    task = next(
        item for item in suite["tasks"] if item["id"] == "grounded-ella-transfer"
    )
    types = [spec["type"] for spec in task["graders"]]
    assert "must_include_facts" in types
    assert "contains_any" not in json.dumps(task["graders"])
    grades = []
    for spec in task["graders"]:
        if spec["type"] == "tool_used":
            grades.append(
                tool_used(
                    ["search_library"],
                    spec.get("any_of"),
                    required=bool(spec.get("required", False)),
                )
            )
        elif spec["type"] == "must_include_facts":
            grades.append(
                must_include_facts(
                    "ZXQELLA7 transfers sparse models across tasks [1].",
                    spec["facts"],
                )
            )
    _, passed = score_task([grade for grade in grades if grade.name != "faithfulness"])
    assert passed is True


@pytest.mark.asyncio
async def test_every_split_suite_reference_passes_graders():
    for name in (
        "ask-regression",
        "ask-quality",
        "chat-regression",
        "chat-quality",
        "acl-regression",
    ):
        suite = load_suite(name)
        for task in suite["tasks"]:
            trial = await grade_reference(task)
            assert trial["passed"], f"{name}/{task['id']}: {trial['grades']}"


def test_pass_metrics_for_three_trials():
    assert pass_at_k([False, True, False], 3) == 1.0
    assert pass_hat_k([True, True, True], 3) == 1.0
    assert pass_hat_k([True, False, True], 3) == 0.0


def test_ask_regression_suite_on_disk():
    path = (
        Path(__file__).resolve().parents[1] / "eval" / "suites" / "ask-regression.json"
    )
    payload = json.loads(path.read_text())
    assert len(payload["tasks"]) >= 10
    assert all(task["kind"] == "regression" for task in payload["tasks"])


def test_summarize_splits_regression_and_quality():
    trials = [
        {
            "task_id": "a",
            "kind": "regression",
            "passed": True,
        },
        {
            "task_id": "b",
            "kind": "capability",
            "passed": True,
        },
        {
            "task_id": "b",
            "kind": "capability",
            "passed": False,
        },
    ]
    summary = _summarize("chat-quality", trials, 2)
    assert summary["regression_failed"] is False
    assert summary["gate"] == "pass"
    assert summary["quality_pass_at_1"] == 1.0
    assert summary["quality_pass_hat_k"] == 0.0
    mixed = _summarize(
        "ask-regression",
        [{"task_id": "x", "kind": "regression", "passed": False}],
        1,
    )
    assert mixed["regression_failed"] is True
    assert mixed["gate"] == "fail"


def test_regression_gate_failed_on_summary():
    assert regression_gate_failed([{"regression_failed": True, "gate": "fail"}])
    assert not regression_gate_failed([{"regression_failed": False, "gate": "pass"}])
