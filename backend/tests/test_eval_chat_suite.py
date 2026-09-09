import json
from pathlib import Path

from eval.graders import outcome, score_task, tool_used
from eval.harness import load_suite
from eval.metrics import pass_at_k, pass_hat_k


def test_chat_suite_has_enough_tasks_and_trials():
    suite = load_suite("chat")
    assert suite["trials"] == 3
    assert len(suite["tasks"]) >= 8
    ids = {task["id"] for task in suite["tasks"]}
    assert "grounded-ella-transfer" in ids
    assert "abstain-sota" in ids
    assert "greeting-no-inventory" in ids
    assert "multi-part-compare-then-gap" in ids
    assert "skill-cite-from-library" in ids


def test_chat_graders_on_fake_transcript():
    suite = load_suite("chat")
    task = next(
        item for item in suite["tasks"] if item["id"] == "grounded-ella-transfer"
    )
    grades = []
    for spec in task["graders"]:
        if spec["type"] == "tool_used":
            grades.append(tool_used(["search_library"], spec["any_of"]))
        elif spec["type"] == "outcome":
            grades.append(
                outcome(
                    "ZXQELLA7 transfers sparse models across tasks [1].",
                    contains_any=spec["contains_any"],
                )
            )
    _, passed = score_task([grade for grade in grades if grade.name != "faithfulness"])
    assert passed is True


def test_pass_metrics_for_three_trials():
    assert pass_at_k([False, True, False], 3) == 1.0
    assert pass_hat_k([True, True, True], 3) == 1.0
    assert pass_hat_k([True, False, True], 3) == 0.0


def test_ask_suite_on_disk():
    path = Path(__file__).resolve().parents[1] / "eval" / "suites" / "ask.json"
    payload = json.loads(path.read_text())
    assert len(payload["tasks"]) >= 20
