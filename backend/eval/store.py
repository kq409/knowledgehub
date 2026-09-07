"""Persist eval runs as JSON on disk so the API and UI can read transcripts."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

RUNS_DIR = Path(__file__).resolve().parent / "runs"


def runs_dir() -> Path:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    return RUNS_DIR


def new_run_id() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def write_run(
    run_id: str, manifest: dict[str, Any], trials: list[dict[str, Any]]
) -> Path:
    folder = runs_dir() / run_id
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    for trial in trials:
        task_id = trial.get("task_id", "task")
        index = trial.get("trial_index", 0)
        name = f"{task_id}__{index}.json"
        (folder / name).write_text(
            json.dumps(trial, indent=2, default=str) + "\n",
            encoding="utf-8",
        )
    return folder


def list_runs() -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    if not RUNS_DIR.exists():
        return items
    for folder in sorted(RUNS_DIR.iterdir(), reverse=True):
        manifest_path = folder / "manifest.json"
        if not folder.is_dir() or not manifest_path.exists():
            continue
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        payload["id"] = folder.name
        items.append(payload)
    return items


def load_run(run_id: str) -> dict[str, Any] | None:
    folder = runs_dir() / run_id
    manifest_path = folder / "manifest.json"
    if not manifest_path.exists():
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    trials: list[dict[str, Any]] = []
    for path in sorted(folder.glob("*__*.json")):
        trials.append(json.loads(path.read_text(encoding="utf-8")))
    manifest["id"] = run_id
    manifest["trials"] = trials
    return manifest
