"""Retrieval and trial-consistency metrics."""

from __future__ import annotations


def _titles_at_k(retrieved: list[str], k: int) -> list[str]:
    return retrieved[: max(0, k)]


def precision_at_k(retrieved: list[str], relevant: list[str], k: int) -> float:
    if k <= 0:
        return 0.0
    top = _titles_at_k(retrieved, k)
    if not top:
        return 0.0
    wanted = set(relevant)
    hits = sum(1 for title in top if title in wanted)
    return hits / len(top)


def recall_at_k(retrieved: list[str], relevant: list[str], k: int) -> float:
    if not relevant:
        return 1.0
    top = set(_titles_at_k(retrieved, k))
    hits = sum(1 for title in relevant if title in top)
    return hits / len(relevant)


def mean_reciprocal_rank(retrieved: list[str], relevant: list[str]) -> float:
    wanted = set(relevant)
    if not wanted:
        return 1.0
    for rank, title in enumerate(retrieved, start=1):
        if title in wanted:
            return 1.0 / rank
    return 0.0


def pass_at_k(trial_passed: list[bool], k: int) -> float:
    """1.0 if any of the first k trials passed, else 0.0."""
    if k <= 0 or not trial_passed:
        return 0.0
    return 1.0 if any(trial_passed[:k]) else 0.0


def pass_hat_k(trial_passed: list[bool], k: int) -> float:
    """1.0 if every one of the first k trials passed, else 0.0."""
    if k <= 0 or len(trial_passed) < k:
        return 0.0
    return 1.0 if all(trial_passed[:k]) else 0.0


def suite_pass_at_k(task_trials: list[list[bool]], k: int) -> float:
    if not task_trials:
        return 0.0
    return sum(pass_at_k(trials, k) for trials in task_trials) / len(task_trials)


def suite_pass_hat_k(task_trials: list[list[bool]], k: int) -> float:
    if not task_trials:
        return 0.0
    return sum(pass_hat_k(trials, k) for trials in task_trials) / len(task_trials)
