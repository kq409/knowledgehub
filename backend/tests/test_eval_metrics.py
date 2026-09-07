from eval.metrics import (
    mean_reciprocal_rank,
    pass_at_k,
    pass_hat_k,
    precision_at_k,
    recall_at_k,
    suite_pass_at_k,
    suite_pass_hat_k,
)


def test_precision_recall_mrr():
    retrieved = ["ELLA", "CNN", "VAE"]
    relevant = ["ELLA"]
    assert precision_at_k(retrieved, relevant, 1) == 1.0
    assert precision_at_k(retrieved, relevant, 3) == 1.0 / 3
    assert recall_at_k(retrieved, relevant, 3) == 1.0
    assert mean_reciprocal_rank(retrieved, relevant) == 1.0
    assert mean_reciprocal_rank(["CNN", "ELLA"], relevant) == 0.5
    assert mean_reciprocal_rank(["CNN"], relevant) == 0.0


def test_recall_empty_relevant_is_one():
    assert recall_at_k(["ELLA"], [], 8) == 1.0
    assert mean_reciprocal_rank(["ELLA"], []) == 1.0


def test_pass_at_k_and_pass_hat_k():
    trials = [True, False, True]
    assert pass_at_k(trials, 1) == 1.0
    assert pass_at_k([False, False, True], 2) == 0.0
    assert pass_at_k([False, False, True], 3) == 1.0
    assert pass_hat_k(trials, 1) == 1.0
    assert pass_hat_k(trials, 2) == 0.0
    assert pass_hat_k([True, True, True], 3) == 1.0


def test_suite_pass_aggregates_per_task():
    tasks = [[True, False], [False, False]]
    assert suite_pass_at_k(tasks, 1) == 0.5
    assert suite_pass_hat_k([[True, True], [True, True]], 2) == 1.0
