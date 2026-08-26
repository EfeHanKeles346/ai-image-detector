from __future__ import annotations

import numpy as np

from experiments.e31_representation_ladder import (
    acceptance,
    evaluate_scores,
    threshold_at_source_fp_budget,
)


def test_threshold_honors_macro_and_worst_source_budgets() -> None:
    scores = np.array([0.1, 0.2, 0.3, 0.4] * 2)
    sources = np.array(["a"] * 4 + ["b"] * 4)
    threshold, evidence = threshold_at_source_fp_budget(
        scores, sources, macro_budget=0.25, worst_budget=0.25
    )
    assert threshold == 0.4
    assert evidence["macro_real_fp"] == 0.25
    assert evidence["worst_real_fp"] == 0.25


def test_threshold_can_choose_zero_false_positives() -> None:
    threshold, evidence = threshold_at_source_fp_budget(
        np.array([0.5, 0.5]),
        np.array(["a", "b"]),
        macro_budget=0.0,
        worst_budget=0.0,
    )
    assert threshold > 0.5
    assert evidence["macro_real_fp"] == 0.0


def test_source_aware_metrics_and_acceptance() -> None:
    labels = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
    scores = np.array([0.1, 0.2, 0.1, 0.2, 0.9, 0.8, 0.9, 0.8, 0.9, 0.8])
    sources = np.array(
        ["real-a", "real-a", "real-b", "real-b", "flux-1-dev", "flux-1-dev",
         "nano-banana", "nano-banana", "nano-banana-pro", "nano-banana-pro"]
    )
    generators = np.array(["", "", "", "", "flux", "flux", "nano", "nano", "pro", "pro"])
    metrics = evaluate_scores(labels, scores, sources, generators, threshold=0.5)
    assert metrics["macro_real_fp"] == 0.0
    assert metrics["current_ai_macro_recall"] == 1.0
    assert acceptance(metrics)["passed"] is True
