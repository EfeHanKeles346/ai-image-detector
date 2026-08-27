from __future__ import annotations

import numpy as np
import pytest

from experiments.e42_train import (
    development_gate,
    select_threshold,
    source_balanced_weights,
    source_folds,
)


def test_source_folds_never_split_a_source_and_populate_each_class_fold() -> None:
    roles, labels, sources = [], [], []
    for label in (0, 1):
        for source_index in range(7):
            for _ in range(source_index + 1):
                roles.append("development")
                labels.append(label)
                sources.append(f"{label}-{source_index}")
    assignments = source_folds(np.asarray(roles), np.asarray(labels), np.asarray(sources))
    assert len(assignments) == 14
    for label in (0, 1):
        assert {assignments[f"{label}-{index}"] for index in range(7)} == set(range(5))


def test_source_balanced_weights_equalize_class_and_source_mass() -> None:
    labels = np.asarray([0, 0, 0, 1, 1])
    sources = np.asarray(["r1", "r1", "r2", "a1", "a1"])
    weights = source_balanced_weights(labels, sources)
    assert weights[labels == 0].sum() == pytest.approx(weights[labels == 1].sum())
    assert weights[sources == "r1"].sum() == pytest.approx(weights[sources == "r2"].sum())


def test_threshold_uses_macro_and_worst_real_source_budgets() -> None:
    rows = [
        {"label": 0, "source": "r1", "score": score} for score in (0.1, 0.2, 0.3, 0.9, 0.95)
    ] + [
        {"label": 0, "source": "r2", "score": score} for score in (0.1, 0.2, 0.3, 0.4, 0.5)
    ] + [
        {"label": 1, "source": "a", "score": score} for score in (0.7, 0.8, 0.9)
    ]
    rates = select_threshold(rows)
    assert rates["real_macro_fp"] <= 0.10
    assert rates["real_worst_source_fp"] <= 0.20


def test_gate_cannot_hide_a_weak_ai_source_behind_robust_auc() -> None:
    clean_metrics = {
        "roc_auc": 0.95,
        "tpr_at_fpr": {"tpr": 0.90},
        "eer": 0.10,
        "balanced_accuracy": 0.90,
        "coverage": 1.0,
    }
    rates = {
        "real_macro_fp": 0.05,
        "real_worst_source_fp": 0.10,
        "ai_macro_recall": 0.90,
        "ai_worst_source_recall": 0.50,
    }
    robust = {"roc_auc": 0.95, "balanced_accuracy": 0.90, "coverage": 1.0}
    assert development_gate(clean_metrics, rates, robust)["passed"] is False
