from __future__ import annotations

import numpy as np

from experiments import e32_r0_train as r0


def test_threshold_respects_each_real_source_budget() -> None:
    labels = np.array([0, 0, 0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.8])
    sources = np.array(["r1", "r1", "r2", "r2", "ai", "ai"])
    threshold = r0.threshold_at_fp_budget(labels, scores, sources, macro_budget=0.0, worst_budget=0.0)
    assert threshold > 0.4


def test_evaluate_reports_source_macro_and_current_family() -> None:
    real_sources = ["vision-base-native", "forchheim-fodb", "csafe-mcsidb-s21"]
    labels = []
    scores = []
    sources = []
    for source in real_sources:
        labels.extend([0, 0])
        scores.extend([0.1, 0.2])
        sources.extend([source, source])
    for source in r0.CURRENT_AI_SOURCES:
        labels.extend([1, 1])
        scores.extend([0.8, 0.9])
        sources.extend([source, source])
    metrics = r0.evaluate(np.array(labels), np.array(scores), np.array(sources), 0.5)
    assert metrics["auc"] == 1.0
    assert metrics["macro_real_fp"] == 0.0
    assert metrics["current_ai_macro_recall"] == 1.0
    assert r0.screen_gate(metrics)["passed"] is True


def test_screen_gate_rejects_weak_current_family() -> None:
    metrics = {
        "auc": 0.9,
        "current_ai_macro_recall": 0.8,
        "current_ai_worst_source_recall": 0.2,
        "macro_real_fp": 0.05,
        "worst_real_fp": 0.10,
    }
    result = r0.screen_gate(metrics)
    assert result["passed"] is False
    assert result["checks"]["current_ai_worst_source_recall_gte_0.40"] is False
