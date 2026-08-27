from __future__ import annotations

import pytest

from experiments.e42_bfree_score import parent_weighted_summary, stress_gate


def test_parent_weighting_prevents_repost_volume_from_dominating() -> None:
    rows = [
        *[
            {"parent_id": "real-many", "label": 0, "score": 0.1, "status": "ok"}
            for _ in range(100)
        ],
        {"parent_id": "real-one", "label": 0, "score": 0.9, "status": "ok"},
        {"parent_id": "ai-one", "label": 1, "score": 0.9, "status": "ok"},
    ]
    summary = parent_weighted_summary(rows, 0.5)
    assert summary["real_parent_weighted_fp"] == pytest.approx(0.5)
    assert summary["ai_parent_weighted_recall"] == pytest.approx(1.0)
    assert summary["parent_weighted_balanced_accuracy"] == pytest.approx(0.75)


def test_stress_gate_requires_both_classes_not_only_balanced_average() -> None:
    result = stress_gate({
        "parent_weighted_balanced_accuracy": 0.80,
        "real_parent_weighted_recall": 0.61,
        "ai_parent_weighted_recall": 0.99,
    })
    assert result["passed"] is False
    assert result["checks"]["real_parent_weighted_recall_gte_0_75"] is False
