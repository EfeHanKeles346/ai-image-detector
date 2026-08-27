from __future__ import annotations

from experiments.e38_final import final_gate


def test_final_gate_requires_every_balanced_metric() -> None:
    metrics = {
        "roc_auc": 0.95,
        "tpr_at_fpr": {"tpr": 0.85},
        "eer": 0.1,
        "balanced_accuracy": 0.9,
        "coverage": 1.0,
    }
    rates = {
        "real_macro_fp": 0.05,
        "real_worst_device_fp": 0.15,
        "ai_macro_recall": 0.85,
        "ai_worst_family_recall": 0.70,
    }
    assert final_gate(metrics, rates)["passed"] is True
    rates["ai_macro_recall"] = 0.79
    assert final_gate(metrics, rates)["passed"] is False


def test_final_gate_rejects_partial_coverage() -> None:
    metrics = {
        "roc_auc": 0.95,
        "tpr_at_fpr": {"tpr": 0.85},
        "eer": 0.1,
        "balanced_accuracy": 0.9,
        "coverage": 0.99,
    }
    rates = {
        "real_macro_fp": 0.05,
        "real_worst_device_fp": 0.15,
        "ai_macro_recall": 0.85,
        "ai_worst_family_recall": 0.70,
    }
    assert final_gate(metrics, rates)["passed"] is False
