from __future__ import annotations

from experiments.e39_final import final_gate


def _passing() -> tuple[dict, dict]:
    return (
        {"roc_auc": 0.95, "tpr_at_fpr": {"tpr": 0.85}, "eer": 0.10, "balanced_accuracy": 0.90, "coverage": 1.0},
        {
            "real_macro_fp": 0.05,
            "real_worst_device_fp": 0.15,
            "ai_macro_recall": 0.85,
            "ai_worst_family_recall": 0.70,
        },
    )


def test_e39_final_gate_requires_every_metric() -> None:
    metrics, rates = _passing()
    assert final_gate(metrics, rates)["passed"] is True
    rates["real_worst_device_fp"] = 0.21
    assert final_gate(metrics, rates)["passed"] is False


def test_e39_final_gate_requires_complete_coverage() -> None:
    metrics, rates = _passing()
    metrics["coverage"] = 439 / 440
    assert final_gate(metrics, rates)["passed"] is False
