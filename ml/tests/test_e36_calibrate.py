from __future__ import annotations

from experiments.e36_calibrate import calibration_gate, select_threshold


def _row(label: int, source: str, score: float) -> dict:
    return {"label": label, "source": source, "score": score}


def test_select_threshold_preserves_ai_coverage_at_first_real_safe_cut() -> None:
    rows = [
        *[_row(0, "phone-a", score) for score in (0.1, 0.2, 0.3, 0.4, 0.9)],
        *[_row(0, "phone-b", score) for score in (0.1, 0.2, 0.3, 0.4, 0.8)],
        *[_row(1, "ai-a", score) for score in (0.7, 0.8, 0.9, 1.0)],
        *[_row(1, "ai-b", score) for score in (0.6, 0.8, 0.9, 1.0)],
    ]
    selected = select_threshold(rows)
    assert selected["threshold"] == 0.9
    assert selected["real_worst_device_fp"] <= 0.20
    assert selected["ai_macro_recall"] == 0.5


def test_gate_requires_real_and_ai_sides_together() -> None:
    metrics = {"roc_auc": 0.95, "tpr_at_fpr": {"tpr": 0.9}, "eer": 0.1, "balanced_accuracy": 0.9}
    frontier = {
        "real_macro_fp": 0.05,
        "real_worst_device_fp": 0.10,
        "ai_macro_recall": 0.85,
        "ai_worst_family_recall": 0.70,
    }
    assert calibration_gate(metrics, frontier)["passed"] is True
    frontier["ai_worst_family_recall"] = 0.50
    assert calibration_gate(metrics, frontier)["passed"] is False
