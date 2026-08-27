from __future__ import annotations

from experiments.e39_freeze import calibration_gate, select_e39_threshold


def _rows(label: int, source: str, scores: tuple[float, ...]) -> list[dict]:
    return [{"label": label, "source": source, "score": score} for score in scores]


def test_select_e39_threshold_requires_both_real_and_ai_budgets() -> None:
    rows = [
        *_rows(0, "camera-a", (0.05,) * 16 + (0.20,) * 4),
        *_rows(0, "camera-b", (0.05,) * 20),
        *_rows(1, "generator-a", (0.30,) * 20),
        *_rows(1, "generator-b", (0.25,) * 20),
    ]
    selected = select_e39_threshold(rows)
    assert selected["threshold"] > 0.05
    assert selected["real_macro_false_positive_rate"] == 0.10
    assert selected["real_worst_false_positive_rate"] == 0.20
    assert selected["ai_worst_recall"] == 1.0


def test_calibration_gate_rejects_partial_coverage() -> None:
    metrics = {
        "roc_auc": 0.95,
        "tpr_at_fpr": {"tpr": 0.90},
        "eer": 0.10,
        "balanced_accuracy": 0.90,
        "coverage": 0.99,
    }
    frontier = {
        "real_macro_false_positive_rate": 0.05,
        "real_worst_false_positive_rate": 0.10,
        "ai_macro_recall": 0.90,
        "ai_worst_recall": 0.80,
    }
    assert calibration_gate(metrics, frontier)["passed"] is False
