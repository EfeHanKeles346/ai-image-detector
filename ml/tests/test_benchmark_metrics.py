from __future__ import annotations

import pytest

from pixelproof.benchmark_metrics import (
    BenchmarkContractError,
    evaluate_binary_scores,
    select_source_robust_threshold,
)


def _rows(label: int, source: str, scores: list[float], condition: str = "original") -> list[dict]:
    return [
        {"label": label, "source": source, "condition": condition, "status": "ok", "score": score}
        for score in scores
    ]


def test_nist_style_metrics_preserve_failures_and_groups() -> None:
    rows = (
        _rows(0, "camera-a", [0.1, 0.2, 0.3])
        + _rows(1, "generator-a", [0.7, 0.8, 0.9])
        + [{"label": 1, "source": "generator-a", "condition": "original", "status": "error"}]
    )
    result = evaluate_binary_scores(rows, threshold=0.5)
    assert result["roc_auc"] == 1.0
    assert result["eer"] == 0.0
    assert result["balanced_accuracy"] == 1.0
    assert result["counts"] == {
        "total": 7,
        "succeeded": 6,
        "failed": 1,
        "real_succeeded": 3,
        "ai_succeeded": 3,
    }
    assert result["coverage"] == pytest.approx(6 / 7)
    assert result["accuracy_counting_failures_wrong"] == pytest.approx(6 / 7)
    assert len(result["per_group"]) == 2


def test_source_robust_threshold_uses_lowest_real_safe_cut() -> None:
    records = (
        _rows(0, "camera-a", [0.05] * 18 + [0.81, 0.82])
        + _rows(0, "camera-b", [0.10] * 18 + [0.79, 0.80])
        + _rows(1, "generator-a", [0.95] * 20)
        + _rows(1, "generator-b", [0.90] * 20)
    )
    result = select_source_robust_threshold(records, min_group_size=20)
    assert result["threshold"] > 0.10
    assert result["threshold"] < 0.79
    assert result["real_macro_false_positive_rate"] == pytest.approx(0.10)
    assert result["real_worst_false_positive_rate"] == pytest.approx(0.10)
    assert result["ai_macro_recall"] == 1.0


def test_threshold_selection_rejects_failed_calibration_rows() -> None:
    records = _rows(0, "camera", [0.1] * 20) + _rows(1, "generator", [0.9] * 20)
    records.append({"label": 0, "source": "camera", "status": "error"})
    with pytest.raises(BenchmarkContractError, match="discard"):
        select_source_robust_threshold(records)


def test_metrics_reject_implicit_or_inverted_labels() -> None:
    with pytest.raises(BenchmarkContractError, match="0=REAL"):
        evaluate_binary_scores(
            [{"label": "real", "source": "camera", "status": "ok", "score": 0.1}],
            threshold=0.5,
        )
