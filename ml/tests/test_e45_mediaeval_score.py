import numpy as np

from experiments.e45_mediaeval_score import (
    AI_CUT,
    BINARY_THRESHOLD,
    REAL_CUT,
    binary_platform_rates,
    bootstrap_primary,
    selective_metrics,
)


def _rows():
    rows = []
    for platform in ("Facebook", "Instagram", "LinkedIn", "X"):
        rows.extend([
            {"label": 0, "platform": platform, "score": 0.01, "status": "ok"},
            {"label": 0, "platform": platform, "score": 0.10, "status": "ok"},
            {"label": 1, "platform": platform, "score": 0.90, "status": "ok"},
            {"label": 1, "platform": platform, "score": 0.99, "status": "ok"},
        ])
    return rows


def test_threshold_order_is_nonempty_selective_band():
    assert REAL_CUT < BINARY_THRESHOLD < AI_CUT


def test_binary_and_selective_metrics_are_perfect_on_separated_rows():
    rates = binary_platform_rates(_rows(), BINARY_THRESHOLD)
    selective = selective_metrics(_rows())
    assert rates["pooled_real_false_ai"] == 0.0
    assert rates["pooled_ai_recall"] == 1.0
    assert selective["automatic_coverage"] == 1.0
    assert selective["covered_accuracy"] == 1.0


def test_selective_metrics_counts_uncertain_without_forcing_a_label():
    rows = _rows()
    rows[0]["score"] = (REAL_CUT + AI_CUT) / 2
    found = selective_metrics(rows)
    assert found["uncertain_rows"] == 1
    assert np.isclose(found["uncertain_rate"], 1 / len(rows))


def test_bootstrap_is_deterministic_and_contains_perfect_values():
    first = bootstrap_primary(_rows(), samples=100, seed=7)
    second = bootstrap_primary(_rows(), samples=100, seed=7)
    assert first == second
    assert first["roc_auc"] == [1.0, 1.0]
    assert first["balanced_accuracy"] == [1.0, 1.0]
