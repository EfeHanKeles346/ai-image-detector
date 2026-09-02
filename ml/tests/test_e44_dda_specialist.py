from __future__ import annotations

from experiments.e44_dda_specialist import SCREEN_PARENTS, screen_gate, select_parent_ids


def test_parent_selection_is_stable_unique_and_score_blind():
    rows = [
        {"parent_id": f"p{index:04d}", "score": 1.0 - index / 1000}
        for index in range(SCREEN_PARENTS + 20)
    ]
    first = select_parent_ids(rows)
    second = select_parent_ids(list(reversed(rows)))
    assert first == second
    assert len(first) == SCREEN_PARENTS
    assert len(set(first)) == SCREEN_PARENTS


def _metrics(**changes):
    values = {"coverage": 1.0, "roc_auc": 0.90, "balanced_accuracy": 0.85}
    values.update(changes)
    return values


def _rates(**changes):
    values = {
        "real_false_positive_rate": 0.10,
        "core_variant_macro_recall": 0.85,
        "all_variant_macro_recall": 0.75,
        "worst_variant_recall": 0.45,
    }
    values.update(changes)
    return values


def test_screen_gate_requires_specialist_and_real_safety():
    assert screen_gate(_metrics(), _rates())["passed"] is True
    failed = screen_gate(_metrics(), _rates(real_false_positive_rate=0.21))
    assert failed["passed"] is False
    assert failed["checks"]["real_fp_lte_0_20"] is False
