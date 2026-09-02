from __future__ import annotations

from experiments.e43_dda_score import VARIANTS, final_gate, variant_rates


def _metrics(**changes):
    values = {
        "roc_auc": 0.95,
        "tpr_at_fpr": {"tpr": 0.85},
        "eer": 0.10,
        "balanced_accuracy": 0.90,
        "coverage": 1.0,
    }
    values.update(changes)
    return values


def _rates(**changes):
    values = {
        "real_false_positive_rate": 0.08,
        "ai_macro_recall": 0.85,
        "ai_worst_variant_recall": 0.70,
    }
    values.update(changes)
    return values


def test_final_gate_requires_every_frozen_check():
    assert final_gate(_metrics(), _rates())["passed"] is True
    failed = final_gate(_metrics(), _rates(ai_worst_variant_recall=0.59))
    assert failed["passed"] is False
    assert failed["checks"]["ai_worst_variant_recall_gte_0_60"] is False


def test_variant_rates_keep_real_and_generator_recall_separate():
    rows = [{"label": 0, "condition": "REAL", "score": 0.1}]
    rows.extend({"label": 1, "condition": variant, "score": 0.9} for variant in VARIANTS)
    rates = variant_rates(rows)
    assert rates["real_false_positive_rate"] == 0.0
    assert rates["ai_macro_recall"] == 1.0
    assert rates["ai_worst_variant_recall"] == 1.0
