from __future__ import annotations

import pytest

from experiments import e42_rr_final as rr


def test_canonical_parent_mapping_survives_rr_derivative_prefixes() -> None:
    assert rr.canonical_stem("real_000001.jpg", "original") == "real_000001"
    assert rr.canonical_stem("transfer_real_000001.png", "transfer") == "real_000001"
    assert rr.canonical_stem("redigital_real_000001.jpg", "redigital") == "real_000001"
    assert rr.source_from_row("real", "real_000001") == "rrdataset_real_pool"
    assert rr.source_from_row("ai", "Medical_&_Public_Health_000837") == "medical_and_public_health"


def _row(record: str, parent: str, condition: str, label: int, sha: str, dhash: str) -> dict:
    return {
        "record_id": record,
        "parent_id": parent,
        "condition": condition,
        "label": label,
        "sha256": sha,
        "dhash": dhash,
    }


def test_audit_allows_derivatives_of_one_parent_but_rejects_prior_overlap() -> None:
    rows = [
        _row("o", "real:1", "original", 0, "a", "1"),
        _row("t", "real:1", "transfer", 0, "b", "2"),
        _row("r", "real:1", "redigital", 0, "c", "3"),
    ]
    passed = rr.audit_rows(rows, set(), set())
    assert passed["passed"] is True
    assert passed["condition_sets_by_parent"] == {"original+redigital+transfer": 1}
    failed = rr.audit_rows(rows, {"b"}, set())
    assert failed["passed"] is False
    assert failed["failures"]["prior_exact_overlap"] == ["t"]


def test_audit_rejects_duplicate_condition_and_cross_parent_exact() -> None:
    rows = [
        _row("a", "real:1", "original", 0, "same", "1"),
        _row("b", "real:1", "original", 0, "b", "2"),
        _row("c", "real:2", "transfer", 0, "same", "3"),
    ]
    result = rr.audit_rows(rows, set(), set())
    assert result["passed"] is False
    assert result["failures"]["duplicate_parent_condition"] == ["real:1"]
    assert result["failures"]["cross_parent_exact"][0]["parents"] == ["real:1", "real:2"]


def _condition(auc: float, balanced: float, coverage: float = 1.0) -> dict:
    return {
        "metrics": {
            "roc_auc": auc,
            "balanced_accuracy": balanced,
            "eer": 0.05,
            "tpr_at_fpr": {"fpr": 0.10, "tpr": 0.90},
            "coverage": coverage,
            "counts": {"real_succeeded": 50, "ai_succeeded": 50},
        },
        "rates": {
            "real_macro_fp": 0.05,
            "real_worst_source_fp": 0.05,
            "ai_macro_recall": 0.90,
            "ai_worst_source_recall": 0.80,
        },
    }


def test_final_gate_requires_clean_and_both_robust_conditions() -> None:
    result = rr.final_gate({
        "original": _condition(0.95, 0.90),
        "transfer": _condition(0.90, 0.85),
        "redigital": _condition(0.89, 0.82),
    })
    assert result["passed"] is True
    failed = rr.final_gate({
        "original": _condition(0.95, 0.90),
        "transfer": _condition(0.84, 0.85),
        "redigital": _condition(0.89, 0.82),
    })
    assert failed["passed"] is False
    assert failed["checks"]["transfer_auc_gte_0_85"] is False
