from __future__ import annotations

import pytest

from experiments.e49_dotting import MODEL_KEYS
from experiments.e49_dotting_score import CONDITIONS, evaluate_diagnostic, validate_rows


def _rows(score_by_condition=None):
    score_by_condition = score_by_condition or {"publisher_original": 0.9, "social_q75": 0.8}
    rows = []
    for condition in CONDITIONS:
        for source in MODEL_KEYS.values():
            for index in range(160):
                parent = f"{source}:{index}"
                rows.append({"record_id": f"{parent}:{condition}", "parent_id": parent,
                             "condition": condition, "source": source, "label": 1,
                             "path": "/tmp/example", "sha256": "a" * 64, "status": "ok",
                             "score": score_by_condition[condition]})
    return rows


def test_dotting_diagnostic_passes_only_ai_recall_contract():
    report = evaluate_diagnostic(_rows())
    assert report["state"] == "e49_d1_ai_diagnostic_pass"
    assert report["gate"] == {"passed": True, "passed_checks": 6, "total_checks": 6}
    assert report["paired_transport"]["ai_recall_loss_original_to_q75"] == 0
    assert "roc_auc" not in report["conditions"]["publisher_original"]


def test_dotting_diagnostic_fails_one_weak_model_without_row_removal():
    rows = _rows()
    weak = next(iter(MODEL_KEYS.values()))
    for row in rows:
        if row["source"] == weak and row["condition"] == "social_q75":
            row["score"] = 0.01
    report = evaluate_diagnostic(rows)
    assert report["state"] == "e49_d1_ai_diagnostic_fail"
    assert report["conditions"]["social_q75"]["worst_model_recall"] == 0
    assert report["conditions"]["social_q75"]["selective_counts"]["confident_real_miss"] == 160


def test_dotting_pair_validation_rejects_source_drift():
    rows = _rows()
    rows[-1]["source"] = "changed"
    with pytest.raises(ValueError, match="source balance"):
        validate_rows(rows, require_scores=True)


def test_dotting_completed_scores_must_be_finite():
    rows = _rows()
    rows[0]["score"] = float("nan")
    with pytest.raises(ValueError, match="invalid completed score"):
        validate_rows(rows, require_scores=True)
