from __future__ import annotations

import numpy as np

from experiments.e37_source_heldout import FOLDS, fold_ids, gate, select_threshold


def _row(label: int, source: str, score: float) -> dict:
    return {"label": label, "source": source, "score": score}


def test_fold_contract_assigns_every_source_once() -> None:
    real = sorted(source for fold in FOLDS for source in fold["real"])
    ai = sorted(source for fold in FOLDS for source in fold["ai"])
    labels = np.asarray([0] * len(real) + [1] * len(ai))
    sources = np.asarray(real + ai)
    assigned = fold_ids(labels, sources)
    assert set(assigned.tolist()) == set(range(5))
    assert len(assigned) == len(real) + len(ai)


def test_fold_contract_rejects_an_unknown_source() -> None:
    try:
        fold_ids(np.asarray([0]), np.asarray(["unknown-phone"]))
    except ValueError as error:
        assert "0 fold assignments" in str(error)
    else:
        raise AssertionError("unknown source should fail closed")


def test_threshold_and_gate_require_both_classes() -> None:
    rows = [
        *[_row(0, "device_001", score) for score in (0.1, 0.2, 0.3, 0.4, 0.9)],
        *[_row(0, "device_002", score) for score in (0.1, 0.2, 0.3, 0.4, 0.8)],
        *[_row(1, "FLUX.2_max", score) for score in (0.9, 0.91, 0.92, 0.93, 0.94)],
        *[_row(1, "GLM-Image", score) for score in (0.85, 0.9, 0.91, 0.92, 0.93)],
    ]
    selected = select_threshold(rows)
    assert selected["threshold"] == np.nextafter(0.8, np.inf)
    assert selected["real_worst_device_fp"] == 0.2
    metrics = {
        "roc_auc": 0.95,
        "tpr_at_fpr": {"tpr": 0.9},
        "eer": 0.1,
        "balanced_accuracy": 0.9,
        "coverage": 1.0,
    }
    assert gate(metrics, selected)["passed"] is True
    selected["ai_worst_family_recall"] = 0.5
    assert gate(metrics, selected)["passed"] is False
