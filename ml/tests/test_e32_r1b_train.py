from __future__ import annotations

import numpy as np
import pytest

from experiments.e32_r1b_train import merge_features


def _contract(record_id: str, value: float) -> dict[str, np.ndarray]:
    return {"record_ids": np.asarray([record_id]), "features": np.asarray([[value, value + 1]], dtype=np.float32)}


def test_merge_features_reorders_to_receipt_and_rebuilds_metadata() -> None:
    rows = [
        {"record_id": "new", "label": "real", "role": "CALIBRATION", "source_id": "iphone"},
        {"record_id": "old", "label": "ai", "role": "TRAIN", "source_id": "qwen"},
    ]
    merged = merge_features(_contract("old", 1), _contract("new", 3), rows)
    assert merged["record_ids"].tolist() == ["new", "old"]
    assert merged["features"].tolist() == [[3, 4], [1, 2]]
    assert merged["labels"].tolist() == [0, 1]


def test_merge_features_rejects_overlap() -> None:
    rows = [{"record_id": "same", "label": "real", "role": "TRAIN", "source_id": "x"}]
    with pytest.raises(ValueError, match="overlap"):
        merge_features(_contract("same", 1), _contract("same", 2), rows)
