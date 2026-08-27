from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from experiments.e41_freeze import NEW_THRESHOLD, OLD_THRESHOLD, head_numeric_sha256


def test_e41_threshold_is_the_single_frozen_transfer() -> None:
    assert OLD_THRESHOLD == 0.17080099880695346
    assert NEW_THRESHOLD == 0.6195540428161622


def test_head_numeric_identity_ignores_artifact_metadata() -> None:
    x = np.asarray([[0.0, 1.0], [1.0, 0.0], [0.2, 0.9], [0.9, 0.2]])
    y = np.asarray([0, 1, 0, 1])
    head = make_pipeline(StandardScaler(), LogisticRegression()).fit(x, y)
    before = head_numeric_sha256(head)
    metadata = {"head": head, "threshold": 0.1}
    metadata["threshold"] = 0.9
    assert head_numeric_sha256(metadata["head"]) == before
