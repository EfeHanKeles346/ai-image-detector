from __future__ import annotations

import numpy as np

from experiments.e44_fusion import _feature, source_label_weights, split_exact, split_fractional


def test_exact_split_is_stable_and_disjoint():
    values = [f"p{index:03d}" for index in range(20)]
    first = split_exact(values, "test", 10, 5)
    second = split_exact(list(reversed(values)), "test", 10, 5)
    assert first == second
    assert list(first.values()).count("fit") == 10
    assert list(first.values()).count("calibration") == 5
    assert list(first.values()).count("development") == 5


def test_fractional_split_retains_three_roles():
    roles = split_fractional([f"p{index}" for index in range(11)], "test")
    assert set(roles.values()) == {"fit", "calibration", "development"}


def test_feature_is_finite_at_probability_edges():
    assert np.isfinite(_feature(0.0, 1.0)).all()


def test_source_label_weights_balance_labels():
    rows = [
        {"label": 0, "population": "real", "source": "a"},
        {"label": 0, "population": "real", "source": "a"},
        {"label": 1, "population": "ai", "source": "b"},
    ]
    weights = source_label_weights(rows)
    assert np.isclose(weights[:2].sum(), weights[2:].sum())
