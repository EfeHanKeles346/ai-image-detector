from __future__ import annotations

import numpy as np

from experiments.e38_fixed_adaptation import (
    ADAPTATION_WEIGHT,
    HISTORICAL_WEIGHT,
    make_head,
    training_weights,
)


def test_training_weights_are_uniform_within_declared_roles() -> None:
    weights = training_weights(3, 2)
    np.testing.assert_array_equal(weights[:3], np.full(3, HISTORICAL_WEIGHT))
    np.testing.assert_array_equal(weights[3:], np.full(2, ADAPTATION_WEIGHT))


def test_training_weights_fail_without_both_roles() -> None:
    for counts in ((0, 2), (3, 0)):
        try:
            training_weights(*counts)
        except ValueError:
            pass
        else:
            raise AssertionError("missing role should fail closed")


def test_fixed_head_has_preregistered_contract() -> None:
    head = make_head().named_steps["logisticregression"]
    assert head.C == 0.0003
    assert head.class_weight == "balanced"
    assert head.random_state == 42
