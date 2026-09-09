import numpy as np
import pytest

from experiments.e53_artifact_replay import compare


def test_exact_replay():
    rows = [{'score': .1, 'predicted_ai': False}, {'score': .9, 'predicted_ai': True}]
    assert compare(rows, np.array([.1, .9]), .5) == 0


def test_tiny_error_that_flips_decision_is_not_allowed():
    rows = [{'score': .5, 'predicted_ai': True}]
    with pytest.raises(ValueError):
        compare(rows, np.array([.5-1e-8]), .5)


def test_numerical_drift_is_rejected_even_without_decision_flip():
    rows = [{'score': .8, 'predicted_ai': True}]
    with pytest.raises(ValueError):
        compare(rows, np.array([.81]), .5)
