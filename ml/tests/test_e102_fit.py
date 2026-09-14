import numpy as np
import pytest
from experiments.e102_fit import new_sensor_gates
from experiments.e92_model import AI_CUT


def test_sensor_failures_cannot_hide_behind_pooled_success():
    rows = [{'label': 0, 'source': 'MIDD:A'}]*5 + [{'label': 0, 'source': 'MIDD:B'}]*95
    scores = np.zeros((100, 4)); scores[:2, 0] = AI_CUT
    result = new_sensor_gates(rows, scores)
    assert result['clean']['pooled_fpr'] == .02
    assert result['clean']['by_sensor_fpr']['MIDD:A'] == .4
    assert not result['clean']['passed']
    assert result['social_q75']['passed']


def test_incomplete_invalid_or_nonreal_new_cohort_fails():
    rows = [{'label': 0, 'source': 'MIDD:A'}]
    for values in [np.zeros((1, 3)), np.full((1, 4), np.nan), np.full((1, 4), 1.01)]:
        with pytest.raises(ValueError):
            new_sensor_gates(rows, values)
    with pytest.raises(ValueError):
        new_sensor_gates([{'label': 1, 'source': 'AI'}], np.zeros((1, 4)))
