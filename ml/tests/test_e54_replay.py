import numpy as np
import pytest

from experiments.e54_replay import compare_scores


def test_nextafter_threshold_keeps_archived_comparison_precision():
    actual=np.array([.5],dtype=np.float32);expected=actual.astype(np.float64)
    cut=np.nextafter(.5,float('inf'))
    assert compare_scores(actual,expected,cut)==(0.,0)


def test_detects_true_decision_change_and_nonfinite():
    error,changed=compare_scores([.6],[.4],.5)
    assert error==pytest.approx(.2) and changed==1
    with pytest.raises(ValueError):compare_scores([float('nan')],[.4],.5)
