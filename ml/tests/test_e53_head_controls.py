import numpy as np
import pytest
from experiments.e53_head_controls import ARMS,feature_map


def test_bounded_controls_and_rowwise_normalization():
    assert len(ARMS)==4
    x=np.array([[3.,4.,100.],[0.,0.,200.]])
    y=feature_map(x,2,True)
    assert np.allclose(y,[[.6,.8],[0.,0.]])
    assert np.array_equal(x,[[3.,4.,100.],[0.,0.,200.]])
    assert np.array_equal(feature_map(x,2,False),x[:,:2])


def test_invalid_features_fail_closed():
    with pytest.raises(ValueError):feature_map(np.ones((1,2)),3,False)
    with pytest.raises(ValueError):feature_map(np.full((1,2),np.nan),2,True)
