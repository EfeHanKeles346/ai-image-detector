import numpy as np
import pytest
from experiments.e53_crop_dedup_full import parity


class Head:
    def predict_proba(self,x):return np.c_[1-x[:,0],x[:,0]]


def test_parity_reports_decision_flip_even_for_small_score_change():
    a=np.array([[.499999,1.]]);b=np.array([[.500001,1.]])
    result=parity(a,b,Head(),.5)
    assert result['max_score_error']<.00005
    assert result['decision_flips']==1


def test_invalid_optimized_features_fail_closed():
    with pytest.raises(ValueError):parity(np.ones((1,2)),np.full((1,2),np.nan),Head(),.5)
