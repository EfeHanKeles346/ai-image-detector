import numpy as np
import pytest
from experiments.e53_report import paired_interval,preservation_guard


def test_paired_identical_predictions_have_exact_zero_deltas():
    y=np.tile([0,1],100);p=np.tile([[0,0],[1,1]],(100,1))
    ci=paired_interval(y,np.repeat(np.arange(10),20),p,p,repetitions=100)
    assert all(v==[0.,0.] for v in ci.values())
    result=preservation_guard(y,np.where(y==0,'camera','AI'),p,p,ci)
    assert not result['passed']  # Unchanged is not an authentic improvement.


def test_real_gain_cannot_hide_ai_loss():
    y=np.r_[np.zeros(100),np.ones(100)].astype(int)
    old=np.ones((200,2),dtype=bool);new=old.copy();new[:100]=False;new[100:120]=False
    ci=paired_interval(y,np.repeat(np.arange(10),20),new,old,repetitions=100)
    result=preservation_guard(y,np.where(y==0,'camera','AI'),new,old,ci)
    assert result['checks']['real_fpr_lower_both']
    assert not result['checks']['ai_recall_no_point_loss_both']
    assert not result['passed']


def test_pairing_shape_and_classes_fail_closed():
    with pytest.raises(ValueError):paired_interval(np.zeros(10),np.arange(10),np.zeros((10,2)),np.zeros((10,2)))
    with pytest.raises(ValueError):paired_interval(np.tile([0,1],5),np.arange(10),np.zeros((10,1)),np.zeros((10,2)))
