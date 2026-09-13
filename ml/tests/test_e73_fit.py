import numpy as np
import pytest
from scipy.special import logit
from experiments import e73_fit as m


def test_confidence_constraint_protects_previously_missed_ai_too():
    x=np.array([[1.,1.],[2.,1.],[-1.,1.],[-2.,1.]])
    logits=logit(m.model.AI_CUT)+np.array([1.,-1.,2.,-.5])
    labels=np.array([1,1,0,0])
    bounds,ai,real,slack=m.constraints_for(x,logits,labels)
    np.testing.assert_array_equal(bounds[0].A,x[:2])
    np.testing.assert_array_equal(bounds[0].lb,[0.,0.])
    assert ai.sum()==2 and real.sum()==1
    assert slack[0]==pytest.approx(.5-m.GUARD_MARGIN)
    # The old decision-only constraint allowed a negative shift for the caught AI;
    # neither caught nor currently missed AI may now lose reference logit margin.
    assert np.all(bounds[0].A@np.zeros(2)>=bounds[0].lb)


def test_confidence_fit_can_rescue_real_while_preserving_all_ai_margins():
    x=np.array([[1.,1.],[2.,1.],[-1.,1.],[-2.,1.]])
    old=logit(m.model.AI_CUT)+np.array([1.,-1.,1.,-1.])
    labels=np.array([1,1,0,0]);weights=np.full(4,.25)
    fitted,result=m.fit_confidence(x,old,labels,weights)
    shift=x@fitted
    assert result['success'] and result['max_constraint_violation']<1e-8
    assert shift[:2].min()>=-1e-8
    assert old[2]+shift[2]<logit(m.model.AI_CUT)
    assert old[3]+shift[3]<logit(m.model.AI_CUT)
    assert result['objective']<m.objective(np.zeros(2),x,old,labels,weights)[0]


def test_confidence_validation_and_no_correct_real_case():
    x=np.array([[1.,1.],[-1.,1.]])
    logits=logit(m.model.AI_CUT)+np.ones(2);labels=np.array([1,0])
    fitted,result=m.fit_confidence(x,logits,labels,np.array([.5,.5]))
    assert result['success'] and result['all_ai_views_protected']==1
    with pytest.raises(ValueError,match='aligned'):m.constraints_for(x,logits[:1],labels)
    with pytest.raises(ValueError,match='two-class'):m.constraints_for(x,logits,np.ones(2))


def test_basis_reset_discards_previous_weights_and_copies_coordinates():
    keys=(*m.model.BASE_KEYS,'clip_center','clip_scale','clip_mean','clip_components','clip_scales')
    previous={k:np.arange(3,dtype=float) for k in keys};previous['weights']=np.ones(129)
    result=m.reset_basis(previous)
    assert not result['weights'].any()
    for k in keys:
        np.testing.assert_array_equal(result[k],previous[k])
        assert not np.shares_memory(result[k],previous[k])
    assert previous['weights'].sum()==129
