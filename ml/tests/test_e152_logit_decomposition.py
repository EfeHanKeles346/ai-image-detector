import numpy as np
import pytest
from experiments.e152_logit_decomposition import decompose, summary


def test_exact_linear_attribution_and_paired_transport_identity():
    rng=np.random.default_rng(152)
    x=rng.normal(size=(48,8));z=rng.normal(size=(48,3))
    old=rng.normal(size=9);new=rng.normal(size=12)
    part=decompose(x,z,old,new)
    np.testing.assert_allclose(part[:,0],x@old[:-1]+old[-1],rtol=0,atol=1e-12)
    np.testing.assert_allclose(part[:,1],np.column_stack([x,z])@new[:-1]+new[-1],rtol=0,atol=1e-12)
    np.testing.assert_allclose(part[:,1]-part[:,0],part[:,2:].sum(axis=1),rtol=0,atol=1e-12)
    views=part.reshape(12,4,5);delta=views[:,1:]-views[:,:1]
    np.testing.assert_allclose(delta[:,:,1]-delta[:,:,0],delta[:,:,2:].sum(axis=2),rtol=0,atol=1e-12)
    np.testing.assert_array_equal(delta[:,:,4],0)
    unchanged=decompose(x,z,old,np.r_[old[:-1],np.zeros(3),old[-1]])
    np.testing.assert_array_equal(unchanged[:,2:],0)


def test_error_cohorts_harm_signs_and_ties_respect_both_classes():
    values=np.array([[.2,-.4,-.6,0,0],[-.2,.4,0,.6,0],[-.2,.4,.6,0,0],
                     [.2,-.4,-.6,0,0],[.2,-.4,-.3,-.3,0]])
    result=summary(values,np.array([1,0,1,0,1]))
    assert result['all']['parents']==5
    assert result['new_errors']['parents']==3
    assert result['rescued_errors']['parents']==2
    assert result['new_errors']['largest_harmful_term']=={
        'semantic_coefficient_change':1,'added_residual':1,'intercept_change':0}
    assert result['new_errors']['largest_harmful_tie']==1
    assert result['rescued_errors']['no_harmful_term']==2
    assert summary(values[[0]],np.array([1]))['rescued_errors']=={'parents':0}


@pytest.mark.parametrize('damage',['old_width','new_width','row_alignment','nonfinite','nonmatrix'])
def test_invalid_logit_inputs_fail_closed(damage):
    x=np.ones((3,2));z=np.ones((3,1));old=np.zeros(3);new=np.zeros(4)
    if damage=='old_width':old=np.zeros(4)
    elif damage=='new_width':new=np.zeros(3)
    elif damage=='row_alignment':z=z[:2]
    elif damage=='nonfinite':new[0]=np.nan
    elif damage=='nonmatrix':x=x.ravel()
    with pytest.raises(ValueError):decompose(x,z,old,new)


@pytest.mark.parametrize('labels',[np.array([[1]]),np.array([2]),np.array([])])
def test_invalid_cohort_labels_rejected(labels):
    with pytest.raises(ValueError):summary(np.zeros((1,5)),labels)
