import numpy as np
import pytest
from experiments.e55_audit import objective_gradient


def test_weighted_duplicate_objective_and_gradient_are_equal():
    rng=np.random.default_rng(55);x=rng.normal(size=(12,4));y=np.arange(12)%2
    weights=rng.uniform(.1,2,size=12);coef=rng.normal(size=4)
    loss,g=objective_gradient(x,y,weights,coef,.2,.01)
    loss2,g2=objective_gradient(np.r_[x,x],np.tile(y,2),np.r_[.8*weights,.2*weights],coef,.2,.01)
    assert loss2==pytest.approx(loss,abs=1e-12)
    np.testing.assert_allclose(g,g2,atol=1e-12)
    for i in range(5):
        param=np.r_[coef,.2];a=param.copy();b=param.copy();a[i]+=1e-6;b[i]-=1e-6
        la,_=objective_gradient(x,y,weights,a[:-1],a[-1],.01)
        lb,_=objective_gradient(x,y,weights,b[:-1],b[-1],.01)
        assert (la-lb)/2e-6==pytest.approx(g[i],abs=1e-7)


def test_objective_rejects_invalid_weights():
    with pytest.raises(ValueError):objective_gradient(np.zeros((2,1)),[0,1],[1,-1],[0],0,.01)
