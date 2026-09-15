import numpy as np
import pytest
from pixelproof import transport_consistency as t,holdout_linear


def test_covariance_matches_explicit_weighted_logit_variance():
    rng=np.random.default_rng(136);x=rng.normal(size=(12,5));weights=np.repeat([.05,.075,.125],4)
    matrix=t.covariance(x,weights);parameters=rng.normal(size=6)
    z=(x@parameters[:-1]+parameters[-1]).reshape(3,4)
    expected=float(weights@((z-z.mean(axis=1,keepdims=True)).ravel()**2))
    assert parameters[:-1]@matrix@parameters[:-1]==pytest.approx(expected)
    assert np.linalg.eigvalsh(matrix).min()>-1e-12
    offsets=np.repeat(rng.normal(size=(3,5)),4,axis=0)
    np.testing.assert_allclose(t.covariance(x+offsets,weights),matrix,atol=1e-12)


def test_objective_gradient_including_unpenalized_intercept():
    rng=np.random.default_rng(136);x=rng.normal(size=(16,3));y=np.repeat([0,1,0,1],4);w=np.full(16,1/16)
    matrix=t.covariance(x,w);p=rng.normal(size=4)
    loss,grad=t.objective(p,x,y,w,matrix);numeric=[]
    for i in range(4):
        delta=np.zeros(4);delta[i]=1e-6
        numeric.append((t.objective(p+delta,x,y,w,matrix)[0]-t.objective(p-delta,x,y,w,matrix)[0])/2e-6)
    np.testing.assert_allclose(grad,numeric,atol=1e-8)
    assert grad[-1]==holdout_linear.objective(p,x,y,w)[1][-1]
    same=np.repeat(rng.normal(size=(4,3)),4,axis=0)
    np.testing.assert_allclose(t.covariance(same,w),0,atol=1e-15)


def test_fit_is_finite_and_reduces_training_variance():
    rng=np.random.default_rng(7);y=np.repeat([0,1]*10,4);x=np.column_stack([y*2-1+rng.normal(0,.6,len(y)),rng.normal(size=len(y))])
    w=np.full(len(y),1/len(y));baseline,_=holdout_linear.fit_head(x,y,w);new,report=t.fit_head(x,y,w)
    matrix=t.covariance(x,w)
    assert report['success'] and report['gradient_max_abs']<=1e-5
    assert new[:-1]@matrix@new[:-1]<=baseline[:-1]@matrix@baseline[:-1]+1e-7


def test_incomplete_or_unequal_weight_views_rejected():
    with pytest.raises(ValueError):t.covariance(np.zeros((7,2)),np.full(7,1/7))
    with pytest.raises(ValueError):t.covariance(np.zeros((8,2)),np.array([.05,.2,.125,.125,.125,.125,.125,.125]))
    with pytest.raises(ValueError):t.fit_head(np.zeros((8,2)),np.array([0,0,1,0,1,1,1,1]),np.full(8,.125))
