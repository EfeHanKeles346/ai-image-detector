import numpy as np
import pytest
from scipy.optimize import check_grad
from pixelproof.holdout_linear import fit_map,project,objective,fit_head,predict


def test_objective_gradient_and_intercept_is_not_penalized():
    rng=np.random.default_rng(12);x=rng.normal(size=(10,3));y=np.arange(10)%2;w=np.ones(10)/10
    p=rng.normal(size=4)
    assert check_grad(lambda z:objective(z,x,y,w)[0],lambda z:objective(z,x,y,w)[1],p)<1e-6
    z=np.zeros((10,3));p=np.array([0.,0.,0.,4.])
    assert objective(p,z,y,w,.01)[0]==objective(p,z,y,w,100.)[0]


def test_held_out_distribution_cannot_change_fitted_transform_or_head():
    rng=np.random.default_rng(3);train=rng.normal(size=(30,5));y=(train[:,0]>0).astype(int)
    a=fit_map(train,2);x=project(train,a);p,_=fit_head(x,y,np.ones(30)/30)
    before={k:v.copy() for k,v in a.items()};before_p=p.copy()
    values=predict(project(np.full((3,5),100.),a),p)
    assert np.isfinite(values).all()
    for key in before:np.testing.assert_array_equal(a[key],before[key])
    np.testing.assert_array_equal(p,before_p)


def test_simple_separation_converges_and_saved_arrays_replay(tmp_path):
    x=np.array([[-2.],[-1.],[1.],[2.]]);y=np.array([0,0,1,1])
    p,report=fit_head(x,y,np.ones(4)/4)
    assert report['success'] and report['gradient_max_abs']<=1e-5
    np.testing.assert_array_equal(predict(x,p)>=.5,y.astype(bool))
    np.savez(tmp_path/'p.npz',parameters=p)
    with np.load(tmp_path/'p.npz') as a:np.testing.assert_array_equal(predict(x,p),predict(x,a['parameters']))


def test_nan_or_non_unit_mass_cannot_fit():
    with pytest.raises(ValueError):fit_map(np.full((10,5),np.nan),2)
    with pytest.raises(ValueError):fit_head(np.zeros((2,2)),[0,1],[1.,1.])
