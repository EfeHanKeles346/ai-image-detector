import numpy as np
import pytest
import torch
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from experiments import e77_model as m


def fixture():
    rng=np.random.default_rng(77);x=rng.normal(size=(80,12)).astype(np.float32);clip=rng.normal(size=(80,9)).astype(np.float32)
    head=make_pipeline(StandardScaler(),LogisticRegression(C=.1)).fit(x,(x[:,0]>0).astype(int))
    old={'base_mean':np.zeros(12),'base_components':rng.normal(size=(64,12)),'base_scales':np.ones(64),
         'clip_center':np.zeros(9),'clip_scale':np.ones(9),'clip_mean':np.zeros(9),
         'clip_components':rng.normal(size=(64,9)),'clip_scales':np.ones(64)}
    previous=m.base.fit_map(head,x,clip,old);previous['weights'][:]=99
    torch.manual_seed(77);net=m.manifold.build_network(9,8,3)
    representation=m.manifold.export_network(net)|{'real_center':np.zeros(9),'real_scale':np.ones(9),
        'residual_center':np.zeros(9),'residual_scale':np.ones(9),'residual_mean':np.zeros(9),
        'residual_components':rng.normal(size=(64,9)),'residual_scales':np.ones(64)}
    return x,clip,head,previous,representation,m.assemble(previous,representation)


def test_original_map_is_exact_and_previous_correction_never_leaks():
    x,clip,head,previous,rep,a=fixture()
    assert a['weights'].shape==(321,) and not a['weights'].any()
    assert np.array_equal(m.predict(head,x,clip,a),head.predict_proba(x)[:,1])
    z=m.project(head,x,clip,a)
    assert np.array_equal(z[:,:256],m.base.project(head,x,clip,previous)[:,:-1])
    assert np.array_equal(z[:,256:320],m.manifold.coordinates(clip,rep))
    assert np.all(z[:,-1]==1)
    for k in m.BASE_KEYS:assert not np.shares_memory(a[k],previous[k])
    for k in m.MANIFOLD_KEYS:assert not np.shares_memory(a[k],rep[k])


def test_serialized_nonlinear_correction_and_batch_parity(tmp_path):
    x,clip,head,_,_,a=fixture();a['weights'][260]=.07
    path=tmp_path/'map.npz';np.savez_compressed(path,**a);expected=m.predict(head,x,clip,a)
    with np.load(path,allow_pickle=False) as saved:
        assert np.array_equal(expected,m.predict(head,x,clip,saved))
        actual=np.concatenate([m.predict(head,x[i:i+7],clip[i:i+7],saved) for i in range(0,len(x),7)])
        np.testing.assert_allclose(actual,expected,atol=1e-7,rtol=0)
        assert np.array_equal(actual>=m.AI_CUT,expected>=m.AI_CUT)


def test_incomplete_pairs_and_malformed_residual_map_rejected():
    x,clip,head,_,_,a=fixture()
    with pytest.raises(ValueError,match='aligned'):m.project(head,x,clip[:-1],a)
    a['residual_components'][0,0]=np.nan
    with pytest.raises(ValueError,match='coordinates'):m.project(head,x,clip,a)
