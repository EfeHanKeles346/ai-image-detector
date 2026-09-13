import numpy as np
import pytest
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from experiments import e74_model as m


def fixture():
    rng=np.random.default_rng(74);x=rng.normal(size=(80,12)).astype(np.float32);clip=rng.normal(size=(80,9)).astype(np.float32)
    head=make_pipeline(StandardScaler(),LogisticRegression(C=.1)).fit(x,(x[:,0]>0).astype(int))
    old={'base_mean':np.zeros(12),'base_components':rng.normal(size=(64,12)),'base_scales':np.ones(64),
         'clip_center':np.zeros(9),'clip_scale':np.ones(9),'clip_mean':np.zeros(9),
         'clip_components':rng.normal(size=(64,9)),'clip_scales':np.ones(64),'weights':np.ones(129)}
    return x,clip,head,old,m.fit_map(head,x,clip,old)


def test_exact_basis_and_reference_preserved_without_previous_weight_leakage():
    x,clip,head,old,a=fixture()
    assert not a['weights'].any() and len(a['weights'])==257
    assert np.array_equal(m.predict(head,x,clip,a),head.predict_proba(x)[:,1])
    expected=m.linear.project(head,x,clip,old)[:,:-1]
    assert np.array_equal(m.project(head,x,clip,a)[:,:128],expected)
    for key in m.KEYS:assert np.array_equal(a[key],old[key]) and not np.shares_memory(a[key],old[key])
    assert old['weights'].sum()==129


def test_interactions_are_train_standardized_bilinear_map():
    x,clip,head,_,a=fixture();z=m.coordinates(head,x,clip,a)
    expected=m.bilinear_sketch(z[:,:64],z[:,64:],a['sketch_hash'],a['sketch_sign'])
    features=m.project(head,x,clip,a)
    np.testing.assert_allclose(features[:,128:256],(expected-a['interaction_mean'])/a['interaction_scale'])
    np.testing.assert_allclose(features[:,128:256].mean(axis=0),0,atol=1e-14)
    np.testing.assert_allclose(features[:,128:256].std(axis=0),1,atol=1e-14)
    assert np.all(features[:,-1]==1)


def test_serialization_and_batch_replay(tmp_path):
    x,clip,head,_,a=fixture();a['weights'][130]=.1
    path=tmp_path/'map.npz';np.savez_compressed(path,**a);expected=m.predict(head,x,clip,a)
    with np.load(path,allow_pickle=False) as saved:
        assert np.array_equal(m.predict(head,x,clip,saved),expected)
        actual=np.concatenate([m.predict(head,x[i:i+7],clip[i:i+7],saved) for i in range(0,80,7)])
        np.testing.assert_allclose(actual,expected,atol=1e-7,rtol=0)
        assert np.array_equal(actual>=m.AI_CUT,expected>=m.AI_CUT)


def test_invalid_alignment_or_sketch_rejected():
    x,clip,head,_,a=fixture()
    with pytest.raises(ValueError,match='aligned'):m.project(head,x,clip[:-1],a)
    a['sketch_hash'][0,0]=128
    with pytest.raises(ValueError,match='sketch'):m.project(head,x,clip,a)
