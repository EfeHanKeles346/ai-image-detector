import numpy as np
import pytest
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from experiments import e69_model as m


def fixture(monkeypatch):
    monkeypatch.setattr(m,'RANK',4)
    rng=np.random.default_rng(69);x=rng.normal(size=(80,12)).astype(np.float32)
    shuffle=(x*.7+rng.normal(scale=.1,size=x.shape)).astype(np.float32)
    head=make_pipeline(StandardScaler(),LogisticRegression(C=.1)).fit(x,(x[:,0]+x[:,1]>0).astype(int))
    a,_=m.fit_basis(head,x,shuffle)
    return x,shuffle,head,a


def test_zero_texture_correction_exactly_preserves_reference(monkeypatch):
    x,b,head,a=fixture(monkeypatch)
    assert np.array_equal(m.predict(head,x,b,a),head.predict_proba(x)[:,1])
    assert m.project(head,x,b,a).shape==(80,9)


def test_texture_branch_uses_shuffle_without_altering_original_projection(monkeypatch):
    x,b,head,a=fixture(monkeypatch);p=m.project(head,x,b,a);q=m.project(head,x,b+.25,a)
    assert np.array_equal(p[:,:4],q[:,:4])
    assert not np.allclose(p[:,4:8],q[:,4:8])
    a['weights'][4]=.1
    assert not np.allclose(m.predict(head,x,b,a),m.predict(head,x,b+.25,a))
    with pytest.raises(ValueError,match='aligned'):m.project(head,x,b[:-1],a)


def test_serialized_two_basis_texture_replays_exactly(tmp_path,monkeypatch):
    x,b,head,a=fixture(monkeypatch);a['weights'][4]=.1
    p=tmp_path/'weights.npz';np.savez_compressed(p,**a)
    with np.load(p,allow_pickle=False) as saved:
        assert np.array_equal(m.predict(head,x,b,a),m.predict(head,x,b,saved))


def test_texture_branch_is_shuffled_features_not_original_minus_shuffle(monkeypatch):
    x,b,head,a=fixture(monkeypatch)
    p=m.project(head,x,b,a);q=m.project(head,x+.25,b,a)
    assert np.array_equal(p[:,4:8],q[:,4:8])
    assert not np.allclose(p[:,:4],q[:,:4])
