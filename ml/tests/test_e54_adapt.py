import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from experiments.e54_adapt import deterministic_order, torch_head, atomic_checkpoint


def test_epoch_order_visits_every_view_once_and_resumes_exactly():
    ids=np.arange(45)
    a=deterministic_order(ids,1,0);b=deterministic_order(ids,1,0)
    np.testing.assert_array_equal(a,b)
    assert set(a)==set(ids)
    assert not np.array_equal(a,deterministic_order(ids,1,1))
    np.testing.assert_array_equal(np.r_[a[:16],b[16:]],a)


def test_converted_head_preserves_probability_and_ai_orientation():
    x=np.random.default_rng(54).normal(size=(20,3072)).astype('float32');y=np.arange(20)%2
    pipe=make_pipeline(StandardScaler(),LogisticRegression(C=.01)).fit(x,y)
    head=torch_head(pipe,torch.device('cpu'))
    actual=torch.sigmoid(head(torch.tensor(x)).flatten()).detach().numpy()
    np.testing.assert_allclose(actual,pipe.predict_proba(x)[:,1],atol=2e-6)


def test_atomic_checkpoint_roundtrip(tmp_path):
    path=tmp_path/'state.pt'
    atomic_checkpoint(path,{'epoch':1,'offset':8,'tensor':torch.arange(4)})
    state=torch.load(path,weights_only=True)
    assert state['offset']==8
    assert not path.with_suffix('.pt.part').exists()
