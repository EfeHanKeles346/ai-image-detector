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


def test_adam_checkpoint_continuation_matches_uninterrupted_steps(tmp_path):
    torch.manual_seed(54)
    inputs=torch.randn(29,4);targets=torch.arange(29).remainder(2).float()
    initial=torch.nn.Linear(4,1).state_dict()
    order=deterministic_order(np.arange(29),0,0)
    def create():
        head=torch.nn.Linear(4,1);head.load_state_dict(initial)
        return head,torch.optim.AdamW(head.parameters(),lr=1e-5,weight_decay=.01)
    def steps(head,optimizer,start,end):
        for offset in range(start,end,8):
            ids=order[offset:offset+8]
            optimizer.zero_grad(set_to_none=True)
            loss=torch.nn.functional.binary_cross_entropy_with_logits(head(inputs[ids]).flatten(),targets[ids])
            loss.backward();torch.nn.utils.clip_grad_norm_(head.parameters(),1.);optimizer.step()
    complete,full_optimizer=create();steps(complete,full_optimizer,0,29)
    interrupted,partial_optimizer=create();steps(interrupted,partial_optimizer,0,16)
    path=tmp_path/'resume.pt'
    atomic_checkpoint(path,{'head':interrupted.state_dict(),'optimizer':partial_optimizer.state_dict(),'offset':16})
    resumed,resume_optimizer=create();state=torch.load(path,weights_only=True)
    resumed.load_state_dict(state['head']);resume_optimizer.load_state_dict(state['optimizer'])
    steps(resumed,resume_optimizer,state['offset'],29)
    for name,parameter in complete.state_dict().items():
        torch.testing.assert_close(parameter,resumed.state_dict()[name],rtol=0,atol=0)
