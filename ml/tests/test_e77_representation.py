import numpy as np
import pytest
import torch
from experiments import e77_representation as m


def test_real_normalization_excludes_ai_and_balances_sources():
    rows=[{'label':0,'source':'A','parent_id':'a'}, {'label':0,'source':'A','parent_id':'b'},
          {'label':0,'source':'B','parent_id':'c'}, {'label':1,'source':'fake','parent_id':'d'}]
    mask,w=m.real_weights(rows)
    assert mask.tolist()==[True]*9+[False]*3
    np.testing.assert_allclose(w.mean(),1)
    np.testing.assert_allclose(w[:6].sum(),w[6:].sum())
    x=np.repeat(np.array([[0.,0.],[2.,2.],[10.,10.],[1e9,1e9]]),3,axis=0)
    a=m.real_scaler(x,mask,w)
    np.testing.assert_allclose(a['real_center'],[5.5,5.5])
    x[-3:]=-1e9
    b=m.real_scaler(x,mask,w)
    assert all(np.array_equal(a[k],b[k]) for k in a)


def test_fixed_epoch_orders_cover_every_view_and_resume_without_rng_state():
    order=m.epoch_order(137,40)
    assert np.array_equal(np.sort(order),np.arange(137))
    np.random.seed(999);np.random.random(50)
    assert np.array_equal(order,m.epoch_order(137,40))
    assert not np.array_equal(order,m.epoch_order(137,41))
    with pytest.raises(ValueError):m.epoch_order(137,100)


def test_serialized_ae_residual_is_absolute_error_and_batch_stable(tmp_path):
    torch.set_num_threads(2);torch.manual_seed(77)
    net=m.build_network(9,8,3).eval()
    a=m.export_network(net)|{'real_center':np.zeros(9),'real_scale':np.ones(9)}
    x=np.random.default_rng(7).normal(size=(41,9)).astype(np.float32)
    with torch.inference_mode():expected=torch.abs(torch.from_numpy(x)-net(torch.from_numpy(x))).numpy()
    path=tmp_path/'ae.npz';np.savez_compressed(path,**a)
    with np.load(path,allow_pickle=False) as saved:
        assert np.array_equal(m.residuals(x,saved),expected)
        np.testing.assert_allclose(m.residuals(x,saved,7),expected,atol=2e-7,rtol=0)
    a['ae_0_weight'][0,0]=np.nan
    with pytest.raises(ValueError,match='state'):m.residuals(x,a)


def test_checkpoint_binds_contract_weights_and_complete_epoch_trace():
    checkpoint={'binding':'contract','weights_sha256':'weights','completed_epochs':20,
                'initial_loss':1.,'losses':[{'epoch':10,'weighted_REAL_L1':.9},{'epoch':20,'weighted_REAL_L1':.8}]}
    assert m.validate_checkpoint(checkpoint,'contract','weights')==20
    for changed in [checkpoint|{'binding':'different'},checkpoint|{'weights_sha256':'different'},
                    checkpoint|{'completed_epochs':21},checkpoint|{'losses':checkpoint['losses'][:1]}]:
        with pytest.raises(ValueError,match='checkpoint'):m.validate_checkpoint(changed,'contract','weights')


def test_cpu_optimizer_checkpoint_resumes_next_update_exactly(tmp_path):
    torch.manual_seed(77);net=m.build_network(5,4,2)
    opt=torch.optim.Adam(net.parameters(),lr=2e-4,foreach=False)
    x=torch.randn(13,5);w=torch.linspace(.5,1.5,13)
    def update(model,optimizer):
        loss=(torch.abs(x-model(x)).mean(dim=1)*w).mean()
        optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step()
    update(net,opt)
    path=tmp_path/'state.pt';torch.save({'model':m.cpu_state(net.state_dict()),'optimizer':m.cpu_state(opt.state_dict())},path)
    update(net,opt)
    state=torch.load(path,weights_only=True);restored=m.build_network(5,4,2)
    restored.load_state_dict(state['model']);new_opt=torch.optim.Adam(restored.parameters(),lr=2e-4,foreach=False)
    new_opt.load_state_dict(state['optimizer']);update(restored,new_opt)
    assert all(torch.equal(v,restored.state_dict()[k]) for k,v in net.state_dict().items())
