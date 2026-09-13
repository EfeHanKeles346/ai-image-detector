import numpy as np
import pytest
import torch
from sklearn.preprocessing import StandardScaler
from experiments import e82_representation as m


def test_class_mass_and_hard_real_weights_remain_global():
    rows=[{'label':0,'source':'A','parent_id':'a'},{'label':0,'source':'A','parent_id':'b'},
          {'label':0,'source':'B','parent_id':'c'},{'label':1,'source':'AI','parent_id':'d'}]
    baseline=np.zeros(12);baseline[:3]=1
    labels,w=m.loss_weights(rows,baseline)
    assert w.mean()==pytest.approx(1)
    assert w[labels==0].sum()==pytest.approx(w[labels==1].sum())
    assert w[0]==pytest.approx(2*w[3])
    assert w[6]==pytest.approx(2*w[3])
    with pytest.raises(ValueError,match='aligned'):m.loss_weights(rows,baseline[:-1])


def test_saved_latent_affine_span_contains_the_classifier_logit(tmp_path):
    torch.set_num_threads(2);torch.manual_seed(82);net=m.build_network(9,8,3)
    a=m.export_network(net)|{'input_center':np.zeros(9),'input_scale':np.ones(9)}
    x=np.random.default_rng(82).normal(size=(73,9))
    raw=m.latent_raw(x,a);scaler=StandardScaler().fit(raw)
    a.update(latent_center=scaler.mean_,latent_scale=scaler.scale_)
    z=m.coordinates(x,a)
    net=net.double().eval()
    with torch.inference_mode():expected=net(torch.from_numpy(x)).flatten().numpy()
    w=a['nn_4_weight'][0].astype(np.float64)
    actual=z@(w*a['latent_scale'])+a['latent_center']@w+a['nn_4_bias'][0]
    np.testing.assert_allclose(actual,expected,atol=1e-12,rtol=0)
    path=tmp_path/'network.npz';np.savez_compressed(path,**a)
    with np.load(path,allow_pickle=False) as saved:
        assert np.array_equal(m.coordinates(x,saved),z)
        np.testing.assert_allclose(m.coordinates(x,saved,batch_size=8),z,atol=1e-12,rtol=0)
    a['nn_2_weight'][0,0]=np.nan
    with pytest.raises(ValueError,match='state'):m.latent_raw(x,a)


def test_fixed_epoch_order_and_checkpoint_bindings():
    order=m.epoch_order(137,40);assert np.array_equal(np.sort(order),np.arange(137))
    np.random.seed(999);np.random.random(100)
    assert np.array_equal(order,m.epoch_order(137,40))
    assert not np.array_equal(order,m.epoch_order(137,41))
    with pytest.raises(ValueError):m.epoch_order(137,100)
    checkpoint={'binding':'contract','weights_sha256':'weights','completed_epochs':20,
                'initial_loss':1.,'losses':[{'epoch':10,'weighted_BCE':.9},{'epoch':20,'weighted_BCE':.8}]}
    assert m.validate_checkpoint(checkpoint,'contract','weights')==20
    for changed in [checkpoint|{'binding':'other'},checkpoint|{'weights_sha256':'other'},
                    checkpoint|{'completed_epochs':21},checkpoint|{'losses':checkpoint['losses'][:1]}]:
        with pytest.raises(ValueError,match='checkpoint'):m.validate_checkpoint(changed,'contract','weights')


def test_adamw_checkpoint_resumes_the_next_supervised_update_exactly(tmp_path):
    torch.manual_seed(82);net=m.build_network(5,4,2)
    opt=torch.optim.AdamW(net.parameters(),lr=2e-4,weight_decay=1e-4,foreach=False)
    x=torch.randn(13,5);labels=(x[:,0]>0).float();w=torch.linspace(.5,1.5,13)
    def update(model,optimizer):
        loss=(torch.nn.functional.binary_cross_entropy_with_logits(model(x).flatten(),labels,reduction='none')*w).mean()
        optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step()
    update(net,opt);path=tmp_path/'state.pt'
    torch.save({'model':m.cpu_state(net.state_dict()),'optimizer':m.cpu_state(opt.state_dict())},path)
    update(net,opt);state=torch.load(path,weights_only=True);restored=m.build_network(5,4,2)
    restored.load_state_dict(state['model']);new_opt=torch.optim.AdamW(restored.parameters(),lr=2e-4,weight_decay=1e-4,foreach=False)
    new_opt.load_state_dict(state['optimizer']);update(restored,new_opt)
    assert all(torch.equal(v,restored.state_dict()[k]) for k,v in net.state_dict().items())
