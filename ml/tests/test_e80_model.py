import numpy as np
import pytest
import torch
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from experiments import e80_model as m
from experiments.e80_fit import full_training_gates


def fixture():
    rng=np.random.default_rng(80);x=rng.normal(size=(96,12)).astype(np.float32)
    clip=rng.normal(size=(96,9)).astype(np.float32);dear=rng.normal(size=(96,80)).astype(np.float32)
    head=make_pipeline(StandardScaler(),LogisticRegression(C=.1)).fit(x,(x[:,0]>0).astype(int))
    old={'base_mean':np.zeros(12),'base_components':rng.normal(size=(64,12)),'base_scales':np.ones(64),
         'clip_center':np.zeros(9),'clip_scale':np.ones(9),'clip_mean':np.zeros(9),
         'clip_components':rng.normal(size=(64,9)),'clip_scales':np.ones(64)}
    previous=m.base.base.fit_map(head,x,clip,old)
    torch.manual_seed(80);net=m.base.manifold.build_network(9,8,3)
    rep=m.base.manifold.export_network(net)|{'real_center':np.zeros(9),'real_scale':np.ones(9),
        'residual_center':np.zeros(9),'residual_scale':np.ones(9),'residual_mean':np.zeros(9),
        'residual_components':rng.normal(size=(64,9)),'residual_scales':np.ones(64)}
    previous=m.base.assemble(previous,rep);previous['weights'][:]=10
    return x,clip,dear,head,previous,m.fit_map(previous,dear)


def test_exact_old_map_and_zero_teacher_replay_without_old_weights():
    x,clip,dear,head,previous,a=fixture()
    assert len(a['weights'])==385 and not a['weights'].any()
    assert np.array_equal(m.predict(head,x,clip,dear,a),head.predict_proba(x)[:,1])
    z=m.project(head,x,clip,dear,a)
    assert np.array_equal(z[:,:320],m.base.project(head,x,clip,previous)[:,:-1])
    np.testing.assert_allclose(z[:,320:384].mean(axis=0),0,atol=1e-14)
    # Randomized SVD gives approximate explained variance; its basis is orthonormal.
    np.testing.assert_allclose(a['dear_components']@a['dear_components'].T,np.eye(64),atol=1e-12)
    assert previous['weights'].sum()==3210
    for k in m.BASE_KEYS:assert not np.shares_memory(a[k],previous[k])


def test_serialized_forensic_map_batch_parity_and_alignment(tmp_path):
    x,clip,dear,head,_,a=fixture();a['weights'][325]=.1
    path=tmp_path/'candidate.npz';np.savez_compressed(path,**a);expected=m.predict(head,x,clip,dear,a)
    with np.load(path,allow_pickle=False) as saved:
        assert np.array_equal(m.predict(head,x,clip,dear,saved),expected)
        actual=np.concatenate([m.predict(head,x[i:i+7],clip[i:i+7],dear[i:i+7],saved) for i in range(0,len(x),7)])
        np.testing.assert_allclose(actual,expected,atol=1e-7,rtol=0)
        assert np.array_equal(actual>=m.AI_CUT,expected>=m.AI_CUT)
    with pytest.raises(ValueError,match='aligned'):m.project(head,x,clip,dear[:-1],a)


def test_full_training_guard_rejects_hidden_small_source_failure():
    rows=[{'parent_id':f'p{i}','label':int(i>=100),'source':'small' if i==0 else 'large'} for i in range(200)]
    rows += [{'parent_id':f'n{i}','label':0,'source':'MIDD:A'} for i in range(20)]
    scores=np.zeros((220,3));scores[100:200]=1;scores[0]=1
    result=full_training_gates(rows,scores,200)
    for values in result.values():
        for value in values.values():
            checks=value['gate']['checks']
            assert checks['pooled_real_false_ai_lte_0_10'] and checks['covered_accuracy_gte_0_95']
            assert not checks['worst_source_real_false_ai_lte_0_20'] and not value['gate']['passed']


def test_low_false_ai_does_not_hide_excessive_uncertainty():
    rows=[{'parent_id':f'p{i}','label':int(i>=100),'source':'real' if i<100 else 'AI'} for i in range(200)]
    rows += [{'parent_id':f'n{i}','label':0,'source':'MIDD:A'} for i in range(20)]
    scores=np.full((220,3),.02);scores[100:200]=.9
    result=full_training_gates(rows,scores,200)
    for values in result.values():
        for value in values.values():
            checks=value['gate']['checks']
            assert checks['pooled_real_false_ai_lte_0_10']
            assert not checks['automatic_coverage_gte_0_80'] and not checks['uncertain_rate_lte_0_20']
    with pytest.raises(ValueError,match='pairing'):full_training_gates(rows,scores[:-1],200)


def test_tiny_numeric_error_cannot_hide_runtime_decision_flip():
    from experiments.e80_fit import runtime_checks
    labels=np.array([0,1]);shifts=np.zeros(2)
    for cut in [m.AI_CUT,m.REAL_CUT]:
        expected=np.array([cut-1e-10,.9]);actual=np.array([cut+1e-10,.9])
        result=runtime_checks(expected,actual,shifts,labels)
        assert result['max_score_error']<1e-6 and not result['passed']
    expected=np.array([0.,.9])
    assert runtime_checks(expected,expected,shifts,labels)['passed']
    assert not runtime_checks(expected,expected,np.array([0.,-2e-8]),labels)['passed']
