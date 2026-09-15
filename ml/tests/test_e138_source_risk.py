import json
import numpy as np
import pytest
from experiments import e138_source_risk as e


def test_error_guards_preserve_individuals_despite_same_total_accuracy():
    r=e.transitions([0,0,1,1],[.1,.7,.2,.9],[.7,.1,.9,.2])
    assert r['0']=={'new_errors':1,'rescued_errors':1}
    assert r['1']=={'new_errors':1,'rescued_errors':1}
    assert r['zero_new_AI_misses'] is False and r['zero_new_REAL_false_alerts'] is False
    with pytest.raises(ValueError):e.transitions([1],[.1],[np.nan])


def test_actual_small_fits_exclude_outer_fold_and_lock_before_metrics(tmp_path,monkeypatch):
    root=tmp_path/'e138';root.mkdir();base=tmp_path/'e131';base.mkdir();evidence=tmp_path/'evidence';evidence.mkdir()
    ml=tmp_path/'repo/ml';ml.mkdir(parents=True);contract=root/'contract.json';contract.write_text('{}')
    conditions=['clean','assigned_transport','q75','social_q75'];folds=np.repeat(np.arange(3),2)
    rows=[dict(parent_id=str(i),label=i%2,source=f'group{i//2}') for i in range(6)]
    prior=dict(rows=rows,outer_fold={str(i):int(folds[i]) for i in range(6)},components={str(i):f'g{i//2}' for i in range(6)})
    (base/'contract.json').write_text(json.dumps(prior));binding=e.digest(base/'contract.json')
    feature=np.column_stack([np.repeat(np.arange(6),4),np.repeat(np.arange(6)%2*6-3,4)+np.tile(np.arange(4)*.1,6)])
    features={k:feature.copy() for k in ('dino','clip','dear','center_control','full_frame')}
    transform=dict(center=np.zeros(2),scale=np.ones(2),mean=np.zeros(2),components=np.eye(2),latent_scale=np.ones(2))
    parameters=np.zeros(9);parameters[1]=1
    for fold in range(3):
        shared={name+'_'+k:v for name in ('dino','clip','dear') for k,v in transform.items()}
        np.savez(base/f'fold{fold}_shared.npz',**shared,contract_sha256=binding)
        for branch in e.BRANCHES:np.savez(base/f'fold{fold}_{branch}.npz',**transform,parameters=parameters,contract_sha256=binding)
    baseline=e.holdout_linear.predict(np.column_stack([feature]*4),parameters).reshape(6,4)
    np.savez(base/'scores.npz',**{b:baseline for b in e.BRANCHES},parents=np.array([str(i) for i in range(6)]),
             conditions=conditions,outer_fold=folds,global_role='TRAIN',usage='INTERNAL_HELD_OUT',contract_sha256=binding)
    for name,value in [('ROOT',root),('CONTRACT',contract),('EVIDENCE',evidence),('ML_ROOT',ml)]:monkeypatch.setattr(e,name,value)
    monkeypatch.setattr(e.previous,'ROOT',base);monkeypatch.setattr(e.previous,'CONTRACT',base/'contract.json')
    monkeypatch.setattr(e.previous,'load_features',lambda rows:features)
    monkeypatch.setattr(e,'validate',lambda:dict(max_seconds=60,conditions=conditions,limits='synthetic'))
    monkeypatch.setattr(e,'resource_check',lambda *a:None)
    fits=[];actual_fit=e.group_robust_linear.fit_head
    def fit(x,*args,**kwargs):
        fits.append(set(x[:,0].astype(int).tolist()))
        assert set(args[1].tolist())=={f'g{int(v)//2}' for v in x[:,0]}
        return actual_fit(x,*args,**kwargs)
    monkeypatch.setattr(e.group_robust_linear,'fit_head',fit)
    actual_metrics=e.previous.metrics
    def metrics(*args,**kwargs):
        assert (root/'locked_scores.json').exists();return actual_metrics(*args,**kwargs)
    monkeypatch.setattr(e.previous,'metrics',metrics)
    report=e.fit()
    for i,selected in enumerate(fits):assert selected==set(np.where(folds!=i//2)[0])
    assert len(fits)==6 and report['promotion_allowed'] is False
    assert report['parents']==6
    with pytest.raises(FileExistsError):e.fit()
