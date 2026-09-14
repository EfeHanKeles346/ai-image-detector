import numpy as np
import pytest
from experiments.e131_source_holdout import metrics


def test_fixed_cut_counts_and_no_vacuous_single_class_auc():
    result=metrics(np.array([0,0,1,1]),np.array([0,.5,.8,.2]))
    assert [result[k] for k in ('tp','fp','fn','tn')]==[1,1,1,1]
    assert result['REAL_FPR']==result['AI_recall']==.5
    negative=metrics(np.array([0,0]),np.array([0.,.2]))
    assert negative['AI_recall'] is None and negative['auc'] is None


@pytest.mark.parametrize('score',[np.array([np.nan,.3]),np.array([1.1,.3]),np.array([.2])])
def test_invalid_holdout_scores_rejected(score):
    with pytest.raises(ValueError):metrics(np.array([0,1]),score)


def test_small_end_to_end_fit_excludes_every_heldout_group_and_locks_before_metrics(monkeypatch,tmp_path):
    import json
    from experiments import e131_source_holdout as m
    rows=[{'parent_id':str(i),'source':'s'+str(i),'role':'TRAIN','label':i%2} for i in range(12)]
    assignment={str(i):i//4 for i in range(12)}
    c={'rows':rows,'outer_fold':assignment,'components':{str(i):str(i) for i in range(12)},
       'max_seconds':100,'components_count':12,'fold_counts':[{'0':2,'1':2}]*3,'limits':'synthetic test',
       'corpus_level_overlap_limit':'synthetic test; not a family-independent benchmark'}
    run=tmp_path/'run';run.mkdir();ev=tmp_path/'evidence';ev.mkdir();ml=tmp_path/'ml';ml.mkdir()
    contract=run/'contract.json';contract.write_text(json.dumps(c))
    monkeypatch.setattr(m,'ROOT',run);monkeypatch.setattr(m,'EVIDENCE',ev);monkeypatch.setattr(m,'ML_ROOT',ml)
    monkeypatch.setattr(m,'CONTRACT',contract);monkeypatch.setattr(m,'validate',lambda:c)
    monkeypatch.setattr(m,'resource_check',lambda deadline:None)
    rng=np.random.default_rng(1)
    data={k:rng.normal(size=(48,5)).astype(np.float32) for k in ('dino','clip','dear','center_control','full_frame')}
    for value in data.values():value[:,0]=np.repeat(np.arange(12),4)
    monkeypatch.setattr(m,'load_features',lambda rs:data)
    seen=[];original_map=m.holdout_linear.fit_map
    def small_map(train,width):
        seen.append(set(train[:,0].astype(int)))
        return original_map(train,2)
    monkeypatch.setattr(m.holdout_linear,'fit_map',small_map)
    original_metrics=m.metrics
    def guarded_metrics(y,score):
        assert (run/'locked_scores.json').exists()
        return original_metrics(y,score)
    monkeypatch.setattr(m,'metrics',guarded_metrics)
    result=m.fit()
    assert result['state']=='E131_paired_internal_source_holdout_complete'
    assert len(seen)==15
    for fold in range(3):
        expected={i for i in range(12) if i//4!=fold}
        assert all(ids==expected for ids in seen[fold*5:(fold+1)*5])
    with np.load(run/'scores.npz') as archive:
        assert np.isfinite(archive['center_control']).all()
        assert np.isfinite(archive['full_frame']).all()
        np.testing.assert_array_equal(archive['outer_fold'],np.arange(12)//4)
