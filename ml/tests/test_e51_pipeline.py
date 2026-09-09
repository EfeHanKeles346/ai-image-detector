import hashlib
from io import BytesIO

import numpy as np
from PIL import Image
import pytest

from experiments.e51_admission import filter_parents
from experiments.e51_pipeline import (
    choose_real_cut, metrics, prepare, real_safe_threshold, views,
)


def population():
    y = np.tile(np.r_[np.zeros(20,dtype=int),np.ones(20,dtype=int)],2)
    scores = np.tile(np.r_[np.linspace(.01,.2,20),np.linspace(.8,.99,20)],2)
    groups = np.tile(np.r_[np.repeat(['camera1','camera2'],10),np.repeat(['ai1','ai2'],10)],2)
    conditions = np.repeat(['original','q75'],40)
    return y,scores,groups,conditions


def test_real_budget_is_enforced_for_both_transports_and_ties():
    y,s,g,c = population()
    s[40:60] = .9
    cut = real_safe_threshold(y,s,g,c)
    assert cut > .9
    for condition in set(c):
        real = (y==0)&(c==condition)
        assert (s[real]>=cut).mean()<=.10
        for group in set(g[real]):
            assert (s[real&(g==group)]>=cut).mean()<=.20


def test_no_detection_is_not_a_successful_real_safe_model():
    y,s,g,c = population()
    result = metrics(y,s,g,1.01,1.0)
    assert result['real_false_ai']==0
    assert result['ai_recall']==0
    assert not result['passed']


def test_cal_band_coverage_and_accuracy_reproduce():
    y,s,g,c = population()
    cut = real_safe_threshold(y,s,g,c)
    real_cut = choose_real_cut(y,s,c,cut)
    assert 0<=real_cut<=cut
    for condition in set(c):
        mask = c==condition
        result = metrics(y[mask],s[mask],g[mask],cut,real_cut)
        assert result['covered_accuracy']>=.95
        assert result['automatic_coverage']>=.8
        assert result['passed']


@pytest.mark.parametrize('bad',[float('nan'),float('inf'),-.1,1.1])
def test_invalid_scores_fail_closed(bad):
    y,s,g,c = population(); s[0] = bad
    with pytest.raises(ValueError): real_safe_threshold(y,s,g,c)
    with pytest.raises(ValueError): metrics(y,s,g,.5,.2)


def test_missing_class_in_one_transport_fails_closed():
    y,s,g,c = population(); y[c=='q75'] = 0
    with pytest.raises(ValueError): real_safe_threshold(y,s,g,c)


def test_parent_exclusion_removes_every_cal_child_and_preserves_inputs():
    train = [{'parent_id':f't{i}'} for i in range(20)]
    cal = [{'parent_id':f'c{i}','condition':c} for i in range(20) for c in ('original','q75')]
    t,k = filter_parents(train,cal,{'t0','c0'})
    assert len(t)==19 and len(k)==38 and len(cal)==40
    with pytest.raises(ValueError): filter_parents(train,cal,{'t0','t1'})
    with pytest.raises(ValueError): filter_parents(train,cal,{'unknown'})


def test_three_train_views_and_two_bound_cal_views():
    rows = views({'train':[{'parent_id':'t'}],
                  'cal':[{'parent_id':'c','condition':c} for c in ('original','q75')]})
    assert len(rows)==len({r['record_id'] for r in rows})==5
    assert {r['condition'] for r in rows if r['role']=='TRAIN'} >= {'clean','q75'}


def test_existing_cal_q75_is_not_compressed_twice(tmp_path):
    raw = BytesIO()
    Image.fromarray(np.random.default_rng(1).integers(0,256,(250,250,3),dtype=np.uint8)).save(raw,format='JPEG',quality=75)
    path = tmp_path/'image.jpg'; path.write_bytes(raw.getvalue())
    row = {'path':str(path),'sha256':hashlib.sha256(raw.getvalue()).hexdigest(),'parent_id':'test','role':'CAL'}
    original = prepare({**row,'condition':'original'})
    q75 = prepare({**row,'condition':'q75'})
    assert np.array_equal(original[0],q75[0])
    assert np.array_equal(original[1],q75[1])
    with pytest.raises(ValueError): prepare({**row,'condition':'original','sha256':'wrong'})
