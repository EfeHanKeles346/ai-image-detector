import json
import numpy as np
import pytest
from experiments import e69_development as d
from experiments.e69_features import shuffle_crops


def test_dev_shuffle_exactly_matches_frozen_train_transform():
    crops=np.random.default_rng(67).integers(0,256,(3,224,224,3),dtype=np.uint8)
    cached=np.stack([crops,crops,crops])
    assert np.array_equal(np.stack(d.shuffle_view(crops)),shuffle_crops(cached)[0])


def test_dev_transport_coverage_and_role_are_mandatory():
    manifest=[dict(parent_id='r',label=0,source='SIDD:GP'),dict(parent_id='a',label=1,source='GPT')]
    rows=[r|{'condition':c,'role':'DEVELOPMENT'} for r in manifest for c in d.CONDITIONS]
    d.validate_pairs(rows,manifest)
    with pytest.raises(ValueError,match='coverage'):d.validate_pairs(rows[:-1],manifest)
    with pytest.raises(ValueError,match='duplicate'):d.validate_pairs(rows[:-1]+[rows[0]],manifest)
    with pytest.raises(ValueError,match='role'):d.validate_pairs([rows[0]|{'role':'TRAIN'},*rows[1:]],manifest)


def test_another_ai_rescue_cannot_hide_a_new_ai_miss():
    rows=[dict(label=1,source='GPT',reference_score=o,score=n) for o,n in [(1,0),(0,1)]]
    result=d.transitions(rows)['1:GPT']
    assert result['old_ai_rate']==result['new_ai_rate']==.5
    assert result['new_ai_misses']==1


def test_failed_train_guard_cannot_create_dev_contract(tmp_path,monkeypatch):
    payload=json.dumps({'dev_scoring_permitted':False})
    report=tmp_path/'fit.json';report.write_text(payload)
    (tmp_path/'e69_fit.json').write_text(payload)
    monkeypatch.setattr(d,'validate_fit',lambda:{})
    monkeypatch.setattr(d,'FIT_REPORT',report);monkeypatch.setattr(d,'EVIDENCE',tmp_path)
    monkeypatch.setattr(d,'CONTRACT',tmp_path/'dev_contract.json')
    with pytest.raises(ValueError,match='TRAIN guard failed'):d.freeze()
    assert not d.CONTRACT.exists()
