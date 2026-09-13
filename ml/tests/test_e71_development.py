import json
import numpy as np
import pytest
from experiments import e71_development as d
from experiments.e71_features import aggregate


def test_dev_clip_aggregation_exactly_matches_frozen_train_operation():
    raw=np.random.default_rng(71).normal(size=(3,3,768)).astype(np.float32)
    assert np.array_equal(np.stack([d.clip_aggregate(r) for r in raw]),aggregate(raw))
    with pytest.raises(ValueError):d.clip_aggregate(raw[0].astype(np.float64))


def test_failed_train_guard_blocks_consumed_dev(tmp_path,monkeypatch):
    payload=json.dumps({'dev_scoring_permitted':False})
    report=tmp_path/'fit.json';report.write_text(payload)
    (tmp_path/'e71_fit.json').write_text(payload)
    monkeypatch.setattr(d,'validate_fit',lambda:{})
    monkeypatch.setattr(d,'FIT_REPORT',report);monkeypatch.setattr(d,'EVIDENCE',tmp_path)
    monkeypatch.setattr(d,'CONTRACT',tmp_path/'dev_contract.json')
    with pytest.raises(ValueError,match='TRAIN guard failed'):d.freeze()
    assert not d.CONTRACT.exists()


def test_reference_replay_requires_pixels_and_both_cut_decisions():
    row={'sha256':'original'};previous=row|{'reference_score':.5}
    assert d.check_reference(row,.5+1e-8,previous,1e-6)<1e-6
    with pytest.raises(ValueError,match='pixel'):d.check_reference({'sha256':'changed'},.5,previous,1e-6)
    with pytest.raises(ValueError,match='score'):d.check_reference(row,.51,previous,1e-6)
    with pytest.raises(ValueError,match='score'):d.check_reference(row,np.nan,previous,1e-6)
    for cut in [d.model.AI_CUT,d.model.REAL_CUT]:
        with pytest.raises(ValueError,match='decision'):
            d.check_reference(row,cut-1e-9,row|{'reference_score':cut+1e-9},1e-6)
