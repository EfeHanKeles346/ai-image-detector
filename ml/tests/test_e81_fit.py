import numpy as np
import pytest
from experiments import e81_fit as m


def test_reuse_exact_map_without_warm_start_or_old_artifact_binding():
    old={'weights':np.full(386,10.),'dear_components':np.arange(12).reshape(3,4),
         'contract_sha256':np.array('old_contract'),'reference_sha256':np.array('reference')}
    new=m.reset_map(old)
    assert set(new)=={'weights','dear_components'} and not new['weights'].any()
    np.testing.assert_array_equal(new['dear_components'],old['dear_components'])
    assert not np.shares_memory(new['dear_components'],old['dear_components'])
    assert old['weights'].sum()==3860
    with pytest.raises(ValueError,match='complete'):m.reset_map({'weights':np.zeros(385)})


def test_freeze_requires_the_measured_source_failure_before_creating_contract(monkeypatch,tmp_path):
    monkeypatch.setattr(m,'ROOT',tmp_path)
    monkeypatch.setattr(m,'validate_previous',lambda:{})
    monkeypatch.setattr(m,'read',lambda path:{'dev_scoring_permitted':True})
    with pytest.raises(ValueError,match='source failure'):m.freeze()
    assert not list(tmp_path.iterdir())


def test_new_report_preserves_the_frozen_predecessor(monkeypatch,tmp_path):
    import json
    evidence=tmp_path/'evidence';evidence.mkdir();old=evidence/'e80_fit.json';old.write_text('previous immutable report')
    monkeypatch.setattr(m,'EVIDENCE',evidence);monkeypatch.setattr(m,'REPORT',tmp_path/'fit.json')
    m.write_report({'state':'new'})
    assert old.read_text()=='previous immutable report'
    assert json.loads((evidence/'e81_fit.json').read_text())=={'state':'new'}
    with pytest.raises(FileExistsError):m.write_report({'state':'changed'})
