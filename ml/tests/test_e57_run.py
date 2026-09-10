import pytest
from types import SimpleNamespace
from experiments import e57_run


def test_pending_phases_and_orphan_refusal(tmp_path,monkeypatch):
    monkeypatch.setattr(e57_run,'ROOT',tmp_path)
    monkeypatch.setattr(e57_run,'MANIFEST',tmp_path/'admitted.json')
    assert [phase for _,phase in e57_run.pending()]==['audit','freeze','extract','fit','report']
    (tmp_path/'admitted.json').touch();(tmp_path/'model_contract.json').touch()
    (tmp_path/'fivek_features.npz').touch()
    with pytest.raises(ValueError,match='orphaned'):e57_run.pending()
    (tmp_path/'features.json').touch()
    assert [phase for _,phase in e57_run.pending()]==['fit','report']
    (tmp_path/'training').mkdir()
    for arm in ('native64','native64_fivek'):
        for fold in range(3):(tmp_path/'training'/f'{arm}_fold{fold}.json').touch()
    (tmp_path/'result.json').touch()
    assert e57_run.pending()==[]


def test_wait_requires_complete_receipts_not_just_exited_process(tmp_path,monkeypatch):
    monkeypatch.setattr(e57_run,'ROOT',tmp_path)
    monkeypatch.setattr(e57_run,'EVIDENCE',tmp_path)
    monkeypatch.setattr(e57_run,'safe',lambda deadline:None)
    monkeypatch.setattr(e57_run.subprocess,'run',lambda *args,**kwargs:SimpleNamespace(returncode=1,stdout=''))
    with pytest.raises(RuntimeError,match='without completed receipt'):e57_run.wait_acquisition(123,1)
    (tmp_path/'download.json').touch();(tmp_path/'e57_v2_download.json').touch()
    e57_run.wait_acquisition(123,1)


def test_wait_rejects_reused_process_id(tmp_path,monkeypatch):
    monkeypatch.setattr(e57_run,'safe',lambda deadline:None)
    monkeypatch.setattr(e57_run.subprocess,'run',lambda *args,**kwargs:SimpleNamespace(returncode=0,stdout='unrelated command'))
    with pytest.raises(RuntimeError,match='different process'):e57_run.wait_acquisition(123,1)
