import hashlib
import json

import pytest

from experiments.e55_resume import pending_stages,power_safe,verify_pause,stop_owned_child


@pytest.mark.parametrize('text,expected',[
    ("Now drawing from 'AC Power'\n 1%; charging; present: true",True),
    ("Now drawing from 'AC Power'\n 100%; charged; present: true",True),
    ("Now drawing from 'Battery Power'\n 99%; discharging",False),
    ("Now drawing from 'AC Power'\n 3%; discharging",False),
    ("Now drawing from 'AC Power'\n unknown",False),
    ("Now drawing from 'AC Power'\n 0%; charging",False),
])
def test_power_is_fail_closed(text,expected):
    assert power_safe(text)==expected


def test_stage_selection_does_not_redo_completed_fits(tmp_path):
    assert pending_stages(tmp_path)==['extract','fit','report']
    (tmp_path/'grayscale_features.npz').touch();(tmp_path/'features.json').touch()
    assert pending_stages(tmp_path)==['fit','report']
    (tmp_path/'training').mkdir()
    for arm in ('duplicate_control','grayscale_20'):
        for fold in range(3):(tmp_path/'training'/f'{arm}_fold{fold}.json').touch()
    assert pending_stages(tmp_path)==['report']
    (tmp_path/'result.json').touch()
    assert pending_stages(tmp_path)==[]


@pytest.mark.parametrize('name',['features.json','grayscale_features.npz'])
def test_partial_finalization_needs_review(tmp_path,name):
    (tmp_path/name).touch()
    with pytest.raises(ValueError):pending_stages(tmp_path)


def test_pause_hashes_verified_and_path_traversal_rejected(tmp_path):
    (tmp_path/'contract.json').write_bytes(b'contract');(tmp_path/'chunks').mkdir()
    chunk=tmp_path/'chunks/000000.npz';chunk.write_bytes(b'content')
    receipt={'contract_sha256':hashlib.sha256(b'contract').hexdigest(),'completed_views':48,
             'chunks':[{'file':'000000.npz','sha256':hashlib.sha256(b'content').hexdigest(),'bytes':7,'views':48}]}
    path=tmp_path/'pause.json';path.write_text(json.dumps(receipt))
    assert verify_pause(tmp_path,path)==48
    chunk.write_bytes(b'changed')
    with pytest.raises(ValueError):verify_pause(tmp_path,path)
    receipt['chunks'][0]['file']='../000000.npz';path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError):verify_pause(tmp_path,path)


def test_stop_only_signals_the_owned_live_child(monkeypatch):
    calls=[]
    monkeypatch.setattr('experiments.e55_resume.os.killpg',lambda pid,sig:calls.append(pid))
    class Child:
        pid=123456
        def poll(self):return None
        def wait(self,timeout):return -15
    stop_owned_child(Child())
    assert calls==[123456]
    calls.clear()
    class Done(Child):
        def poll(self):return 0
    stop_owned_child(Done())
    assert calls==[]
