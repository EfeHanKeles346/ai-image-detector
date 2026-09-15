import importlib.util
import json
from pathlib import Path
import pytest


def load(monkeypatch,tmp_path):
    monkeypatch.setenv('PIXELPROOF_DATA_ROOT',str(tmp_path))
    folder=Path(__file__).resolve().parents[1]/'tools';monkeypatch.syspath_prepend(str(folder))
    spec=importlib.util.spec_from_file_location('registered_runner_test',folder/'run_registered_research.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def test_every_upstream_lifecycle_must_release_resources(monkeypatch,tmp_path):
    m=load(monkeypatch,tmp_path)
    first={'state':'complete','E92_restored':True}
    second={'state':'complete','E92_restored_after_DEV':True}
    third={'state':'complete','E92_restored_after_pilot':True}
    m.upstream_ready(first,second,third)
    m.upstream_ready(first,{'state':'skipped_train_failed'},third)
    for a,b,c in [({},second,third),(first,{},third),(first,second,{}),
                  ({'state':'complete','E92_restored':False},second,third),
                  (first,second,{'state':'complete','E92_restored_after_pilot':False})]:
        with pytest.raises(RuntimeError):m.upstream_ready(a,b,c)


@pytest.mark.parametrize('stage_name,method',[('e136_pipeline','model1_ablation_ready'),
                                           ('e137_pipeline','model1_source_risk_ready')])
def test_model1_waits_for_preceding_resources(monkeypatch,tmp_path,stage_name,method):
    m=load(monkeypatch,tmp_path)
    stage=tmp_path/stage_name;stage.mkdir();status=stage/'status.json'
    for value in ({'state':'running','E92_restored':True},
                  {'state':'complete','E92_restored':False},{}):
        status.write_text(json.dumps(value))
        with pytest.raises(RuntimeError):getattr(m,method)(tmp_path)
    status.write_text(json.dumps({'state':'complete','E92_restored':True}))
    getattr(m,method)(tmp_path)
