import importlib.util
import sys
from pathlib import Path
import pytest


def load(monkeypatch, tmp_path):
    monkeypatch.setenv('PIXELPROOF_DATA_ROOT', str(tmp_path))
    tools = Path(__file__).resolve().parents[1]/'tools'
    monkeypatch.syspath_prepend(str(tools))
    spec = importlib.util.spec_from_file_location('test_e119_runner', tools/'run_e119_after_model1.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def test_waits_for_both_gpu_workflows_and_accepts_scientific_train_skip(monkeypatch, tmp_path):
    m = load(monkeypatch, tmp_path)
    first = {'state': 'complete', 'E92_restored': True}
    assert not m.ready({'state': 'running'}, {'state': 'waiting'})
    assert not m.ready(first, {'state': 'running'})
    assert m.ready(first, {'state': 'skipped_train_failed'})
    assert m.ready(first, {'state': 'complete', 'E92_restored_after_DEV': True})


@pytest.mark.parametrize('first,second', [
    ({'state':'failed'}, {}),
    ({'state':'complete','E92_restored':False}, {'state':'skipped_train_failed'}),
    ({'state':'complete','E92_restored':True}, {'state':'complete','E92_restored_after_DEV':False}),
])
def test_failure_never_releases_pilot(monkeypatch, tmp_path, first, second):
    with pytest.raises(RuntimeError): load(monkeypatch,tmp_path).ready(first,second)
