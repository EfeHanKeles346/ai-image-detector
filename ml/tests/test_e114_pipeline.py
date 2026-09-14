import importlib.util
import json
from pathlib import Path
import time
import pytest


def load_runner(tmp_path, monkeypatch):
    folder = Path(__file__).parents[1]/'tools'
    monkeypatch.syspath_prepend(str(folder))
    monkeypatch.setenv('PIXELPROOF_DATA_ROOT', str(tmp_path))
    spec = importlib.util.spec_from_file_location('dependent_context_pipeline_test', folder/'run_e114_after_training.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    (tmp_path/'e112_pipeline').mkdir()
    return module


@pytest.mark.parametrize('state,restored', [('failed', True), ('complete', False)])
def test_upstream_failure_or_unrestored_api_denies_dependent_job(tmp_path, monkeypatch, state, restored):
    module = load_runner(tmp_path, monkeypatch)
    (tmp_path/'e112_pipeline/status.json').write_text(json.dumps({'state': state, 'E92_restored': restored}))
    with pytest.raises(RuntimeError): module.wait_training(time.monotonic()+1)


def test_only_complete_upstream_lifecycle_releases_dependency(tmp_path, monkeypatch):
    module = load_runner(tmp_path, monkeypatch)
    (tmp_path/'e112_pipeline/status.json').write_text(json.dumps({'state': 'complete', 'E92_restored': True}))
    assert module.wait_training(time.monotonic()+1) is None


def test_dependency_wait_is_bounded(tmp_path, monkeypatch):
    module = load_runner(tmp_path, monkeypatch)
    with pytest.raises(TimeoutError): module.wait_training(time.monotonic()-1)
