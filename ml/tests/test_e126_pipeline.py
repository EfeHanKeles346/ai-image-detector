import importlib.util
import json
from pathlib import Path
import time
import pytest


def load_runner(tmp_path, monkeypatch):
    folder = Path(__file__).parents[1]/'tools'
    monkeypatch.syspath_prepend(str(folder))
    monkeypatch.setenv('PIXELPROOF_DATA_ROOT', str(tmp_path))
    spec = importlib.util.spec_from_file_location('dependent_fullframe_pipeline_test', folder/'run_e126_after_training.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    (tmp_path/'e124_pipeline').mkdir()
    return module


@pytest.mark.parametrize('state,restored', [('failed', True), ('complete', False)])
def test_upstream_failure_or_unrestored_api_denies_dependent_job(tmp_path, monkeypatch, state, restored):
    module = load_runner(tmp_path, monkeypatch)
    (tmp_path/'e124_pipeline/status.json').write_text(json.dumps({'state': state, 'E92_restored': restored}))
    with pytest.raises(RuntimeError): module.wait_training(time.monotonic()+1)


def test_only_complete_upstream_lifecycle_releases_dependency(tmp_path, monkeypatch):
    module = load_runner(tmp_path, monkeypatch)
    (tmp_path/'e124_pipeline/status.json').write_text(json.dumps({'state': 'complete', 'E92_restored': True}))
    assert module.wait_training(time.monotonic()+1) is None


def test_dependency_wait_is_bounded(tmp_path, monkeypatch):
    module = load_runner(tmp_path, monkeypatch)
    with pytest.raises(TimeoutError): module.wait_training(time.monotonic()-1)


def test_restored_model_name_alone_is_not_artifact_proof(tmp_path, monkeypatch):
    module = load_runner(tmp_path, monkeypatch)
    current = {'status': 'ready', 'model_id': 'E92', 'artifact_sha256': module.E92_SHA}
    assert module.ready_e92(current)
    assert not module.ready_e92(current | {'artifact_sha256': 'different'})
    assert not module.ready_e92(None)
