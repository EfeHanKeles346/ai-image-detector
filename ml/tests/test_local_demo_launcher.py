import importlib.util
import json
from pathlib import Path
import pytest


def load():
    path = Path(__file__).resolve().parents[1] / 'tools/serve_local_demo.py'
    spec = importlib.util.spec_from_file_location('demo_launcher_test', path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def test_preflight_fails_without_assets_and_sets_offline_cors(tmp_path, monkeypatch):
    m = load(); monkeypatch.setattr(m, 'REPO', tmp_path)
    (tmp_path / 'evidence').mkdir(); data = tmp_path / 'data'; data.mkdir()
    (tmp_path / 'evidence/e93_runtime_manifest.json').write_text(json.dumps({'data_files':{'weight':'sha'}}))
    with pytest.raises(ValueError): m.environment(data)
    (data / 'weight').write_bytes(b'fixture')
    env = m.environment(data)
    assert env['HF_HUB_OFFLINE'] == env['TRANSFORMERS_OFFLINE'] == '1'
    assert env['PIXELPROOF_DATA_ROOT'] == str(data)
    assert env['PIXELPROOF_CORS_ORIGINS'] == m.ORIGINS


def test_reuse_requires_exact_identity_policy_and_cors():
    m = load()
    correct = dict(status='ready', model_id='E92', artifact_sha256=m.EXPECTED_SHA,
        guard_id='e92-paired-v2', display_policy='e92-primary-reference-advisory-v2', research_only=True,
        downloads_allowed=False, _cors_ok=True)
    assert m.ready(correct)
    for k in correct:
        wrong = dict(correct); wrong.pop(k)
        assert not m.ready(wrong)
    assert not m.ready(None)


def test_unrelated_listener_is_not_replaced(monkeypatch):
    m = load(); monkeypatch.setattr(m, 'environment', lambda root: {})
    class Probe:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def settimeout(self, seconds): pass
        def connect_ex(self, address): return 0
    monkeypatch.setattr(m.socket, 'socket', Probe)
    monkeypatch.setattr(m, 'health', lambda: {'status':'ready','model_id':'unrelated'})
    def forbidden(*args, **kwargs): raise AssertionError('Unrelated service must not be replaced')
    monkeypatch.setattr(m.subprocess, 'Popen', forbidden)
    with pytest.raises(RuntimeError, match='Mevcut süreç durdurulmadı'): m.main('/unused')
