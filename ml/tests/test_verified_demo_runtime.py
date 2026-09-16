import json
import joblib
import pytest
from pixelproof import e92_demo
from pixelproof.primary_demo_policy import PrimaryDemoEngine
from pixelproof.verified_demo_runtime import verify_manifest


def test_changed_manifest_is_rejected_before_weight_deserialization(tmp_path, monkeypatch):
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps(dict(code_files={}, data_files={}, model_id='E92', research_only=True, schema_version=1)))
    monkeypatch.setattr(e92_demo, 'MANIFEST', manifest)

    def forbidden(*args, **kwargs):
        raise AssertionError('Unverified manifest reached model deserialization')

    monkeypatch.setattr(joblib, 'load', forbidden)
    with pytest.raises(ValueError, match='manifest'):
        PrimaryDemoEngine()


def test_exact_repository_manifest_and_changed_digest(tmp_path):
    verify_manifest()
    data = json.loads(e92_demo.MANIFEST.read_text())
    data['data_files']['e92/correction.npz'] = '0' * 64
    changed = tmp_path / 'changed.json'
    changed.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='manifest'):
        verify_manifest(changed)
