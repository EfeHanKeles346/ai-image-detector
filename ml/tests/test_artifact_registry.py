import hashlib
import json

from pixelproof.artifact_registry import verify_registry
from pixelproof.serve import ModelRuntime


def write_manifest(root, artifacts):
    (root / "artifacts.manifest.json").write_text(json.dumps({
        "schema_version": 1,
        "artifacts": artifacts,
    }))


def test_file_artifact_hash_is_verified_offline(tmp_path):
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    model = artifacts / "model.bin"
    model.write_bytes(b"frozen model")
    expected = hashlib.sha256(model.read_bytes()).hexdigest()
    entry = {
        "id": "model",
        "kind": "file",
        "group": "core",
        "path": "artifacts/model.bin",
        "sha256": expected,
    }
    write_manifest(tmp_path, [entry])

    assert verify_registry(tmp_path, groups={"core"}) == {
        "ok": True,
        "checked": ["model"],
        "issues": [],
    }

    model.write_bytes(b"tampered")
    report = verify_registry(tmp_path, groups={"core"})
    assert report["ok"] is False
    assert "SHA-256 mismatch" in report["issues"][0]


def test_optional_artifacts_are_checked_only_when_requested(tmp_path):
    write_manifest(tmp_path, [{
        "id": "optional-model",
        "kind": "file",
        "group": "optional",
        "optional": True,
        "path": "artifacts/missing.bin",
        "sha256": "0" * 64,
    }])
    assert verify_registry(tmp_path)["ok"] is True
    report = verify_registry(tmp_path, include_optional=True)
    assert report["ok"] is False
    assert "missing" in report["issues"][0]


def test_missing_core_artifacts_yield_actionable_health_not_import_traceback(tmp_path):
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    write_manifest(tmp_path, [{
        "id": "missing-core",
        "kind": "file",
        "group": "core",
        "path": "artifacts/missing.pt",
        "sha256": "0" * 64,
    }])

    runtime = ModelRuntime(artifacts)
    assert runtime.ensure_loaded() is False
    health = runtime.health()
    assert health["status"] == "unavailable"
    assert health["core_ready"] is False
    assert "missing" in health["load_errors"]["artifact_manifest"]
