import hashlib
import json

import numpy as np
import pytest
import torch
from PIL import Image

import pixelproof.project_model as project_model
from pixelproof.project_model import ProjectModelContractError, load_project_model


class TinyModel(torch.nn.Module):
    def forward(self, images):
        return images.mean(dim=(1, 2, 3))


def write_checkpoint(root, *, arm="resnet18", inference=None):
    artifacts = root / "artifacts"
    artifacts.mkdir(exist_ok=True)
    path = artifacts / "project.pt"
    payload = {
        "arm": arm,
        "seed": 2024,
        "model": TinyModel().state_dict(),
        "training": {"validation_auc": 0.91},
        "inference": inference or {
            "tile_px": 128,
            "texture_floor": 0.04,
            "normalization": "imagenet",
            "selected_aggregation": "top3",
            "threshold": 0.75,
            "calibration_fraction": 0.5,
            "split_seed": 2026,
        },
    }
    torch.save(payload, path)
    return path


def write_manifest(root, path, digest=None):
    expected = digest or hashlib.sha256(path.read_bytes()).hexdigest()
    (root / "artifacts.manifest.json").write_text(json.dumps({
        "schema_version": 1,
        "artifacts": [{
            "id": project_model.ARTIFACT_ID,
            "kind": "file",
            "group": project_model.ARTIFACT_GROUP,
            "path": str(path.relative_to(root)),
            "sha256": expected,
            "revision": "test",
        }],
    }))


def test_valid_checkpoint_loads_metadata_scores_and_aggregation(tmp_path, monkeypatch):
    path = write_checkpoint(tmp_path)
    write_manifest(tmp_path, path)
    monkeypatch.setattr(project_model, "create_model", lambda *args, **kwargs: TinyModel())

    loaded = load_project_model(tmp_path, torch.device("cpu"))
    scores = loaded.score_tiles([
        Image.fromarray(np.zeros((128, 128, 3), dtype=np.uint8)),
        Image.fromarray(np.full((128, 128, 3), 255, dtype=np.uint8)),
    ])

    assert loaded.metadata.seed == 2024
    assert loaded.metadata.aggregation == "top3"
    assert loaded.metadata.sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    assert scores.shape == (2,)
    assert np.all(np.isfinite(scores))
    assert loaded.aggregate(scores) == pytest.approx(scores.mean())
    assert loaded.triggered(0.75)


def test_image_scoring_uses_checkpoint_policy_and_respects_tile_cap(tmp_path, monkeypatch):
    path = write_checkpoint(tmp_path)
    write_manifest(tmp_path, path)
    monkeypatch.setattr(project_model, "create_model", lambda *args, **kwargs: TinyModel())
    loaded = load_project_model(tmp_path, torch.device("cpu"))

    result = loaded.score_image(Image.new("RGB", (2304, 2304)), max_tiles=16)

    # The spatial sampler is capped at sixteen; the checkpoint's texture floor
    # may conservatively reduce that set further (this flat fixture falls back to three).
    assert 0 < result.tile_count <= 16
    assert len(result.positions) == len(result.textures) == len(result.tile_scores)
    assert result.score == pytest.approx(loaded.aggregate(result.tile_scores))


def test_missing_checkpoint_is_rejected_before_deserialization(tmp_path):
    missing = tmp_path / "artifacts/project.pt"
    write_manifest(tmp_path, missing, "0" * 64)

    with pytest.raises(ProjectModelContractError, match="missing"):
        load_project_model(tmp_path)


def test_tampered_checkpoint_is_rejected_before_deserialization(tmp_path):
    path = write_checkpoint(tmp_path)
    expected = hashlib.sha256(path.read_bytes()).hexdigest()
    write_manifest(tmp_path, path, expected)
    path.write_bytes(b"not the verified checkpoint")

    with pytest.raises(ProjectModelContractError, match="SHA-256 mismatch"):
        load_project_model(tmp_path)


def test_incompatible_checkpoint_contract_is_rejected(tmp_path):
    path = write_checkpoint(tmp_path, arm="small_cnn")
    write_manifest(tmp_path, path)

    with pytest.raises(ProjectModelContractError, match="checkpoint arm"):
        load_project_model(tmp_path)
