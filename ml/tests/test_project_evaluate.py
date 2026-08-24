import csv
import json

import numpy as np
import pytest
from PIL import Image

from pixelproof.project_evaluate import (
    EvaluationConfigurationError,
    evaluate_folders,
    write_report,
)
from pixelproof.project_model import ProjectImageScore, ProjectModelMetadata


class FakeProjectModel:
    def __init__(self):
        self.calls = 0
        self.device = "cpu"
        self.metadata = ProjectModelMetadata(
            artifact_id="fixture-model",
            sha256="a" * 64,
            revision="fixture-revision",
            arm="resnet18",
            seed=2024,
            tile_px=128,
            texture_floor=0.04,
            normalization="imagenet",
            aggregation="top3",
            threshold=0.5,
            calibration_fraction=0.5,
            split_seed=2026,
            validation_auc=0.91,
        )

    def score_image(self, picture, max_tiles=256):
        self.calls += 1
        score = float(np.asarray(picture, dtype=np.float64).mean() / 255.0)
        return ProjectImageScore(
            score=score,
            triggered=score >= self.metadata.threshold,
            tile_scores=(score,),
            textures=(0.1,),
            positions=((0, 0),),
            width=picture.width,
            height=picture.height,
        )


def save_image(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (128, 128), color=(value, value, value)).save(path)


def test_folder_evaluation_scores_each_image_once_and_reports_fixed_threshold_metrics(tmp_path):
    real = tmp_path / "real"
    ai = tmp_path / "ai"
    save_image(real / "phone" / "dark.png", 0)
    save_image(real / "phone" / "mid.png", 64)
    save_image(ai / "generator-a" / "light.png", 192)
    save_image(ai / "generator-b" / "white.webp", 255)
    model = FakeProjectModel()

    report = evaluate_folders(
        model,
        real,
        ai,
        repo_root=tmp_path,
        command=["pixelproof-evaluate-project", "--fixture"],
    )

    assert model.calls == 4
    assert report["schema_version"] == 1
    assert report["model"]["sha256"] == "a" * 64
    assert report["provenance"]["command"] == ["pixelproof-evaluate-project", "--fixture"]
    assert report["counts"]["total"] == {
        "discovered": 4,
        "succeeded": 4,
        "failed": 0,
        "read_failures": 0,
        "decode_failures": 0,
        "inference_failures": 0,
    }
    assert report["metrics"] == {
        "roc_auc": 1.0,
        "recall_at_stored_threshold": 1.0,
        "false_positive_rate_at_stored_threshold": 0.0,
        "accuracy_at_stored_threshold": 1.0,
        "confusion": {"tp": 2, "fn": 0, "fp": 0, "tn": 2},
    }
    assert {(row["class_name"], row["source"]) for row in report["per_folder"]} == {
        ("real", "phone"),
        ("ai", "generator-a"),
        ("ai", "generator-b"),
    }


def test_decode_failures_remain_in_json_and_csv_instead_of_disappearing(tmp_path):
    real = tmp_path / "real"
    ai = tmp_path / "ai"
    save_image(real / "valid.png", 0)
    save_image(ai / "valid.png", 255)
    (real / "broken.jpg").write_bytes(b"not an image")
    report = evaluate_folders(FakeProjectModel(), real, ai, repo_root=tmp_path)

    assert report["counts"]["total"]["discovered"] == 3
    assert report["counts"]["total"]["succeeded"] == 2
    assert report["counts"]["total"]["decode_failures"] == 1
    failed = [row for row in report["predictions"] if row["status"] == "error"]
    assert len(failed) == 1
    assert failed[0]["path"] == "real/broken.jpg"
    assert failed[0]["error_stage"] == "decode"

    json_path, csv_path = write_report(report, tmp_path / "report")
    stored = json.loads(json_path.read_text())
    with csv_path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert stored["counts"]["total"]["failed"] == 1
    assert len(rows) == 3
    assert sum(row["status"] == "error" for row in rows) == 1


def test_invalid_dataset_and_nonempty_output_fail_loudly(tmp_path):
    real = tmp_path / "real"
    ai = real / "ai"
    save_image(real / "real.png", 0)
    save_image(ai / "ai.png", 255)
    with pytest.raises(EvaluationConfigurationError, match="separate"):
        evaluate_folders(FakeProjectModel(), real, ai, repo_root=tmp_path)

    report_dir = tmp_path / "existing"
    report_dir.mkdir()
    (report_dir / "keep.txt").write_text("do not replace")
    with pytest.raises(EvaluationConfigurationError, match="absent or empty"):
        write_report({"predictions": []}, report_dir)
