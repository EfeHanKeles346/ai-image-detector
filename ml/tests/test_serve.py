import io

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image

from pixelproof.image_input import ImageLimits
from pixelproof.project_model import ProjectModelMetadata, ProjectTileModel
from pixelproof.serve import MAX_TILES, ModelRuntime, RuntimeUnavailable, create_app


class FakeRuntime:
    def __init__(self, decision_ready=True):
        self.decision_ready = decision_ready
        self.last_image = None
        self.load_calls = 0

    def ensure_loaded(self):
        self.load_calls += 1
        return True

    def health(self):
        return {
            "status": "ready" if self.decision_ready else "degraded",
            "core_ready": True,
            "decision_ready": self.decision_ready,
            "load_errors": {},
        }

    def predict(self, picture, byte_size, method):
        self.last_image = picture.copy()
        decision = None
        if self.decision_ready:
            decision = {
                "label": "insufficient",
                "triggered_by": [],
                "arms": {},
                "caveats": [],
                "bytes_per_pixel": byte_size / (picture.width * picture.height),
                "provenance": "test",
            }
        return {
            "p_ai": 0.4,
            "verdict": "uncertain",
            "method": "cnn" if method == "auto" else method,
            "method_label": "Test",
            "auto_selected": method == "auto",
            "engine": "fake",
            "resolution": "overridden-by-route",
            "enough_evidence": True,
            "tile_map": None,
            "project_model": None,
            "decision": decision,
        }


def encoded_image(mode="RGB", size=(64, 64), image_format="PNG", **save_options):
    picture = Image.new(mode, size, save_options.pop("color", 0))
    output = io.BytesIO()
    picture.save(output, format=image_format, **save_options)
    return output.getvalue()


def post(client, raw, method="auto", filename="image.png", content_type="image/png"):
    return client.post(
        "/predict",
        data={"method": method},
        files={"image": (filename, raw, content_type)},
    )


def test_invalid_bytes_and_unsupported_method_are_explicit_4xx():
    app = create_app(runtime=FakeRuntime(), allowed_origins=[])
    with TestClient(app) as client:
        assert post(client, b"not an image").status_code == 415
        response = post(client, encoded_image(), method="invented")
        assert response.status_code == 422
        assert "Bilinmeyen yöntem" in response.json()["detail"]


def test_byte_pixel_dimension_and_aspect_limits_are_enforced_before_inference():
    limits = ImageLimits(
        max_upload_bytes=128,
        max_pixels=300,
        max_dimension=100,
        max_aspect_ratio=20,
        evidence_floor_px=8,
    )
    runtime = FakeRuntime()
    app = create_app(runtime=runtime, limits=limits, allowed_origins=[])
    with TestClient(app) as client:
        assert post(client, b"x" * 129).status_code == 413
        assert post(client, encoded_image(size=(20, 20))).status_code == 413
        assert post(client, encoded_image(size=(101, 2))).status_code == 413

    aspect_limits = ImageLimits(max_upload_bytes=1_000_000, max_aspect_ratio=20)
    app = create_app(runtime=runtime, limits=aspect_limits, allowed_origins=[])
    with TestClient(app) as client:
        response = post(client, encoded_image(size=(300, 10)))
        assert response.status_code == 422
        assert "oranı" in response.json()["detail"]
    assert runtime.last_image is None


def test_exif_orientation_and_white_transparency_background_are_shared_inputs():
    runtime = FakeRuntime()
    app = create_app(runtime=runtime, allowed_origins=[])
    exif = Image.Exif()
    exif[274] = 6
    oriented = encoded_image(size=(60, 90), image_format="JPEG", exif=exif)

    with TestClient(app) as client:
        response = post(client, oriented, filename="oriented.jpg", content_type="image/jpeg")
        assert response.status_code == 200
        assert response.json()["resolution"] == "90x60"
        assert runtime.last_image.size == (90, 60)

        transparent = encoded_image(
            mode="RGBA",
            size=(64, 64),
            color=(255, 0, 0, 0),
        )
        response = post(client, transparent)
        assert response.status_code == 200
        assert runtime.last_image.mode == "RGB"
        assert runtime.last_image.getpixel((0, 0)) == (255, 255, 255)


def test_both_dimensions_gate_official_verdict_but_keep_research_signal():
    runtime = FakeRuntime()
    app = create_app(runtime=runtime, allowed_origins=[])
    with TestClient(app) as client:
        response = post(client, encoded_image(size=(40, 400)))
    assert response.status_code == 200
    payload = response.json()
    assert payload["enough_evidence"] is False
    assert payload["decision"] is None
    assert payload["p_ai"] == 0.4


def test_health_is_truthful_when_verdict_arms_are_unavailable_and_valid_input_runs():
    runtime = FakeRuntime(decision_ready=False)
    app = create_app(runtime=runtime, allowed_origins=[])
    with TestClient(app) as client:
        health = client.get("/health").json()
        response = post(client, encoded_image())
    assert health["status"] == "degraded"
    assert health["core_ready"] is True
    assert health["decision_ready"] is False
    assert response.status_code == 200
    assert response.json()["decision"] is None
    assert runtime.load_calls == 1


def test_cors_accepts_only_the_configured_browser_origin():
    app = create_app(runtime=FakeRuntime(), allowed_origins=["https://pixelproof.example"])
    with TestClient(app) as client:
        allowed = client.options(
            "/predict",
            headers={
                "Origin": "https://pixelproof.example",
                "Access-Control-Request-Method": "POST",
            },
        )
        denied = client.options(
            "/predict",
            headers={
                "Origin": "https://evil.example",
                "Access-Control-Request-Method": "POST",
            },
        )
    assert allowed.headers["access-control-allow-origin"] == "https://pixelproof.example"
    assert "access-control-allow-origin" not in denied.headers


class TinyProjectModel(torch.nn.Module):
    def forward(self, images):
        return images.mean(dim=(1, 2, 3))


def ready_project_runtime(tmp_path):
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    runtime = ModelRuntime(artifacts)
    runtime.project_model = ProjectTileModel(
        TinyProjectModel(),
        torch.device("cpu"),
        ProjectModelMetadata(
            artifact_id="test-project-model",
            sha256="a" * 64,
            revision="test-revision",
            arm="resnet18",
            seed=2024,
            tile_px=128,
            texture_floor=0.0,
            normalization="imagenet",
            aggregation="top3",
            threshold=0.75,
            calibration_fraction=0.5,
            split_seed=2026,
            validation_auc=0.91,
        ),
    )
    runtime.load_attempted = True
    return runtime


def test_project_model_api_returns_traceable_research_result_for_small_image(tmp_path):
    runtime = ready_project_runtime(tmp_path)
    health = runtime.health()
    assert health["status"] == "ready"
    assert health["project_model_ready"] is True
    assert health["core_ready"] is False
    assert health["decision_ready"] is False
    app = create_app(runtime=runtime, allowed_origins=[])
    with TestClient(app) as client:
        response = post(client, encoded_image(size=(64, 64)), method="project_model")

    assert response.status_code == 200
    payload = response.json()
    assert payload["method"] == "project_model"
    assert payload["project_model"] == {
        "score": payload["p_ai"],
        "threshold": 0.75,
        "triggered": False,
        "research_only": True,
        "limitation": payload["project_model"]["limitation"],
        "artifact_id": "test-project-model",
        "artifact_sha256": "a" * 64,
        "revision": "test-revision",
        "seed": 2024,
        "aggregation": "top3",
        "tile_px": 128,
        "tile_count": 1,
    }
    assert "%86,2" in payload["project_model"]["limitation"]
    assert len(payload["tile_map"]["tiles"]) == 1


def test_project_model_large_image_is_bounded_and_matches_shared_scorer(tmp_path):
    runtime = ready_project_runtime(tmp_path)
    picture = Image.fromarray(np.zeros((2304, 2304, 3), dtype=np.uint8))
    direct = runtime.project_model.score_image(picture, max_tiles=MAX_TILES)
    result = runtime.predict(picture, byte_size=1024, method="project_model")

    assert direct.tile_count == MAX_TILES
    assert result["project_model"]["tile_count"] == MAX_TILES
    assert result["p_ai"] == pytest.approx(direct.score, abs=0.0001)
    assert result["project_model"]["score"] == pytest.approx(direct.score, abs=0.0001)


def test_project_model_unavailable_does_not_fall_back_to_unverified_method(tmp_path):
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    runtime = ModelRuntime(artifacts)
    runtime.load_attempted = True
    runtime.load_errors["project_model"] = "verified artifact missing"

    with pytest.raises(RuntimeUnavailable, match="verified artifact missing"):
        runtime.predict(Image.new("RGB", (128, 128)), 100, "project_model")
