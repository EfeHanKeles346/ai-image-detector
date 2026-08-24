import pytest

from pixelproof.demo import DemoError, validate_health, validate_prediction


def health_payload():
    return {
        "status": "ready",
        "project_model_ready": True,
        "project_model": {
            "artifact_id": "e20-tile-resnet18-seed2024",
            "sha256": "a" * 64,
        },
        "load_errors": {},
    }


def prediction_payload():
    return {
        "method": "project_model",
        "verdict": "uncertain",
        "project_model": {
            "score": 0.25,
            "threshold": 0.9895,
            "triggered": False,
            "research_only": True,
            "artifact_sha256": "a" * 64,
            "tile_count": 12,
        },
    }


def test_demo_contract_accepts_ready_project_model_and_research_prediction():
    validate_health(health_payload())
    assert validate_prediction(prediction_payload())["score"] == 0.25


def test_demo_contract_rejects_degraded_health_and_authenticity_verdict():
    health = health_payload()
    health["status"] = "degraded"
    with pytest.raises(DemoError, match="not ready"):
        validate_health(health)

    prediction = prediction_payload()
    prediction["verdict"] = "real"
    with pytest.raises(DemoError, match="authenticity-certifying"):
        validate_prediction(prediction)


def test_demo_contract_rejects_untraceable_or_invalid_project_results():
    prediction = prediction_payload()
    prediction["project_model"]["artifact_sha256"] = "short"
    with pytest.raises(DemoError, match="SHA-256"):
        validate_prediction(prediction)

    prediction = prediction_payload()
    prediction["project_model"]["score"] = 2.0
    with pytest.raises(DemoError, match="invalid score"):
        validate_prediction(prediction)
