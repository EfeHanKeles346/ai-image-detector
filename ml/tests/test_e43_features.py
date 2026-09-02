import pytest

from experiments.e43_features import feature_rows


def _manifest(count: int, state: str = "e43_rr_roles_frozen_before_features"):
    roles = ("train", "calibration", "development")
    return {
        "state": state,
        "rows": [
            {"record_id": f"row-{index}", "e43_role": roles[index % 3]}
            for index in range(count)
        ],
    }


def test_e43_feature_rows_require_frozen_state(monkeypatch):
    monkeypatch.setattr("experiments.e43_features.EXPECTED_ROWS", 3)
    with pytest.raises(ValueError, match="state changed"):
        feature_rows(_manifest(3, state="wrong"))


def test_e43_feature_rows_require_exact_unique_population(monkeypatch):
    monkeypatch.setattr("experiments.e43_features.EXPECTED_ROWS", 3)
    assert len(feature_rows(_manifest(3))) == 3
    duplicate = _manifest(3)
    duplicate["rows"][2]["record_id"] = "row-1"
    with pytest.raises(ValueError, match="population changed"):
        feature_rows(duplicate)


def test_e43_feature_rows_require_all_roles(monkeypatch):
    monkeypatch.setattr("experiments.e43_features.EXPECTED_ROWS", 3)
    manifest = _manifest(3)
    manifest["rows"][2]["e43_role"] = "train"
    with pytest.raises(ValueError, match="roles changed"):
        feature_rows(manifest)
