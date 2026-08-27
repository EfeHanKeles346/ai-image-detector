from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from experiments import e33_r1c


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("normal_000001.jpg", "everyday_life"),
        ("production_000002.png", "labor_and_production"),
        ("Culture_&_Religion_000003.png", "culture_and_religion"),
        ("War_&_Conflict_Scenes_000004.png", "war_and_conflict"),
    ],
)
def test_scenario_contract_is_explicit(name: str, expected: str) -> None:
    assert e33_r1c.scenario_from_filename(name) == expected


def test_scenario_contract_rejects_unknown_name() -> None:
    with pytest.raises(ValueError, match="undeclared"):
        e33_r1c.scenario_from_filename("camera_1.jpg")


def test_manifest_preserves_explicit_label_direction_and_scenario(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "calibration"
    for class_name in ("real", "ai"):
        folder = root / class_name
        folder.mkdir(parents=True)
        Image.new("RGB", (8, 8)).save(folder / "normal_000001.jpg")
    receipt = tmp_path / "calibration_extraction_receipt.json"
    receipt.write_text('{"state":"calibration_extraction_complete"}')
    monkeypatch.setattr(e33_r1c, "OUTPUT_ROOT", tmp_path)
    monkeypatch.setattr(e33_r1c, "MANIFEST", tmp_path / "manifest.json")
    monkeypatch.setattr(e33_r1c, "MANIFEST_EVIDENCE", tmp_path / "evidence.json")
    result = e33_r1c.build_manifest(root)
    payload = __import__("json").loads((tmp_path / "manifest.json").read_text())
    assert result["real_count"] == result["ai_count"] == 1
    assert {row["label"] for row in payload["rows"]} == {0, 1}
    assert {row["source"] for row in payload["rows"]} == {"everyday_life"}
    assert payload["source_semantics"] == "scenario groups, not camera-pipeline identities"
