from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


MODULE_PATH = Path(__file__).parents[1] / "experiments/e32_gap_acquisition.py"
EXPERIMENTS = str(MODULE_PATH.parent)
if EXPERIMENTS not in sys.path:
    sys.path.insert(0, EXPERIMENTS)
SPEC = importlib.util.spec_from_file_location("e32_gap_acquisition", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
e32 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = e32
SPEC.loader.exec_module(e32)


def _fixture_spec() -> dict:
    readme = b"explicit generator recipe"
    return {
        "id": "qwen-image-2512",
        "repo_id": "fixture/repo",
        "revision": "a" * 40,
        "readme_sha256": hashlib.sha256(readme).hexdigest(),
        "license_tag": "license:cc-by-sa-4.0",
        "family": "Qwen Image 2512",
        "reported_images": 8,
        "reported_prompt_groups": 2,
        "native_format": "JPEG XL",
        "expected_width": 1328,
        "expected_height": 1328,
        "include_prefixes": ["data/"],
        "exclude_prefixes": [],
    }


def _fixture_info() -> SimpleNamespace:
    siblings = []
    for category, stem in (("animal", "animal_00001"), ("object", "object_00001")):
        for variation in range(4):
            root = f"data/{category}/dataset_{variation}/{stem}_{variation}"
            siblings.append(SimpleNamespace(rfilename=f"{root}.jxl", size=100 + variation))
            siblings.append(SimpleNamespace(rfilename=f"{root}.txt", size=10))
    return SimpleNamespace(sha="a" * 40, tags=["license:cc-by-sa-4.0"], siblings=siblings)


def test_freeze_keeps_complete_prompt_groups_and_sidecars():
    spec = _fixture_spec()
    readme = b"explicit generator recipe"
    source = e32.freeze_source(
        spec,
        prompt_group_limit=2,
        info_loader=lambda repo, revision: _fixture_info(),
        readme_loader=lambda url, digest: readme,
    )
    assert source["selected_prompt_groups"] == 2
    assert source["selected_images"] == 8
    assert len(source["assets"]) == 16
    assert source["selected_category_group_counts"] == {"animal": 1, "object": 1}


def test_round_robin_does_not_let_one_category_dominate():
    selected = e32._round_robin_groups(
        {"large": [f"large:{index}" for index in range(10)], "small": ["small:0", "small:1"]},
        4,
    )
    assert selected == ["large:0", "small:0", "large:1", "small:1"]


@pytest.mark.parametrize("path", ["../escape.jxl", "/absolute.jxl"])
def test_destination_rejects_unsafe_source_path(path):
    with pytest.raises(ValueError):
        e32._destination("source", path)


def test_reference_hierarchy_is_not_a_generated_flux_group():
    with pytest.raises(ValueError, match="unexpected source hierarchy"):
        e32._category_and_group(
            "flux2-klein-9b", "data_edit/reference/dataset_0/edit_00001_0.jxl"
        )


def test_bulk_download_requires_matching_decoder_smoke(tmp_path, monkeypatch):
    selection = tmp_path / "selection.json"
    selection.write_text("{}")
    smoke = tmp_path / "smoke.json"
    monkeypatch.setattr(e32, "DETAILED_SELECTION", selection)
    monkeypatch.setattr(e32, "SMOKE_EVIDENCE", smoke)
    with pytest.raises(PermissionError, match="missing"):
        e32._require_smoke_gate()

    smoke.write_text(
        '{"state":"decoder_smoke_passed","selection_sha256":"wrong"}'
    )
    with pytest.raises(PermissionError, match="different"):
        e32._require_smoke_gate()
