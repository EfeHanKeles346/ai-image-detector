from __future__ import annotations

from copy import deepcopy
import zipfile

import pytest

from experiments.e36_acquisition import (
    CAL_AI,
    CAL_REAL,
    FINAL_AI,
    FINAL_REAL,
    HF_REPO,
    HF_REVISION,
    ZENODO_FILES,
    ZENODO_ID,
    ZENODO_VERSION,
    inspect_zip,
    validate_huggingface,
    validate_zenodo,
    _assert_final_unlocked,
)


def _zenodo() -> dict:
    return {
        "id": ZENODO_ID,
        "metadata": {"version": ZENODO_VERSION, "license": {"id": "cc-by-4.0"}},
        "files": [
            {
                "key": name,
                "size": size,
                "checksum": f"md5:{md5}",
                "links": {"self": f"https://zenodo.org/api/records/{ZENODO_ID}/files/{name}/content"},
            }
            for name, (size, md5) in ZENODO_FILES.items()
        ],
    }


def test_real_source_binds_licence_files_and_disjoint_roles() -> None:
    result = validate_zenodo(_zenodo())
    assert set(CAL_REAL).isdisjoint(FINAL_REAL)
    assert all(result[name]["role"] == "cal" for name in CAL_REAL)
    assert all(result[name]["role"] == "locked_final" for name in FINAL_REAL)


def test_real_source_rejects_checksum_drift() -> None:
    payload = deepcopy(_zenodo())
    payload["files"][0]["checksum"] = "md5:" + "0" * 32
    with pytest.raises(ValueError, match="contract changed"):
        validate_zenodo(payload)


def _hf() -> dict:
    siblings = []
    for family, ids in ((*[(family, range(101, 201)) for family in CAL_AI], *[(family, range(1, 41)) for family in FINAL_AI])):
        for index in ids:
            siblings.append({
                "rfilename": f"images/{family}/{index:06d}_hash.png",
                "size": 1000 + index,
                "lfs": {"sha256": f"{index:064x}"[-64:]},
            })
    return {"id": HF_REPO, "sha": HF_REVISION, "cardData": {"license": "apache-2.0"}, "siblings": siblings}


def test_ai_source_freezes_family_disjoint_cal_and_final() -> None:
    result = validate_huggingface(_hf())
    assert set(CAL_AI).isdisjoint(FINAL_AI)
    assert len(result["cal"]) == 600
    assert len(result["locked_final"]) == 240


def _info(name: str) -> zipfile.ZipInfo:
    item = zipfile.ZipInfo(name)
    item.file_size = 10
    item.compress_size = 10
    item.CRC = 0
    return item


def test_zip_inventory_exposes_conditions_and_rejects_traversal() -> None:
    result = inspect_zip([_info("device/view_000/a.jpg"), _info("device/view_001/a.jpg")])
    assert [row["condition"] for row in result["images"]] == ["view_000", "view_001"]
    with pytest.raises(ValueError, match="unsafe"):
        inspect_zip([_info("device/../escape.jpg")])


def test_final_unlock_fails_closed_without_bound_evidence(monkeypatch, tmp_path) -> None:
    import experiments.e36_acquisition as acquisition

    monkeypatch.setattr(acquisition, "E38_EVIDENCE", tmp_path / "missing.json")
    monkeypatch.setattr(acquisition, "E38_CANDIDATE", tmp_path / "missing.joblib")
    with pytest.raises(ValueError, match="evidence"):
        _assert_final_unlocked()
