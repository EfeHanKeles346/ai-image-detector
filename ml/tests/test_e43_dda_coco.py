import zipfile

import pytest

from experiments.e43_dda_coco import (
    COCO_BYTES,
    COCO_ETAG,
    EXPECTED_VARIANT_COUNTS,
    audit_parent_rows,
    real_ids_from_infos,
    synthetic_ids_from_names,
    validate_coco_headers,
)


def _info(name: str, size: int = 10) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name)
    info.file_size = size
    return info


def test_coco_headers_bind_size_and_etag():
    result = validate_coco_headers({
        "content-length": str(COCO_BYTES),
        "etag": COCO_ETAG,
        "last-modified": "Wed, 11 Jul 2018 05:08:47 GMT",
    })
    assert result["bytes"] == COCO_BYTES


def test_coco_real_schema_rejects_wrong_path():
    infos = [_info(f"val2017/{index:012d}.jpg") for index in range(5_000)]
    assert len(real_ids_from_infos(infos)) == 5_000
    infos[-1] = _info("train2017/000000004999.jpg")
    with pytest.raises(ValueError, match="unexpected COCO member"):
        real_ids_from_infos(infos)


def test_synthetic_structure_binds_every_variant_count():
    names = []
    for variant, count in EXPECTED_VARIANT_COUNTS.items():
        names.extend(f"DDA-COCO/{variant}/val2017/{index:012d}.jpg" for index in range(count))
    grouped = synthetic_ids_from_names(names)
    assert {variant: len(ids) for variant, ids in grouped.items()} == EXPECTED_VARIANT_COUNTS


def test_synthetic_structure_rejects_unknown_variant():
    with pytest.raises(ValueError, match="unexpected DDA-COCO member"):
        synthetic_ids_from_names(["DDA-COCO/unknown/val2017/000000000001.jpg"])


def _row(record: str, parent: str, condition: str, label: int, sha: str, dhash: str) -> dict:
    return {
        "record_id": record,
        "parent_id": parent,
        "condition": condition,
        "label": label,
        "sha256": sha,
        "dhash": dhash,
    }


def _parent(parent: str, real_sha: str, real_dhash: str) -> list[dict]:
    rows = [_row(f"{parent}:REAL", parent, "REAL", 0, real_sha, real_dhash)]
    rows.extend(
        _row(f"{parent}:{variant}", parent, variant, 1, f"{parent}:{variant}", f"d:{parent}:{variant}")
        for variant in EXPECTED_VARIANT_COUNTS
    )
    return rows


def test_parent_audit_excludes_whole_protected_parent():
    rows = _parent("p1", "real-1", "d-real-1") + _parent("p2", "real-2", "d-real-2")
    result = audit_parent_rows(rows, {"p1:FLUX.1"}, set())
    assert result["passed"] is True
    assert result["excluded_parent_ids"] == ["p1"]
    assert result["protected_exact_overlap_records"] == ["p1:FLUX.1"]


def test_parent_audit_exact_duplicate_policy_and_dhash_diagnostic():
    rows = _parent("p1", "same-real", "same-dhash") + _parent("p2", "same-real", "same-dhash")
    result = audit_parent_rows(rows, set(), set())
    assert result["excluded_parent_ids"] == ["p2"]
    assert len(result["cross_parent_exact_groups"]) == 1
    assert len(result["cross_parent_dhash_diagnostic"]) == 1


def test_parent_audit_cross_label_exact_excludes_every_touched_parent():
    rows = _parent("p1", "shared", "d1") + _parent("p2", "real-2", "d2")
    rows[-1]["sha256"] = "shared"
    result = audit_parent_rows(rows, set(), set())
    assert result["excluded_parent_ids"] == ["p1", "p2"]
    assert len(result["cross_label_exact_groups"]) == 1
