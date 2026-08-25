from __future__ import annotations

import json

import pytest

from pixelproof.data_contract import (
    DataRecord,
    DataRole,
    enforce_byte_ceiling,
    final_run_receipt,
    load_manifest,
    select_stratified,
    shortcut_audit,
    validate_records,
)


REVISION = "1" * 40
SHA = "a" * 64


def row(index: int, *, role=DataRole.DEVELOPMENT_TEST, label="real", **changes):
    values = {
        "record_id": f"row-{index}",
        "role": role,
        "source_id": "source/example",
        "source_revision": REVISION,
        "source_key": f"{index:03d}",
        "label": label,
        "group": "camera" if label == "real" else "generator",
        "transport": "native",
        "path": f"images/{index}.jpg",
        "generator": None if label == "real" else "generator",
        "camera_pipeline": "camera" if label == "real" else None,
        "content_id": f"content-{index}",
        "sha256": f"{index:064x}",
        "bytes": 10_000 + index,
        "width": 100 + index,
        "height": 80 + index,
        "image_format": "JPEG",
    }
    values.update(changes)
    return DataRecord(**values)


def test_role_loader_refuses_training_access_to_test_rows(tmp_path):
    path = tmp_path / "manifest.json"
    value = row(1)
    path.write_text(json.dumps({"schema_version": 1, "records": [{**value.__dict__, "role": value.role.value}]}))
    with pytest.raises(PermissionError, match="requested train"):
        load_manifest(path, required_role=DataRole.TRAIN)


def test_derived_record_cannot_cross_parent_role_or_underlying_content():
    parent = row(1)
    derived = row(
        2,
        role=DataRole.LOCKED_FINAL_TEST,
        parent_id=parent.record_id,
        content_id=parent.content_id,
    )
    with pytest.raises(ValueError, match="crossed"):
        validate_records([parent, derived])


def test_exact_and_underlying_content_cannot_leak_across_roles():
    first = row(1)
    with pytest.raises(ValueError, match="exact image"):
        validate_records([first, row(2, role=DataRole.CALIBRATION, sha256=first.sha256)])
    with pytest.raises(ValueError, match="underlying content"):
        validate_records(
            [first, row(2, role=DataRole.CALIBRATION, content_id=first.content_id)]
        )


def test_stratified_selection_is_deterministic_and_fails_incomplete_cells():
    records = [
        {"label": label, "regime": regime, "source_key": f"{index:02d}"}
        for label in ("real", "ai")
        for regime in ("texture", "structure")
        for index in (2, 0, 1)
    ]
    selected = select_stratified(
        records, group_fields=("label", "regime"), per_group=2
    )
    assert len(selected) == 8
    assert {item["source_key"] for item in selected} == {"00", "01"}
    with pytest.raises(ValueError, match="needs 4"):
        select_stratified(records, group_fields=("label", "regime"), per_group=4)


def test_byte_ceiling_checks_every_file_and_total():
    assert enforce_byte_ceiling([10, 20], total_ceiling=30, per_file_ceiling=20) == 30
    with pytest.raises(ValueError, match="per-file"):
        enforce_byte_ceiling([21], total_ceiling=30, per_file_ceiling=20)
    with pytest.raises(ValueError, match="above"):
        enforce_byte_ceiling([20, 20], total_ceiling=30)


def test_metadata_shortcut_probe_flags_perfect_format_and_geometry_split():
    records = []
    for index in range(20):
        records.append(
            row(index, label="real", width=120, height=80, image_format="JPEG", bytes=10_000)
        )
        records.append(
            row(
                100 + index,
                label="ai",
                width=512,
                height=512,
                image_format="PNG",
                bytes=500_000,
            )
        )
    report = shortcut_audit(records)
    assert report["metadata_probe_auc"] > 0.95
    assert not report["pass"]
    assert any("format" in issue for issue in report["issues"])


def test_final_receipt_is_locked_to_final_rows_and_cannot_be_overwritten(tmp_path):
    final_rows = [row(1, role=DataRole.LOCKED_FINAL_TEST)]
    receipt = tmp_path / "receipt.json"
    result = final_run_receipt(
        receipt,
        records=final_rows,
        candidate_id="candidate",
        candidate_sha256=SHA,
        threshold_contract_sha256="b" * 64,
        results_sha256="c" * 64,
    )
    assert result["record_count"] == 1
    with pytest.raises(FileExistsError):
        final_run_receipt(
            receipt,
            records=final_rows,
            candidate_id="candidate",
            candidate_sha256=SHA,
            threshold_contract_sha256="b" * 64,
            results_sha256="c" * 64,
        )
