"""Receipt-bound, unscored realization audit for IPN-NFID DEVELOPMENT."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from pixelproof.project_paths import ML_ROOT


EXPERIMENTS_DIR = Path(__file__).resolve().parent
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))

import e32_r1b_acquisition as acquisition  # noqa: E402
import e32_source_realization as realization  # noqa: E402


OUTPUT = acquisition.OUTPUT_ROOT / "r1b_ipn_realization.json"
EVIDENCE = ML_ROOT.parent / "evidence" / "e32_r1b_ipn_realization.json"
SCENE_RE = re.compile(r"_(\d+)\s+natural\s+(landscape|portrait)\.jpe?g$", re.IGNORECASE)


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def _write_atomic(path: Path, value: Any) -> bytes:
    raw = _json_bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_bytes(raw)
    temporary.replace(path)
    return raw


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def scene_group(name: str) -> str:
    match = SCENE_RE.search(name)
    if match is None:
        raise ValueError(f"unparseable IPN natural scene name: {name}")
    number, orientation = match.groups()
    return f"ipn:{orientation.lower()}:{int(number):02d}"


def cross_scene_perceptual_groups(
    records: Sequence[Mapping[str, Any]], groups: Sequence[Sequence[str]]
) -> list[list[str]]:
    scene_by_key = {str(row["source_key"]): str(row["scene_group"]) for row in records}
    return [list(group) for group in groups if len({scene_by_key[key] for key in group}) > 1]


def audit() -> dict[str, Any]:
    selection_raw = acquisition.SELECTION.read_bytes()
    selection = json.loads(selection_raw)
    if selection.get("state") != "selection_frozen_no_selected_bytes_claimed":
        raise ValueError("R1b selection is not frozen")
    receipt_raw = acquisition.IPN_RECEIPT.read_bytes()
    receipt = json.loads(receipt_raw)
    if receipt.get("state") != "ipn_download_complete_md5_verified":
        raise ValueError("IPN download receipt is incomplete")
    if receipt.get("selection_sha256") != _sha256(selection_raw):
        raise ValueError("IPN download receipt lost selection binding")
    assets = selection["ipn"]["assets"]
    receipt_rows = {int(row["file_id"]): row for row in receipt["rows"]}
    if len(assets) != acquisition.IPN_EXPECTED_COUNT or len(receipt_rows) != len(assets):
        raise ValueError("IPN receipt count changed")

    records: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for asset in assets:
        file_id = int(asset["file_id"])
        downloaded = receipt_rows.get(file_id)
        if downloaded is None:
            failures.append({"file_id": file_id, "reason": "missing_receipt_row"})
            continue
        expected_path = (
            acquisition.OUTPUT_ROOT
            / "real"
            / "ipn_nfid"
            / "natural"
            / str(asset["device"])
            / f"{file_id}.jpg"
        )
        if downloaded.get("path") != str(expected_path.relative_to(acquisition.OUTPUT_ROOT)):
            failures.append({"file_id": file_id, "reason": "receipt_path_changed"})
            continue
        if not expected_path.is_file():
            failures.append({"file_id": file_id, "reason": "missing_file"})
            continue
        if expected_path.stat().st_size != int(asset["bytes"]):
            failures.append({"file_id": file_id, "reason": "byte_count_mismatch"})
            continue
        try:
            record = realization._image_record(expected_path)
        except Exception as error:
            failures.append({"file_id": file_id, "reason": f"decode_failure:{type(error).__name__}"})
            continue
        if acquisition._md5(expected_path) != asset["md5"]:
            failures.append({"file_id": file_id, "reason": "md5_changed_after_download"})
        record.update(
            {
                "source_key": str(expected_path.relative_to(acquisition.OUTPUT_ROOT)),
                "file_id": file_id,
                "upstream_name": asset["name"],
                "device": asset["device"],
                "scene_group": scene_group(str(asset["name"])),
                "role": "development",
                "label": "real",
            }
        )
        records.append(record)

    exact_groups = realization._duplicates(records, "sha256")
    perceptual_groups = realization._confirmed_perceptual_duplicates(records)
    cross_scene_groups = cross_scene_perceptual_groups(records, perceptual_groups)
    protected_exact, protected_dhash, protected_manifests = realization._protected_hashes()
    peer_exact, peer_perceptual, peer_legacy, peer_reports = realization._passed_peer_hashes(
        "ipn-nfid-natural-development"
    )
    protected_hits = [
        row["source_key"]
        for row in records
        if row["sha256"] in protected_exact or row["dhash"] in protected_dhash
    ]
    peer_hits = [
        row["source_key"]
        for row in records
        if row["sha256"] in peer_exact
        or (row["dhash"], row["phash"]) in peer_perceptual
        or row["dhash"] in peer_legacy
    ]
    if exact_groups:
        failures.append({"reason": "within_source_exact_duplicates", "count": len(exact_groups)})
    if cross_scene_groups:
        failures.append({"reason": "cross_scene_perceptual_duplicates", "count": len(cross_scene_groups)})
    if protected_hits:
        failures.append({"reason": "protected_overlap", "count": len(protected_hits)})
    if peer_hits:
        failures.append({"reason": "passed_peer_overlap", "count": len(peer_hits)})
    if len(records) != acquisition.IPN_EXPECTED_COUNT:
        failures.append({"reason": "realized_count_mismatch", "count": len(records)})

    report = {
        "schema_version": 1,
        "experiment": "E32/C4-R1b-IPN-development-realization",
        "state": "development_realization_rejected" if failures else "development_realization_passed_unscored",
        "selection_sha256": _sha256(selection_raw),
        "download_receipt_sha256": _sha256(receipt_raw),
        "role": "development",
        "label": "real",
        "realized_images": len(records),
        "bytes": sum(int(row["bytes"]) for row in records),
        "device_counts": dict(sorted(Counter(row["device"] for row in records).items())),
        "scene_group_counts": dict(sorted(Counter(row["scene_group"] for row in records).items())),
        "format_counts": dict(sorted(Counter(row["decoded_format"] for row in records).items())),
        "mode_counts": dict(sorted(Counter(row["mode"] for row in records).items())),
        "exif_present": sum(bool(row["exif_present"]) for row in records),
        "unique_sha256": len({row["sha256"] for row in records}),
        "perceptual_same_scene_groups": len(perceptual_groups) - len(cross_scene_groups),
        "cross_scene_perceptual_groups": cross_scene_groups,
        "protected_overlap": protected_hits,
        "passed_peer_overlap": peer_hits,
        "protected_manifest_files": protected_manifests,
        "passed_peer_audit_files": peer_reports,
        "failures": failures,
        "records": records,
        "boundary": "Data-quality realization only. No detector loaded; IPN remains DEVELOPMENT and cannot fit decisions.",
    }
    raw = _write_atomic(OUTPUT, report)
    compact = {key: value for key, value in report.items() if key not in {"records", "cross_scene_perceptual_groups", "protected_overlap", "passed_peer_overlap", "failures"}}
    compact.update(
        {
            "failure_reason_counts": dict(sorted(Counter(row["reason"] for row in failures).items())),
            "detailed_report_bytes": len(raw),
            "detailed_report_sha256": _sha256(raw),
            "detailed_report_external_path": str(OUTPUT.relative_to(acquisition.OUTPUT_ROOT)),
        }
    )
    _write_atomic(EVIDENCE, compact)
    return compact


def main(argv: Iterable[str] | None = None) -> int:
    argparse.ArgumentParser(description=__doc__).parse_args(argv)
    print(json.dumps(audit(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
