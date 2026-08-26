"""Append eligible iPhone 14 devices to the frozen C3 roles without changing old rows."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from pixelproof.project_paths import DATA_ROOT, ML_ROOT


EXPERIMENTS_DIR = Path(__file__).resolve().parent
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))

import e32_role_manifest as c3  # noqa: E402


C3_MANIFEST = DATA_ROOT / "e32" / "c3_role_manifest.json"
C3_EVIDENCE = ML_ROOT.parent / "evidence" / "e32_c3_role_manifest.json"
IPHONE = DATA_ROOT / "e32" / "r1b_csafe_iphone14_eligibility.json"
IPHONE_EVIDENCE = ML_ROOT.parent / "evidence" / "e32_r1b_csafe_iphone14_eligibility.json"
OUTPUT = DATA_ROOT / "e32" / "r1b_role_manifest.json"
EVIDENCE = ML_ROOT.parent / "evidence" / "e32_r1b_role_manifest.json"
EXPECTED_C3_RECORDS_SHA = "568e8e26caf08636f5f60356f96bd44017d6f0a6e2ffdb4384cdac771f6ed887"


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _write_atomic(path: Path, value: Any) -> bytes:
    raw = _json_bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_bytes(raw)
    temporary.replace(path)
    return raw


def append_records(
    old_records: Sequence[Mapping[str, Any]], iphone_rows: Sequence[Mapping[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    eligible = [row for row in iphone_rows if row.get("eligible") is True]
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in eligible:
        groups[str(row["device"])].append(row)
    if len(groups) != 10:
        raise ValueError(f"expected ten iPhone devices, found {len(groups)}")
    calibration = c3.calibration_groups(
        {device: len(rows) for device, rows in groups.items()}, round(0.20 * len(eligible))
    )
    if len(calibration) != 2:
        raise ValueError(f"expected two calibration devices, selected {sorted(calibration)}")
    appended = []
    role_groups: dict[str, set[str]] = defaultdict(set)
    for row in sorted(eligible, key=lambda item: str(item["source_key"])):
        device = str(row["device"])
        role = "CALIBRATION" if device in calibration else "TRAIN"
        group = f"device:{device}"
        role_groups[role].add(group)
        source_key = str(row["source_key"])
        appended.append(
            {
                "record_id": _sha256(f"csafe-mcsidb-iphone14:{source_key}".encode()),
                "source_id": "csafe-mcsidb-iphone14",
                "source_key": source_key,
                "label": "real",
                "role": role,
                "role_group": group,
                "parent_group": str(row["parent_group"]),
                "sha256": str(row["sha256"]),
                "decoded_format": str(row["decoded_format"]),
                "width": int(row["width"]),
                "height": int(row["height"]),
                "device": device,
                "scene_group": None,
                "model_name": None,
            }
        )
    if role_groups["TRAIN"] & role_groups["CALIBRATION"]:
        raise ValueError("iPhone device leaked across roles")
    combined = [dict(row) for row in old_records] + appended
    if combined[: len(old_records)] != list(old_records):
        raise ValueError("C3 row order/content changed")
    if len({row["record_id"] for row in combined}) != len(combined):
        raise ValueError("R1b record id collision")
    summary = {
        "selected_rows": len(appended),
        "role_counts": dict(sorted(Counter(row["role"] for row in appended).items())),
        "role_group_counts": {role: len(role_groups[role]) for role in ("TRAIN", "CALIBRATION")},
        "calibration_devices": sorted(calibration),
        "train_devices": sorted(set(groups) - calibration),
    }
    return combined, summary


def freeze() -> dict[str, Any]:
    c3_raw = C3_MANIFEST.read_bytes()
    c3_compact = json.loads(C3_EVIDENCE.read_text())
    if c3_compact.get("detailed_report_sha256") != _sha256(c3_raw):
        raise ValueError("C3 manifest binding changed")
    old = json.loads(c3_raw)
    if old.get("records_sha256") != EXPECTED_C3_RECORDS_SHA:
        raise ValueError("C3 record list changed")
    iphone_raw = IPHONE.read_bytes()
    iphone_compact = json.loads(IPHONE_EVIDENCE.read_text())
    if iphone_compact.get("detailed_report_sha256") != _sha256(iphone_raw):
        raise ValueError("iPhone eligibility binding changed")
    iphone = json.loads(iphone_raw)
    if iphone.get("state") != "eligibility_frozen_role_free":
        raise ValueError("iPhone eligibility is not frozen")

    records, iphone_summary = append_records(old["records"], iphone["records"])
    role_counts = Counter(row["role"] for row in records)
    class_counts = Counter(row["label"] for row in records)
    role_class_counts = {
        role: dict(sorted(Counter(row["label"] for row in records if row["role"] == role).items()))
        for role in ("TRAIN", "CALIBRATION")
    }
    if len(records) != 26_682 or class_counts != {"ai": 11_344, "real": 15_338}:
        raise ValueError(f"R1b controlled counts changed: {len(records)}/{class_counts}")
    report = {
        "schema_version": 1,
        "experiment": "E32/C4-R1b-controlled-role-extension",
        "state": "train_calibration_manifest_frozen",
        "seed": c3.SEED,
        "calibration_fraction_target": c3.CALIBRATION_FRACTION,
        "c3_manifest_sha256": c3_compact["detailed_report_sha256"],
        "c3_records_sha256": old["records_sha256"],
        "iphone_eligibility_sha256": iphone_compact["detailed_report_sha256"],
        "controlled_change": "append eligible iPhone14 REAL only; preserve every C3 record and role",
        "class_counts": dict(sorted(class_counts.items())),
        "role_counts": dict(sorted(role_counts.items())),
        "role_class_counts": role_class_counts,
        "iphone_source": iphone_summary,
        "records_sha256": _sha256(_json_bytes(records)),
        "records": records,
        "checks": {
            "c3_prefix_exact": records[: len(old["records"])] == old["records"],
            "unique_record_ids": len({row["record_id"] for row in records}) == len(records),
            "iphone_device_role_overlap_count": 0,
            "development_or_locked_rows": 0,
        },
        "limitations": [
            "Class counts are intentionally not balanced; class-weighted heads isolate the effect of added authentic Apple coverage.",
            "iPhone14 TRAIN/CALIBRATION is device-disjoint but not source-disjoint; IPN remains the absent-source DEVELOPMENT gate.",
        ],
    }
    raw = _write_atomic(OUTPUT, report)
    compact = {key: value for key, value in report.items() if key != "records"}
    compact.update(
        {
            "detailed_report_bytes": len(raw),
            "detailed_report_sha256": _sha256(raw),
            "detailed_report_external_path": str(OUTPUT.relative_to(DATA_ROOT / "e32")),
        }
    )
    _write_atomic(EVIDENCE, compact)
    return compact


def main(argv: Iterable[str] | None = None) -> int:
    argparse.ArgumentParser(description=__doc__).parse_args(argv)
    print(json.dumps(freeze(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
