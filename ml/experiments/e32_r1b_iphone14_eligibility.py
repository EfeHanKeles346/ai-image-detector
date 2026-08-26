"""Freeze eligible iPhone 14 parents after deterministic duplicate-component exclusion."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from pixelproof.project_paths import DATA_ROOT, ML_ROOT


AUDIT = DATA_ROOT / "e32" / "audits" / "csafe-mcsidb-iphone14.json"
AUDIT_EVIDENCE = ML_ROOT.parent / "evidence" / "e32_csafe-mcsidb-iphone14_realization.json"
OUTPUT = DATA_ROOT / "e32" / "r1b_csafe_iphone14_eligibility.json"
EVIDENCE = ML_ROOT.parent / "evidence" / "e32_r1b_csafe_iphone14_eligibility.json"
EXPECTED_EXCLUDED = {
    "real/csafe_iphone14/natural/iPhone14_5/telephoto/IMG_1290.JPG",
    "real/csafe_iphone14/natural/iPhone14_5/telephoto/IMG_1291.JPG",
}


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


def duplicate_component_keys(groups: Sequence[Sequence[str]]) -> set[str]:
    return {str(key) for group in groups for key in group}


def freeze() -> dict[str, Any]:
    audit_raw = AUDIT.read_bytes()
    compact = json.loads(AUDIT_EVIDENCE.read_text())
    if compact.get("detailed_report_sha256") != _sha256(audit_raw):
        raise ValueError("iPhone14 realization binding changed")
    audit = json.loads(audit_raw)
    if audit.get("state") != "source_realization_rejected_no_role_assignment":
        raise ValueError("iPhone14 realization is not the expected stopped state")
    if audit.get("failure_reason_counts") is not None:
        raise ValueError("detailed audit unexpectedly contains compact-only fields")
    failures = audit.get("failures", [])
    if failures != [{"count": "1", "reason": "within_source_confirmed_perceptual_duplicates"}]:
        raise ValueError(f"iPhone14 audit has non-correctable failures: {failures}")
    excluded = duplicate_component_keys(audit["duplicate_groups"]["perceptual_confirmed"])
    if excluded != EXPECTED_EXCLUDED:
        raise ValueError(f"iPhone14 duplicate component changed: {sorted(excluded)}")
    records = []
    for row in audit["records"]:
        source_key = str(row["source_key"])
        records.append(
            {
                "source_key": source_key,
                "eligible": source_key not in excluded,
                "exclusion_reason": None if source_key not in excluded else "confirmed_perceptual_duplicate_component",
                "sha256": row["sha256"],
                "dhash": row["dhash"],
                "phash": row["phash"],
                "device": row["device"],
                "lens": row["lens"],
                "camera_pipeline": row["camera_pipeline"],
                "parent_group": row["parent_group"],
                "decoded_format": row["decoded_format"],
                "width": row["width"],
                "height": row["height"],
                "bytes": row["bytes"],
                "label": "real",
            }
        )
    eligible = [row for row in records if row["eligible"]]
    if len(records) != 3_996 or len(eligible) != 3_994:
        raise ValueError("iPhone14 eligibility count changed")
    report = {
        "schema_version": 1,
        "experiment": "E32/C4-R1b-csafe-iphone14-eligibility",
        "state": "eligibility_frozen_role_free",
        "source_id": "csafe-mcsidb-iphone14",
        "realization_sha256": compact["detailed_report_sha256"],
        "selected_parents": len(records),
        "eligible_parents": len(eligible),
        "excluded_parents": len(excluded),
        "excluded_source_keys": sorted(excluded),
        "eligible_format_counts": {
            name: sum(row["decoded_format"] == name for row in eligible) for name in ("JPEG", "MPO")
        },
        "records": records,
        "boundary": "Role-free eligibility only; files remain intact and no TRAIN/CALIBRATION assignment occurs.",
    }
    raw = _write_atomic(OUTPUT, report)
    result = {key: value for key, value in report.items() if key != "records"}
    result.update(
        {
            "detailed_report_bytes": len(raw),
            "detailed_report_sha256": _sha256(raw),
            "detailed_report_external_path": str(OUTPUT.relative_to(DATA_ROOT / "e32")),
        }
    )
    _write_atomic(EVIDENCE, result)
    return result


def main(argv: Iterable[str] | None = None) -> int:
    argparse.ArgumentParser(description=__doc__).parse_args(argv)
    print(json.dumps(freeze(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
