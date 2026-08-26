"""Realize iPhone 14 natural parents without opening protected model scores."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from pixelproof.e32_candidate import image_paths
from pixelproof.project_paths import ML_ROOT


EXPERIMENTS_DIR = Path(__file__).resolve().parent
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))

import e32_r1b_acquisition as acquisition  # noqa: E402
import e32_r1b_csafe_iphone14 as iphone14  # noqa: E402
import e32_source_realization as realization  # noqa: E402


EXPECTED_OWNER_IDENTITY = "390e3c210ee61d70252d7e4714b8640463f44d57760942d25a1bdf7eab5aac09"
IPN_AUDIT = acquisition.OUTPUT_ROOT / "r1b_ipn_realization.json"


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def owner_exact_hashes(folder: Path) -> tuple[set[str], str, int]:
    paths = image_paths([str(folder)])
    identity = [
        {
            "name": path.name,
            "bytes": path.stat().st_size,
            "sha256": realization._sha256_file(path),
        }
        for path in paths
    ]
    identity_sha = hashlib.sha256(_json_bytes(identity)).hexdigest()
    return {row["sha256"] for row in identity}, identity_sha, len(identity)


def audit(owner_gallery: Path) -> dict[str, Any]:
    extraction_raw = iphone14.EXTRACTION.read_bytes()
    compact_extraction = json.loads(
        (iphone14.EVIDENCE_ROOT / "e32_r1b_csafe_iphone14_natural_extraction.json").read_text()
    )
    if compact_extraction.get("detailed_report_sha256") != realization._sha256_bytes(extraction_raw):
        raise ValueError("iPhone14 extraction binding changed")
    extraction = json.loads(extraction_raw)
    if extraction.get("state") != "natural_extraction_complete_role_free":
        raise ValueError("iPhone14 extraction state changed")

    ipn = json.loads(IPN_AUDIT.read_text())
    if ipn.get("state") != "development_realization_passed_unscored":
        raise ValueError("IPN protected audit is not a clean unscored realization")
    ipn_exact = {str(row["sha256"]) for row in ipn["records"]}
    ipn_perceptual = {(str(row["dhash"]), str(row["phash"])) for row in ipn["records"]}
    owner_exact, owner_identity, owner_count = owner_exact_hashes(owner_gallery)
    if owner_identity != EXPECTED_OWNER_IDENTITY or owner_count != 210:
        raise ValueError(
            f"owner gallery identity changed: expected {EXPECTED_OWNER_IDENTITY}/210, "
            f"found {owner_identity}/{owner_count}"
        )

    records: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    ipn_hits: list[str] = []
    owner_hits: list[str] = []
    for selected in extraction["records"]:
        source_key = str(selected["source_key"])
        path = acquisition.OUTPUT_ROOT / source_key
        if not path.is_file():
            failures.append({"source_key": source_key, "reason": "missing_file"})
            continue
        if path.stat().st_size != int(selected["bytes"]):
            failures.append({"source_key": source_key, "reason": "byte_count_mismatch"})
            continue
        try:
            record = realization._image_record(path)
        except Exception as error:
            failures.append({"source_key": source_key, "reason": f"decode_failure:{type(error).__name__}"})
            continue
        if record["sha256"] != selected["sha256"]:
            failures.append({"source_key": source_key, "reason": "extraction_sha256_changed"})
        if record["sha256"] in ipn_exact or (record["dhash"], record["phash"]) in ipn_perceptual:
            ipn_hits.append(source_key)
        if record["sha256"] in owner_exact:
            owner_hits.append(source_key)
        record.update(
            {
                "source_key": source_key,
                "device": selected["device"],
                "lens": selected["lens"],
                "camera_pipeline": selected["camera_pipeline"],
                "parent_group": selected["parent_group"],
                "native_social_state": "native",
            }
        )
        records.append(record)
    if ipn_hits:
        failures.append({"reason": "protected_ipn_overlap", "count": str(len(ipn_hits))})
    if owner_hits:
        failures.append({"reason": "protected_owner_exact_overlap", "count": str(len(owner_hits))})

    realization._finalize(
        source_id="csafe-mcsidb-iphone14",
        kind="real",
        selection_raw=extraction_raw,
        expected_images=int(extraction["parent_count"]),
        records=records,
        failures=failures,
        extra={
            "device_counts": dict(sorted(Counter(row["device"] for row in records).items())),
            "lens_counts": dict(sorted(Counter(row["lens"] for row in records).items())),
            "camera_pipeline_counts": dict(sorted(Counter(row["camera_pipeline"] for row in records).items())),
            "native_social_state_counts": {"native": len(records)},
            "protected_ipn_exact_or_perceptual_overlaps": len(ipn_hits),
            "protected_ipn_realization_sha256": realization._sha256_file(IPN_AUDIT),
            "protected_owner_exact_overlaps": len(owner_hits),
            "protected_owner_gallery_identity_sha256": owner_identity,
            "protected_owner_count": owner_count,
            "owner_boundary": "Only raw exact SHA identity was read; no owner detector score or perceptual feature was computed.",
        },
    )
    return json.loads(
        (iphone14.EVIDENCE_ROOT / "e32_csafe-mcsidb-iphone14_realization.json").read_text()
    )


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("owner_gallery", type=Path)
    args = parser.parse_args(argv)
    print(json.dumps(audit(args.owner_gallery.resolve()), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
