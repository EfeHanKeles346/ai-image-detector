"""Safe inventory, natural-only freeze and extraction for E32/R1b CSAFE iPhone 14."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping
from zipfile import ZipFile

from pixelproof.project_paths import ML_ROOT


EXPERIMENTS_DIR = Path(__file__).resolve().parent
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))

import e32_archive_inventory as archive_tools  # noqa: E402
import e32_r1b_acquisition as acquisition  # noqa: E402


ARCHIVE = acquisition.OUTPUT_ROOT / "real" / "csafe" / "archives" / "iPhone14.zip"
INVENTORY = acquisition.OUTPUT_ROOT / "r1b_csafe_iphone14_inventory.json"
SELECTION = acquisition.OUTPUT_ROOT / "r1b_csafe_iphone14_natural_selection.json"
EXTRACTION = acquisition.OUTPUT_ROOT / "r1b_csafe_iphone14_natural_extraction.json"
EVIDENCE_ROOT = ML_ROOT.parent / "evidence"
LENSES = {"front", "telephoto", "ultra", "wide"}
CONTENT_TYPES = {"blank", "natural"}
EXPECTED_COUNTS = {"blank": 4_000, "natural": 3_996}
EXPECTED_LENS_COUNTS = {
    "blank:front": 1_000,
    "blank:telephoto": 1_000,
    "blank:ultra": 1_000,
    "blank:wide": 1_000,
    "natural:front": 998,
    "natural:telephoto": 1_000,
    "natural:ultra": 998,
    "natural:wide": 1_000,
}
CHUNK_BYTES = 8 * 1024**2


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def _write_atomic(path: Path, value: Any) -> bytes:
    raw = _json_bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_bytes(raw)
    temporary.replace(path)
    return raw


def parse_member(name: str) -> dict[str, str]:
    parts = PurePosixPath(name).parts
    if len(parts) != 5 or parts[0] != "iPhone14":
        raise ValueError(f"unexpected iPhone14 member hierarchy: {name}")
    _, device, content_type, lens, filename = parts
    if not re.fullmatch(r"iPhone14_(?:10|[1-9])", device):
        raise ValueError(f"unexpected iPhone14 device: {name}")
    if content_type not in CONTENT_TYPES:
        raise ValueError(f"unexpected iPhone14 content type: {name}")
    if lens not in LENSES:
        raise ValueError(f"unexpected iPhone14 lens: {name}")
    if PurePosixPath(filename).suffix.lower() != ".jpg":
        raise ValueError(f"unexpected iPhone14 suffix: {name}")
    return {"device": device, "content_type": content_type, "lens": lens, "filename": filename}


def _bound_download() -> tuple[bytes, dict[str, Any]]:
    selection_raw = acquisition.SELECTION.read_bytes()
    receipt = json.loads(acquisition.CSAFE_RECEIPT.read_text())
    if receipt.get("state") != "csafe_iphone14_download_complete_md5_verified":
        raise ValueError("iPhone14 download receipt is incomplete")
    if receipt.get("selection_sha256") != _sha256(selection_raw):
        raise ValueError("iPhone14 download receipt lost selection binding")
    if int(receipt.get("bytes", -1)) != acquisition.CSAFE["bytes"]:
        raise ValueError("iPhone14 receipt byte count changed")
    if receipt.get("md5") != acquisition.CSAFE["md5"]:
        raise ValueError("iPhone14 receipt MD5 changed")
    return selection_raw, receipt


def inventory() -> dict[str, Any]:
    selection_raw, receipt = _bound_download()
    archive = archive_tools._inventory_zip(ARCHIVE, acquisition.CSAFE["bytes"])
    content_counts: Counter[str] = Counter()
    lens_counts: Counter[str] = Counter()
    devices: Counter[str] = Counter()
    for member in archive["members"]:
        parsed = parse_member(str(member["name"]))
        content_counts[parsed["content_type"]] += 1
        lens_counts[f"{parsed['content_type']}:{parsed['lens']}"] += 1
        devices[parsed["device"]] += 1
    if dict(content_counts) != EXPECTED_COUNTS:
        raise ValueError(f"iPhone14 content counts changed: {dict(content_counts)}")
    if dict(lens_counts) != EXPECTED_LENS_COUNTS:
        raise ValueError(f"iPhone14 lens counts changed: {dict(lens_counts)}")
    if len(devices) != 10 or set(devices.values()) != {798, 799, 800}:
        raise ValueError(f"iPhone14 device counts changed: {dict(devices)}")
    detailed = {
        "schema_version": 1,
        "experiment": "E32/C4-R1b-csafe-iphone14-inventory",
        "state": "archive_inventory_passed_internal_rows_unselected",
        "source_id": "csafe-mcsidb-iphone14",
        "license": "CC BY 4.0",
        "acquisition_selection_sha256": _sha256(selection_raw),
        "download_receipt_sha256": _sha256(acquisition.CSAFE_RECEIPT.read_bytes()),
        "published_md5": receipt["md5"],
        "content_counts": dict(sorted(content_counts.items())),
        "lens_counts": dict(sorted(lens_counts.items())),
        "device_counts": dict(sorted(devices.items())),
        **archive,
        "boundary": "CRC/path inventory only; no member selected, extracted or assigned a role.",
    }
    raw = _write_atomic(INVENTORY, detailed)
    compact = {key: value for key, value in detailed.items() if key != "members"}
    compact.update(
        {
            "detailed_report_bytes": len(raw),
            "detailed_report_sha256": _sha256(raw),
            "detailed_report_external_path": str(INVENTORY.relative_to(acquisition.OUTPUT_ROOT)),
        }
    )
    _write_atomic(EVIDENCE_ROOT / "e32_r1b_csafe_iphone14_inventory.json", compact)
    return compact


def freeze_natural() -> dict[str, Any]:
    inventory_raw = INVENTORY.read_bytes()
    compact = json.loads((EVIDENCE_ROOT / "e32_r1b_csafe_iphone14_inventory.json").read_text())
    if compact.get("detailed_report_sha256") != _sha256(inventory_raw):
        raise ValueError("iPhone14 inventory binding changed")
    payload = json.loads(inventory_raw)
    if payload.get("state") != "archive_inventory_passed_internal_rows_unselected":
        raise ValueError("iPhone14 inventory state changed")
    records = []
    for member in payload["members"]:
        parsed = parse_member(str(member["name"]))
        if parsed["content_type"] != "natural":
            continue
        records.append(
            {
                "archive_member": member["name"],
                "expected_bytes": member["bytes"],
                "expected_crc32": member["crc32"],
                "device": parsed["device"],
                "lens": parsed["lens"],
                "camera_pipeline": f"{parsed['device']}:{parsed['lens']}",
                "parent_group": f"csafe-iphone14:{parsed['device']}:{parsed['lens']}:{PurePosixPath(parsed['filename']).stem}",
                "filename": parsed["filename"],
            }
        )
    records.sort(key=lambda row: row["archive_member"])
    if len(records) != EXPECTED_COUNTS["natural"]:
        raise ValueError("iPhone14 natural selection count changed")
    detailed = {
        "schema_version": 1,
        "experiment": "E32/C4-R1b-csafe-iphone14-natural-selection",
        "state": "natural_selection_frozen_no_member_bytes_read",
        "source_id": "csafe-mcsidb-iphone14",
        "inventory_sha256": compact["detailed_report_sha256"],
        "selected": len(records),
        "excluded_blank": EXPECTED_COUNTS["blank"],
        "device_counts": dict(sorted(Counter(row["device"] for row in records).items())),
        "lens_counts": dict(sorted(Counter(row["lens"] for row in records).items())),
        "records": records,
        "boundary": "Natural-only metadata selection; no member payload opened and no role assigned.",
    }
    raw = _write_atomic(SELECTION, detailed)
    result = {key: value for key, value in detailed.items() if key != "records"}
    result.update(
        {
            "detailed_report_bytes": len(raw),
            "detailed_report_sha256": _sha256(raw),
            "detailed_report_external_path": str(SELECTION.relative_to(acquisition.OUTPUT_ROOT)),
        }
    )
    _write_atomic(EVIDENCE_ROOT / "e32_r1b_csafe_iphone14_natural_selection.json", result)
    return result


def extract_natural() -> dict[str, Any]:
    selection_raw = SELECTION.read_bytes()
    compact = json.loads((EVIDENCE_ROOT / "e32_r1b_csafe_iphone14_natural_selection.json").read_text())
    if compact.get("detailed_report_sha256") != _sha256(selection_raw):
        raise ValueError("iPhone14 natural selection binding changed")
    selection = json.loads(selection_raw)
    if selection.get("state") != "natural_selection_frozen_no_member_bytes_read":
        raise ValueError("iPhone14 natural selection state changed")
    records = []
    with ZipFile(ARCHIVE) as archive:
        for index, selected in enumerate(selection["records"], start=1):
            member_name = str(selected["archive_member"])
            info = archive.getinfo(member_name)
            if info.file_size != int(selected["expected_bytes"]):
                raise ValueError(f"iPhone14 selected member size changed: {member_name}")
            if f"{info.CRC:08x}" != selected["expected_crc32"]:
                raise ValueError(f"iPhone14 selected member CRC changed: {member_name}")
            destination = (
                acquisition.OUTPUT_ROOT
                / "real"
                / "csafe_iphone14"
                / "natural"
                / selected["device"]
                / selected["lens"]
                / selected["filename"]
            )
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                if destination.stat().st_size != info.file_size:
                    raise ValueError(f"existing iPhone14 extraction changed: {destination}")
                state = "already_complete"
            else:
                partial = destination.with_suffix(destination.suffix + ".partial")
                if partial.exists():
                    raise ValueError(f"stale iPhone14 extraction partial: {partial}")
                with archive.open(info) as source, partial.open("xb") as target:
                    while chunk := source.read(CHUNK_BYTES):
                        target.write(chunk)
                if partial.stat().st_size != info.file_size:
                    raise ValueError(f"iPhone14 extracted size changed: {member_name}")
                os.replace(partial, destination)
                state = "extracted"
            records.append(
                {
                    "source_key": str(destination.relative_to(acquisition.OUTPUT_ROOT)),
                    "archive_member": member_name,
                    "device": selected["device"],
                    "lens": selected["lens"],
                    "camera_pipeline": selected["camera_pipeline"],
                    "parent_group": selected["parent_group"],
                    "bytes": info.file_size,
                    "sha256": _sha256_file(destination),
                    "state": state,
                }
            )
            if index % 200 == 0 or index == len(selection["records"]):
                print(f"CSAFE iPhone14 natural {index}/{len(selection['records'])} complete", flush=True)
    receipt = {
        "schema_version": 1,
        "experiment": "E32/C4-R1b-csafe-iphone14-natural-extraction",
        "state": "natural_extraction_complete_role_free",
        "selection_sha256": compact["detailed_report_sha256"],
        "parent_count": len(records),
        "bytes": sum(row["bytes"] for row in records),
        "records": records,
        "boundary": "Natural parents are role-free; no TRAIN/CALIBRATION assignment occurs here.",
    }
    raw = _write_atomic(EXTRACTION, receipt)
    result = {key: value for key, value in receipt.items() if key != "records"}
    result.update(
        {
            "detailed_report_bytes": len(raw),
            "detailed_report_sha256": _sha256(raw),
            "detailed_report_external_path": str(EXTRACTION.relative_to(acquisition.OUTPUT_ROOT)),
        }
    )
    _write_atomic(EVIDENCE_ROOT / "e32_r1b_csafe_iphone14_natural_extraction.json", result)
    return result


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("inventory", "freeze-natural", "extract-natural"))
    args = parser.parse_args(argv)
    result = {"inventory": inventory, "freeze-natural": freeze_natural, "extract-natural": extract_natural}[args.command]()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
