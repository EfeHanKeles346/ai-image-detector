"""Freeze and acquire the licensed E33 RRDataset benchmark archives."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import tarfile
from pathlib import Path
from typing import Any, Mapping

import requests

from pixelproof.project_paths import DATA_ROOT, ML_ROOT


RECORD_ID = 14963880
API_URL = f"https://zenodo.org/api/records/{RECORD_ID}"
RECORD_URL = f"https://zenodo.org/records/{RECORD_ID}"
OUTPUT_ROOT = DATA_ROOT / "e33_rrdataset"
SELECTION = OUTPUT_ROOT / "acquisition_selection.json"
EVIDENCE = ML_ROOT.parent / "evidence" / "e33_rrdataset_acquisition.json"
MIN_FREE_BYTES = 100 * 1024**3
MAX_MEMBER_BYTES = 100 * 1024**2
MAX_EXPANDED_BYTES = {"cal": 40 * 1024**3, "test": 240 * 1024**3}
TOP_LEVEL = {
    "cal": "RRDataset_original_train_val",
    "test": "RRDataset_test",
}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
FILES = {
    "cal": {
        "name": "RRDataset_original_train_val.tar.gz",
        "bytes": 2_163_176_547,
        "md5": "2f4498c3690d8f4c7a30d2e41dd34500",
        "role": "r1c_cal_candidate",
    },
    "test": {
        "name": "RRDataset_test.tar.gz",
        "bytes": 20_117_869_400,
        "md5": "13c3ff3d61986170cc0c8cf76a35cd4b",
        "role": "locked_final_test",
    },
}


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def _write_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_bytes(_json_bytes(value))
    temporary.replace(path)


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _md5(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024**2):
            digest.update(chunk)
    return digest.hexdigest()


def validate_record(payload: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    if int(payload.get("id", -1)) != RECORD_ID:
        raise ValueError("RRDataset Zenodo record id changed")
    if payload.get("metadata", {}).get("license", {}).get("id") != "cc-by-4.0":
        raise ValueError("RRDataset licence is not CC BY 4.0")
    published = {item.get("key"): item for item in payload.get("files", [])}
    selected: dict[str, dict[str, Any]] = {}
    for role, expected in FILES.items():
        item = published.get(expected["name"])
        if item is None:
            raise ValueError(f"RRDataset file disappeared: {expected['name']}")
        checksum = str(item.get("checksum", ""))
        found = (int(item.get("size", -1)), checksum)
        wanted = (expected["bytes"], f"md5:{expected['md5']}")
        if found != wanted:
            raise ValueError(f"RRDataset contract changed for {expected['name']}: {found}")
        url = str(item.get("links", {}).get("self", ""))
        if not url.startswith("https://zenodo.org/api/records/") or not url.endswith("/content"):
            raise ValueError(f"RRDataset download URL is not the pinned Zenodo content URL: {url}")
        selected[role] = {**expected, "url": url}
    return selected


def freeze() -> dict[str, Any]:
    response = requests.get(API_URL, timeout=(20, 120))
    response.raise_for_status()
    selected = validate_record(response.json())
    detailed = {
        "schema_version": 1,
        "experiment": "E33/R1c-RRDataset",
        "state": "selection_frozen_no_archive_bytes_claimed",
        "record_id": RECORD_ID,
        "record_url": RECORD_URL,
        "license": "CC BY 4.0",
        "label_invariant": {"real": 0, "ai": 1},
        "files": selected,
        "boundaries": [
            "Only the original train/validation archive may propose the R1c threshold.",
            "The test archive is locked until the R1c candidate contract is frozen.",
            "No owner-gallery or IPN score may select the R1c threshold.",
            "No archive image byte was downloaded by this metadata freeze.",
        ],
    }
    raw = _json_bytes(detailed)
    _write_atomic(SELECTION, detailed)
    compact = {
        "schema_version": 1,
        "experiment": detailed["experiment"],
        "state": detailed["state"],
        "record_id": RECORD_ID,
        "record_url": RECORD_URL,
        "license": detailed["license"],
        "detailed_selection_sha256": _sha256(raw),
        "detailed_selection_bytes": len(raw),
        "files": {
            role: {key: item[key] for key in ("name", "bytes", "md5", "role")}
            for role, item in selected.items()
        },
        "new_archive_bytes_downloaded_by_freeze": 0,
    }
    _write_atomic(EVIDENCE, compact)
    return compact


def _selection() -> dict[str, Any]:
    payload = json.loads(SELECTION.read_text())
    if payload.get("state") != "selection_frozen_no_archive_bytes_claimed":
        raise ValueError("E33 acquisition selection is not frozen")
    if payload.get("license") != "CC BY 4.0" or payload.get("record_id") != RECORD_ID:
        raise ValueError("E33 acquisition selection contract changed")
    return payload


def download(role: str) -> dict[str, Any]:
    selected = _selection()["files"][role]
    destination = OUTPUT_ROOT / "archives" / selected["name"]
    destination.parent.mkdir(parents=True, exist_ok=True)
    expected_bytes = int(selected["bytes"])
    expected_md5 = str(selected["md5"])
    if destination.exists():
        if destination.stat().st_size != expected_bytes or _md5(destination) != expected_md5:
            raise ValueError(f"completed archive contract mismatch: {destination}")
        state = "already_complete"
    else:
        partial = destination.with_suffix(destination.suffix + ".partial")
        current = partial.stat().st_size if partial.exists() else 0
        if current > expected_bytes:
            raise ValueError(f"partial exceeds expected size: {partial}")
        free = shutil.disk_usage(OUTPUT_ROOT).free
        remaining = expected_bytes - current
        if free < remaining + MIN_FREE_BYTES:
            raise OSError(f"insufficient free space: need {remaining + MIN_FREE_BYTES:,}, have {free:,}")
        command = [
            "/usr/bin/curl", "--fail", "--location", "--silent", "--show-error",
            "--connect-timeout", "30", "--retry", "8", "--retry-delay", "5",
            "--retry-all-errors", "--speed-limit", "1024", "--speed-time", "120",
            "--output", str(partial),
        ]
        if current:
            command.extend(["--continue-at", "-"])
        command.append(str(selected["url"]))
        subprocess.run(command, check=True)
        if partial.stat().st_size != expected_bytes:
            raise ValueError(f"download size mismatch: {destination}")
        if _md5(partial) != expected_md5:
            raise ValueError(f"download MD5 mismatch: {destination}")
        partial.replace(destination)
        state = "downloaded"
    receipt = {
        "schema_version": 1,
        "state": f"{role}_archive_complete_md5_verified",
        "transfer_state": state,
        "selection_sha256": _sha256(SELECTION.read_bytes()),
        "role": role,
        "path": str(destination.relative_to(OUTPUT_ROOT)),
        "bytes": expected_bytes,
        "md5": expected_md5,
    }
    _write_atomic(OUTPUT_ROOT / f"{role}_download_receipt.json", receipt)
    return receipt


def _safe_member_name(name: str, expected_top: str) -> Path:
    path = Path(name)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError(f"unsafe tar path: {name!r}")
    if path.parts[0] != expected_top:
        raise ValueError(f"unexpected tar root: {name!r}")
    return path


def inspect_members(
    members: list[tarfile.TarInfo], *, role: str
) -> dict[str, Any]:
    if role not in FILES:
        raise ValueError(f"unknown role {role!r}")
    names: set[str] = set()
    image_count = 0
    image_bytes = 0
    other_files: list[str] = []
    by_split_class: dict[str, int] = {}
    for member in members:
        path = _safe_member_name(member.name, TOP_LEVEL[role])
        if member.name in names:
            raise ValueError(f"duplicate tar member: {member.name}")
        names.add(member.name)
        if member.issym() or member.islnk() or member.isdev() or member.isfifo():
            raise ValueError(f"unsupported tar member type: {member.name}")
        if not (member.isdir() or member.isfile()):
            raise ValueError(f"unknown tar member type: {member.name}")
        if member.size < 0 or member.size > MAX_MEMBER_BYTES:
            raise ValueError(f"implausible tar member size: {member.name}")
        if not member.isfile():
            continue
        if path.suffix.lower() in IMAGE_SUFFIXES:
            if len(path.parts) < 4:
                raise ValueError(f"image lacks split/class path: {member.name}")
            split, class_name = path.parts[1:3]
            if class_name not in {"real", "ai"}:
                raise ValueError(f"image class is not explicit real/ai: {member.name}")
            key = f"{split}/{class_name}"
            by_split_class[key] = by_split_class.get(key, 0) + 1
            image_count += 1
            image_bytes += member.size
        else:
            other_files.append(member.name)
    if image_bytes > MAX_EXPANDED_BYTES[role]:
        raise ValueError("archive expansion exceeds the frozen safety ceiling")
    if not image_count:
        raise ValueError("archive contains no supported image")
    return {
        "member_count": len(members),
        "image_count": image_count,
        "image_bytes": image_bytes,
        "by_split_class": dict(sorted(by_split_class.items())),
        "other_files": sorted(other_files),
    }


def inventory(role: str) -> dict[str, Any]:
    selection = _selection()
    item = selection["files"][role]
    archive = OUTPUT_ROOT / "archives" / item["name"]
    if not archive.is_file() or archive.stat().st_size != int(item["bytes"]):
        raise FileNotFoundError(f"verified {role} archive is not complete")
    receipt_path = OUTPUT_ROOT / f"{role}_download_receipt.json"
    receipt = json.loads(receipt_path.read_text())
    if receipt.get("state") != f"{role}_archive_complete_md5_verified":
        raise ValueError(f"{role} download receipt is not verified")
    with tarfile.open(archive, mode="r:gz") as bundle:
        summary = inspect_members(bundle.getmembers(), role=role)
    result = {
        "schema_version": 1,
        "state": f"{role}_archive_inventory_passed",
        "selection_sha256": _sha256(SELECTION.read_bytes()),
        "download_receipt_sha256": _sha256(receipt_path.read_bytes()),
        "archive": {key: item[key] for key in ("name", "bytes", "md5", "role")},
        **summary,
    }
    _write_atomic(OUTPUT_ROOT / f"{role}_archive_inventory.json", result)
    return result


def extract_calibration() -> dict[str, Any]:
    inventory_path = OUTPUT_ROOT / "cal_archive_inventory.json"
    inventory_payload = json.loads(inventory_path.read_text())
    if inventory_payload.get("state") != "cal_archive_inventory_passed":
        raise ValueError("calibration archive inventory has not passed")
    archive = OUTPUT_ROOT / "archives" / FILES["cal"]["name"]
    final_root = OUTPUT_ROOT / "calibration"
    temporary_root = OUTPUT_ROOT / "calibration.partial"
    if final_root.exists():
        return {"state": "calibration_already_extracted", "path": str(final_root)}
    if temporary_root.exists():
        raise FileExistsError(f"partial extraction requires manual audit: {temporary_root}")
    temporary_root.mkdir(parents=True)
    extracted = 0
    extracted_bytes = 0
    try:
        with tarfile.open(archive, mode="r:gz") as bundle:
            for member in bundle:
                path = _safe_member_name(member.name, TOP_LEVEL["cal"])
                if not member.isfile() or len(path.parts) < 4 or path.parts[1] != "val":
                    continue
                if path.parts[2] not in {"real", "ai"} or path.suffix.lower() not in IMAGE_SUFFIXES:
                    continue
                relative = Path(*path.parts[2:])
                destination = temporary_root / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                source = bundle.extractfile(member)
                if source is None:
                    raise ValueError(f"cannot read tar member: {member.name}")
                with source, destination.open("xb") as output:
                    shutil.copyfileobj(source, output, length=8 * 1024**2)
                if destination.stat().st_size != member.size:
                    raise ValueError(f"extracted size mismatch: {member.name}")
                extracted += 1
                extracted_bytes += member.size
        expected = sum(
            int(count)
            for key, count in inventory_payload["by_split_class"].items()
            if key.startswith("val/")
        )
        if extracted != expected:
            raise ValueError(f"calibration extraction count mismatch: {extracted} != {expected}")
        temporary_root.replace(final_root)
    except Exception:
        raise
    result = {
        "schema_version": 1,
        "state": "calibration_extraction_complete",
        "inventory_sha256": _sha256(inventory_path.read_bytes()),
        "path": str(final_root.relative_to(OUTPUT_ROOT)),
        "image_count": extracted,
        "image_bytes": extracted_bytes,
    }
    _write_atomic(OUTPUT_ROOT / "calibration_extraction_receipt.json", result)
    return result


def status() -> dict[str, Any]:
    result: dict[str, Any] = {"selection_exists": SELECTION.exists(), "files": {}}
    for role, item in FILES.items():
        destination = OUTPUT_ROOT / "archives" / item["name"]
        partial = destination.with_suffix(destination.suffix + ".partial")
        result["files"][role] = {
            "complete_bytes": destination.stat().st_size if destination.exists() else 0,
            "partial_bytes": partial.stat().st_size if partial.exists() else 0,
            "expected_bytes": item["bytes"],
        }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=(
            "freeze", "download-cal", "download-test", "inventory-cal", "inventory-test",
            "extract-cal", "status",
        ),
    )
    args = parser.parse_args()
    if args.command == "freeze":
        result = freeze()
    elif args.command == "download-cal":
        result = download("cal")
    elif args.command == "download-test":
        result = download("test")
    elif args.command == "inventory-cal":
        result = inventory("cal")
    elif args.command == "inventory-test":
        result = inventory("test")
    elif args.command == "extract-cal":
        result = extract_calibration()
    else:
        result = status()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
