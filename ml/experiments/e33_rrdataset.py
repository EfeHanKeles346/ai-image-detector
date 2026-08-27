"""Freeze and acquire the licensed E33 RRDataset benchmark archives."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
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
    parser.add_argument("command", choices=("freeze", "download-cal", "download-test", "status"))
    args = parser.parse_args()
    if args.command == "freeze":
        result = freeze()
    elif args.command == "download-cal":
        result = download("cal")
    elif args.command == "download-test":
        result = download("test")
    else:
        result = status()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
