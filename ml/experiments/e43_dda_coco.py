"""Acquire and audit the official COCO real companion for E43 DDA-COCO."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
from typing import Any, Iterable, Mapping, Sequence
import zipfile

import requests

from pixelproof.project_paths import DATA_ROOT, ML_ROOT


ROOT = DATA_ROOT / "e43_dda_coco"
COCO_URL = "https://s3.amazonaws.com/images.cocodataset.org/zips/val2017.zip"
COCO_BYTES = 815_585_330
COCO_ETAG = '"d366be60d3dc737327160d62453e3973-98"'
COCO_ARCHIVE = ROOT / "archives" / "val2017.zip"
COCO_RECEIPT = ROOT / "real_download_receipt.json"
COCO_INVENTORY = ROOT / "real_inventory.json"
STRUCTURE_REPORT = ROOT / "paired_structure.json"
STRUCTURE_EVIDENCE = ML_ROOT.parent / "evidence" / "e43_dda_coco_structure.json"
DDA_ARCHIVE = DATA_ROOT / "e34_dda_coco" / "archives" / "DDA-COCO.zip"
DDA_SHA256 = "8cd600779aaecef21605b07bff9ab3963a7fb9b9614a3d9a0588cd4a5e099c24"
MIN_FREE_BYTES = 100 * 1024**3
REAL_RE = re.compile(r"^val2017/(\d{12})\.jpg$")
SYNTHETIC_RE = re.compile(r"^DDA-COCO/([^/]+)/val2017/(\d{12})\.jpg$")
VARIANTS = (
    "FLUX.1",
    "sd-vae-ft-ema",
    "sd-vae-ft-mse",
    "sdxl-vae",
    "stable-diffusion-2-1",
    "stable-diffusion-3.5-large",
)
EXPECTED_VARIANT_COUNTS = {
    "FLUX.1": 4_971,
    "sd-vae-ft-ema": 5_000,
    "sd-vae-ft-mse": 5_000,
    "sdxl-vae": 5_000,
    "stable-diffusion-2-1": 5_000,
    "stable-diffusion-3.5-large": 4_998,
}


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024**2):
            digest.update(chunk)
    return digest.hexdigest()


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_bytes(_json_bytes(value))
    temporary.replace(path)


def validate_coco_headers(headers: Mapping[str, str]) -> dict[str, Any]:
    if int(headers.get("content-length", -1)) != COCO_BYTES:
        raise ValueError("COCO val2017 size changed")
    if headers.get("etag") != COCO_ETAG:
        raise ValueError("COCO val2017 ETag changed")
    return {"bytes": COCO_BYTES, "etag": COCO_ETAG, "last_modified": headers.get("last-modified")}


def download_real() -> dict[str, Any]:
    if COCO_RECEIPT.exists():
        raise FileExistsError("COCO val2017 receipt already exists; no silent replacement")
    response = requests.head(COCO_URL, timeout=(20, 120))
    response.raise_for_status()
    source = validate_coco_headers({key.lower(): value for key, value in response.headers.items()})
    COCO_ARCHIVE.parent.mkdir(parents=True, exist_ok=True)
    partial = COCO_ARCHIVE.with_suffix(".zip.partial")
    current = partial.stat().st_size if partial.exists() else 0
    if current > COCO_BYTES:
        raise ValueError("COCO val2017 partial exceeds expected size")
    remaining = COCO_BYTES - current
    if shutil.disk_usage(ROOT).free < remaining + MIN_FREE_BYTES:
        raise OSError("insufficient free space for COCO val2017")
    command = [
        "/usr/bin/curl", "--fail", "--location", "--silent", "--show-error",
        "--connect-timeout", "30", "--retry", "8", "--retry-delay", "5",
        "--retry-all-errors", "--speed-limit", "1024", "--speed-time", "120",
        "--output", str(partial),
    ]
    if current:
        command.extend(["--continue-at", "-"])
    command.append(COCO_URL)
    subprocess.run(command, check=True)
    if partial.stat().st_size != COCO_BYTES:
        raise ValueError("COCO val2017 download size mismatch")
    sha256 = _digest(partial)
    partial.replace(COCO_ARCHIVE)
    receipt = {
        "schema_version": 1,
        "state": "coco_val2017_download_complete_sha256_bound",
        "url": COCO_URL,
        **source,
        "sha256": sha256,
        "network_bytes": remaining,
        "model_scores_created": 0,
    }
    _write(COCO_RECEIPT, receipt)
    return receipt


def real_ids_from_infos(infos: Sequence[zipfile.ZipInfo]) -> set[str]:
    ids: set[str] = set()
    for info in infos:
        if info.is_dir():
            continue
        path = PurePosixPath(info.filename)
        if path.is_absolute() or ".." in path.parts or info.flag_bits & 0x1:
            raise ValueError(f"unsafe COCO ZIP member: {info.filename}")
        match = REAL_RE.fullmatch(info.filename)
        if match is None or info.file_size <= 0 or info.file_size > 50 * 1024**2:
            raise ValueError(f"unexpected COCO member: {info.filename}")
        if match.group(1) in ids:
            raise ValueError("duplicate COCO image ID")
        ids.add(match.group(1))
    if len(ids) != 5_000:
        raise ValueError(f"COCO val2017 count changed: {len(ids)}")
    return ids


def inventory_real() -> dict[str, Any]:
    if COCO_INVENTORY.exists():
        raise FileExistsError("COCO val2017 inventory already exists")
    receipt = json.loads(COCO_RECEIPT.read_text())
    if receipt.get("state") != "coco_val2017_download_complete_sha256_bound":
        raise ValueError("COCO val2017 receipt state changed")
    if COCO_ARCHIVE.stat().st_size != COCO_BYTES or _digest(COCO_ARCHIVE) != receipt["sha256"]:
        raise ValueError("COCO val2017 archive changed after acquisition")
    with zipfile.ZipFile(COCO_ARCHIVE) as bundle:
        infos = bundle.infolist()
        ids = real_ids_from_infos(infos)
        bad = bundle.testzip()
        expanded = sum(info.file_size for info in infos if not info.is_dir())
    if bad is not None:
        raise ValueError(f"COCO val2017 CRC failure: {bad}")
    report = {
        "schema_version": 1,
        "state": "coco_val2017_inventory_passed",
        "archive_sha256": receipt["sha256"],
        "image_count": len(ids),
        "expanded_bytes": expanded,
        "id_list_sha256": hashlib.sha256(("\n".join(sorted(ids)) + "\n").encode()).hexdigest(),
        "zip_crc_passed": True,
        "model_scores_created": 0,
    }
    _write(COCO_INVENTORY, report)
    return report


def synthetic_ids_from_names(names: Iterable[str]) -> dict[str, set[str]]:
    grouped = {variant: set() for variant in VARIANTS}
    for name in names:
        if name.endswith("/"):
            continue
        match = SYNTHETIC_RE.fullmatch(name)
        if match is None or match.group(1) not in grouped:
            raise ValueError(f"unexpected DDA-COCO member: {name}")
        variant, parent = match.groups()
        if parent in grouped[variant]:
            raise ValueError("duplicate DDA-COCO parent within variant")
        grouped[variant].add(parent)
    counts = {variant: len(ids) for variant, ids in grouped.items()}
    if counts != EXPECTED_VARIANT_COUNTS:
        raise ValueError(f"DDA-COCO variant counts changed: {counts}")
    return grouped


def audit_structure() -> dict[str, Any]:
    if STRUCTURE_REPORT.exists() or STRUCTURE_EVIDENCE.exists():
        raise FileExistsError("DDA-COCO paired structure already exists")
    inventory = json.loads(COCO_INVENTORY.read_text())
    if inventory.get("state") != "coco_val2017_inventory_passed":
        raise ValueError("COCO val2017 inventory state changed")
    if DDA_ARCHIVE.stat().st_size != 4_301_452_066 or _digest(DDA_ARCHIVE) != DDA_SHA256:
        raise ValueError("DDA-COCO archive changed")
    with zipfile.ZipFile(COCO_ARCHIVE) as real_bundle:
        real_ids = real_ids_from_infos(real_bundle.infolist())
    with zipfile.ZipFile(DDA_ARCHIVE) as synthetic_bundle:
        grouped = synthetic_ids_from_names(info.filename for info in synthetic_bundle.infolist())
    if any(not ids <= real_ids for ids in grouped.values()):
        raise ValueError("DDA-COCO contains a parent absent from official COCO val2017")
    complete = set.intersection(real_ids, *grouped.values())
    report = {
        "schema_version": 1,
        "state": "e43_dda_coco_structure_passed_before_decode",
        "real_archive_sha256": inventory["archive_sha256"],
        "dda_archive_sha256": DDA_SHA256,
        "real_count": len(real_ids),
        "variant_counts": {variant: len(grouped[variant]) for variant in VARIANTS},
        "complete_parent_count": len(complete),
        "complete_row_count": len(complete) * (1 + len(VARIANTS)),
        "complete_parent_ids_sha256": hashlib.sha256(
            ("\n".join(sorted(complete)) + "\n").encode()
        ).hexdigest(),
        "model_scores_created": 0,
        "next": "Decode/hash complete seven-view parent groups and audit protected overlap before scoring.",
    }
    _write(STRUCTURE_REPORT, report)
    _write(STRUCTURE_EVIDENCE, report)
    return report


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("download-real", "inventory-real", "audit-structure"))
    args = parser.parse_args(argv)
    actions = {
        "download-real": download_real,
        "inventory-real": inventory_real,
        "audit-structure": audit_structure,
    }
    print(json.dumps(actions[args.command](), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
