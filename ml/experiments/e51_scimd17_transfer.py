"""Acquire the frozen SCIMD-17 TRAIN-only archive without opening image bodies."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import ssl
from typing import Any, Iterable, Mapping
from urllib.error import HTTPError
from urllib.request import Request, urlopen
import zipfile

import certifi

from experiments.e51_transfer import ROUTE_CONTRACT_SHA256
from pixelproof.project_paths import DATA_ROOT, ML_ROOT


ZENODO_RECORD = 17_317_613
ZENODO_DOI = "10.5281/zenodo.17317613"
ZENODO_VERSION = "Version  1.1"
ZENODO_LICENSE = "cc-by-4.0"
ARCHIVE_NAME = "SCIMD-17.zip"
ARCHIVE_BYTES = 174_438_734
ARCHIVE_MD5 = "37da574c9e8d9c0fd3a7c9bedc5d72a6"
METADATA_URL = f"https://zenodo.org/api/records/{ZENODO_RECORD}"
CONTENT_URL = f"{METADATA_URL}/files/{ARCHIVE_NAME}/content"
MAX_MEMBER_BYTES = 20 * 1024**2
MAX_EXPANDED_BYTES = 4 * 1024**3
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}

ROOT = DATA_ROOT / "e51"
CONTRACT = ROOT / "route" / "contract_untransferred.json"
PAYLOAD = ROOT / "payloads" / "scimd17_train" / ARCHIVE_NAME
STAGING = ROOT / "staging" / "scimd17_train" / f"{ARCHIVE_NAME}.part"
RECEIPT = ROOT / "receipts" / "scimd17_archive_download_unscored.json"
EVIDENCE = ML_ROOT.parent / "evidence" / "e51_scimd17_download.json"


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def _write_atomic(path: Path, value: Any) -> bytes:
    raw = _json_bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_bytes(raw)
    temporary.replace(path)
    return raw


def _digest(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm, usedforsecurity=False)
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_record(payload: Mapping[str, Any]) -> dict[str, Any]:
    metadata = payload.get("metadata") or {}
    if (
        int(payload.get("id", -1)) != ZENODO_RECORD
        or payload.get("doi") != ZENODO_DOI
        or metadata.get("version") != ZENODO_VERSION
        or (metadata.get("license") or {}).get("id") != ZENODO_LICENSE
    ):
        raise ValueError("SCIMD-17 Zenodo identity/version/licence changed")
    files = {str(row.get("key")): row for row in payload.get("files") or []}
    item = files.get(ARCHIVE_NAME)
    if item is None:
        raise ValueError("SCIMD-17 archive disappeared")
    if (
        int(item.get("size", -1)) != ARCHIVE_BYTES
        or item.get("checksum") != f"md5:{ARCHIVE_MD5}"
        or (item.get("links") or {}).get("self") != CONTENT_URL
    ):
        raise ValueError("SCIMD-17 archive byte identity changed")
    return {"url": CONTENT_URL, "bytes": ARCHIVE_BYTES, "md5": ARCHIVE_MD5}


def summarize_archive(path: Path) -> dict[str, Any]:
    with zipfile.ZipFile(path) as archive:
        infos = archive.infolist()
    names = [info.filename for info in infos]
    if len(names) != len(set(names)):
        raise ValueError("SCIMD-17 archive contains duplicate paths")
    suffixes: Counter[str] = Counter()
    image_files = 0
    expanded_bytes = 0
    for info in infos:
        member = PurePosixPath(info.filename)
        mode = info.external_attr >> 16
        if member.is_absolute() or ".." in member.parts or (mode & 0o170000) == 0o120000:
            raise ValueError(f"unsafe SCIMD-17 archive member: {info.filename}")
        if info.file_size > MAX_MEMBER_BYTES:
            raise ValueError(f"oversized SCIMD-17 archive member: {info.filename}")
        expanded_bytes += info.file_size
        if not info.is_dir():
            suffix = member.suffix.lower()
            suffixes[suffix or "<none>"] += 1
            if suffix in IMAGE_SUFFIXES:
                image_files += 1
    if expanded_bytes > MAX_EXPANDED_BYTES or image_files == 0:
        raise ValueError("SCIMD-17 archive expansion/image inventory is unsafe")
    return {
        "members": len(infos),
        "files": sum(not info.is_dir() for info in infos),
        "image_files": image_files,
        "expanded_bytes": expanded_bytes,
        "suffixes": dict(sorted(suffixes.items())),
    }


def _open_json(url: str) -> dict[str, Any]:
    context = ssl.create_default_context(cafile=certifi.where())
    request = Request(url, headers={"User-Agent": "pixelproof-e51/1"})
    with urlopen(request, timeout=60, context=context) as response:
        return json.load(response)


def _download_resumable(url: str, destination: Path, expected_bytes: int) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    existing = destination.stat().st_size if destination.exists() else 0
    if existing > expected_bytes:
        destination.unlink()
        existing = 0
    if existing == expected_bytes:
        return
    headers = {"User-Agent": "pixelproof-e51/1"}
    if existing:
        headers["Range"] = f"bytes={existing}-"
    context = ssl.create_default_context(cafile=certifi.where())
    request = Request(url, headers=headers)
    try:
        response = urlopen(request, timeout=120, context=context)
    except HTTPError as error:
        if error.code == 416 and destination.stat().st_size == expected_bytes:
            return
        raise
    with response:
        status = response.getcode()
        if existing and status == 206:
            content_range = response.headers.get("Content-Range", "")
            if not content_range.startswith(f"bytes {existing}-"):
                raise ValueError("SCIMD-17 resume range changed")
            mode = "ab"
        else:
            existing = 0
            mode = "wb"
        with destination.open(mode) as target:
            shutil.copyfileobj(response, target, length=8 * 1024**2)
    if destination.stat().st_size != expected_bytes:
        raise ValueError("SCIMD-17 transfer ended at an unexpected byte count")


def download() -> dict[str, Any]:
    if RECEIPT.exists() or EVIDENCE.exists():
        raise FileExistsError("SCIMD-17 receipt already exists; no silent replacement")
    contract_raw = CONTRACT.read_bytes()
    if hashlib.sha256(contract_raw).hexdigest() != ROUTE_CONTRACT_SHA256:
        raise ValueError("E51 route contract changed before SCIMD-17 transfer")
    train = (json.loads(contract_raw).get("roles") or {}).get("train") or {}
    auxiliary = train.get("auxiliary") or {}
    if (
        auxiliary.get("source") != "SCIMD-17"
        or auxiliary.get("zenodo_record") != ZENODO_RECORD
        or auxiliary.get("archive_bytes") != ARCHIVE_BYTES
        or auxiliary.get("archive_md5") != ARCHIVE_MD5
        or auxiliary.get("restriction") != "224x224 resized REAL hard negatives; TRAIN only"
    ):
        raise ValueError("E51 SCIMD-17 TRAIN boundary changed")
    selected = validate_record(_open_json(METADATA_URL))
    free_before = shutil.disk_usage(ROOT).free
    if free_before < ARCHIVE_BYTES + 10 * 1024**3:
        raise OSError("insufficient free space for SCIMD-17 plus 10 GiB reserve")
    _download_resumable(selected["url"], STAGING, ARCHIVE_BYTES)
    if _digest(STAGING, "md5") != ARCHIVE_MD5:
        raise ValueError("SCIMD-17 archive MD5 changed")
    archive_sha256 = _digest(STAGING, "sha256")
    inventory = summarize_archive(STAGING)
    PAYLOAD.parent.mkdir(parents=True, exist_ok=True)
    STAGING.replace(PAYLOAD)
    payload = {
        "schema_version": 1,
        "state": "e51_scimd17_train_archive_downloaded_unopened_unscored",
        "role": "TRAIN_REAL_HARD_NEGATIVE_ONLY",
        "restriction": "224x224 resized REAL hard negatives; never CAL, DEVELOPMENT or final",
        "route_contract_sha256": ROUTE_CONTRACT_SHA256,
        "zenodo_record": ZENODO_RECORD,
        "doi": ZENODO_DOI,
        "version": ZENODO_VERSION,
        "license": ZENODO_LICENSE,
        "archive_name": ARCHIVE_NAME,
        "archive_bytes": ARCHIVE_BYTES,
        "archive_md5": ARCHIVE_MD5,
        "archive_sha256": archive_sha256,
        "inventory": inventory,
        "free_bytes_before": free_before,
        "free_bytes_after": shutil.disk_usage(ROOT).free,
        "image_bodies_decoded": 0,
        "model_scores_created": 0,
    }
    raw = _write_atomic(RECEIPT, payload)
    evidence = {**payload, "inventory": inventory, "receipt_bytes": len(raw),
                "receipt_sha256": hashlib.sha256(raw).hexdigest()}
    _write_atomic(EVIDENCE, evidence)
    return evidence


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("download",))
    parser.parse_args(argv)
    print(json.dumps(download(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
