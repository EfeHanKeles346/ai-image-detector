"""Acquire and inventory the official MediaEval ITW-SM validation archive.

This stage is intentionally score-blind.  It binds the public source, resumes the
archive transfer, verifies the immutable HTTP identity and performs a safe ZIP/CRC
inventory.  It never extracts an image and never loads a detector.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import subprocess
from typing import Any, Iterable, Mapping, Sequence
import zipfile
import zlib

import requests

from pixelproof.project_paths import DATA_ROOT, ML_ROOT


URL = "https://artifacts.mever.iti.gr/MediaEval2025/itw-sm-sid-val.zip"
SOURCE_REPOSITORY = "https://github.com/mever-team/mediaeval2026-sid"
EXPECTED_BYTES = 3_553_693_205
EXPECTED_ETAG = '"68555a02-d3d10e15"'
EXPECTED_LAST_MODIFIED = "Fri, 20 Jun 2025 12:54:26 GMT"
EXPECTED_IMAGES = {"0_real": 5_000, "1_fake": 5_000}
MIN_FREE_BYTES = 100 * 1024**3

ROOT = DATA_ROOT / "e45_mediaeval_itwsm"
ARCHIVE = ROOT / "archives" / "itw-sm-sid-val.zip"
RECEIPT = ROOT / "acquisition_receipt.json"
INVENTORY = ROOT / "unscored_inventory.json"
EVIDENCE = ML_ROOT.parent / "evidence" / "e45_mediaeval_acquisition.json"


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def _write_atomic(path: Path, value: Any) -> bytes:
    raw = _json_bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_bytes(raw)
    temporary.replace(path)
    return raw


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024**2):
            digest.update(chunk)
    return digest.hexdigest()


def validate_headers(headers: Mapping[str, str]) -> dict[str, Any]:
    normalized = {key.lower(): value for key, value in headers.items()}
    if int(normalized.get("content-length", -1)) != EXPECTED_BYTES:
        raise ValueError("MediaEval archive size changed")
    if normalized.get("etag") != EXPECTED_ETAG:
        raise ValueError("MediaEval archive ETag changed")
    if normalized.get("last-modified") != EXPECTED_LAST_MODIFIED:
        raise ValueError("MediaEval archive Last-Modified changed")
    if normalized.get("accept-ranges", "").lower() != "bytes":
        raise ValueError("MediaEval archive no longer advertises byte-range resume")
    return {
        "bytes": EXPECTED_BYTES,
        "etag": EXPECTED_ETAG,
        "last_modified": EXPECTED_LAST_MODIFIED,
        "accept_ranges": "bytes",
    }


def classify_member(name: str) -> tuple[str, str]:
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError(f"unsafe MediaEval ZIP member: {name!r}")
    parts = path.parts
    if len(parts) == 2 and parts[0] in EXPECTED_IMAGES:
        group = parts[0]
    elif len(parts) == 3 and parts[1] in EXPECTED_IMAGES:
        # Some official archives wrap the two declared class folders in one root.
        group = parts[1]
    else:
        raise ValueError(f"unexpected MediaEval ZIP member: {name!r}")
    if path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
        raise ValueError(f"unexpected MediaEval image suffix: {name!r}")
    return group, "image"


def inspect_infos(infos: Sequence[zipfile.ZipInfo]) -> dict[str, Any]:
    files = [info for info in infos if not info.is_dir()]
    if len({info.filename for info in files}) != len(files):
        raise ValueError("duplicate MediaEval ZIP member name")
    counts: Counter[str] = Counter()
    roots: set[str] = set()
    expanded = 0
    for info in files:
        path = PurePosixPath(info.filename)
        if info.flag_bits & 0x1:
            raise ValueError(f"encrypted MediaEval ZIP member: {info.filename!r}")
        group, kind = classify_member(info.filename)
        if kind != "image" or info.file_size <= 0 or info.file_size > 100 * 1024**2:
            raise ValueError(f"invalid MediaEval image member: {info.filename!r}")
        counts[group] += 1
        expanded += info.file_size
        roots.add(path.parts[0] if len(path.parts) == 3 else "")
    if dict(counts) != EXPECTED_IMAGES:
        raise ValueError(f"MediaEval class counts changed: {dict(counts)}")
    if len(roots) != 1:
        raise ValueError(f"MediaEval ZIP has inconsistent roots: {sorted(roots)}")
    return {
        "files": len(files),
        "images_by_class": dict(sorted(counts.items())),
        "expanded_bytes": expanded,
        "archive_root": next(iter(roots)),
    }


def crc_failures(bundle: zipfile.ZipFile) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    for index, info in enumerate(bundle.infolist()):
        if info.is_dir():
            continue
        try:
            with bundle.open(info) as stream:
                while stream.read(8 * 1024**2):
                    pass
        except (OSError, RuntimeError, zipfile.BadZipFile, zlib.error) as error:
            failures.append({
                "index": index,
                "member": info.filename,
                "header_offset": info.header_offset,
                "compressed_bytes": info.compress_size,
                "expanded_bytes": info.file_size,
                "error_type": type(error).__name__,
                "error": str(error),
            })
    return failures


def _head() -> dict[str, Any]:
    response = requests.head(URL, allow_redirects=True, timeout=(20, 120))
    response.raise_for_status()
    if response.url != URL:
        raise ValueError(f"MediaEval source redirected unexpectedly: {response.url}")
    return validate_headers(response.headers)


def acquire() -> dict[str, Any]:
    if RECEIPT.exists() or EVIDENCE.exists():
        raise FileExistsError("MediaEval acquisition receipt already exists; no silent replacement")
    source = _head()
    ARCHIVE.parent.mkdir(parents=True, exist_ok=True)
    partial = ARCHIVE.with_suffix(ARCHIVE.suffix + ".partial")
    current = partial.stat().st_size if partial.exists() else 0
    if current > EXPECTED_BYTES:
        raise ValueError("MediaEval partial exceeds expected size")
    remaining = EXPECTED_BYTES - current
    free_before = shutil.disk_usage(ROOT).free
    if free_before < remaining + MIN_FREE_BYTES:
        raise OSError("insufficient free space for MediaEval archive plus 100 GiB reserve")
    command = [
        "/usr/bin/curl",
        "--fail",
        "--location",
        "--silent",
        "--show-error",
        "--connect-timeout",
        "30",
        "--retry",
        "12",
        "--retry-delay",
        "5",
        "--retry-all-errors",
        "--speed-limit",
        "1024",
        "--speed-time",
        "180",
        "--output",
        str(partial),
    ]
    if current:
        command.extend(["--continue-at", "-"])
    command.append(URL)
    subprocess.run(command, check=True)
    if partial.stat().st_size != EXPECTED_BYTES:
        raise ValueError("MediaEval archive size mismatch after transfer")
    sha256 = _digest(partial)
    partial.replace(ARCHIVE)
    receipt = {
        "schema_version": 1,
        "state": "mediaeval_itwsm_archive_complete_unscored",
        "source_repository": SOURCE_REPOSITORY,
        "url": URL,
        **source,
        "sha256": sha256,
        "path": str(ARCHIVE),
        "network_bytes_this_run": remaining,
        "free_bytes_before": free_before,
        "free_bytes_after": shutil.disk_usage(ROOT).free,
        "role": "E45_UNTOUCHED_FINAL",
        "terms": "Research-only ITW-SM use; no redistribution; preserve accepted terms.",
        "model_scores_created": 0,
    }
    _write_atomic(RECEIPT, receipt)
    return receipt


def inventory() -> dict[str, Any]:
    if INVENTORY.exists():
        raise FileExistsError("MediaEval inventory already exists; no silent replacement")
    receipt = json.loads(RECEIPT.read_text())
    if receipt.get("state") != "mediaeval_itwsm_archive_complete_unscored":
        raise ValueError("MediaEval acquisition receipt state changed")
    if ARCHIVE.stat().st_size != EXPECTED_BYTES or _digest(ARCHIVE) != receipt["sha256"]:
        raise ValueError("MediaEval archive changed after acquisition")
    with zipfile.ZipFile(ARCHIVE) as bundle:
        summary = inspect_infos(bundle.infolist())
        failures = crc_failures(bundle)
    usable_files = summary["files"] - len(failures)
    state = (
        "mediaeval_itwsm_inventory_passed_unscored"
        if not failures
        else "mediaeval_itwsm_inventory_crc_failure_unscored"
    )
    report = {
        "schema_version": 1,
        "state": state,
        "archive_sha256": receipt["sha256"],
        "archive_bytes": EXPECTED_BYTES,
        **summary,
        "usable_files": usable_files,
        "official_archive_coverage": usable_files / summary["files"],
        "crc_failures": failures,
        "zip_crc_passed": not failures,
        "images_decoded": 0,
        "model_scores_created": 0,
        "boundary": (
            "Archive and per-member CRC inventory only; no image was extracted or scored. "
            "Any failed member is ineligible for the zero-score manifest."
        ),
    }
    raw = _write_atomic(INVENTORY, report)
    evidence = {
        **report,
        "source_repository": SOURCE_REPOSITORY,
        "url": URL,
        "detailed_inventory_bytes": len(raw),
        "detailed_inventory_sha256": hashlib.sha256(raw).hexdigest(),
    }
    _write_atomic(EVIDENCE, evidence)
    return evidence


def status() -> dict[str, Any]:
    partial = ARCHIVE.with_suffix(ARCHIVE.suffix + ".partial")
    return {
        "archive_exists": ARCHIVE.is_file(),
        "archive_bytes": ARCHIVE.stat().st_size if ARCHIVE.is_file() else 0,
        "partial_exists": partial.is_file(),
        "partial_bytes": partial.stat().st_size if partial.is_file() else 0,
        "receipt_exists": RECEIPT.is_file(),
        "inventory_exists": INVENTORY.is_file(),
        "expected_bytes": EXPECTED_BYTES,
        "model_scores_created": 0,
    }


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("status", "acquire", "inventory"))
    args = parser.parse_args(argv)
    result = {"status": status, "acquire": acquire, "inventory": inventory}[args.command]()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
