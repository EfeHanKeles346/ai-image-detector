"""Acquire the score-blind E46 development and final sources.

SynthWildX is downloaded file-by-file from the publisher's frozen X URL list.
The TrueFake Facebook archive is transferred with Google Drive confirmation and
byte-range resume.  This module validates bytes and image structure only; it
never imports or runs a detector.
"""

from __future__ import annotations

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import html
import io
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
from typing import Any, Iterable, Mapping, Sequence

from PIL import Image
import requests

from pixelproof.project_paths import DATA_ROOT, ML_ROOT


SYNTHWILDX_LIST_URL = (
    "https://raw.githubusercontent.com/grip-unina/"
    "ClipBased-SyntheticImageDetection/main/data/synthwildx/list.csv"
)
SYNTHWILDX_LIST_BYTES = 206_017
SYNTHWILDX_LIST_SHA256 = "a40a374e5fdd9531b0927facf0e047f9e50c3a379936c37eec5e13bbd9a28188"
SYNTHWILDX_COUNTS = {"dalle3": 500, "firefly": 500, "midjourney_v5": 500, "real": 500}
SYNTHWILDX_NAMESPACE = "E46_SYNTHWILDX_ROLE_V1"

TRUEFAKE_FILE_ID = "10cQq48JtpRZgrHuckMyeFOwPvZZHDMXd"
TRUEFAKE_LANDING_URL = f"https://drive.usercontent.google.com/download?id={TRUEFAKE_FILE_ID}"
TRUEFAKE_BYTES = 4_207_525_545
TRUEFAKE_LAST_MODIFIED = "Sat, 01 Nov 2025 18:21:12 GMT"
TRUEFAKE_FILENAME = "Facebook.tar.gz"
MIN_FREE_BYTES = 100 * 1024**3

ROOT = DATA_ROOT / "e46"
SYNTHWILDX_ROOT = ROOT / "synthwildx"
SYNTHWILDX_LIST = SYNTHWILDX_ROOT / "list.csv"
SYNTHWILDX_IMAGES = SYNTHWILDX_ROOT / "images"
SYNTHWILDX_MANIFEST = SYNTHWILDX_ROOT / "manifest_unscored.json"
SYNTHWILDX_RECEIPT = SYNTHWILDX_ROOT / "acquisition_receipt.json"
TRUEFAKE_ROOT = ROOT / "truefake_facebook"
TRUEFAKE_ARCHIVE = TRUEFAKE_ROOT / TRUEFAKE_FILENAME
TRUEFAKE_RECEIPT = TRUEFAKE_ROOT / "acquisition_receipt.json"
EVIDENCE = ML_ROOT.parent / "evidence" / "e46_acquisition.json"


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def _write_atomic(path: Path, value: Any) -> bytes:
    raw = _json_bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_bytes(raw)
    temporary.replace(path)
    return raw


def _digest_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024**2):
            digest.update(chunk)
    return digest.hexdigest()


def parse_synthwildx_list(raw: bytes) -> list[dict[str, str]]:
    if len(raw) != SYNTHWILDX_LIST_BYTES or _digest_bytes(raw) != SYNTHWILDX_LIST_SHA256:
        raise ValueError("SynthWildX publisher list changed")
    reader = csv.DictReader(io.StringIO(raw.decode("utf-8")))
    if reader.fieldnames != ["url", "filename", "typ"]:
        raise ValueError(f"SynthWildX columns changed: {reader.fieldnames}")
    rows: list[dict[str, str]] = []
    names: set[str] = set()
    counts = {name: 0 for name in SYNTHWILDX_COUNTS}
    for source in reader:
        name = source["filename"]
        path = PurePosixPath(name)
        typ = source["typ"]
        if (
            path.is_absolute() or ".." in path.parts or len(path.parts) != 2
            or path.parts[0] != typ or path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}
        ):
            raise ValueError(f"unsafe SynthWildX filename: {name!r}")
        if not source["url"].startswith("https://pbs.twimg.com/media/"):
            raise ValueError(f"unexpected SynthWildX host: {source['url']!r}")
        if typ not in counts or name in names:
            raise ValueError(f"invalid SynthWildX row: {source!r}")
        counts[typ] += 1
        names.add(name)
        rows.append(dict(source))
    if counts != SYNTHWILDX_COUNTS:
        raise ValueError(f"SynthWildX counts changed: {counts}")
    return rows


def assign_roles(rows: Sequence[Mapping[str, str]]) -> dict[str, str]:
    """Assign exactly 60/40 CAL/DEVELOPMENT within every publisher type."""
    output: dict[str, str] = {}
    for typ, expected in SYNTHWILDX_COUNTS.items():
        names = [row["filename"] for row in rows if row["typ"] == typ]
        names.sort(key=lambda name: (hashlib.sha256(
            f"{SYNTHWILDX_NAMESPACE}|{typ}|{name}".encode()
        ).digest(), name))
        if len(names) != expected:
            raise ValueError(f"SynthWildX role input changed for {typ}")
        cut = 3 * expected // 5
        output.update({name: "CAL" if index < cut else "DEVELOPMENT"
                       for index, name in enumerate(names)})
    return output


def _validate_image(raw: bytes) -> dict[str, Any]:
    if not raw or len(raw) > 100 * 1024**2:
        raise ValueError("invalid SynthWildX payload size")
    with Image.open(io.BytesIO(raw)) as opened:
        opened.verify()
    with Image.open(io.BytesIO(raw)) as opened:
        width, height = opened.size
        image_format = opened.format
    if width < 16 or height < 16:
        raise ValueError("SynthWildX image geometry is too small")
    return {"bytes": len(raw), "sha256": _digest_bytes(raw), "width": width, "height": height,
            "format": image_format}


def _download_one(row: Mapping[str, str]) -> dict[str, Any]:
    target = SYNTHWILDX_IMAGES / row["filename"]
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file():
        raw = target.read_bytes()
        return {**_validate_image(raw), "network": "reused"}
    headers = {"User-Agent": "Mozilla/5.0 (PixelProof academic research)"}
    error = "unknown failure"
    for _ in range(4):
        try:
            response = requests.get(row["url"], headers=headers, timeout=(20, 120))
            response.raise_for_status()
            facts = _validate_image(response.content)
            partial = target.with_suffix(target.suffix + ".part")
            partial.write_bytes(response.content)
            partial.replace(target)
            return {**facts, "network": "downloaded", "resolved_url": response.url}
        except (OSError, requests.RequestException, ValueError) as caught:
            error = f"{type(caught).__name__}: {caught}"
    return {"error": error, "network": "failed"}


def acquire_synthwildx(workers: int = 12) -> dict[str, Any]:
    if SYNTHWILDX_RECEIPT.exists() or SYNTHWILDX_MANIFEST.exists():
        raise FileExistsError("SynthWildX receipt already exists; no silent replacement")
    response = requests.get(SYNTHWILDX_LIST_URL, timeout=(20, 120))
    response.raise_for_status()
    raw = response.content
    rows = parse_synthwildx_list(raw)
    roles = assign_roles(rows)
    SYNTHWILDX_ROOT.mkdir(parents=True, exist_ok=True)
    SYNTHWILDX_LIST.write_bytes(raw)
    results: dict[str, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_download_one, row): row["filename"] for row in rows}
        for index, future in enumerate(as_completed(futures), start=1):
            results[futures[future]] = future.result()
            if index % 100 == 0 or index == len(rows):
                print(f"SynthWildX {index}/{len(rows)}", flush=True)
    manifest_rows = []
    for row in rows:
        result = results[row["filename"]]
        manifest_rows.append({
            **dict(row),
            "label": 0 if row["typ"] == "real" else 1,
            "role": roles[row["filename"]],
            "path": str(SYNTHWILDX_IMAGES / row["filename"]),
            "status": "ok" if "sha256" in result else "failed",
            **result,
        })
    successful = [row for row in manifest_rows if row["status"] == "ok"]
    failures = [row for row in manifest_rows if row["status"] != "ok"]
    manifest_raw = _write_atomic(SYNTHWILDX_MANIFEST, {"schema_version": 1, "rows": manifest_rows})
    receipt = {
        "schema_version": 1,
        "state": "e46_synthwildx_complete_unscored" if not failures else "e46_synthwildx_partial_unscored",
        "source_list_url": SYNTHWILDX_LIST_URL,
        "source_list_bytes": len(raw),
        "source_list_sha256": _digest_bytes(raw),
        "declared_rows": len(rows),
        "successful_rows": len(successful),
        "failed_rows": len(failures),
        "successful_bytes": sum(int(row["bytes"]) for row in successful),
        "role_counts": {
            role: sum(row["status"] == "ok" and row["role"] == role for row in manifest_rows)
            for role in ("CAL", "DEVELOPMENT")
        },
        "type_counts": {
            typ: sum(row["status"] == "ok" and row["typ"] == typ for row in manifest_rows)
            for typ in SYNTHWILDX_COUNTS
        },
        "manifest_bytes": len(manifest_raw),
        "manifest_sha256": _digest_bytes(manifest_raw),
        "model_scores_created": 0,
        "failures": [{"filename": row["filename"], "error": row["error"]} for row in failures],
    }
    _write_atomic(SYNTHWILDX_RECEIPT, receipt)
    _update_evidence()
    return receipt


def parse_drive_landing(body: str) -> dict[str, str]:
    fields = {key: html.unescape(value) for key, value in re.findall(
        r'name="([^"]+)" value="([^"]*)"', body
    )}
    match = re.search(r">([^<]+\.tar\.gz)</a> \(([^)]+)\)", body)
    if fields.get("id") != TRUEFAKE_FILE_ID or fields.get("confirm") != "t" or not fields.get("uuid"):
        raise ValueError("TrueFake Google Drive confirmation changed")
    if not match or match.group(1) != TRUEFAKE_FILENAME or match.group(2) != "3.9G":
        raise ValueError("TrueFake advertised file identity changed")
    return fields


def _truefake_source() -> tuple[str, dict[str, Any]]:
    session = requests.Session()
    landing = session.get(TRUEFAKE_LANDING_URL, timeout=(20, 120))
    landing.raise_for_status()
    fields = parse_drive_landing(landing.text)
    prepared = requests.Request(
        "GET", "https://drive.usercontent.google.com/download", params=fields
    ).prepare().url
    probe = session.get(prepared, headers={"Range": "bytes=0-0"}, stream=True, timeout=(20, 120))
    try:
        probe.raise_for_status()
        if probe.status_code != 206 or probe.headers.get("content-range") != f"bytes 0-0/{TRUEFAKE_BYTES}":
            raise ValueError("TrueFake archive byte identity changed")
        if probe.headers.get("last-modified") != TRUEFAKE_LAST_MODIFIED:
            raise ValueError("TrueFake Last-Modified changed")
        if probe.headers.get("accept-ranges", "").lower() != "bytes":
            raise ValueError("TrueFake source no longer supports resume")
        facts = {
            "bytes": TRUEFAKE_BYTES,
            "last_modified": TRUEFAKE_LAST_MODIFIED,
            "accept_ranges": "bytes",
            "content_disposition": probe.headers.get("content-disposition"),
        }
    finally:
        probe.close()
    return prepared, facts


def acquire_truefake() -> dict[str, Any]:
    if TRUEFAKE_RECEIPT.exists():
        raise FileExistsError("TrueFake receipt already exists; no silent replacement")
    confirmed_url, source = _truefake_source()
    TRUEFAKE_ROOT.mkdir(parents=True, exist_ok=True)
    partial = TRUEFAKE_ARCHIVE.with_suffix(TRUEFAKE_ARCHIVE.suffix + ".partial")
    current = partial.stat().st_size if partial.exists() else 0
    if current > TRUEFAKE_BYTES:
        raise ValueError("TrueFake partial exceeds frozen size")
    remaining = TRUEFAKE_BYTES - current
    free_before = shutil.disk_usage(TRUEFAKE_ROOT).free
    if free_before < remaining + MIN_FREE_BYTES:
        raise OSError("insufficient disk space for TrueFake plus 100 GiB reserve")
    command = [
        "/usr/bin/curl", "--fail", "--location", "--silent", "--show-error",
        "--connect-timeout", "30", "--retry", "12", "--retry-delay", "5",
        "--retry-all-errors", "--speed-limit", "1024", "--speed-time", "180",
        "--output", str(partial),
    ]
    if current:
        command.extend(["--continue-at", "-"])
    command.append(confirmed_url)
    subprocess.run(command, check=True)
    if partial.stat().st_size != TRUEFAKE_BYTES:
        raise ValueError("TrueFake archive size mismatch after transfer")
    sha256 = _digest(partial)
    partial.replace(TRUEFAKE_ARCHIVE)
    receipt = {
        "schema_version": 1,
        "state": "e46_truefake_facebook_archive_complete_unscored_uninventoried",
        "landing_url": TRUEFAKE_LANDING_URL,
        "advertised_filename": TRUEFAKE_FILENAME,
        **source,
        "sha256": sha256,
        "path": str(TRUEFAKE_ARCHIVE),
        "network_bytes_this_run": remaining,
        "free_bytes_before": free_before,
        "free_bytes_after": shutil.disk_usage(TRUEFAKE_ROOT).free,
        "role": "E46_UNTOUCHED_FINAL",
        "model_scores_created": 0,
    }
    _write_atomic(TRUEFAKE_RECEIPT, receipt)
    _update_evidence()
    return receipt


def _update_evidence() -> None:
    payload = {
        "schema_version": 1,
        "state": "e46_acquisition_progress_unscored",
        "synthwildx": json.loads(SYNTHWILDX_RECEIPT.read_text()) if SYNTHWILDX_RECEIPT.exists() else None,
        "truefake_facebook": json.loads(TRUEFAKE_RECEIPT.read_text()) if TRUEFAKE_RECEIPT.exists() else None,
        "model_scores_created": 0,
    }
    _write_atomic(EVIDENCE, payload)


def status() -> dict[str, Any]:
    partial = TRUEFAKE_ARCHIVE.with_suffix(TRUEFAKE_ARCHIVE.suffix + ".partial")
    return {
        "synthwildx_receipt": SYNTHWILDX_RECEIPT.is_file(),
        "synthwildx_images": sum(1 for path in SYNTHWILDX_IMAGES.glob("*/*") if path.is_file()),
        "truefake_archive": TRUEFAKE_ARCHIVE.is_file(),
        "truefake_archive_bytes": TRUEFAKE_ARCHIVE.stat().st_size if TRUEFAKE_ARCHIVE.is_file() else 0,
        "truefake_partial_bytes": partial.stat().st_size if partial.is_file() else 0,
        "truefake_expected_bytes": TRUEFAKE_BYTES,
        "model_scores_created": 0,
    }


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("status", "acquire-synthwildx", "acquire-truefake"))
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args(argv)
    if args.command == "status":
        result = status()
    elif args.command == "acquire-synthwildx":
        result = acquire_synthwildx(args.workers)
    else:
        result = acquire_truefake()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
