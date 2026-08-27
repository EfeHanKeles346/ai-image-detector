"""Freeze and acquire the licensed E34 DDA-COCO aligned benchmark archive."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import re
import shutil
import subprocess
import time
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

import requests

from pixelproof.project_paths import DATA_ROOT, ML_ROOT


REPO_ID = "Junwei-Xi/DDA-COCO"
REVISION = "8c9330a3b374bcac46a8045a0e3c09ebcf7868fb"
FILENAME = "DDA-COCO.zip"
EXPECTED_BYTES = 4_301_452_066
EXPECTED_SHA256 = "8cd600779aaecef21605b07bff9ab3963a7fb9b9614a3d9a0588cd4a5e099c24"
API_URL = f"https://huggingface.co/api/datasets/{REPO_ID}"
DOWNLOAD_URL = f"https://huggingface.co/datasets/{REPO_ID}/resolve/{REVISION}/{FILENAME}?download=true"
OUTPUT_ROOT = DATA_ROOT / "e34_dda_coco"
SELECTION = OUTPUT_ROOT / "acquisition_selection.json"
EVIDENCE = ML_ROOT.parent / "evidence" / "e34_dda_acquisition.json"
MIN_FREE_BYTES = 100 * 1024**3
MAX_MEMBER_BYTES = 100 * 1024**2
MAX_EXPANDED_BYTES = 40 * 1024**3
CONTENT_RANGE_RE = re.compile(r"^bytes (\d+)-(\d+)/(\d+)$")
RETRY_DELAYS = (0, 5, 20, 60, 180)


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def _write_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_bytes(_json_bytes(value))
    temporary.replace(path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024**2):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def validate_repository(payload: Mapping[str, Any]) -> dict[str, Any]:
    if payload.get("id") != REPO_ID or payload.get("sha") != REVISION:
        raise ValueError("DDA-COCO repository identity/revision changed")
    if payload.get("cardData", {}).get("license") != "apache-2.0":
        raise ValueError("DDA-COCO licence is not Apache-2.0")
    item = next(
        (entry for entry in payload.get("siblings", []) if entry.get("rfilename") == FILENAME),
        None,
    )
    if item is None:
        raise ValueError("DDA-COCO archive disappeared")
    return {
        "repo_id": REPO_ID,
        "revision": REVISION,
        "license": "Apache-2.0",
        "filename": FILENAME,
        "bytes": EXPECTED_BYTES,
        "sha256": EXPECTED_SHA256,
        "url": DOWNLOAD_URL,
        "role": "paired_train_cal_candidate",
    }


def freeze() -> dict[str, Any]:
    response = requests.get(API_URL, timeout=(20, 120))
    response.raise_for_status()
    source = validate_repository(response.json())
    detailed = {
        "schema_version": 1,
        "experiment": "E34/R1c-P-DDA-COCO",
        "state": "selection_frozen_no_archive_bytes_claimed",
        "source": source,
        "boundaries": [
            "DDA-COCO becomes TRAIN/CAL only and cannot support a PixelProof DDA benchmark claim.",
            "No member enters a role before ZIP safety, exact parent links and protected-hash audit.",
            "All variants of one COCO parent stay in one role.",
            "No archive image byte was downloaded by metadata freeze.",
        ],
    }
    raw = _json_bytes(detailed)
    _write_atomic(SELECTION, detailed)
    compact = {
        "schema_version": 1,
        "experiment": detailed["experiment"],
        "state": detailed["state"],
        "source": {key: source[key] for key in ("repo_id", "revision", "license", "filename", "bytes", "sha256", "role")},
        "detailed_selection_bytes": len(raw),
        "detailed_selection_sha256": _sha256(raw),
        "new_archive_bytes_downloaded_by_freeze": 0,
    }
    _write_atomic(EVIDENCE, compact)
    return compact


def _selection() -> dict[str, Any]:
    payload = json.loads(SELECTION.read_text())
    if payload.get("state") != "selection_frozen_no_archive_bytes_claimed":
        raise ValueError("E34 selection is not frozen")
    return payload


def download() -> dict[str, Any]:
    source = _selection()["source"]
    destination = OUTPUT_ROOT / "archives" / FILENAME
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.stat().st_size != EXPECTED_BYTES or _sha256_file(destination) != EXPECTED_SHA256:
            raise ValueError("completed DDA-COCO archive contract mismatch")
        transfer_state = "already_complete"
    else:
        partial = destination.with_suffix(destination.suffix + ".partial")
        current = partial.stat().st_size if partial.exists() else 0
        if current > EXPECTED_BYTES:
            raise ValueError("DDA-COCO partial exceeds expected size")
        free = shutil.disk_usage(OUTPUT_ROOT).free
        remaining = EXPECTED_BYTES - current
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
        command.append(str(source["url"]))
        subprocess.run(command, check=True)
        if partial.stat().st_size != EXPECTED_BYTES:
            raise ValueError("DDA-COCO download size mismatch")
        if _sha256_file(partial) != EXPECTED_SHA256:
            raise ValueError("DDA-COCO SHA-256 mismatch")
        partial.replace(destination)
        transfer_state = "downloaded"
    receipt = {
        "schema_version": 1,
        "state": "archive_complete_sha256_verified",
        "transfer_state": transfer_state,
        "selection_sha256": _sha256(SELECTION.read_bytes()),
        "path": str(destination.relative_to(OUTPUT_ROOT)),
        "bytes": EXPECTED_BYTES,
        "sha256": EXPECTED_SHA256,
    }
    _write_atomic(OUTPUT_ROOT / "download_receipt.json", receipt)
    return receipt


def parse_content_range(value: str | None) -> tuple[int, int, int]:
    match = CONTENT_RANGE_RE.fullmatch(value or "")
    if match is None:
        raise ValueError(f"invalid Content-Range: {value!r}")
    return tuple(int(part) for part in match.groups())


def range_plan(start: int, total: int, workers: int = 4) -> list[tuple[int, int]]:
    if not 0 <= start < total or workers <= 0:
        raise ValueError("invalid range plan")
    remaining = total - start
    count = min(workers, remaining)
    base, extra = divmod(remaining, count)
    ranges = []
    cursor = start
    for index in range(count):
        size = base + (1 if index < extra else 0)
        ranges.append((cursor, cursor + size - 1))
        cursor += size
    return ranges


def _download_range(url: str, path: Path, start: int, end: int, total: int) -> dict[str, Any]:
    expected = end - start + 1
    current = path.stat().st_size if path.exists() else 0
    if current > expected:
        raise ValueError(f"range partial exceeds expected size: {path}")
    if current == expected:
        return {"path": path, "bytes": current, "state": "already_complete"}
    for delay in RETRY_DELAYS:
        if delay:
            time.sleep(delay)
        current = path.stat().st_size if path.exists() else 0
        request_start = start + current
        try:
            with requests.get(
                url,
                headers={"Range": f"bytes={request_start}-{end}"},
                stream=True,
                timeout=(30, 180),
            ) as response:
                response.raise_for_status()
                found = parse_content_range(response.headers.get("Content-Range"))
                if response.status_code != 206 or found != (request_start, end, total):
                    raise ValueError(f"range response mismatch: {(response.status_code, found)}")
                with path.open("ab") as output:
                    for chunk in response.iter_content(8 * 1024**2):
                        if chunk:
                            output.write(chunk)
            break
        except (requests.RequestException, OSError) as error:
            if delay == RETRY_DELAYS[-1]:
                raise RuntimeError(f"range download failed: {start}-{end}") from error
    found_bytes = path.stat().st_size
    if found_bytes != expected:
        raise ValueError(f"range length mismatch: {path} ({found_bytes} != {expected})")
    return {"path": path, "bytes": found_bytes, "state": "downloaded"}


def download_ranges(workers: int = 4) -> dict[str, Any]:
    source = _selection()["source"]
    destination = OUTPUT_ROOT / "archives" / FILENAME
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return download()
    prefix = destination.with_suffix(destination.suffix + ".partial")
    prefix_bytes = prefix.stat().st_size if prefix.exists() else 0
    if not 0 < prefix_bytes < EXPECTED_BYTES:
        return download()
    _remaining = EXPECTED_BYTES - prefix_bytes
    free = shutil.disk_usage(OUTPUT_ROOT).free
    if free < EXPECTED_BYTES + _remaining + MIN_FREE_BYTES:
        raise OSError("insufficient free space for verified parallel assembly")
    plan = range_plan(prefix_bytes, EXPECTED_BYTES, workers)
    paths = [
        destination.with_name(f"{FILENAME}.range-{start}-{end}.partial")
        for start, end in plan
    ]
    with ThreadPoolExecutor(max_workers=len(plan)) as executor:
        futures = {
            executor.submit(_download_range, str(source["url"]), path, start, end, EXPECTED_BYTES): (start, end)
            for (start, end), path in zip(plan, paths, strict=True)
        }
        for future in as_completed(futures):
            start, end = futures[future]
            future.result()
            print(f"DDA range {start}-{end} complete", flush=True)
    assembled = destination.with_suffix(destination.suffix + ".parallel.partial")
    if assembled.exists():
        raise FileExistsError(f"parallel assembly already exists: {assembled}")
    with assembled.open("xb") as output:
        for part in (prefix, *paths):
            with part.open("rb") as source_file:
                shutil.copyfileobj(source_file, output, length=8 * 1024**2)
    if assembled.stat().st_size != EXPECTED_BYTES or _sha256_file(assembled) != EXPECTED_SHA256:
        raise ValueError("parallel DDA-COCO assembly failed final size/SHA-256")
    assembled.replace(destination)
    for part in (prefix, *paths):
        part.unlink()
    receipt = {
        "schema_version": 1,
        "state": "archive_complete_sha256_verified",
        "transfer_state": "parallel_ranges_assembled",
        "selection_sha256": _sha256(SELECTION.read_bytes()),
        "path": str(destination.relative_to(OUTPUT_ROOT)),
        "bytes": EXPECTED_BYTES,
        "sha256": EXPECTED_SHA256,
        "preserved_prefix_bytes": prefix_bytes,
        "range_count": len(plan),
    }
    _write_atomic(OUTPUT_ROOT / "download_receipt.json", receipt)
    return receipt


def inspect_zip(infos: list[zipfile.ZipInfo]) -> dict[str, Any]:
    names: set[str] = set()
    file_count = 0
    image_count = 0
    expanded_bytes = 0
    by_top: dict[str, int] = {}
    for info in infos:
        path = PurePosixPath(info.filename)
        if path.is_absolute() or ".." in path.parts or not path.parts:
            raise ValueError(f"unsafe ZIP path: {info.filename!r}")
        if info.filename in names:
            raise ValueError(f"duplicate ZIP member: {info.filename}")
        names.add(info.filename)
        mode = info.external_attr >> 16
        if mode & 0o170000 == 0o120000:
            raise ValueError(f"ZIP symlink is forbidden: {info.filename}")
        if info.is_dir():
            continue
        if info.flag_bits & 0x1:
            raise ValueError(f"encrypted ZIP member: {info.filename}")
        if info.file_size < 0 or info.file_size > MAX_MEMBER_BYTES:
            raise ValueError(f"implausible ZIP member size: {info.filename}")
        file_count += 1
        expanded_bytes += info.file_size
        by_top[path.parts[0]] = by_top.get(path.parts[0], 0) + 1
        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
            image_count += 1
    if expanded_bytes > MAX_EXPANDED_BYTES:
        raise ValueError("DDA-COCO expansion exceeds frozen safety ceiling")
    if not image_count:
        raise ValueError("DDA-COCO archive contains no supported image")
    return {
        "member_count": len(infos),
        "file_count": file_count,
        "image_count": image_count,
        "expanded_bytes": expanded_bytes,
        "by_top": dict(sorted(by_top.items())),
    }


def inventory() -> dict[str, Any]:
    archive = OUTPUT_ROOT / "archives" / FILENAME
    receipt = json.loads((OUTPUT_ROOT / "download_receipt.json").read_text())
    if receipt.get("state") != "archive_complete_sha256_verified":
        raise ValueError("DDA-COCO download receipt is not verified")
    with zipfile.ZipFile(archive) as bundle:
        result = inspect_zip(bundle.infolist())
        bad = bundle.testzip()
    if bad is not None:
        raise ValueError(f"DDA-COCO CRC failure: {bad}")
    output = {
        "schema_version": 1,
        "state": "archive_inventory_passed",
        "download_receipt_sha256": _sha256((OUTPUT_ROOT / "download_receipt.json").read_bytes()),
        **result,
    }
    _write_atomic(OUTPUT_ROOT / "archive_inventory.json", output)
    return output


def status() -> dict[str, Any]:
    destination = OUTPUT_ROOT / "archives" / FILENAME
    partial = destination.with_suffix(destination.suffix + ".partial")
    return {
        "selection_exists": SELECTION.exists(),
        "complete_bytes": destination.stat().st_size if destination.exists() else 0,
        "partial_bytes": partial.stat().st_size if partial.exists() else 0,
        "expected_bytes": EXPECTED_BYTES,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "download", "download-ranges", "inventory", "status"))
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.command == "download-ranges":
        result = download_ranges(args.workers)
    else:
        result = {"freeze": freeze, "download": download, "inventory": inventory, "status": status}[args.command]()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
