"""E32/C1 — frozen, resumable authentic-photo acquisition on the external data root.

The default ``freeze-real`` command downloads only small source metadata and writes an exact
selection receipt. Image bytes require an explicit download command. Third-party bytes and the
detailed receipt stay under ``$PIXELPROOF_DATA_ROOT/e32``; Git receives only compact aggregate
evidence.

Run from ``ml/``::

    PIXELPROOF_DATA_ROOT=/Volumes/LaCie/pixelproof-datasets \
      PYTHONPATH=src .venv/bin/python experiments/e32_data_system.py freeze-real
    PIXELPROOF_DATA_ROOT=/Volumes/LaCie/pixelproof-datasets \
      PYTHONPATH=src .venv/bin/python experiments/e32_data_system.py download-real --source vision
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence
from urllib.parse import urlparse

import requests

from pixelproof.project_paths import DATA_ROOT, ML_ROOT


REGISTRY_PATH = ML_ROOT / "e32_sources.json"
OUTPUT_ROOT = DATA_ROOT / "e32"
DETAILED_SELECTION = OUTPUT_ROOT / "real_acquisition_selection.json"
COMPACT_EVIDENCE = ML_ROOT.parent / "evidence" / "e32_real_acquisition_selection.json"
MIN_FREE_BYTES = 100 * 1024**3
VISION_WORKERS = 4
CSAFE_RANGE_WORKERS = 4
CHUNK_BYTES = 8 * 1024**2
RETRY_DELAYS = (0, 5, 20, 60, 180)
IMAGE_SUFFIXES = {".jpg", ".jpeg"}
CONTENT_RANGE = re.compile(r"^bytes (\d+)-(\d+)/(\d+)$")


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _revision(material: str) -> str:
    return _sha256(material.encode())[:40]


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def _write_atomic(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_bytes(raw)
    temporary.replace(path)


def registry() -> dict[str, Mapping[str, Any]]:
    payload = json.loads(REGISTRY_PATH.read_text())
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported E32 source registry schema")
    sources = payload.get("sources", [])
    output = {str(source["id"]): source for source in sources}
    if len(output) != len(sources):
        raise ValueError("duplicate E32 source id")
    return output


def _request(method: str, url: str, *, stream: bool = False, headers: Mapping[str, str] | None = None) -> requests.Response:
    last_error: Exception | None = None
    for delay in RETRY_DELAYS:
        if delay:
            time.sleep(delay)
        try:
            response = requests.request(
                method,
                url,
                stream=stream,
                headers=dict(headers or {}),
                timeout=(20, 120),
                allow_redirects=True,
            )
            response.raise_for_status()
            return response
        except requests.exceptions.SSLError:
            raise
        except (requests.RequestException, OSError) as error:
            last_error = error
    raise RuntimeError(f"request failed after {len(RETRY_DELAYS)} attempts: {url}") from last_error


def _get_verified(url: str, expected_sha256: str) -> bytes:
    result = subprocess.run(
        [
            "/usr/bin/curl",
            "--fail",
            "--location",
            "--silent",
            "--show-error",
            "--connect-timeout",
            "30",
            "--retry",
            "5",
            "--retry-all-errors",
            url,
        ],
        check=True,
        capture_output=True,
    )
    raw = result.stdout
    found = _sha256(raw)
    if found != expected_sha256:
        raise ValueError(f"upstream metadata changed for {url}: expected {expected_sha256}, found {found}")
    return raw


def parse_vision_native_urls(raw: bytes) -> list[dict[str, str]]:
    """Select only native JPEG parents; reject derivatives, flat fields and videos."""
    assets = []
    for line in raw.decode("utf-8").splitlines():
        url = line.strip()
        if not url or "/images/nat/" not in url:
            continue
        parsed = urlparse(url)
        relative = PurePosixPath(parsed.path).relative_to("/VISION/dataset")
        if relative.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        device = relative.parts[0]
        if not device.startswith("D") or "_" not in device:
            raise ValueError(f"unparseable VISION device path: {relative}")
        assets.append(
            {
                "url": url,
                "source_key": relative.as_posix(),
                "device": device.split("_", 1)[0],
                "camera_pipeline": device,
                "filename": relative.name,
            }
        )
    keys = [asset["source_key"] for asset in assets]
    if len(keys) != len(set(keys)):
        raise ValueError("VISION selection contains duplicate source keys")
    return sorted(assets, key=lambda asset: asset["source_key"])


def _freeze_vision(source: Mapping[str, Any]) -> dict[str, Any]:
    raw = _get_verified(str(source["index_url"]), str(source["index_sha256"]))
    _get_verified(str(source["readme_url"]), str(source["readme_sha256"]))
    assets = parse_vision_native_urls(raw)
    devices = sorted({asset["device"] for asset in assets})
    if len(assets) != int(source["expected_parent_count"]):
        raise ValueError(f"VISION expected {source['expected_parent_count']} parents, found {len(assets)}")
    if len(devices) != int(source["expected_devices"]):
        raise ValueError(f"VISION expected {source['expected_devices']} devices, found {len(devices)}")
    return {
        "source_id": source["id"],
        "source_revision": _revision(f"{source['revision']}:{source['index_sha256']}"),
        "role": source["assigned_role"],
        "label": "real",
        "license": source["license"],
        "parent_count": len(assets),
        "device_count": len(devices),
        "assets": assets,
    }


def _freeze_fodb(source: Mapping[str, Any]) -> dict[str, Any]:
    _get_verified(str(source["readme_url"]), str(source["readme_sha256"]))
    archives = []
    for item in source["archives"]:
        if int(item["bytes"]) <= 0 or not str(item["url"]).startswith("https://"):
            raise ValueError("FODB archive lacks a safe URL or byte declaration")
        archives.append(dict(item))
    return {
        "source_id": source["id"],
        "source_revision": _revision(f"{source['revision']}:{source['readme_sha256']}"),
        "role": source["assigned_role"],
        "label": "real",
        "license": source["license"],
        "parent_count_after_orig_extraction": int(source["expected_parent_count"]),
        "device_count": int(source["expected_devices"]),
        "declared_download_bytes": sum(int(item["bytes"]) for item in archives),
        "archives": archives,
    }


def _freeze_csafe(source: Mapping[str, Any]) -> dict[str, Any]:
    payload = _request("GET", str(source["article_api"])).json()
    if int(payload.get("id", -1)) != int(source["article_id"]):
        raise ValueError("CSAFE Figshare article id changed")
    if int(payload.get("version", -1)) != 1:
        raise ValueError("CSAFE Figshare article version changed")
    if payload.get("license", {}).get("name") != source["license"]:
        raise ValueError("CSAFE licence changed")
    expected = source["archive"]
    found = next((item for item in payload.get("files", []) if int(item.get("id", -1)) == int(expected["file_id"])), None)
    if found is None:
        raise ValueError("CSAFE selected archive disappeared")
    for field, remote_field in (("name", "name"), ("bytes", "size"), ("md5", "computed_md5")):
        if str(expected[field]) != str(found.get(remote_field)):
            raise ValueError(f"CSAFE selected archive {field} changed")
    archive = dict(expected)
    archive["url"] = str(found["download_url"])
    return {
        "source_id": source["id"],
        "source_revision": _revision(f"{source['doi']}:{source['revision']}:{expected['md5']}"),
        "role": source["assigned_role"],
        "label": "real",
        "license": source["license"],
        "selection_state": "archive_frozen_internal_rows_pending_inventory",
        "declared_download_bytes": int(expected["bytes"]),
        "archive": archive,
    }


def freeze_real() -> dict[str, Any]:
    sources = registry()
    frozen = [
        _freeze_vision(sources["vision-base-native"]),
        _freeze_fodb(sources["forchheim-fodb"]),
        _freeze_csafe(sources["csafe-mcsidb-s21"]),
    ]
    detailed = {
        "schema_version": 1,
        "experiment": "E32/C1",
        "state": "selection_frozen_no_image_bytes_claimed",
        "label_invariant": {"real": 0, "ai": 1},
        "data_root_name": DATA_ROOT.name,
        "sources": frozen,
        "boundaries": [
            "The owner gallery is absent.",
            "VISION flat fields, videos and social derivatives are absent.",
            "FODB orig files are parents; social variants inherit the same parent after extraction.",
            "CSAFE internal rows cannot enter a split until its downloaded archive is inventoried.",
            "No listed real candidate may enter a locked final arm after candidate training reads it.",
        ],
    }
    detailed_raw = _json_bytes(detailed)
    _write_atomic(DETAILED_SELECTION, detailed_raw)
    compact_sources = []
    for item in frozen:
        compact_sources.append(
            {
                key: value
                for key, value in item.items()
                if key not in {"assets", "archives", "archive"}
            }
        )
    compact = {
        "schema_version": 1,
        "experiment": "E32/C1",
        "state": detailed["state"],
        "detailed_selection_sha256": _sha256(detailed_raw),
        "detailed_selection_bytes": len(detailed_raw),
        "sources": compact_sources,
        "vision_selected_parent_count": frozen[0]["parent_count"],
        "fodb_expected_orig_parent_count": frozen[1]["parent_count_after_orig_extraction"],
        "csafe_state": frozen[2]["selection_state"],
        "declared_archive_download_bytes": frozen[1]["declared_download_bytes"] + frozen[2]["declared_download_bytes"],
        "new_image_bytes_downloaded_by_freeze": 0,
    }
    _write_atomic(COMPACT_EVIDENCE, _json_bytes(compact))
    return compact


def _safe_destination(relative: str) -> Path:
    pure = PurePosixPath(relative)
    if pure.is_absolute() or ".." in pure.parts or not pure.parts:
        raise ValueError(f"unsafe relative destination: {relative!r}")
    destination = OUTPUT_ROOT.joinpath(*pure.parts)
    destination.resolve().relative_to(OUTPUT_ROOT.resolve())
    return destination


def _ensure_capacity(expected_remaining: int) -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(OUTPUT_ROOT).free
    required = expected_remaining + MIN_FREE_BYTES
    if free < required:
        raise OSError(f"insufficient free space: need {required:,} bytes including floor, have {free:,}")


def _stream_download(url: str, destination: Path, expected_bytes: int | None = None) -> dict[str, Any]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        found_bytes = destination.stat().st_size
        if expected_bytes is None or found_bytes == expected_bytes:
            return {
                "path": destination.relative_to(OUTPUT_ROOT).as_posix(),
                "bytes": found_bytes,
                "state": "already_complete",
            }
        raise ValueError(
            f"completed file size mismatch for {destination}: "
            f"expected {expected_bytes}, found {found_bytes}"
        )
    partial = destination.with_suffix(destination.suffix + ".partial")
    offset = partial.stat().st_size if partial.exists() else 0
    if expected_bytes is not None and offset > expected_bytes:
        raise ValueError(f"partial file exceeds expected size: {partial}")
    if expected_bytes is not None and offset == expected_bytes:
        partial.replace(destination)
        return {
            "path": destination.relative_to(OUTPUT_ROOT).as_posix(),
            "bytes": expected_bytes,
            "state": "recovered_complete_partial",
        }
    _ensure_capacity(max(0, (expected_bytes or 0) - offset))
    command = [
        "/usr/bin/curl",
        "--fail",
        "--location",
        "--silent",
        "--show-error",
        "--connect-timeout",
        "30",
        "--retry",
        "8",
        "--retry-delay",
        "5",
        "--retry-all-errors",
        "--speed-limit",
        "1024",
        "--speed-time",
        "120",
        "--output",
        str(partial),
    ]
    if offset:
        command.extend(["--continue-at", "-"])
    command.append(url)
    subprocess.run(command, check=True)
    size = partial.stat().st_size
    if expected_bytes is not None and size != expected_bytes:
        raise ValueError(f"download size mismatch for {url}: expected {expected_bytes}, found {size}")
    partial.replace(destination)
    return {"path": destination.relative_to(OUTPUT_ROOT).as_posix(), "bytes": size, "state": "downloaded"}


def _load_selection() -> dict[str, Any]:
    if not DETAILED_SELECTION.exists():
        raise FileNotFoundError("freeze-real must succeed before image download")
    payload = json.loads(DETAILED_SELECTION.read_text())
    if payload.get("state") != "selection_frozen_no_image_bytes_claimed":
        raise ValueError("unexpected E32 detailed selection state")
    return payload


def _source(payload: Mapping[str, Any], source_id: str) -> Mapping[str, Any]:
    try:
        return next(item for item in payload["sources"] if item["source_id"] == source_id)
    except StopIteration as error:
        raise KeyError(f"missing frozen source {source_id}") from error


def download_vision(payload: Mapping[str, Any]) -> dict[str, Any]:
    source = _source(payload, "vision-base-native")
    assets = list(source["assets"])
    results = []

    def fetch(asset: Mapping[str, str]) -> dict[str, Any]:
        destination = _safe_destination(f"real/vision/{asset['source_key']}")
        result = _stream_download(asset["url"], destination)
        return {**result, "source_key": asset["source_key"], "camera_pipeline": asset["camera_pipeline"]}

    with ThreadPoolExecutor(max_workers=VISION_WORKERS) as executor:
        futures = [executor.submit(fetch, asset) for asset in assets]
        for index, future in enumerate(as_completed(futures), start=1):
            results.append(future.result())
            if index % 100 == 0 or index == len(futures):
                print(f"VISION {index}/{len(futures)} complete", flush=True)
    return {"source_id": source["source_id"], "files": len(results), "bytes": sum(item["bytes"] for item in results)}


def download_fodb(payload: Mapping[str, Any]) -> dict[str, Any]:
    source = _source(payload, "forchheim-fodb")
    results = []
    for archive in source["archives"]:
        destination = _safe_destination(f"real/fodb/archives/{archive['name']}")
        results.append(_stream_download(archive["url"], destination, int(archive["bytes"])))
        print(f"FODB {archive['name']} complete", flush=True)
    return {"source_id": source["source_id"], "files": len(results), "bytes": sum(item["bytes"] for item in results)}


def download_csafe(payload: Mapping[str, Any]) -> dict[str, Any]:
    source = _source(payload, "csafe-mcsidb-s21")
    archive = source["archive"]
    destination = _safe_destination(f"real/csafe/archives/{archive['name']}")
    result = _stream_download(archive["url"], destination, int(archive["bytes"]))
    hasher = hashlib.md5(usedforsecurity=False)
    with destination.open("rb") as handle:
        while chunk := handle.read(CHUNK_BYTES):
            hasher.update(chunk)
    digest = hasher.hexdigest()
    if digest != archive["md5"]:
        raise ValueError(f"CSAFE MD5 mismatch: expected {archive['md5']}, found {digest}")
    print(f"CSAFE {archive['name']} complete and MD5 verified", flush=True)
    return {"source_id": source["source_id"], "files": 1, "bytes": result["bytes"], "md5": digest}


def _range_plan(start: int, stop: int, workers: int) -> list[tuple[int, int]]:
    if start < 0 or stop <= start or workers <= 0:
        raise ValueError("invalid byte-range plan")
    remaining = stop - start
    width = (remaining + workers - 1) // workers
    return [
        (left, min(stop - 1, left + width - 1))
        for left in range(start, stop, width)
    ]


def _parse_content_range(value: str | None) -> tuple[int, int, int]:
    match = CONTENT_RANGE.fullmatch(value or "")
    if match is None:
        raise ValueError(f"invalid Content-Range: {value!r}")
    return tuple(int(item) for item in match.groups())


def _download_range(url: str, path: Path, start: int, end: int, total: int) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    expected = end - start + 1
    for attempt, delay in enumerate(RETRY_DELAYS):
        current = path.stat().st_size if path.exists() else 0
        if current == expected:
            return {"path": path, "bytes": current, "state": "already_complete"}
        if current > expected:
            raise ValueError(f"range partial exceeds expected size: {path}")
        if delay:
            time.sleep(delay)
        requested = start + current
        try:
            with requests.get(
                url,
                headers={"Range": f"bytes={requested}-{end}"},
                stream=True,
                timeout=(20, 120),
                allow_redirects=True,
            ) as response:
                response.raise_for_status()
                if response.status_code != 206:
                    raise ValueError(f"range request returned HTTP {response.status_code}")
                found_start, found_end, found_total = _parse_content_range(
                    response.headers.get("Content-Range")
                )
                if (found_start, found_end, found_total) != (requested, end, total):
                    raise ValueError(
                        "range response mismatch: "
                        f"expected {(requested, end, total)}, found {(found_start, found_end, found_total)}"
                    )
                with path.open("ab") as handle:
                    for chunk in response.iter_content(CHUNK_BYTES):
                        if chunk:
                            handle.write(chunk)
        except (requests.RequestException, OSError) as error:
            if attempt == len(RETRY_DELAYS) - 1:
                raise RuntimeError(f"range failed after retries: {start}-{end}") from error
            continue
    found = path.stat().st_size
    if found != expected:
        raise ValueError(f"range size mismatch: expected {expected}, found {found}")
    return {"path": path, "bytes": found, "state": "downloaded"}


def _assemble_ranges(
    prefix: Path,
    ranges: Sequence[Path],
    destination: Path,
    expected_bytes: int,
    expected_md5: str,
) -> dict[str, Any]:
    assembled = destination.with_suffix(destination.suffix + ".parallel.partial")
    if assembled.exists():
        raise ValueError(f"stale assembled partial requires review: {assembled}")
    digest = hashlib.md5(usedforsecurity=False)
    written = 0
    with assembled.open("xb") as target:
        for source in (prefix, *ranges):
            with source.open("rb") as handle:
                while chunk := handle.read(CHUNK_BYTES):
                    target.write(chunk)
                    digest.update(chunk)
                    written += len(chunk)
    if written != expected_bytes:
        raise ValueError(f"assembled byte mismatch: expected {expected_bytes}, found {written}")
    found_md5 = digest.hexdigest()
    if found_md5 != expected_md5:
        raise ValueError(f"assembled MD5 mismatch: expected {expected_md5}, found {found_md5}")
    os.replace(assembled, destination)
    return {"bytes": written, "md5": found_md5}


def download_csafe_ranges(payload: Mapping[str, Any]) -> dict[str, Any]:
    source = _source(payload, "csafe-mcsidb-s21")
    archive = source["archive"]
    destination = _safe_destination(f"real/csafe/archives/{archive['name']}")
    if destination.exists():
        return download_csafe(payload)
    prefix = destination.with_suffix(destination.suffix + ".partial")
    if not prefix.is_file() or prefix.stat().st_size <= 0:
        raise ValueError("parallel CSAFE recovery requires a non-empty contiguous prefix")
    expected = int(archive["bytes"])
    prefix_bytes = prefix.stat().st_size
    if prefix_bytes >= expected:
        raise ValueError("CSAFE prefix is not a strict partial")
    plan = _range_plan(prefix_bytes, expected, CSAFE_RANGE_WORKERS)
    _ensure_capacity((expected - prefix_bytes) + expected)

    def fetch(bounds: tuple[int, int]) -> dict[str, Any]:
        start, end = bounds
        part = destination.with_name(f"{destination.name}.range-{start}-{end}.partial")
        result = _download_range(str(archive["url"]), part, start, end, expected)
        print(f"CSAFE range {start}-{end} complete", flush=True)
        return result

    with ThreadPoolExecutor(max_workers=CSAFE_RANGE_WORKERS) as executor:
        results = list(executor.map(fetch, plan))
    range_paths = [Path(result["path"]) for result in results]
    assembled = _assemble_ranges(
        prefix,
        range_paths,
        destination,
        expected,
        str(archive["md5"]),
    )
    prefix.unlink()
    for path in range_paths:
        path.unlink()
    print("CSAFE parallel ranges assembled and MD5 verified", flush=True)
    return {
        "source_id": source["source_id"],
        "files": 1,
        "bytes": assembled["bytes"],
        "md5": assembled["md5"],
        "preserved_prefix_bytes": prefix_bytes,
        "range_count": len(plan),
    }


def status() -> dict[str, Any]:
    files = []
    if OUTPUT_ROOT.exists():
        for path in sorted(OUTPUT_ROOT.rglob("*")):
            if path.is_file() and not path.name.startswith("._"):
                files.append({"path": path.relative_to(OUTPUT_ROOT).as_posix(), "bytes": path.stat().st_size})
    return {
        "output_root": str(OUTPUT_ROOT),
        "free_bytes": shutil.disk_usage(OUTPUT_ROOT if OUTPUT_ROOT.exists() else DATA_ROOT).free,
        "file_count": len(files),
        "physical_bytes": sum(item["bytes"] for item in files),
        "partial_files": [item for item in files if item["path"].endswith(".partial")],
    }


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("freeze-real")
    download = subparsers.add_parser("download-real")
    download.add_argument("--source", required=True, choices=("vision", "fodb", "csafe"))
    subparsers.add_parser("download-csafe-ranges")
    subparsers.add_parser("status")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "freeze-real":
        result = freeze_real()
    elif args.command == "status":
        result = status()
    elif args.command == "download-csafe-ranges":
        result = download_csafe_ranges(_load_selection())
    else:
        payload = _load_selection()
        result = {
            "vision": download_vision,
            "fodb": download_fodb,
            "csafe": download_csafe,
        }[args.source](payload)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
