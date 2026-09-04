"""Download only the byte-bound E49-C OpenFake assets, with resume and decode checks."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
from io import BytesIO
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from PIL import Image
import requests

from experiments.e49_openfake import CONFIG, MODEL_KEYS_C, PAGE_SIZE, REPO_ID, REVISION
from experiments.e49_openfake_assets import (
    ALLOWED_HEAD_CONTENT_TYPES,
    ASSET_CONTRACT,
    _resolve_wanted_assets,
)
from pixelproof.project_paths import DATA_ROOT, ML_ROOT


ASSET_CONTRACT_SHA256 = "7b71449e0e7d9ea22973f021af2d4ec49cc395e3fffe6816ffc123274a571415"
EXPECTED_ROWS = 960
EXPECTED_BYTES = 241_736_938
MAX_PIXELS = 50_000_000

ROOT = DATA_ROOT / "e49" / "openfake"
PAYLOAD_ROOT = ROOT / "payloads"
RECEIPT = ROOT / "download_receipt_unscored.json"
EVIDENCE = ML_ROOT.parent / "evidence" / "e49_c_openfake_download.json"


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
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _asset_rows() -> list[dict[str, Any]]:
    raw = ASSET_CONTRACT.read_bytes()
    if hashlib.sha256(raw).hexdigest() != ASSET_CONTRACT_SHA256:
        raise ValueError("E49-C OpenFake asset contract changed")
    payload = json.loads(raw)
    rows = payload.get("rows") or []
    if (
        payload.get("state") != "e49_c_openfake_assets_frozen_untransferred_unscored"
        or payload.get("openfake_expected_bytes") != EXPECTED_BYTES
        or len(rows) != EXPECTED_ROWS
        or payload.get("new_image_bytes_downloaded") != 0
        or payload.get("model_scores_created") != 0
    ):
        raise ValueError("E49-C OpenFake asset boundary changed")
    return rows


def destination(row: Mapping[str, Any]) -> Path:
    model = str(row["model"])
    if model not in MODEL_KEYS_C:
        raise ValueError("E49-C unexpected model in download contract")
    return PAYLOAD_ROOT / model / f"{int(row['row_index']):06d}.jpg"


def inspect_file(path: Path, expected: Mapping[str, Any]) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size != int(expected["content_length"]):
        raise ValueError(f"E49-C payload byte length changed: {path.name}")
    with Image.open(path) as opened:
        opened.verify()
    with Image.open(path) as opened:
        width, height = opened.size
        if opened.format != "JPEG" or width * height > MAX_PIXELS:
            raise ValueError(f"E49-C payload format/geometry unsafe: {path.name}")
        if (width, height) != (int(expected["width"]), int(expected["height"])):
            raise ValueError(f"E49-C payload dimensions changed: {path.name}")
    return {
        "record_id": expected["record_id"],
        "row_index": int(expected["row_index"]),
        "model": expected["model"],
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": _digest(path),
        "format": "JPEG",
        "width": width,
        "height": height,
        "status": "downloaded_unscored",
    }


def _download(item: Mapping[str, Any]) -> dict[str, Any]:
    row = item["row"]
    path = destination(row)
    if path.is_file():
        return inspect_file(path, row)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".jpg.part")
    last_status = "unknown"
    for attempt in range(6):
        try:
            with requests.get(
                str(item["url"]),
                stream=True,
                headers={"User-Agent": "PixelProof-E49C/1.0 frozen-asset-transfer"},
                timeout=(15, 120),
            ) as response:
                last_status = str(response.status_code)
                response.raise_for_status()
                length = int(response.headers.get("Content-Length", "-1"))
                content_type = str(response.headers.get("Content-Type", "")).split(";", 1)[0].lower()
                etag = str(response.headers.get("ETag", "")).strip('"')
                if length != int(row["content_length"]) or content_type not in ALLOWED_HEAD_CONTENT_TYPES:
                    raise ValueError("asset response size/type differs from HEAD contract")
                if row.get("etag") and etag != row["etag"]:
                    raise ValueError("asset response ETag differs from HEAD contract")
                with temporary.open("wb") as stream:
                    for block in response.iter_content(1024 * 1024):
                        if block:
                            stream.write(block)
                if temporary.stat().st_size != length:
                    raise ValueError("asset body length differs from response")
                temporary.replace(path)
            return inspect_file(path, row)
        except (requests.RequestException, OSError, ValueError):
            if temporary.exists():
                temporary.unlink()
            if attempt == 5:
                break
    raise RuntimeError(f"E49-C download failed at row {row['row_index']}, HTTP {last_status}")


def _unexpected_files(expected: set[Path]) -> list[str]:
    if not PAYLOAD_ROOT.exists():
        return []
    return sorted(
        str(path) for path in PAYLOAD_ROOT.rglob("*")
        if path.is_file() and not path.name.startswith("._") and path not in expected
    )


def download() -> dict[str, Any]:
    if RECEIPT.exists() or EVIDENCE.exists():
        raise FileExistsError("E49-C OpenFake download receipt already exists")
    rows = _asset_rows()
    expected_paths = {destination(row) for row in rows}
    extras = _unexpected_files(expected_paths)
    if extras:
        raise ValueError(f"E49-C payload root has unexpected files: {extras[:3]}")

    by_page: dict[int, dict[int, Mapping[str, Any]]] = defaultdict(dict)
    completed: list[dict[str, Any]] = []
    for row in rows:
        path = destination(row)
        if path.is_file():
            completed.append(inspect_file(path, row))
        else:
            index = int(row["row_index"])
            by_page[(index // PAGE_SIZE) * PAGE_SIZE][index] = row
    already_present = len(completed)

    pages = sorted(by_page.items())
    for page_number, (offset, wanted) in enumerate(pages, start=1):
        items, _method = _resolve_wanted_assets(offset, wanted)
        with ThreadPoolExecutor(max_workers=8) as pool:
            completed.extend(pool.map(_download, items))
        if page_number % 24 == 0 or page_number == len(pages):
            print(
                f"E49-C download pages {page_number}/{len(pages)}, files {len(completed)}/{len(rows)}",
                flush=True,
            )

    completed.sort(key=lambda row: (str(row["model"]), int(row["row_index"])))
    counts = Counter(str(row["model"]) for row in completed)
    total_bytes = sum(int(row["bytes"]) for row in completed)
    if (
        len(completed) != EXPECTED_ROWS
        or len({row["record_id"] for row in completed}) != EXPECTED_ROWS
        or counts != Counter({model: 192 for model in MODEL_KEYS_C})
        or total_bytes != EXPECTED_BYTES
        or _unexpected_files(expected_paths)
    ):
        raise ValueError("E49-C completed payload inventory differs from frozen contract")
    payload = {
        "schema_version": 1,
        "state": "e49_c_openfake_downloaded_decoded_unscored",
        "role": "FINAL_AI_COMPONENT_RESERVE_PENDING_DECONTAMINATION_AND_COMPLETE_E49",
        "asset_contract_sha256": ASSET_CONTRACT_SHA256,
        "repository": REPO_ID,
        "revision": REVISION,
        "files": len(completed),
        "bytes": total_bytes,
        "counts": dict(sorted(counts.items())),
        "already_present_at_run_start": already_present,
        "downloaded_this_run": len(completed) - already_present,
        "asset_urls_stored": 0,
        "rows": completed,
        "model_scores_created": 0,
        "boundary": "Exact frozen transfer plus JPEG decode only; no detector, metric or training access.",
    }
    raw = _write_atomic(RECEIPT, payload)
    evidence = {key: payload[key] for key in (
        "schema_version", "state", "role", "asset_contract_sha256", "repository", "revision",
        "files", "bytes", "counts", "already_present_at_run_start", "downloaded_this_run",
        "asset_urls_stored", "model_scores_created",
    )}
    evidence.update({"receipt_bytes": len(raw), "receipt_sha256": hashlib.sha256(raw).hexdigest()})
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
