"""Download the exact E49 Commons reserve with restart-safe byte and image validation."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Iterable, Mapping

from PIL import Image
import requests

from experiments.e49_acquisition import (
    COMMONS_CATEGORIES,
    COMMONS_RESERVE_PER_DEVICE,
    OPEN_COMPONENTS_V2_CONTRACT,
)
from pixelproof.project_paths import DATA_ROOT, ML_ROOT


CONTRACT_SHA256 = "1d4e184c27cb87cf832045a23b6966f382673c3bcd8342a900c07130bd9182aa"
EXPECTED_FILES = 1_100
EXPECTED_BYTES = 2_706_581_778
MAX_PIXELS = 100_000_000
ALLOWED_DECODED_FORMATS = {"JPEG", "MPO"}
DOWNLOAD_WORKERS = 1
REQUEST_PACING_SECONDS = 0.75
USER_AGENT = "PixelProof-E49/1.0 (https://github.com/EfeHanKeles346/ai-image-detector; research)"

ROOT = DATA_ROOT / "e49" / "open_components_v2"
PAYLOAD_ROOT = ROOT / "commons_reserve"
RECEIPT = ROOT / "commons_download_receipt_unscored.json"
EVIDENCE = ML_ROOT.parent / "evidence" / "e49_commons_download.json"


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def _write_atomic(path: Path, value: Any) -> bytes:
    raw = _json_bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_bytes(raw)
    temporary.replace(path)
    return raw


def _hash(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _slug(source: str) -> str:
    return "".join(character.lower() if character.isalnum() else "-" for character in source).strip("-")


def destination(row: Mapping[str, Any]) -> Path:
    return PAYLOAD_ROOT / _slug(str(row["source"])) / f"{int(row['pageid'])}.jpg"


def inspect_file(path: Path, expected: Mapping[str, Any]) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size != int(expected["bytes"]):
        raise ValueError(f"E49 Commons payload byte length changed: {path.name}")
    sha1 = _hash(path, "sha1")
    if sha1 != str(expected["commons_sha1"]):
        raise ValueError(f"E49 Commons SHA1 changed: {path.name}")
    with Image.open(path) as opened:
        opened.verify()
    with Image.open(path) as opened:
        opened.load()
        width, height = opened.size
        decoded_format = str(opened.format or "UNKNOWN").upper()
        # Apple phone originals may be JPEG files with an MPF segment, which Pillow reports as MPO.
        if decoded_format not in ALLOWED_DECODED_FORMATS or width <= 0 or height <= 0 or width * height > MAX_PIXELS:
            raise ValueError(f"E49 Commons payload format/geometry unsafe: {path.name}")
        if (width, height) != (int(expected["width"]), int(expected["height"])):
            raise ValueError(f"E49 Commons payload dimensions changed: {path.name}")
        exif = opened.getexif()
        make = str(exif.get(271, "")).strip()
        model = str(exif.get(272, "")).strip()
        orientation = int(exif.get(274, 1) or 1)
    return {
        **{key: expected[key] for key in (
            "identity", "rank", "label", "source", "category", "pageid", "title",
            "revision_timestamp", "uploader", "description_url", "license", "license_url",
        )},
        "path": str(path), "bytes": path.stat().st_size, "sha1": sha1,
        "sha256": _hash(path, "sha256"), "format": decoded_format,
        "width": width, "height": height, "exif_make": make, "exif_model": model,
        "exif_orientation": orientation, "status": "downloaded_decoded_unscored",
    }


def _download(row: Mapping[str, Any]) -> dict[str, Any]:
    path = destination(row)
    if path.is_file():
        return inspect_file(path, row)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".jpg.part")
    if temporary.exists():
        temporary.unlink()
    last_error: Exception | None = None
    for attempt in range(6):
        try:
            time.sleep(REQUEST_PACING_SECONDS)
            with requests.get(
                str(row["url"]), stream=True,
                headers={"User-Agent": USER_AGENT},
                timeout=(20, 180),
            ) as response:
                if response.status_code == 429:
                    retry_after = response.headers.get("Retry-After", "")
                    delay = int(retry_after) if retry_after.isdigit() else min(5 * 2 ** attempt, 60)
                    time.sleep(min(max(delay, 5), 60))
                    response.raise_for_status()
                response.raise_for_status()
                content_type = str(response.headers.get("Content-Type", "")).split(";", 1)[0].lower()
                declared = int(response.headers.get("Content-Length", "-1"))
                if content_type not in {"image/jpeg", "image/jpg"}:
                    raise ValueError(f"unexpected response type: {content_type}")
                if declared >= 0 and declared != int(row["bytes"]):
                    raise ValueError(f"response byte length changed: {declared}/{row['bytes']}")
                with temporary.open("wb") as stream:
                    for block in response.iter_content(1024 * 1024):
                        if block:
                            stream.write(block)
            if temporary.stat().st_size != int(row["bytes"]):
                raise ValueError("downloaded byte length differs from contract")
            temporary.replace(path)
            return inspect_file(path, row)
        except (OSError, requests.RequestException, ValueError) as error:
            last_error = error
            if temporary.exists():
                temporary.unlink()
            if attempt < 5:
                time.sleep(min(2 ** attempt, 30))
    raise RuntimeError(f"E49 Commons download failed at {row['identity']}: {last_error}")


def _rows() -> list[dict[str, Any]]:
    raw = OPEN_COMPONENTS_V2_CONTRACT.read_bytes()
    if hashlib.sha256(raw).hexdigest() != CONTRACT_SHA256:
        raise ValueError("E49 open-components V2 contract changed")
    payload = json.loads(raw)
    rows = payload.get("commons", {}).get("rows") or []
    if (
        payload.get("state") != "e49_open_components_v2_frozen_untransferred_unscored"
        or payload.get("model_scores_created") != 0
        or payload.get("new_image_bytes_downloaded") != 0
        or len(rows) != EXPECTED_FILES
        or payload.get("commons", {}).get("network_bytes") != EXPECTED_BYTES
    ):
        raise ValueError("E49 Commons source boundary changed")
    return rows


def _unexpected(expected: set[Path]) -> list[str]:
    if not PAYLOAD_ROOT.exists():
        return []
    return sorted(
        str(path) for path in PAYLOAD_ROOT.rglob("*")
        if path.is_file() and not path.name.startswith("._") and path not in expected
    )


def download() -> dict[str, Any]:
    if RECEIPT.exists() or EVIDENCE.exists():
        raise FileExistsError("E49 Commons download receipt already exists")
    rows = _rows()
    expected = {destination(row) for row in rows}
    extras = _unexpected(expected)
    if extras:
        raise ValueError(f"E49 Commons payload root has unexpected files: {extras[:3]}")
    completed = []
    missing = []
    for row in rows:
        path = destination(row)
        if path.is_file():
            completed.append(inspect_file(path, row))
        else:
            missing.append(row)
    already_present = len(completed)
    with ThreadPoolExecutor(max_workers=DOWNLOAD_WORKERS) as pool:
        for index, result in enumerate(pool.map(_download, missing), start=1):
            completed.append(result)
            if index % 50 == 0 or index == len(missing):
                print(f"E49 Commons files {len(completed)}/{len(rows)}", flush=True)
    completed.sort(key=lambda row: (str(row["source"]), str(row["rank"]), str(row["identity"])))
    counts = Counter(str(row["source"]) for row in completed)
    if (
        len(completed) != EXPECTED_FILES
        or len({row["identity"] for row in completed}) != EXPECTED_FILES
        or counts != Counter({category.removeprefix("Category:Taken with "): COMMONS_RESERVE_PER_DEVICE
                              for category in COMMONS_CATEGORIES})
        or sum(int(row["bytes"]) for row in completed) != EXPECTED_BYTES
        or _unexpected(expected)
    ):
        raise ValueError("E49 Commons completed inventory differs from frozen contract")
    payload = {
        "schema_version": 1, "state": "e49_commons_downloaded_decoded_unscored",
        "role": "FINAL_REAL_COMPONENT_RESERVE_PENDING_DECONTAMINATION_AND_COMPLETE_E49",
        "source_contract_sha256": CONTRACT_SHA256,
        "files": len(completed), "bytes": EXPECTED_BYTES,
        "counts": dict(sorted(counts.items())), "already_present_at_run_start": already_present,
        "downloaded_this_run": len(completed) - already_present, "rows": completed,
        "model_scores_created": 0,
        "boundary": "Exact frozen Commons transfer and JPEG audit only; no detector or metric access.",
    }
    raw = _write_atomic(RECEIPT, payload)
    evidence = {
        "schema_version": 1, "state": payload["state"], "role": payload["role"],
        "source_contract_sha256": CONTRACT_SHA256, "files": payload["files"],
        "bytes": payload["bytes"], "counts": payload["counts"],
        "exif_make_present": sum(bool(row["exif_make"]) for row in completed),
        "exif_model_present": sum(bool(row["exif_model"]) for row in completed),
        "receipt_bytes": len(raw), "receipt_sha256": hashlib.sha256(raw).hexdigest(),
        "model_scores_created": 0,
    }
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
