"""Transfer byte-bound E51 payloads without opening a detector or metric."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import time
from typing import Any, Iterable, Mapping

from kaggle.api.kaggle_api_extended import KaggleApi
from PIL import Image

from pixelproof.project_paths import DATA_ROOT, ML_ROOT


ROUTE_CONTRACT_SHA256 = "975e8164477c7234292ba87449007f0ee4c8b65eb582f25a8b0d81140ec315e4"
IEEE_REF = "sp-society-camera-model-identification"
IEEE_FILES = 2_640
IEEE_BYTES = 837_665_909
IEEE_WORKERS = 6
MAX_PIXELS = 100_000_000

ROOT = DATA_ROOT / "e51"
CONTRACT = ROOT / "route" / "contract_untransferred.json"
PAYLOAD_ROOT = ROOT / "payloads" / "ieee_spcup_test"
STAGING_ROOT = ROOT / "staging" / "ieee_spcup_test"
RECEIPT = ROOT / "receipts" / "ieee_spcup_download_unscored.json"
EVIDENCE = ML_ROOT.parent / "evidence" / "e51_ieee_download.json"


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


def ieee_destination(row: Mapping[str, Any]) -> Path:
    remote = PurePosixPath(str(row["remote_path"]))
    if (
        remote.is_absolute()
        or ".." in remote.parts
        or len(remote.parts) != 3
        or remote.parts[:2] != ("test", "test")
        or remote.suffix.lower() not in {".tif", ".tiff"}
    ):
        raise ValueError(f"unsafe IEEE path: {remote}")
    return PAYLOAD_ROOT / remote.name


def inspect_ieee_file(path: Path, expected: Mapping[str, Any]) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size != int(expected["expected_bytes"]):
        raise ValueError(f"IEEE payload byte length changed: {path.name}")
    with Image.open(path) as image:
        image.verify()
    with Image.open(path) as image:
        width, height = image.size
        decoded_format = str(image.format or "UNKNOWN").upper()
        # The publisher names these files .tif but the frozen competition bodies are PNG.
        if decoded_format != "PNG" or (width, height) != (512, 512):
            raise ValueError(f"IEEE payload format/geometry changed: {path.name}")
        if width * height > MAX_PIXELS:
            raise ValueError(f"IEEE payload exceeds safe pixel limit: {path.name}")
    return {
        **{key: expected[key] for key in (
            "identity", "label", "role", "source", "transport_cell", "remote_path",
        )},
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": _digest(path),
        "format": decoded_format,
        "width": width,
        "height": height,
        "state": "downloaded_decoded_unscored",
    }


def _route_rows() -> list[dict[str, Any]]:
    raw = CONTRACT.read_bytes()
    if hashlib.sha256(raw).hexdigest() != ROUTE_CONTRACT_SHA256:
        raise ValueError("E51 route contract changed before IEEE transfer")
    payload = json.loads(raw)
    role = payload.get("roles", {}).get("development_real", {})
    rows = role.get("rows") or []
    if (
        payload.get("state") != "e51_route_frozen_untransferred_unscored"
        or payload.get("new_image_bytes_downloaded") != 0
        or payload.get("model_scores_created") != 0
        or role.get("reference") != IEEE_REF
        or role.get("test_files") != IEEE_FILES
        or role.get("test_bytes") != IEEE_BYTES
        or len(rows) != IEEE_FILES
        or sum(int(row["expected_bytes"]) for row in rows) != IEEE_BYTES
    ):
        raise ValueError("E51 IEEE route boundary changed")
    return rows


def _download_ieee(row: Mapping[str, Any]) -> dict[str, Any]:
    destination = ieee_destination(row)
    if destination.exists():
        return inspect_ieee_file(destination, row)
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = STAGING_ROOT / destination.stem
    stage.mkdir(parents=True, exist_ok=True)
    downloaded = stage / destination.name
    last_error: Exception | None = None
    for attempt in range(6):
        try:
            if downloaded.exists() and downloaded.stat().st_size != int(row["expected_bytes"]):
                downloaded.unlink()
            api = KaggleApi()
            api.authenticate()
            api.competition_download_file(
                IEEE_REF,
                str(row["remote_path"]),
                path=str(stage),
                force=not downloaded.exists(),
                quiet=True,
            )
            result = inspect_ieee_file(downloaded, row)
            downloaded.replace(destination)
            stage.rmdir()
            return {**result, "path": str(destination)}
        except Exception as error:  # Kaggle SDK exceptions vary by release.
            last_error = error
            if attempt < 5:
                time.sleep(min(2.0**attempt, 30.0))
    raise RuntimeError(f"IEEE transfer failed at {row['identity']}: {last_error}")


def _unexpected(expected: set[Path]) -> list[str]:
    if not PAYLOAD_ROOT.exists():
        return []
    return sorted(
        str(path) for path in PAYLOAD_ROOT.rglob("*")
        if path.is_file() and not path.name.startswith("._") and path not in expected
    )


def download_ieee() -> dict[str, Any]:
    if RECEIPT.exists() or EVIDENCE.exists():
        raise FileExistsError("E51 IEEE receipt already exists; no silent replacement")
    rows = _route_rows()
    expected_paths = {ieee_destination(row) for row in rows}
    extras = _unexpected(expected_paths)
    if extras:
        raise ValueError(f"unexpected files in E51 IEEE payload root: {extras[:3]}")
    free_before = shutil.disk_usage(ROOT).free
    if free_before < IEEE_BYTES + 10 * 1024**3:
        raise OSError("insufficient free space for IEEE transfer plus 10 GiB reserve")
    completed = []
    missing = []
    for row in rows:
        destination = ieee_destination(row)
        if destination.is_file():
            completed.append(inspect_ieee_file(destination, row))
        else:
            missing.append(row)
    already_present = len(completed)
    with ThreadPoolExecutor(max_workers=IEEE_WORKERS) as pool:
        for index, result in enumerate(pool.map(_download_ieee, missing), start=1):
            completed.append(result)
            if index % 100 == 0 or index == len(missing):
                print(f"E51 IEEE files {len(completed)}/{len(rows)}", flush=True)
    completed.sort(key=lambda row: row["identity"])
    cells = Counter(row["transport_cell"] for row in completed)
    if (
        len(completed) != IEEE_FILES
        or len({row["identity"] for row in completed}) != IEEE_FILES
        or sum(int(row["bytes"]) for row in completed) != IEEE_BYTES
        or cells != {"postprocessed": 1_320, "unaltered": 1_320}
        or _unexpected(expected_paths)
    ):
        raise ValueError("completed IEEE payload differs from the frozen route")
    payload = {
        "schema_version": 1,
        "state": "e51_ieee_downloaded_decoded_unscored",
        "role": "DEVELOPMENT_REAL",
        "route_contract_sha256": ROUTE_CONTRACT_SHA256,
        "competition": IEEE_REF,
        "files": len(completed),
        "bytes": IEEE_BYTES,
        "transport_cells": dict(sorted(cells.items())),
        "already_present_at_run_start": already_present,
        "downloaded_this_run": len(completed) - already_present,
        "free_bytes_before": free_before,
        "free_bytes_after": shutil.disk_usage(ROOT).free,
        "rows": completed,
        "model_scores_created": 0,
        "boundary": "Exact transfer and decode only; no model, metric, threshold or label adaptation.",
    }
    raw = _write_atomic(RECEIPT, payload)
    evidence = {key: payload[key] for key in (
        "schema_version", "state", "role", "route_contract_sha256", "competition", "files",
        "bytes", "transport_cells", "already_present_at_run_start", "downloaded_this_run",
        "free_bytes_before", "free_bytes_after", "model_scores_created",
    )}
    evidence.update({
        "receipt_bytes": len(raw),
        "receipt_sha256": hashlib.sha256(raw).hexdigest(),
        "payload_identity_sha256": hashlib.sha256(json.dumps(
            [row["identity"] for row in completed], separators=(",", ":")
        ).encode()).hexdigest(),
    })
    _write_atomic(EVIDENCE, evidence)
    return evidence


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("download-ieee",))
    args = parser.parse_args(argv)
    if args.command == "download-ieee":
        print(json.dumps(download_ieee(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
