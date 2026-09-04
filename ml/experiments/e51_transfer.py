"""Transfer byte-bound E51 payloads without opening a detector or metric."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import ssl
import threading
from typing import Any, Iterable, Mapping
import zipfile

import certifi
import fsspec
from huggingface_hub import HfApi, hf_hub_download
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.competitions.types.competition_api_service import ApiDownloadDataFilesRequest
from kagglesdk.datasets.types.dataset_api_service import ApiDownloadDatasetRequest
from PIL import Image

from experiments.e51_data_route import DATAPOINT_REVISION, DATAPOINT_REPO, DATAPOINT_SHARD_BYTES
from pixelproof.project_paths import DATA_ROOT, ML_ROOT


ROUTE_CONTRACT_SHA256 = "975e8164477c7234292ba87449007f0ee4c8b65eb582f25a8b0d81140ec315e4"
IEEE_REF = "sp-society-camera-model-identification"
IEEE_FILES = 2_640
IEEE_BYTES = 837_665_909
IEEE_ARCHIVE_FILES = 5_391
IEEE_ARCHIVE_BYTES = 11_447_649_387
IEEE_ARCHIVE_CONTAINER_BYTES = 11_333_585_079
IEEE_RANGE_BLOCK_BYTES = 8 * 1024**2
MAX_PIXELS = 100_000_000
SCMI30_REF = "goyalpuneet/sci30iitrpr"
SCMI30_VERSION = 2
SCMI30_FILES = 9_940
SCMI30_ARCHIVE_BYTES = 35_592_872_377
SCMI30_ARCHIVE_CONTAINER_BYTES = 34_429_117_013
SCMI30_SELECTED_FILES = 1_200
SCMI30_SELECTED_BYTES = 4_247_339_334
SCMI30_RANGE_BLOCK_BYTES = 1024**2
SCMI30_WORKERS = 4

_SCMI30_PROGRESS_LOCK = threading.Lock()
_SCMI30_PROGRESS = 0

ROOT = DATA_ROOT / "e51"
CONTRACT = ROOT / "route" / "contract_untransferred.json"
PAYLOAD_ROOT = ROOT / "payloads" / "ieee_spcup_test"
STAGING_ROOT = ROOT / "staging" / "ieee_spcup_test"
RECEIPT = ROOT / "receipts" / "ieee_spcup_download_unscored.json"
EVIDENCE = ML_ROOT.parent / "evidence" / "e51_ieee_download.json"
DATAPOINT_PAYLOAD_ROOT = ROOT / "payloads" / "datapoint_repository"
DATAPOINT_RECEIPT = ROOT / "receipts" / "datapoint_shards_download_unscored.json"
DATAPOINT_EVIDENCE = ML_ROOT.parent / "evidence" / "e51_datapoint_download.json"
SCMI30_PAYLOAD_ROOT = ROOT / "payloads" / "scmi30_cal"
SCMI30_STAGING_ROOT = ROOT / "staging" / "scmi30_cal"
SCMI30_RECEIPT = ROOT / "receipts" / "scmi30_cal_download_unscored.json"
SCMI30_EVIDENCE = ML_ROOT.parent / "evidence" / "e51_scmi30_download.json"


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


def validate_selected_archive_members(
    members: Mapping[str, tuple[int, int]], rows: Iterable[Mapping[str, Any]],
) -> dict[str, int]:
    rows = list(rows)
    selected_compressed_bytes = 0
    selected_files = 0
    for row in rows:
        name = str(row["remote_path"])
        found = members.get(name)
        if found is None or found[0] != int(row["expected_bytes"]):
            raise ValueError(f"IEEE archive member changed: {name}")
        selected_files += 1
        selected_compressed_bytes += found[1]
    return {
        "selected_files": selected_files,
        "selected_uncompressed_bytes": sum(int(row["expected_bytes"]) for row in rows),
        "selected_compressed_bytes": selected_compressed_bytes,
    }


def _competition_archive_url() -> str:
    api = KaggleApi()
    api.authenticate()
    with api.build_kaggle_client() as client:
        request = ApiDownloadDataFilesRequest()
        request.competition_name = IEEE_REF
        response = client.competitions.competition_api_client.download_data_files(request)
    return str(response.url)


def _dataset_archive_url() -> str:
    api = KaggleApi()
    api.authenticate()
    with api.build_kaggle_client() as client:
        request = ApiDownloadDatasetRequest()
        request.owner_slug = "goyalpuneet"
        request.dataset_slug = "sci30iitrpr"
        request.dataset_version_number = SCMI30_VERSION
        response = client.datasets.dataset_api_client.download_dataset(request)
    return str(response.request.url)


def scmi30_destination(row: Mapping[str, Any]) -> Path:
    remote = PurePosixPath(str(row["remote_path"]))
    if (
        remote.is_absolute()
        or ".." in remote.parts
        or len(remote.parts) < 4
        or remote.parts[0] != "SCMI30-IITRPR"
        or remote.parts[1] not in {"Random", "Similar"}
        or remote.parts[2] != str(row.get("device_folder", ""))
        or remote.suffix.lower() not in {".jpg", ".jpeg"}
    ):
        raise ValueError(f"unsafe SCMI30 path: {remote}")
    return SCMI30_PAYLOAD_ROOT.joinpath(*remote.parts[1:])


def inspect_scmi30_file(path: Path, expected: Mapping[str, Any]) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size != int(expected["expected_bytes"]):
        raise ValueError(f"SCMI30 payload byte length changed: {path.name}")
    with Image.open(path) as image:
        image.verify()
    with Image.open(path) as image:
        image.load()
        width, height = image.size
        decoded_format = str(image.format or "UNKNOWN").upper()
        if decoded_format not in {"JPEG", "MPO"} or min(width, height) <= 0:
            raise ValueError(f"SCMI30 payload format/geometry changed: {path.name}")
        if width * height > MAX_PIXELS:
            raise ValueError(f"SCMI30 payload exceeds safe pixel limit: {path.name}")
        exif = image.getexif()
        make = str(exif.get(271, "")).strip()
        model = str(exif.get(272, "")).strip()
    return {
        **{key: expected[key] for key in (
            "identity", "rank", "label", "role", "source", "device_id", "device_folder",
            "branch", "remote_path",
        )},
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": _digest(path),
        "format": decoded_format,
        "width": width,
        "height": height,
        "exif_make": make,
        "exif_model": model,
        "state": "downloaded_decoded_unscored",
    }


def _scmi30_contract() -> list[dict[str, Any]]:
    raw = CONTRACT.read_bytes()
    if hashlib.sha256(raw).hexdigest() != ROUTE_CONTRACT_SHA256:
        raise ValueError("E51 route contract changed before SCMI30 transfer")
    payload = json.loads(raw)
    role = payload.get("roles", {}).get("cal", {})
    rows = role.get("rows") or []
    if (
        payload.get("state") != "e51_route_frozen_untransferred_unscored"
        or payload.get("new_image_bytes_downloaded") != 0
        or payload.get("model_scores_created") != 0
        or role.get("reference") != SCMI30_REF
        or role.get("version") != SCMI30_VERSION
        or len(rows) != SCMI30_SELECTED_FILES
        or sum(int(row["expected_bytes"]) for row in rows) != SCMI30_SELECTED_BYTES
    ):
        raise ValueError("E51 SCMI30 route boundary changed")
    return rows


def _download_scmi30_chunk(url: str, rows: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    global _SCMI30_PROGRESS
    results = []
    ssl_context = ssl.create_default_context(cafile=certifi.where())
    with fsspec.open(
        url,
        "rb",
        block_size=SCMI30_RANGE_BLOCK_BYTES,
        cache_type="readahead",
        ssl=ssl_context,
    ) as remote_archive:
        with zipfile.ZipFile(remote_archive) as archive:
            for row in rows:
                destination = scmi30_destination(row)
                stage_name = hashlib.sha256(str(row["identity"]).encode()).hexdigest()[:20] + ".part"
                temporary = SCMI30_STAGING_ROOT / stage_name
                if temporary.exists():
                    temporary.unlink()
                with archive.open(str(row["remote_path"]), "r") as source, temporary.open("wb") as target:
                    shutil.copyfileobj(source, target, length=1024 * 1024)
                result = inspect_scmi30_file(temporary, row)
                destination.parent.mkdir(parents=True, exist_ok=True)
                temporary.replace(destination)
                results.append({**result, "path": str(destination)})
                with _SCMI30_PROGRESS_LOCK:
                    _SCMI30_PROGRESS += 1
                    if _SCMI30_PROGRESS % 50 == 0:
                        print(f"E51 SCMI30 files {_SCMI30_PROGRESS}/{SCMI30_SELECTED_FILES}", flush=True)
    return results


def validate_datapoint_remote(files: Mapping[str, int]) -> dict[str, int]:
    expected = {
        f"data/images/images-{index:04d}.parquet": size
        for index, size in DATAPOINT_SHARD_BYTES.items()
    }
    found = {name: files.get(name, -1) for name in expected}
    if found != expected:
        raise ValueError("Datapoint pinned shard inventory changed")
    return expected


def _datapoint_contract() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    raw = CONTRACT.read_bytes()
    if hashlib.sha256(raw).hexdigest() != ROUTE_CONTRACT_SHA256:
        raise ValueError("E51 route contract changed before Datapoint transfer")
    payload = json.loads(raw)
    role = payload.get("roles", {}).get("development_ai", {})
    rows = role.get("rows") or []
    expected_shards = [
        f"data/images/images-{index:04d}.parquet" for index in sorted(DATAPOINT_SHARD_BYTES)
    ]
    if (
        payload.get("state") != "e51_route_frozen_untransferred_unscored"
        or payload.get("new_image_bytes_downloaded") != 0
        or payload.get("model_scores_created") != 0
        or role.get("repo") != DATAPOINT_REPO
        or role.get("revision") != DATAPOINT_REVISION
        or role.get("source_shards") != expected_shards
        or role.get("source_shard_bytes") != sum(DATAPOINT_SHARD_BYTES.values())
        or len(rows) != 920
    ):
        raise ValueError("E51 Datapoint route boundary changed")
    return role, rows


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
    ssl_context = ssl.create_default_context(cafile=certifi.where())
    url = _competition_archive_url()
    STAGING_ROOT.mkdir(parents=True, exist_ok=True)
    with fsspec.open(
        url,
        "rb",
        block_size=IEEE_RANGE_BLOCK_BYTES,
        cache_type="readahead",
        ssl=ssl_context,
    ) as remote_archive:
        with zipfile.ZipFile(remote_archive) as archive:
            infos = archive.infolist()
            if len(infos) != IEEE_ARCHIVE_FILES or sum(info.file_size for info in infos) != IEEE_ARCHIVE_BYTES:
                raise ValueError("IEEE official ZIP inventory changed")
            members = {info.filename: (info.file_size, info.compress_size) for info in infos}
            if len(members) != len(infos):
                raise ValueError("IEEE official ZIP contains duplicate member names")
            archive_summary = validate_selected_archive_members(members, rows)
            if (
                archive_summary["selected_files"] != IEEE_FILES
                or archive_summary["selected_uncompressed_bytes"] != IEEE_BYTES
            ):
                raise ValueError("IEEE selected ZIP population changed")
            for index, row in enumerate(missing, start=1):
                destination = ieee_destination(row)
                temporary = STAGING_ROOT / f"{destination.name}.part"
                if temporary.exists():
                    temporary.unlink()
                with archive.open(str(row["remote_path"]), "r") as source, temporary.open("wb") as target:
                    shutil.copyfileobj(source, target, length=1024 * 1024)
                result = inspect_ieee_file(temporary, row)
                destination.parent.mkdir(parents=True, exist_ok=True)
                temporary.replace(destination)
                completed.append({**result, "path": str(destination)})
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
        "archive_transport": "official_zip_http_range_selected_members_only",
        "archive_container_bytes": IEEE_ARCHIVE_CONTAINER_BYTES,
        "selected_compressed_bytes": archive_summary["selected_compressed_bytes"],
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
        "archive_transport", "archive_container_bytes", "selected_compressed_bytes",
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


def download_datapoint() -> dict[str, Any]:
    """Fetch only the seven byte-bound Parquet shards; do not open their image columns."""
    if DATAPOINT_RECEIPT.exists() or DATAPOINT_EVIDENCE.exists():
        raise FileExistsError("E51 Datapoint receipt already exists; no silent replacement")
    role, rows = _datapoint_contract()
    info = HfApi().dataset_info(DATAPOINT_REPO, revision=DATAPOINT_REVISION, files_metadata=True)
    if info.sha != DATAPOINT_REVISION or info.gated != "manual":
        raise ValueError("Datapoint repository identity/access changed")
    remote_files = {item.rfilename: int(item.size or -1) for item in info.siblings}
    expected = validate_datapoint_remote(remote_files)
    free_before = shutil.disk_usage(ROOT).free
    expected_bytes = sum(expected.values())
    if free_before < expected_bytes + 10 * 1024**3:
        raise OSError("insufficient free space for Datapoint transfer plus 10 GiB reserve")
    completed = []
    for index, (name, size) in enumerate(expected.items(), start=1):
        path = Path(hf_hub_download(
            repo_id=DATAPOINT_REPO,
            repo_type="dataset",
            revision=DATAPOINT_REVISION,
            filename=name,
            local_dir=DATAPOINT_PAYLOAD_ROOT,
        ))
        if path.resolve() != (DATAPOINT_PAYLOAD_ROOT / PurePosixPath(name)).resolve():
            raise ValueError("Datapoint downloader resolved an unexpected destination")
        if not path.is_file() or path.stat().st_size != size:
            raise ValueError(f"Datapoint shard byte length changed: {name}")
        completed.append({
            "remote_path": name,
            "path": str(path),
            "bytes": size,
            "sha256": _digest(path),
            "state": "downloaded_unopened_unscored",
        })
        print(f"E51 Datapoint shards {index}/{len(expected)}", flush=True)
    if sum(row["bytes"] for row in completed) != expected_bytes:
        raise ValueError("Datapoint completed bytes changed")
    payload = {
        "schema_version": 1,
        "state": "e51_datapoint_shards_downloaded_unopened_unscored",
        "role": "DEVELOPMENT_AI_RESERVE",
        "route_contract_sha256": ROUTE_CONTRACT_SHA256,
        "repo": DATAPOINT_REPO,
        "revision": DATAPOINT_REVISION,
        "shards": len(completed),
        "bytes": expected_bytes,
        "reserved_rows": len(rows),
        "target_rows_after_realization": int(role["target_per_model_after_realization"]) * 5,
        "free_bytes_before": free_before,
        "free_bytes_after": shutil.disk_usage(ROOT).free,
        "rows": completed,
        "image_columns_opened": False,
        "model_scores_created": 0,
    }
    raw = _write_atomic(DATAPOINT_RECEIPT, payload)
    evidence = {key: payload[key] for key in (
        "schema_version", "state", "role", "route_contract_sha256", "repo", "revision",
        "shards", "bytes", "reserved_rows", "target_rows_after_realization", "free_bytes_before",
        "free_bytes_after", "image_columns_opened", "model_scores_created",
    )}
    evidence.update({
        "receipt_bytes": len(raw),
        "receipt_sha256": hashlib.sha256(raw).hexdigest(),
    })
    _write_atomic(DATAPOINT_EVIDENCE, evidence)
    return evidence


def download_scmi30() -> dict[str, Any]:
    """Range-read only the 1,200 frozen CAL members from the official 34 GB ZIP."""
    if SCMI30_RECEIPT.exists() or SCMI30_EVIDENCE.exists():
        raise FileExistsError("E51 SCMI30 receipt already exists; no silent replacement")
    rows = _scmi30_contract()
    expected_paths = {scmi30_destination(row) for row in rows}
    if SCMI30_PAYLOAD_ROOT.exists():
        extras = sorted(
            str(path) for path in SCMI30_PAYLOAD_ROOT.rglob("*")
            if path.is_file() and not path.name.startswith("._") and path not in expected_paths
        )
        if extras:
            raise ValueError(f"unexpected files in E51 SCMI30 payload root: {extras[:3]}")
    free_before = shutil.disk_usage(ROOT).free
    if free_before < SCMI30_SELECTED_BYTES + 10 * 1024**3:
        raise OSError("insufficient free space for SCMI30 transfer plus 10 GiB reserve")
    completed = []
    missing = []
    for row in rows:
        destination = scmi30_destination(row)
        if destination.is_file():
            completed.append(inspect_scmi30_file(destination, row))
        else:
            missing.append(row)
    already_present = len(completed)
    ssl_context = ssl.create_default_context(cafile=certifi.where())
    url = _dataset_archive_url()
    SCMI30_STAGING_ROOT.mkdir(parents=True, exist_ok=True)
    with fsspec.open(
        url,
        "rb",
        block_size=SCMI30_RANGE_BLOCK_BYTES,
        cache_type="readahead",
        ssl=ssl_context,
    ) as remote_archive:
        with zipfile.ZipFile(remote_archive) as archive:
            infos = archive.infolist()
            if len(infos) != SCMI30_FILES or sum(info.file_size for info in infos) != SCMI30_ARCHIVE_BYTES:
                raise ValueError("SCMI30 official ZIP inventory changed")
            info_by_name = {info.filename: info for info in infos}
            if len(info_by_name) != len(infos):
                raise ValueError("SCMI30 official ZIP contains duplicate member names")
            members = {name: (info.file_size, info.compress_size) for name, info in info_by_name.items()}
            archive_summary = validate_selected_archive_members(members, rows)
            if (
                archive_summary["selected_files"] != SCMI30_SELECTED_FILES
                or archive_summary["selected_uncompressed_bytes"] != SCMI30_SELECTED_BYTES
            ):
                raise ValueError("SCMI30 selected ZIP population changed")
            offsets = {name: info.header_offset for name, info in info_by_name.items()}
    missing.sort(key=lambda row: offsets[str(row["remote_path"])])
    chunk_size = max(1, (len(missing) + SCMI30_WORKERS - 1) // SCMI30_WORKERS)
    chunks = [missing[index:index + chunk_size] for index in range(0, len(missing), chunk_size)]
    global _SCMI30_PROGRESS
    _SCMI30_PROGRESS = already_present
    with ThreadPoolExecutor(max_workers=min(SCMI30_WORKERS, len(chunks) or 1)) as pool:
        for results in pool.map(lambda chunk: _download_scmi30_chunk(url, chunk), chunks):
            completed.extend(results)
    print(f"E51 SCMI30 files {len(completed)}/{len(rows)}", flush=True)
    completed.sort(key=lambda row: row["identity"])
    devices = Counter(row["device_id"] for row in completed)
    branches = Counter(row["branch"] for row in completed)
    if (
        len(completed) != SCMI30_SELECTED_FILES
        or len({row["identity"] for row in completed}) != SCMI30_SELECTED_FILES
        or sum(int(row["bytes"]) for row in completed) != SCMI30_SELECTED_BYTES
        or set(devices.values()) != {40}
        or branches != {"Random": 600, "Similar": 600}
    ):
        raise ValueError("completed SCMI30 payload differs from frozen route")
    payload = {
        "schema_version": 1,
        "state": "e51_scmi30_cal_downloaded_decoded_unscored",
        "role": "CAL_REAL",
        "route_contract_sha256": ROUTE_CONTRACT_SHA256,
        "dataset": SCMI30_REF,
        "version": SCMI30_VERSION,
        "files": len(completed),
        "bytes": SCMI30_SELECTED_BYTES,
        "devices": len(devices),
        "per_device": 40,
        "branches": dict(sorted(branches.items())),
        "already_present_at_run_start": already_present,
        "downloaded_this_run": len(completed) - already_present,
        "range_workers": SCMI30_WORKERS,
        "archive_transport": "official_zip_http_range_selected_members_only",
        "archive_container_bytes": SCMI30_ARCHIVE_CONTAINER_BYTES,
        "selected_compressed_bytes": archive_summary["selected_compressed_bytes"],
        "free_bytes_before": free_before,
        "free_bytes_after": shutil.disk_usage(ROOT).free,
        "rows": completed,
        "model_scores_created": 0,
    }
    raw = _write_atomic(SCMI30_RECEIPT, payload)
    evidence = {key: payload[key] for key in (
        "schema_version", "state", "role", "route_contract_sha256", "dataset", "version",
        "files", "bytes", "devices", "per_device", "branches", "already_present_at_run_start",
        "downloaded_this_run", "range_workers", "archive_transport", "archive_container_bytes",
        "selected_compressed_bytes", "free_bytes_before", "free_bytes_after", "model_scores_created",
    )}
    evidence.update({
        "receipt_bytes": len(raw),
        "receipt_sha256": hashlib.sha256(raw).hexdigest(),
        "payload_identity_sha256": hashlib.sha256(json.dumps(
            [row["identity"] for row in completed], separators=(",", ":")
        ).encode()).hexdigest(),
        "exif_make_present": sum(bool(row["exif_make"]) for row in completed),
        "exif_model_present": sum(bool(row["exif_model"]) for row in completed),
    })
    _write_atomic(SCMI30_EVIDENCE, evidence)
    return evidence


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("download-ieee", "download-datapoint", "download-scmi30"))
    args = parser.parse_args(argv)
    if args.command == "download-ieee":
        print(json.dumps(download_ieee(), indent=2, sort_keys=True))
    elif args.command == "download-datapoint":
        print(json.dumps(download_datapoint(), indent=2, sort_keys=True))
    else:
        print(json.dumps(download_scmi30(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
