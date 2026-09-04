"""Freeze an ungated OpenFake reserve for E49-B without image or detector access."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import time
from typing import Any, Iterable, Mapping, Sequence

from huggingface_hub import HfApi
import requests
import pyarrow.parquet as pq

from pixelproof.project_paths import DATA_ROOT, ML_ROOT


NAMESPACE = "E49_B_OPENFAKE_V1"
NAMESPACE_C = "E49_C_OPENFAKE_V1"
REPO_ID = "ComplexDataLab/OpenFake"
REVISION = "3fd1109dc3258874243fa31c5bda9ee24260163b"
LICENSE = "cc-by-nc-4.0"
CONFIG = "core"
SPLIT = "test"
EXPECTED_ROWS = 91_398
PAGE_SIZE = 100
# Two cold Viewer pages at a time stays below the public service's observed 429 boundary.
FETCH_WORKERS = 2
TARGET_PER_MODEL = 160
RESERVE_PER_MODEL = 192
MODEL_KEYS = {
    "gpt-image-2": "GPT Image 2",
    "nano-banana-pro": "Nano Banana Pro",
    "seedream-v5.0": "Seedream v5.0",
    "flux.2-klein-9b": "FLUX.2 Klein 9B",
    "midjourney-7": "Midjourney 7",
}
MODEL_KEYS_C = {
    "gpt-image-2": "GPT Image 2",
    "z-image-turbo": "Z-Image Turbo",
    "seedream-v5.0": "Seedream v5.0",
    "flux.2-klein-9b": "FLUX.2 Klein 9B",
    "midjourney-7": "Midjourney 7",
}
ROWS_ENDPOINT = "https://datasets-server.huggingface.co/rows"
RESOLVE_ENDPOINT = "https://huggingface.co/datasets"
EXPECTED_SHARDS = 13
MAX_METADATA_RANGE_BYTES = 64 * 1024**2

ROOT = DATA_ROOT / "e49" / "openfake"
PAGE_CACHE = ROOT / "metadata_pages"
CONTRACT = ROOT / "source_contract_unscored.json"
EVIDENCE = ML_ROOT.parent / "evidence" / "e49_b_openfake_contract.json"
CONTRACT_C = ROOT / "source_contract_e49c_unscored.json"
EVIDENCE_C = ML_ROOT.parent / "evidence" / "e49_c_openfake_contract.json"


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def _write_atomic(path: Path, value: Any) -> bytes:
    raw = _json_bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_bytes(raw)
    temporary.replace(path)
    return raw


def _rank(model: str, row_index: int, *, namespace: str = NAMESPACE) -> str:
    identity = f"{REPO_ID}|{REVISION}|{CONFIG}|{SPLIT}|{model}|{row_index}"
    return hashlib.sha256(f"{namespace}|{identity}".encode()).hexdigest()


def validate_repository(info: Any) -> None:
    card_data = info.card_data or {}
    if info.id != REPO_ID or info.sha != REVISION:
        raise ValueError("E49-B OpenFake repository identity changed")
    if info.gated is not False:
        raise ValueError("E49-B OpenFake is no longer ungated")
    if str(card_data.get("license", "")).lower() != LICENSE:
        raise ValueError("E49-B OpenFake licence changed")


def compact_page(
    payload: Mapping[str, Any], *, offset: int, requested_length: int = PAGE_SIZE,
) -> list[dict[str, Any]]:
    """Validate a Viewer page and discard prompts plus expiring asset URLs."""
    if int(payload.get("num_rows_total", -1)) != EXPECTED_ROWS or payload.get("partial") is True:
        raise ValueError("E49-B OpenFake Viewer split identity changed")
    rows = payload.get("rows")
    if not isinstance(rows, list) or len(rows) > requested_length:
        raise ValueError("E49-B OpenFake Viewer page shape changed")
    if offset + len(rows) < EXPECTED_ROWS and len(rows) != requested_length:
        raise ValueError("E49-B OpenFake Viewer returned an incomplete page")

    compact: list[dict[str, Any]] = []
    for position, wrapped in enumerate(rows):
        row_index = int(wrapped.get("row_idx", -1))
        if row_index != offset + position or wrapped.get("truncated_cells"):
            raise ValueError(f"E49-B OpenFake row order/content changed at {row_index}")
        row = wrapped.get("row") or {}
        image = row.get("image") or {}
        src = str(image.get("src", ""))
        marker = f"/--/{REVISION}/--/{CONFIG}/{SPLIT}/{row_index}/image/"
        width, height = int(image.get("width") or 0), int(image.get("height") or 0)
        if marker not in src or width <= 0 or height <= 0:
            raise ValueError(f"E49-B OpenFake asset identity/geometry changed at {row_index}")
        compact.append({
            "row_index": row_index,
            "label": str(row.get("label", "")),
            "model": str(row.get("model", "")),
            "type": str(row.get("type", "")),
            "release_date": str(row.get("release_date") or ""),
            "width": width,
            "height": height,
        })
    return compact


def eligible(row: Mapping[str, Any], *, model_keys: Mapping[str, str] = MODEL_KEYS) -> bool:
    row_type = str(row.get("type", "")).strip().lower()
    return (
        row.get("label") == "fake"
        and row.get("model") in model_keys
        and bool(row_type)
        and row_type != "video"
    )


def select_reserve(
    rows: Sequence[Mapping[str, Any]], *, reserve_per_model: int = RESERVE_PER_MODEL,
    model_keys: Mapping[str, str] = MODEL_KEYS, namespace: str = NAMESPACE,
) -> list[dict[str, Any]]:
    """Freeze exact rows by namespace hash after source-stratified metadata qualification."""
    output: list[dict[str, Any]] = []
    for model, display_name in model_keys.items():
        candidates = []
        for source in rows:
            if source.get("model") != model or not eligible(source, model_keys=model_keys):
                continue
            row = dict(source)
            row_index = int(row["row_index"])
            row["record_id"] = f"openfake:{CONFIG}:{SPLIT}:{row_index}"
            row["parent_id"] = row["record_id"]
            row["rank"] = _rank(model, row_index, namespace=namespace)
            row["source"] = display_name
            row["generator"] = display_name
            row["label"] = 1
            candidates.append(row)
        candidates.sort(key=lambda row: (row["rank"], row["record_id"]))
        if len(candidates) < reserve_per_model:
            raise ValueError(
                f"E49-B OpenFake {model} reserve unavailable: {len(candidates)}/{reserve_per_model}"
            )
        output.extend(candidates[:reserve_per_model])
    return sorted(output, key=lambda row: (row["source"], row["rank"], row["record_id"]))


def first_complete_prefix(
    rows: Sequence[Mapping[str, Any]], *, page_size: int = PAGE_SIZE,
    reserve_per_model: int = RESERVE_PER_MODEL, model_keys: Mapping[str, str] = MODEL_KEYS,
) -> tuple[int | None, Counter[str]]:
    """Apply the frozen page-prefix stop rule to any equivalent metadata transport."""
    counts: Counter[str] = Counter()
    for index, row in enumerate(rows, start=1):
        if eligible(row, model_keys=model_keys):
            counts[str(row["model"])] += 1
        if (index % page_size == 0 or index == len(rows)) and all(
            counts[model] >= reserve_per_model for model in model_keys
        ):
            return index, counts
    return None, counts


class CountedRangeReader(io.RawIOBase):
    """Seekable HTTP range reader that refuses unbounded metadata transfers."""

    def __init__(self, url: str, size: int) -> None:
        super().__init__()
        self.url = url
        self.size = size
        self.position = 0
        self.transferred_bytes = 0
        self.requests_made = 0
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": "PixelProof-E49B/1.0 projected-metadata-audit"})

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.position

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            position = offset
        elif whence == io.SEEK_CUR:
            position = self.position + offset
        elif whence == io.SEEK_END:
            position = self.size + offset
        else:
            raise ValueError("E49-B invalid range-reader whence")
        if position < 0 or position > self.size:
            raise ValueError("E49-B range-reader seek outside source")
        self.position = position
        return position

    def read(self, size: int = -1) -> bytes:
        if self.position >= self.size:
            return b""
        if size is None or size < 0:
            size = self.size - self.position
        size = min(size, self.size - self.position)
        if size > MAX_METADATA_RANGE_BYTES:
            raise ValueError(f"E49-B refused oversized metadata range: {size}")
        start, end = self.position, self.position + size - 1
        last_error: Exception | None = None
        for attempt in range(6):
            try:
                with self._session.get(
                    self.url,
                    headers={"Range": f"bytes={start}-{end}"},
                    stream=True,
                    timeout=(15, 90),
                ) as response:
                    if response.status_code != 206:
                        response.raise_for_status()
                        raise ValueError(f"E49-B source ignored Range: {response.status_code}")
                    expected_range = f"bytes {start}-{end}/{self.size}"
                    if response.headers.get("Content-Range") != expected_range:
                        raise ValueError("E49-B source returned the wrong byte range")
                    data = response.raw.read(size + 1)
                if len(data) != size:
                    raise ValueError(f"E49-B short metadata range: {len(data)}/{size}")
                self.position += size
                self.transferred_bytes += size
                self.requests_made += 1
                return data
            except (requests.RequestException, ValueError) as error:
                last_error = error
                if attempt == 5:
                    break
                time.sleep(min(2 ** attempt, 20))
        raise RuntimeError(f"E49-B metadata range failed: {start}-{end}: {last_error}")

    def readinto(self, buffer: Any) -> int:
        data = self.read(len(buffer))
        buffer[:len(data)] = data
        return len(data)

    def close(self) -> None:
        self._session.close()
        super().close()


def _read_projected_shard(file_row: Mapping[str, Any]) -> dict[str, Any]:
    name = str(file_row["path"])
    size = int(file_row["bytes"])
    url = f"{RESOLVE_ENDPOINT}/{REPO_ID}/resolve/{REVISION}/{name}"
    with CountedRangeReader(url, size) as reader:
        parquet = pq.ParquetFile(reader)
        required = {"label", "model", "type", "release_date"}
        if not required.issubset(parquet.schema.names):
            raise ValueError(f"E49-B projected schema changed: {name}")
        table = parquet.read(columns=sorted(required))
        columns = table.to_pydict()
        rows = [{
            "label": str(label),
            "model": str(model),
            "type": str(row_type),
            "release_date": str(release_date or ""),
            "width": 0,
            "height": 0,
        } for label, model, release_date, row_type in zip(
            columns["label"], columns["model"], columns["release_date"], columns["type"], strict=True
        )]
        return {
            "path": name,
            "repository_bytes": size,
            "lfs_sha256": str(file_row["sha256"]),
            "rows": rows,
            "range_bytes_transferred": reader.transferred_bytes,
            "range_requests": reader.requests_made,
        }


def _projected_metadata() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    info = HfApi().dataset_info(REPO_ID, revision=REVISION, files_metadata=True)
    validate_repository(info)
    by_name = {sibling.rfilename: sibling for sibling in info.siblings or []}
    files = []
    for index in range(EXPECTED_SHARDS):
        name = f"core/test-{index:05d}-of-{EXPECTED_SHARDS:05d}.parquet"
        sibling = by_name.get(name)
        if sibling is None or sibling.size is None or sibling.lfs is None:
            raise ValueError(f"E49-B pinned source shard metadata missing: {name}")
        files.append({"path": name, "bytes": sibling.size, "sha256": sibling.lfs.sha256})
    with ThreadPoolExecutor(max_workers=2) as pool:
        projected = list(pool.map(_read_projected_shard, files))

    rows: list[dict[str, Any]] = []
    audit = []
    for shard in projected:
        start = len(rows)
        for local_index, row in enumerate(shard.pop("rows")):
            rows.append({**row, "row_index": start + local_index})
        audit.append({**shard, "row_start": start, "row_end_exclusive": len(rows)})
    if len(rows) != EXPECTED_ROWS:
        raise ValueError(f"E49-B projected row count changed: {len(rows)}/{EXPECTED_ROWS}")
    return rows, audit


def _validate_viewer_prefix(projected: Sequence[Mapping[str, Any]]) -> int:
    validated = 0
    for offset in range(0, EXPECTED_ROWS, PAGE_SIZE):
        path = PAGE_CACHE / f"{offset:06d}.json"
        if not path.is_file():
            break
        cached = _cached_page(path, offset)
        for cached_row, projected_row in zip(cached, projected[offset:offset + len(cached)], strict=True):
            for key in ("row_index", "label", "model", "type", "release_date"):
                if cached_row[key] != projected_row[key]:
                    raise ValueError(f"E49-B Viewer/Parquet mismatch at row {cached_row['row_index']}")
        validated += len(cached)
    return validated


def _continuous_viewer_metadata() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for offset in range(0, EXPECTED_ROWS, PAGE_SIZE):
        path = PAGE_CACHE / f"{offset:06d}.json"
        if not path.is_file():
            break
        rows.extend(_cached_page(path, offset))
    return rows


def _cached_page(path: Path, offset: int) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text())
    if (
        payload.get("state") != "e49_b_openfake_compact_metadata_page"
        or payload.get("revision") != REVISION
        or int(payload.get("offset", -1)) != offset
        or int(payload.get("expected_rows", -1)) != EXPECTED_ROWS
    ):
        raise ValueError(f"E49-B invalid cached page: {path.name}")
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise ValueError(f"E49-B invalid cached rows: {path.name}")
    for position, row in enumerate(rows):
        if int(row.get("row_index", -1)) != offset + position:
            raise ValueError(f"E49-B cached row order changed: {path.name}")
        if set(row) != {"row_index", "label", "model", "type", "release_date", "width", "height"}:
            raise ValueError(f"E49-B cached schema changed: {path.name}")
    return rows


def _fetch_page(session: requests.Session, offset: int) -> list[dict[str, Any]]:
    path = PAGE_CACHE / f"{offset:06d}.json"
    if path.is_file():
        return _cached_page(path, offset)
    params = {"dataset": REPO_ID, "config": CONFIG, "split": SPLIT,
              "offset": offset, "length": PAGE_SIZE}
    last_error: Exception | None = None
    for attempt in range(7):
        try:
            response = session.get(ROWS_ENDPOINT, params=params, timeout=(15, 90))
            response.raise_for_status()
            rows = compact_page(response.json(), offset=offset)
            _write_atomic(path, {
                "schema_version": 1,
                "state": "e49_b_openfake_compact_metadata_page",
                "repository": REPO_ID,
                "revision": REVISION,
                "config": CONFIG,
                "split": SPLIT,
                "expected_rows": EXPECTED_ROWS,
                "offset": offset,
                "rows": rows,
            })
            return rows
        except (requests.RequestException, ValueError, json.JSONDecodeError) as error:
            last_error = error
            if attempt == 6:
                break
            time.sleep(min(2 ** attempt, 30))
    raise RuntimeError(f"E49-B Viewer page failed at {offset}: {last_error}")


def _fetch_page_isolated(offset: int) -> list[dict[str, Any]]:
    """Use one Session per worker; requests.Session is not shared across threads."""
    with requests.Session() as session:
        session.headers.update({"User-Agent": "PixelProof-E49B/1.0 metadata-only research audit"})
        return _fetch_page(session, offset)


def bind() -> dict[str, Any]:
    if CONTRACT.exists() or EVIDENCE.exists():
        raise FileExistsError("E49-B OpenFake contract already exists")
    info = HfApi().dataset_info(REPO_ID, revision=REVISION)
    validate_repository(info)

    scanned: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    stop_offset: int | None = None
    prefetched_end: int = 0
    offsets = list(range(0, EXPECTED_ROWS, PAGE_SIZE))
    for batch_start in range(0, len(offsets), FETCH_WORKERS):
        batch = offsets[batch_start:batch_start + FETCH_WORKERS]
        with ThreadPoolExecutor(max_workers=FETCH_WORKERS) as pool:
            futures = {offset: pool.submit(_fetch_page_isolated, offset) for offset in batch}
            pages = {offset: futures[offset].result() for offset in batch}
        prefetched_end = min(batch[-1] + len(pages[batch[-1]]), EXPECTED_ROWS)
        for offset in batch:
            page = pages[offset]
            scanned.extend(page)
            counts.update(str(row["model"]) for row in page if eligible(row))
            if offset % 5_000 == 0:
                summary = ", ".join(f"{key}={counts[key]}" for key in MODEL_KEYS)
                print(
                    f"E49-B metadata {min(offset + len(page), EXPECTED_ROWS)}/{EXPECTED_ROWS}: {summary}",
                    flush=True,
                )
            if all(counts[model] >= RESERVE_PER_MODEL for model in MODEL_KEYS):
                stop_offset = offset + len(page)
                break
        if stop_offset is not None:
            break
    if stop_offset is None:
        missing = {model: counts[model] for model in MODEL_KEYS if counts[model] < RESERVE_PER_MODEL}
        failure = {
            "schema_version": 1,
            "state": "e49_b_openfake_rejected_population_shortfall_unscored",
            "role": "REJECTED_SOURCE_QUALIFICATION_ONLY",
            "repository": REPO_ID,
            "revision": REVISION,
            "license": LICENSE,
            "config": CONFIG,
            "split": SPLIT,
            "expected_split_rows": EXPECTED_ROWS,
            "scanned_rows": len(scanned),
            "eligible_counts": {model: counts[model] for model in MODEL_KEYS},
            "required_per_model": RESERVE_PER_MODEL,
            "missing_cells": missing,
            "selected_rows": 0,
            "image_assets_requested": 0,
            "new_image_bytes_downloaded": 0,
            "model_scores_created": 0,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "boundary": (
                "Preregistered source population failed before identity selection. No replacement "
                "model, image access, detector access, metric or retry is permitted inside E49-B."
            ),
        }
        _write_atomic(EVIDENCE, failure)
        return failure

    selected = select_reserve(scanned)
    selected_counts = Counter(str(row["model"]) for row in selected)
    identity_sha256 = hashlib.sha256(
        "\n".join(sorted(str(row["record_id"]) for row in selected)).encode()
    ).hexdigest()
    payload = {
        "schema_version": 1,
        "state": "e49_b_openfake_frozen_untransferred_unscored",
        "role": "FINAL_AI_COMPONENT_CANDIDATE_PENDING_BYTE_FEASIBILITY_AND_COMPLETE_E49",
        "repository": REPO_ID,
        "revision": REVISION,
        "license": LICENSE,
        "gated": False,
        "config": CONFIG,
        "split": SPLIT,
        "expected_split_rows": EXPECTED_ROWS,
        "viewer_endpoint": ROWS_ENDPOINT,
        "scan": {
            "page_size": PAGE_SIZE,
            "prefix_start": 0,
            "prefix_end_exclusive": stop_offset,
            "prefetched_end_exclusive": prefetched_end,
            "eligible_counts": {model: counts[model] for model in MODEL_KEYS},
            "stop_rule": "first_complete_100_row_page_with_192_eligible_rows_per_exact_model",
            "stored_fields": ["row_index", "label", "model", "type", "release_date", "width", "height"],
            "excluded_fields": ["prompt", "expiring_asset_url"],
        },
        "selection": {
            "namespace": NAMESPACE,
            "target_per_model": TARGET_PER_MODEL,
            "reserve_per_model": RESERVE_PER_MODEL,
            "selected_counts": dict(sorted(selected_counts.items())),
            "reserve_identity_sha256": identity_sha256,
        },
        "rows": selected,
        "image_assets_requested": 0,
        "new_image_bytes_downloaded": 0,
        "model_scores_created": 0,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "boundary": (
            "Metadata-only source freeze. No prompt retained, asset requested, image downloaded, "
            "detector loaded or metric opened. Viewer JPEG byte feasibility remains mandatory."
        ),
    }
    raw = _write_atomic(CONTRACT, payload)
    evidence = {
        "schema_version": 1,
        "state": payload["state"],
        "role": payload["role"],
        "repository": REPO_ID,
        "revision": REVISION,
        "license": LICENSE,
        "expected_split_rows": EXPECTED_ROWS,
        "prefix_end_exclusive": stop_offset,
        "prefetched_end_exclusive": prefetched_end,
        "eligible_counts": payload["scan"]["eligible_counts"],
        "selected_counts": payload["selection"]["selected_counts"],
        "reserve_identity_sha256": identity_sha256,
        "contract_bytes": len(raw),
        "contract_sha256": hashlib.sha256(raw).hexdigest(),
        "image_assets_requested": 0,
        "new_image_bytes_downloaded": 0,
        "model_scores_created": 0,
    }
    _write_atomic(EVIDENCE, evidence)
    return evidence


def bind_projected() -> dict[str, Any]:
    """Complete the identical population audit via exact-revision scalar column ranges."""
    if CONTRACT.exists() or EVIDENCE.exists():
        raise FileExistsError("E49-B OpenFake result already exists")
    rows, shard_audit = _projected_metadata()
    validated_viewer_rows = _validate_viewer_prefix(rows)
    stop_offset, counts = first_complete_prefix(rows)
    range_bytes = sum(int(shard["range_bytes_transferred"]) for shard in shard_audit)
    range_requests = sum(int(shard["range_requests"]) for shard in shard_audit)
    source_bytes = sum(int(shard["repository_bytes"]) for shard in shard_audit)
    transport = {
        "method": "exact_revision_http_range_projected_parquet_columns",
        "columns": ["label", "model", "type", "release_date"],
        "image_column_requested": False,
        "prompt_column_requested": False,
        "shards": shard_audit,
        "source_parquet_bytes_not_downloaded": source_bytes,
        "range_bytes_transferred": range_bytes,
        "range_requests": range_requests,
        "viewer_prefix_rows_cross_validated": validated_viewer_rows,
    }
    if stop_offset is None:
        missing = {model: counts[model] for model in MODEL_KEYS if counts[model] < RESERVE_PER_MODEL}
        failure = {
            "schema_version": 1,
            "state": "e49_b_openfake_rejected_population_shortfall_unscored",
            "role": "REJECTED_SOURCE_QUALIFICATION_ONLY",
            "repository": REPO_ID,
            "revision": REVISION,
            "license": LICENSE,
            "config": CONFIG,
            "split": SPLIT,
            "expected_split_rows": EXPECTED_ROWS,
            "scanned_rows": len(rows),
            "eligible_counts": {model: counts[model] for model in MODEL_KEYS},
            "required_per_model": RESERVE_PER_MODEL,
            "missing_cells": missing,
            "metadata_transport": transport,
            "selected_rows": 0,
            "image_assets_requested": 0,
            "new_image_bytes_downloaded": 0,
            "model_scores_created": 0,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "boundary": (
                "Preregistered source population failed before identity selection. No replacement "
                "model, image access, detector access, metric or retry is permitted inside E49-B."
            ),
        }
        _write_atomic(EVIDENCE, failure)
        return failure

    selected = select_reserve(rows[:stop_offset])
    selected_counts = Counter(str(row["model"]) for row in selected)
    identity_sha256 = hashlib.sha256(
        "\n".join(sorted(str(row["record_id"]) for row in selected)).encode()
    ).hexdigest()
    payload = {
        "schema_version": 1,
        "state": "e49_b_openfake_frozen_untransferred_unscored",
        "role": "FINAL_AI_COMPONENT_CANDIDATE_PENDING_BYTE_FEASIBILITY_AND_COMPLETE_E49",
        "repository": REPO_ID,
        "revision": REVISION,
        "license": LICENSE,
        "gated": False,
        "config": CONFIG,
        "split": SPLIT,
        "expected_split_rows": EXPECTED_ROWS,
        "scan": {
            "page_size": PAGE_SIZE,
            "prefix_start": 0,
            "prefix_end_exclusive": stop_offset,
            "eligible_counts": {model: counts[model] for model in MODEL_KEYS},
            "stop_rule": "first_complete_100_row_page_with_192_eligible_rows_per_exact_model",
            "metadata_transport": transport,
            "geometry_state": "pending_revision_bound_asset_bind",
        },
        "selection": {
            "namespace": NAMESPACE,
            "target_per_model": TARGET_PER_MODEL,
            "reserve_per_model": RESERVE_PER_MODEL,
            "selected_counts": dict(sorted(selected_counts.items())),
            "reserve_identity_sha256": identity_sha256,
        },
        "rows": selected,
        "image_assets_requested": 0,
        "new_image_bytes_downloaded": 0,
        "model_scores_created": 0,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "boundary": "Projected metadata freeze only; image identity/geometry/byte binding remains mandatory.",
    }
    raw = _write_atomic(CONTRACT, payload)
    evidence = {
        "schema_version": 1,
        "state": payload["state"],
        "role": payload["role"],
        "repository": REPO_ID,
        "revision": REVISION,
        "license": LICENSE,
        "expected_split_rows": EXPECTED_ROWS,
        "prefix_end_exclusive": stop_offset,
        "eligible_counts": payload["scan"]["eligible_counts"],
        "metadata_range_bytes_transferred": range_bytes,
        "metadata_range_requests": range_requests,
        "viewer_prefix_rows_cross_validated": validated_viewer_rows,
        "selected_counts": payload["selection"]["selected_counts"],
        "reserve_identity_sha256": identity_sha256,
        "contract_bytes": len(raw),
        "contract_sha256": hashlib.sha256(raw).hexdigest(),
        "image_assets_requested": 0,
        "new_image_bytes_downloaded": 0,
        "model_scores_created": 0,
    }
    _write_atomic(EVIDENCE, evidence)
    return evidence


def bind_successor() -> dict[str, Any]:
    """Freeze E49-C identities from the pre-existing, cross-validated Viewer prefix."""
    if CONTRACT_C.exists() or EVIDENCE_C.exists():
        raise FileExistsError("E49-C OpenFake contract already exists")
    info = HfApi().dataset_info(REPO_ID, revision=REVISION)
    validate_repository(info)
    if not EVIDENCE.is_file():
        raise FileNotFoundError("E49-B rejection evidence is required before E49-C")
    e49b_raw = EVIDENCE.read_bytes()
    e49b = json.loads(e49b_raw)
    if (
        e49b.get("state") != "e49_b_openfake_rejected_population_shortfall_unscored"
        or e49b.get("missing_cells") != {"nano-banana-pro": 60}
        or e49b.get("selected_rows") != 0
        or e49b.get("new_image_bytes_downloaded") != 0
        or e49b.get("model_scores_created") != 0
    ):
        raise ValueError("E49-B rejection boundary changed")

    rows = _continuous_viewer_metadata()
    stop_offset, counts = first_complete_prefix(rows, model_keys=MODEL_KEYS_C)
    if stop_offset is None:
        raise ValueError(f"E49-C validated Viewer prefix cannot fill cells: {dict(counts)}")
    selected = select_reserve(
        rows[:stop_offset], model_keys=MODEL_KEYS_C, namespace=NAMESPACE_C,
    )
    selected_counts = Counter(str(row["model"]) for row in selected)
    identity_sha256 = hashlib.sha256(
        "\n".join(sorted(str(row["record_id"]) for row in selected)).encode()
    ).hexdigest()
    payload = {
        "schema_version": 1,
        "state": "e49_c_openfake_frozen_untransferred_unscored",
        "role": "FINAL_AI_COMPONENT_CANDIDATE_PENDING_ASSET_BYTE_FEASIBILITY_AND_COMPLETE_E49",
        "predecessor": {
            "experiment": "E49-B",
            "evidence_sha256": hashlib.sha256(e49b_raw).hexdigest(),
            "repair": "replace_only_underfilled_nano_banana_pro_with_z_image_turbo",
        },
        "repository": REPO_ID,
        "revision": REVISION,
        "license": LICENSE,
        "gated": False,
        "config": CONFIG,
        "split": SPLIT,
        "expected_split_rows": EXPECTED_ROWS,
        "viewer_endpoint": ROWS_ENDPOINT,
        "scan": {
            "page_size": PAGE_SIZE,
            "available_continuous_cached_rows": len(rows),
            "prefix_start": 0,
            "prefix_end_exclusive": stop_offset,
            "eligible_counts_at_stop": {model: counts[model] for model in MODEL_KEYS_C},
            "stop_rule": "first_complete_100_row_page_with_192_eligible_rows_per_exact_model",
            "stored_fields": ["row_index", "label", "model", "type", "release_date", "width", "height"],
            "new_metadata_bytes_transferred": 0,
        },
        "selection": {
            "namespace": NAMESPACE_C,
            "target_per_model": TARGET_PER_MODEL,
            "reserve_per_model": RESERVE_PER_MODEL,
            "selected_counts": dict(sorted(selected_counts.items())),
            "reserve_identity_sha256": identity_sha256,
        },
        "rows": selected,
        "image_assets_requested": 0,
        "new_image_bytes_downloaded": 0,
        "model_scores_created": 0,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "boundary": (
            "Identity-only successor freeze from previously validated metadata. No asset URL, "
            "image byte, detector access or metric; fresh asset feasibility is the next gate."
        ),
    }
    raw = _write_atomic(CONTRACT_C, payload)
    evidence = {
        "schema_version": 1,
        "state": payload["state"],
        "role": payload["role"],
        "predecessor": payload["predecessor"],
        "repository": REPO_ID,
        "revision": REVISION,
        "license": LICENSE,
        "available_continuous_cached_rows": len(rows),
        "prefix_end_exclusive": stop_offset,
        "eligible_counts_at_stop": payload["scan"]["eligible_counts_at_stop"],
        "selected_counts": payload["selection"]["selected_counts"],
        "reserve_identity_sha256": identity_sha256,
        "contract_bytes": len(raw),
        "contract_sha256": hashlib.sha256(raw).hexdigest(),
        "new_metadata_bytes_transferred": 0,
        "image_assets_requested": 0,
        "new_image_bytes_downloaded": 0,
        "model_scores_created": 0,
    }
    _write_atomic(EVIDENCE_C, evidence)
    return evidence


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("bind", "bind-projected", "bind-successor"))
    args = parser.parse_args(argv)
    if args.command == "bind":
        result = bind()
    elif args.command == "bind-projected":
        result = bind_projected()
    else:
        result = bind_successor()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
