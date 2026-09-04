"""Freeze an ungated OpenFake reserve for E49-B without image or detector access."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Iterable, Mapping, Sequence

from huggingface_hub import HfApi
import requests

from pixelproof.project_paths import DATA_ROOT, ML_ROOT


NAMESPACE = "E49_B_OPENFAKE_V1"
REPO_ID = "ComplexDataLab/OpenFake"
REVISION = "3fd1109dc3258874243fa31c5bda9ee24260163b"
LICENSE = "cc-by-nc-4.0"
CONFIG = "core"
SPLIT = "test"
EXPECTED_ROWS = 91_398
PAGE_SIZE = 100
FETCH_WORKERS = 8
TARGET_PER_MODEL = 160
RESERVE_PER_MODEL = 192
MODEL_KEYS = {
    "gpt-image-2": "GPT Image 2",
    "nano-banana-pro": "Nano Banana Pro",
    "seedream-v5.0": "Seedream v5.0",
    "flux.2-klein-9b": "FLUX.2 Klein 9B",
    "midjourney-7": "Midjourney 7",
}
ROWS_ENDPOINT = "https://datasets-server.huggingface.co/rows"

ROOT = DATA_ROOT / "e49" / "openfake"
PAGE_CACHE = ROOT / "metadata_pages"
CONTRACT = ROOT / "source_contract_unscored.json"
EVIDENCE = ML_ROOT.parent / "evidence" / "e49_b_openfake_contract.json"


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def _write_atomic(path: Path, value: Any) -> bytes:
    raw = _json_bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_bytes(raw)
    temporary.replace(path)
    return raw


def _rank(model: str, row_index: int) -> str:
    identity = f"{REPO_ID}|{REVISION}|{CONFIG}|{SPLIT}|{model}|{row_index}"
    return hashlib.sha256(f"{NAMESPACE}|{identity}".encode()).hexdigest()


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


def eligible(row: Mapping[str, Any]) -> bool:
    row_type = str(row.get("type", "")).strip().lower()
    return (
        row.get("label") == "fake"
        and row.get("model") in MODEL_KEYS
        and bool(row_type)
        and row_type != "video"
    )


def select_reserve(
    rows: Sequence[Mapping[str, Any]], *, reserve_per_model: int = RESERVE_PER_MODEL,
) -> list[dict[str, Any]]:
    """Freeze exact rows by namespace hash after source-stratified metadata qualification."""
    output: list[dict[str, Any]] = []
    for model, display_name in MODEL_KEYS.items():
        candidates = []
        for source in rows:
            if source.get("model") != model or not eligible(source):
                continue
            row = dict(source)
            row_index = int(row["row_index"])
            row["record_id"] = f"openfake:{CONFIG}:{SPLIT}:{row_index}"
            row["parent_id"] = row["record_id"]
            row["rank"] = _rank(model, row_index)
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
    offsets = list(range(0, EXPECTED_ROWS, PAGE_SIZE))
    for batch_start in range(0, len(offsets), FETCH_WORKERS):
        batch = offsets[batch_start:batch_start + FETCH_WORKERS]
        with ThreadPoolExecutor(max_workers=FETCH_WORKERS) as pool:
            futures = {offset: pool.submit(_fetch_page_isolated, offset) for offset in batch}
            pages = {offset: futures[offset].result() for offset in batch}
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
        raise ValueError(f"E49-B OpenFake cannot fill preregistered cells: {missing}")

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


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("bind",))
    parser.parse_args(argv)
    print(json.dumps(bind(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
