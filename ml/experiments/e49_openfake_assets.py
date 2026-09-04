"""Bind E49-C OpenFake asset geometry and bytes without downloading image bodies."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import threading
import time
from typing import Any, Iterable, Mapping, Sequence

import requests

from experiments.e49_openfake import (
    CONFIG,
    CONTRACT_C,
    EXPECTED_ROWS,
    MODEL_KEYS_C,
    PAGE_SIZE,
    REPO_ID,
    REVISION,
    ROWS_ENDPOINT,
    compact_page,
)
from pixelproof.project_paths import DATA_ROOT, ML_ROOT


IDENTITY_CONTRACT_SHA256 = "0abae56af862c9b402ef5ef594a21181cbbb7f72ba7495a491b0389bfdfcd702"
COMMONS_RESERVE_BYTES = 2_706_581_778
MAX_NETWORK_BYTES = 4 * 1024**3
MAX_ASSET_BYTES = 16 * 1024**2
EXPECTED_ROWS_SELECTED = 960
ALLOWED_HEAD_CONTENT_TYPES = {"image/jpeg", "binary/octet-stream", "application/octet-stream"}
PAGE_FETCH_WORKERS = 4
HEAD_WORKERS = 12
VIEWER_MIN_INTERVAL_SECONDS = 0.8
VIEWER_429_COOLDOWN_SECONDS = 60.0

_viewer_lock = threading.Lock()
_viewer_next_request = 0.0

ROOT = DATA_ROOT / "e49" / "openfake"
HEAD_CACHE = ROOT / "asset_heads"
ASSET_CONTRACT = ROOT / "asset_contract_untransferred_unscored.json"
EVIDENCE = ML_ROOT.parent / "evidence" / "e49_c_openfake_asset_contract.json"


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def _write_atomic(path: Path, value: Any) -> bytes:
    raw = _json_bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_bytes(raw)
    temporary.replace(path)
    return raw


def _selected_rows() -> list[dict[str, Any]]:
    raw = CONTRACT_C.read_bytes()
    if hashlib.sha256(raw).hexdigest() != IDENTITY_CONTRACT_SHA256:
        raise ValueError("E49-C identity contract changed")
    payload = json.loads(raw)
    rows = payload.get("rows") or []
    counts = Counter(str(row.get("model")) for row in rows)
    if (
        payload.get("state") != "e49_c_openfake_frozen_untransferred_unscored"
        or payload.get("revision") != REVISION
        or len(rows) != EXPECTED_ROWS_SELECTED
        or counts != Counter({model: 192 for model in MODEL_KEYS_C})
        or payload.get("image_assets_requested") != 0
        or payload.get("new_image_bytes_downloaded") != 0
        or payload.get("model_scores_created") != 0
    ):
        raise ValueError("E49-C identity boundary changed")
    return rows


def extract_asset_urls(
    payload: Mapping[str, Any], *, offset: int, wanted: Mapping[int, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Validate the page and return wanted signed URLs only to the in-memory caller."""
    compact = {int(row["row_index"]): row for row in compact_page(payload, offset=offset)}
    wrapped = {int(row["row_idx"]): row for row in payload.get("rows", [])}
    output = []
    for row_index, expected in sorted(wanted.items()):
        if row_index not in compact or row_index not in wrapped:
            raise ValueError(f"E49-C selected row missing from Viewer page: {row_index}")
        current = compact[row_index]
        for key in ("model", "release_date", "width", "height"):
            if current[key] != expected[key]:
                raise ValueError(f"E49-C selected metadata changed at {row_index}: {key}")
        row = wrapped[row_index].get("row") or {}
        src = str((row.get("image") or {}).get("src", ""))
        marker = f"/--/{REVISION}/--/{CONFIG}/test/{row_index}/image/"
        if marker not in src:
            raise ValueError(f"E49-C selected asset revision changed at {row_index}")
        output.append({"row": dict(expected), "url": src})
    return output


def _fetch_rows(offset: int, length: int, *, attempts: int) -> Mapping[str, Any]:
    global _viewer_next_request
    params = {"dataset": REPO_ID, "config": CONFIG, "split": "test",
              "offset": offset, "length": length}
    last_status = "unknown"
    for attempt in range(attempts):
        try:
            with _viewer_lock:
                delay = _viewer_next_request - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
                _viewer_next_request = time.monotonic() + VIEWER_MIN_INTERVAL_SECONDS
            response = requests.get(
                ROWS_ENDPOINT,
                params=params,
                headers={"User-Agent": "PixelProof-E49C/1.0 asset-head-audit"},
                timeout=(10, 30),
            )
            last_status = str(response.status_code)
            if response.status_code == 429:
                with _viewer_lock:
                    _viewer_next_request = max(
                        _viewer_next_request,
                        time.monotonic() + VIEWER_429_COOLDOWN_SECONDS,
                    )
            response.raise_for_status()
            return response.json()
        except (requests.RequestException, json.JSONDecodeError):
            if attempt == attempts - 1:
                break
            time.sleep(min(2 ** attempt, 45))
    raise RuntimeError(f"E49-C Viewer rows {offset}+{length} failed with HTTP {last_status}")


def _resolve_wanted_assets(
    offset: int, wanted: Mapping[int, Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], str]:
    """Fall back to exact selected rows when one unrelated row breaks a 100-row page."""
    try:
        payload = _fetch_rows(offset, PAGE_SIZE, attempts=2)
        return extract_asset_urls(payload, offset=offset, wanted=wanted), "page100"
    except RuntimeError:
        output = []
        for row_index, expected in sorted(wanted.items()):
            payload = _fetch_rows(row_index, 1, attempts=6)
            output.extend(extract_asset_urls(payload, offset=row_index, wanted={row_index: expected}))
        return output, "selected_row_fallback"


def _head_asset(item: Mapping[str, Any]) -> dict[str, Any]:
    row = item["row"]
    row_index = int(row["row_index"])
    cache = HEAD_CACHE / f"{row_index:06d}.json"
    if cache.is_file():
        cached = json.loads(cache.read_text())
        if (
            cached.get("state") != "e49_c_openfake_asset_head"
            or cached.get("revision") != REVISION
            or cached.get("record_id") != row["record_id"]
            or cached.get("asset_url_stored") is not False
        ):
            raise ValueError(f"E49-C asset HEAD cache changed: {row_index}")
        return cached

    last_status = "unknown"
    for attempt in range(6):
        try:
            response = requests.head(
                str(item["url"]),
                allow_redirects=True,
                headers={"User-Agent": "PixelProof-E49C/1.0 asset-head-audit"},
                timeout=(15, 60),
            )
            last_status = str(response.status_code)
            response.raise_for_status()
            content_length = int(response.headers.get("Content-Length", "-1"))
            content_type = str(response.headers.get("Content-Type", "")).split(";", 1)[0].lower()
            if content_type not in ALLOWED_HEAD_CONTENT_TYPES or not (0 < content_length <= MAX_ASSET_BYTES):
                raise ValueError(f"invalid JPEG HEAD: {content_type} / {content_length}")
            payload = {
                "schema_version": 1,
                "state": "e49_c_openfake_asset_head",
                "record_id": row["record_id"],
                "row_index": row_index,
                "model": row["model"],
                "revision": REVISION,
                "width": int(row["width"]),
                "height": int(row["height"]),
                "content_type": content_type,
                "content_length": content_length,
                "etag": str(response.headers.get("ETag", "")).strip('"'),
                "asset_url_stored": False,
                "image_body_bytes_downloaded": 0,
            }
            _write_atomic(cache, payload)
            return payload
        except (requests.RequestException, ValueError):
            if attempt == 5:
                break
            time.sleep(min(2 ** attempt, 30))
    raise RuntimeError(f"E49-C asset HEAD failed at row {row_index}, HTTP {last_status}")


def bind_assets() -> dict[str, Any]:
    if ASSET_CONTRACT.exists() or EVIDENCE.exists():
        raise FileExistsError("E49-C asset contract already exists")
    selected = _selected_rows()
    by_page: dict[int, dict[int, Mapping[str, Any]]] = defaultdict(dict)
    for row in selected:
        row_index = int(row["row_index"])
        by_page[(row_index // PAGE_SIZE) * PAGE_SIZE][row_index] = row

    heads: list[dict[str, Any]] = []
    pages = sorted(by_page.items())
    for batch_start in range(0, len(pages), PAGE_FETCH_WORKERS):
        batch = pages[batch_start:batch_start + PAGE_FETCH_WORKERS]
        need_page = {
            offset: wanted for offset, wanted in batch
            if any(not (HEAD_CACHE / f"{index:06d}.json").is_file() for index in wanted)
        }
        items = []
        with ThreadPoolExecutor(max_workers=PAGE_FETCH_WORKERS) as pool:
            futures = {
                offset: pool.submit(_resolve_wanted_assets, offset, wanted)
                for offset, wanted in need_page.items()
            }
            resolved = {offset: future.result() for offset, future in futures.items()}
        for offset, wanted in batch:
            if offset in resolved:
                current, _method = resolved[offset]
                items.extend(current)
            else:
                items.extend({"row": row, "url": ""} for row in wanted.values())
        with ThreadPoolExecutor(max_workers=HEAD_WORKERS) as pool:
            heads.extend(pool.map(_head_asset, items))
        completed_pages = batch_start + len(batch)
        if completed_pages % 24 == 0 or completed_pages == len(pages):
            print(
                f"E49-C asset HEAD pages {completed_pages}/{len(by_page)}, rows {len(heads)}/{len(selected)}",
                flush=True,
            )

    heads.sort(key=lambda row: (str(row["model"]), int(row["row_index"])))
    if len(heads) != EXPECTED_ROWS_SELECTED or len({row["record_id"] for row in heads}) != len(heads):
        raise ValueError("E49-C asset HEAD coverage changed")
    openfake_bytes = sum(int(row["content_length"]) for row in heads)
    global_bytes = COMMONS_RESERVE_BYTES + openfake_bytes
    if global_bytes > MAX_NETWORK_BYTES:
        raise ValueError(f"E49-C global byte ceiling failed: {global_bytes}/{MAX_NETWORK_BYTES}")
    counts = Counter(str(row["model"]) for row in heads)
    if counts != Counter({model: 192 for model in MODEL_KEYS_C}):
        raise ValueError("E49-C asset source quotas changed")

    payload = {
        "schema_version": 1,
        "state": "e49_c_openfake_assets_frozen_untransferred_unscored",
        "role": "FINAL_AI_COMPONENT_CANDIDATE_PENDING_TRANSFER_AND_COMPLETE_E49",
        "identity_contract_sha256": IDENTITY_CONTRACT_SHA256,
        "repository": REPO_ID,
        "revision": REVISION,
        "selected_rows": len(heads),
        "selected_counts": dict(sorted(counts.items())),
        "viewer_pages_requested": len(by_page),
        "asset_resolution_methods": ["page100", "selected_row_fallback_on_page_failure"],
        "asset_head_requests": len(heads),
        "asset_urls_stored": 0,
        "openfake_expected_bytes": openfake_bytes,
        "commons_expected_bytes": COMMONS_RESERVE_BYTES,
        "global_expected_network_bytes": global_bytes,
        "global_network_ceiling_bytes": MAX_NETWORK_BYTES,
        "global_headroom_bytes": MAX_NETWORK_BYTES - global_bytes,
        "rows": heads,
        "new_image_bytes_downloaded": 0,
        "model_scores_created": 0,
        "boundary": "HEAD metadata only; signed URLs discarded and no image body or detector accessed.",
    }
    raw = _write_atomic(ASSET_CONTRACT, payload)
    evidence = {key: payload[key] for key in (
        "schema_version", "state", "role", "identity_contract_sha256", "repository", "revision",
        "selected_rows", "selected_counts", "viewer_pages_requested", "asset_head_requests",
        "asset_resolution_methods", "asset_urls_stored", "openfake_expected_bytes", "commons_expected_bytes",
        "global_expected_network_bytes", "global_network_ceiling_bytes", "global_headroom_bytes",
        "new_image_bytes_downloaded", "model_scores_created",
    )}
    evidence.update({"contract_bytes": len(raw), "contract_sha256": hashlib.sha256(raw).hexdigest()})
    _write_atomic(EVIDENCE, evidence)
    return evidence


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("bind",))
    parser.parse_args(argv)
    print(json.dumps(bind_assets(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
