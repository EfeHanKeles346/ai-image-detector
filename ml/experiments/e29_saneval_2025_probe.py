"""E29 — compact, pinned 2025-generator recall probe for CF-ViT.

The source dataset is much larger than the owner's 100 MB budget.  This command
therefore resolves a fixed, balanced set of Hugging Face dataset-server rows,
preflights their cached JPEG byte sizes, and refuses to download if the strict
100,000,000-byte image ceiling would be exceeded.  The cached JPEGs are a known
web recompression of SANEval's documented raw PNG outputs, not native files.

Run from ``ml/``::

    PYTHONPATH=src .venv/bin/python experiments/e29_saneval_2025_probe.py
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from huggingface_hub import snapshot_download
from huggingface_hub.utils import get_session
from PIL import Image

from pixelproof.image_input import decode_image
from pixelproof.project_paths import DATA_ROOT
from pixelproof.verdict import CF_T_AI

sys.path.insert(0, str(Path(__file__).parent))
from e21_external_detector_benchmark import (  # noqa: E402
    CommunityForensicsDetector,
    select_device,
)


DATASET_ID = "saneval-ann/saneval-sample"
DATASET_REVISION = "e9e188f6018b3d491708f29e7a387f5043dc8841"
DATASET_URL = f"https://huggingface.co/datasets/{DATASET_ID}"
ROWS_URL = "https://datasets-server.huggingface.co/rows"
DATASET_LICENSE = "MIT"
MAX_IMAGE_BYTES = 100_000_000
ROWS_PER_REQUEST = 100
TOTAL_SOURCE_ROWS = 600
ROWS_PER_GROUP = 2
TARGET_MODELS = (
    "GPT Image 1",
    "Imagen 4.0",
    "Imagen 4.0 Ultra",
    "Nano Banana",
    "Seedream 3.0",
)
TARGET_TYPES = ("color", "numeracy", "shape", "spatial", "texture")
TARGET_SPLITS = ("hard", "simple")
EXPECTED_COUNT = (
    len(TARGET_MODELS) * len(TARGET_TYPES) * len(TARGET_SPLITS) * ROWS_PER_GROUP
)


@dataclass(frozen=True)
class SourceRow:
    row_idx: int
    model: str
    split: str
    prompt_type: str
    image_url: str
    width: int
    height: int


def _request(method: str, url: str, *, request_timeout: float = 20, **kwargs: Any):
    """Retry transient dataset-server/cache failures without hiding final errors."""
    last_error: Exception | None = None
    for attempt in range(5):
        try:
            response = get_session().request(method, url, timeout=request_timeout, **kwargs)
            response.raise_for_status()
            return response
        except Exception as error:  # network status is reported after bounded retries
            last_error = error
            if attempt < 4:
                time.sleep(0.5 * (2**attempt))
    raise RuntimeError(f"{method} {url} failed after five attempts: {last_error}")


def _cached_payload(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        cached = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if cached.get("revision") != DATASET_REVISION:
        return None
    payload = cached.get("payload")
    if not isinstance(payload, dict) or len(payload.get("rows", [])) != ROWS_PER_REQUEST:
        return None
    # Dataset-server URLs are signed. Do not resume metadata too close to expiry.
    for item in payload["rows"]:
        url = str(item.get("row", {}).get("image", {}).get("src", ""))
        marker = "Expires="
        if marker not in url:
            return None
        try:
            expires = int(url.split(marker, 1)[1].split("&", 1)[0])
        except ValueError:
            return None
        if expires <= time.time() + 3600:
            return None
    return payload


def fetch_source_rows(cache_dir: Path | None = None) -> list[SourceRow]:
    rows: list[SourceRow] = []
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
    for offset in range(0, TOTAL_SOURCE_ROWS, ROWS_PER_REQUEST):
        cache_path = cache_dir / f"rows_{offset:03d}.json" if cache_dir else None
        payload = _cached_payload(cache_path) if cache_path else None
        if payload is None:
            print(f"fetching source rows {offset}–{offset + ROWS_PER_REQUEST - 1}", flush=True)
            response = _request(
                "GET",
                ROWS_URL,
                params={
                    "dataset": DATASET_ID,
                    "config": "default",
                    "split": "train",
                    "offset": offset,
                    "length": ROWS_PER_REQUEST,
                    "revision": DATASET_REVISION,
                },
            )
            revision = response.headers.get("x-revision")
            if revision != DATASET_REVISION:
                raise RuntimeError(
                    f"dataset revision changed: expected {DATASET_REVISION}, got {revision}"
                )
            payload = response.json()
            if len(payload.get("rows", [])) != ROWS_PER_REQUEST:
                raise RuntimeError(f"source chunk {offset} is incomplete")
            if cache_path is not None:
                temporary = cache_path.with_suffix(".json.part")
                temporary.write_text(
                    json.dumps({"revision": revision, "payload": payload}) + "\n"
                )
                temporary.replace(cache_path)
        else:
            print(f"resuming cached source rows {offset}–{offset + ROWS_PER_REQUEST - 1}", flush=True)
        for item in payload.get("rows", []):
            row = item.get("row", {})
            image = row.get("image", {})
            rows.append(
                SourceRow(
                    row_idx=int(item["row_idx"]),
                    model=str(row.get("model", "")),
                    split=str(row.get("split", "")),
                    prompt_type=str(row.get("type", "")),
                    image_url=str(image.get("src", "")),
                    width=int(image.get("width", 0)),
                    height=int(image.get("height", 0)),
                )
            )
    if len(rows) != TOTAL_SOURCE_ROWS:
        raise RuntimeError(f"expected {TOTAL_SOURCE_ROWS} source rows, received {len(rows)}")
    return rows


def select_rows(rows: list[SourceRow]) -> list[SourceRow]:
    """Take the two lowest row ids per frozen model/type/difficulty group."""
    grouped: dict[tuple[str, str, str], list[SourceRow]] = defaultdict(list)
    for row in rows:
        if (
            row.model in TARGET_MODELS
            and row.prompt_type in TARGET_TYPES
            and row.split in TARGET_SPLITS
        ):
            if not row.image_url or row.width <= 0 or row.height <= 0:
                raise RuntimeError(f"row {row.row_idx} has an invalid image cell")
            grouped[(row.model, row.prompt_type, row.split)].append(row)

    selected: list[SourceRow] = []
    for model in TARGET_MODELS:
        for prompt_type in TARGET_TYPES:
            for split in TARGET_SPLITS:
                key = (model, prompt_type, split)
                candidates = sorted(grouped.get(key, []), key=lambda row: row.row_idx)
                if len(candidates) < ROWS_PER_GROUP:
                    raise RuntimeError(f"source group {key!r} has only {len(candidates)} rows")
                selected.extend(candidates[:ROWS_PER_GROUP])

    selected.sort(key=lambda row: row.row_idx)
    if len(selected) != EXPECTED_COUNT or len({row.row_idx for row in selected}) != EXPECTED_COUNT:
        raise RuntimeError("frozen row selection did not produce 100 unique source rows")
    return selected


def validate_content_lengths(lengths: list[int], ceiling: int = MAX_IMAGE_BYTES) -> int:
    if len(lengths) != EXPECTED_COUNT:
        raise RuntimeError(f"expected {EXPECTED_COUNT} content lengths, got {len(lengths)}")
    if any(length <= 0 for length in lengths):
        raise RuntimeError("every selected image must declare a positive Content-Length")
    total = sum(lengths)
    if total > ceiling:
        raise RuntimeError(f"download would be {total:,} bytes, above {ceiling:,}-byte ceiling")
    return total


def _content_length(row: SourceRow) -> int:
    response = _request("HEAD", row.image_url, request_timeout=30)
    try:
        return int(response.headers["content-length"])
    except (KeyError, ValueError) as error:
        raise RuntimeError(f"row {row.row_idx} has no valid Content-Length") from error


def _slug(value: str) -> str:
    return "".join(character.lower() if character.isalnum() else "_" for character in value).strip("_")


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def download_subset(rows: list[SourceRow], output_dir: Path) -> dict[str, Any]:
    print(f"preflighting {len(rows)} cached JPEG assets", flush=True)
    with ThreadPoolExecutor(max_workers=8) as executor:
        lengths = list(executor.map(_content_length, rows))
    expected_total = validate_content_lengths(lengths)
    print(f"preflight: {expected_total:,} bytes (< {MAX_IMAGE_BYTES:,})", flush=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    total = 0
    seen_hashes: set[str] = set()
    for index, (row, expected_length) in enumerate(zip(rows, lengths), 1):
        filename = f"{row.row_idx:04d}_{_slug(row.model)}.jpg"
        path = output_dir / filename
        if path.is_file():
            raw = path.read_bytes()
        else:
            response = _request("GET", row.image_url, request_timeout=60)
            raw = response.content
        if len(raw) != expected_length:
            raise RuntimeError(
                f"row {row.row_idx} changed size: HEAD={expected_length}, GET={len(raw)}"
            )
        total += len(raw)
        if total > MAX_IMAGE_BYTES:
            raise RuntimeError("download exceeded the hard byte ceiling")
        digest = _sha256(raw)
        if digest in seen_hashes:
            raise RuntimeError(f"row {row.row_idx} duplicates an earlier selected image")
        seen_hashes.add(digest)
        with Image.open(io.BytesIO(raw)) as source:
            if source.format != "JPEG":
                raise RuntimeError(
                    f"row {row.row_idx} cache representation is {source.format}, expected JPEG"
                )
        picture = decode_image(raw)
        if picture.size != (row.width, row.height):
            raise RuntimeError(
                f"row {row.row_idx} geometry changed: API={(row.width, row.height)}, "
                f"decoded={picture.size}"
            )
        if not path.is_file():
            temporary = path.with_suffix(".jpg.part")
            temporary.write_bytes(raw)
            temporary.replace(path)
        records.append(
            {
                **asdict(row),
                "image_url": None,
                "path": filename,
                "bytes": len(raw),
                "sha256": digest,
                "decoded_format": "JPEG",
            }
        )
        if index % 10 == 0 or index == len(rows):
            print(f"downloaded {index}/{len(rows)} ({total:,} bytes)", flush=True)

    if total != expected_total:
        raise RuntimeError(f"download total changed: expected {expected_total}, got {total}")
    content_set_sha256 = _sha256(
        "\n".join(f"{record['row_idx']}:{record['sha256']}" for record in records).encode()
    )
    manifest = {
        "dataset": {
            "id": DATASET_ID,
            "url": DATASET_URL,
            "revision": DATASET_REVISION,
            "license": DATASET_LICENSE,
            "representation": "Hugging Face dataset-server cached JPEG; source card says raw PNG",
        },
        "selection": {
            "rule": "two lowest row ids per target model x type x split",
            "models": list(TARGET_MODELS),
            "types": list(TARGET_TYPES),
            "splits": list(TARGET_SPLITS),
            "count": len(records),
        },
        "download": {
            "image_bytes": total,
            "hard_ceiling_bytes": MAX_IMAGE_BYTES,
            "unique_sha256": len(seen_hashes),
            "content_set_sha256": content_set_sha256,
        },
        "records": records,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def _summary(scores: list[float]) -> dict[str, Any]:
    values = np.asarray(scores, dtype=np.float64)
    return {
        "n": int(len(values)),
        "triggered": int((values >= CF_T_AI).sum()),
        "recall": float((values >= CF_T_AI).mean()),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def score_subset(manifest: dict[str, Any], output_dir: Path, device_name: str) -> dict[str, Any]:
    device = select_device(device_name)
    local_model = Path(
        snapshot_download(
            "buildborderless/CommunityForensics-DeepfakeDet-ViT",
            local_files_only=True,
        )
    )
    detector = CommunityForensicsDetector(local_model, device, allow_download=False)
    scores: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for index, record in enumerate(manifest["records"], 1):
        path = output_dir / record["path"]
        try:
            score, width, height = detector.score(path)
            scores.append(
                {
                    "row_idx": record["row_idx"],
                    "model": record["model"],
                    "type": record["prompt_type"],
                    "split": record["split"],
                    "score": score,
                    "triggered": score >= CF_T_AI,
                    "width": width,
                    "height": height,
                }
            )
        except Exception as error:
            failures.append(
                {"row_idx": record["row_idx"], "error": f"{type(error).__name__}: {error}"}
            )
        if index % 10 == 0 or index == len(manifest["records"]):
            print(f"scored {index}/{len(manifest['records'])}; failures={len(failures)}", flush=True)
    if failures or len(scores) != EXPECTED_COUNT:
        raise RuntimeError(f"CF-ViT did not score all inputs: {failures}")

    grouped: dict[str, dict[str, list[float]]] = {
        "model": defaultdict(list),
        "type": defaultdict(list),
        "split": defaultdict(list),
    }
    for record in scores:
        grouped["model"][record["model"]].append(record["score"])
        grouped["type"][record["type"]].append(record["score"])
        grouped["split"][record["split"]].append(record["score"])
    results = {
        "experiment": "E29",
        "dataset_revision": DATASET_REVISION,
        "content_set_sha256": manifest["download"]["content_set_sha256"],
        "image_bytes": manifest["download"]["image_bytes"],
        "detector": {
            "id": detector.detector_id,
            "weights_sha256": detector.weights_sha256,
            "threshold": CF_T_AI,
            "device": str(device),
            "input_contract": detector.metadata["input_contract"],
        },
        "overall": _summary([record["score"] for record in scores]),
        "by_model": {key: _summary(value) for key, value in sorted(grouped["model"].items())},
        "by_type": {key: _summary(value) for key, value in sorted(grouped["type"].items())},
        "by_split": {key: _summary(value) for key, value in sorted(grouped["split"].items())},
        "failures": failures,
        "limitations": [
            "AI-only recall probe; false positives, specificity, accuracy and AUC are undefined",
            "dataset-server JPEG cache, not SANEval's native PNG representation",
            "100-row diagnostic subset; not the full SANEval benchmark",
        ],
        "scores": scores,
    }
    (output_dir / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DATA_ROOT / "e29_saneval_2025")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--download-only", action="store_true")
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    rows = select_rows(fetch_source_rows(output_dir / "_row_cache"))
    manifest = download_subset(rows, output_dir)
    print(
        f"verified subset: n={manifest['selection']['count']} "
        f"bytes={manifest['download']['image_bytes']:,} "
        f"sha={manifest['download']['content_set_sha256']}",
        flush=True,
    )
    if args.download_only:
        return
    results = score_subset(manifest, output_dir, args.device)
    print(json.dumps({"overall": results["overall"], "by_model": results["by_model"]}, indent=2))


if __name__ == "__main__":
    main()
