"""Materialize the preregistered standardized E32 R0 model inputs."""

from __future__ import annotations

import hashlib
import io
import json
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable, Mapping

import pyarrow.parquet as pq
from PIL import Image, ImageOps

from pixelproof.project_paths import DATA_ROOT, ML_ROOT


REPO_ROOT = ML_ROOT.parent
E32_ROOT = DATA_ROOT / "e32"
MANIFEST_PATH = E32_ROOT / "c3_role_manifest.json"
MANIFEST_EVIDENCE = REPO_ROOT / "evidence" / "e32_c3_role_manifest.json"
OUTPUT_ROOT = E32_ROOT / "model_inputs" / "r0_global_jpeg90"
DETAILED_OUTPUT = E32_ROOT / "r0_input_receipt.json"
COMPACT_EVIDENCE = REPO_ROOT / "evidence" / "e32_r0_input_receipt.json"
INPUT_SIZE = 224
RESIZE_SHORT_SIDE = 256
JPEG_QUALITY = 90
WORKERS = 6
PARQUET_SOURCES = {
    "nano-banana-local": ("bitmind__nano-banana", "image"),
    "communityforensics-ai-local": ("OwensLab__CommunityForensics-Small", "image_data"),
}


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _write_atomic(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_bytes(raw)
    temporary.replace(path)


def raw_image(value: Any) -> bytes:
    if isinstance(value, (bytes, bytearray, memoryview)):
        return bytes(value)
    if isinstance(value, Mapping):
        for key in ("bytes", "data"):
            payload = value.get(key)
            if isinstance(payload, (bytes, bytearray, memoryview)):
                return bytes(payload)
    raise ValueError("Parquet image cell has no byte payload")


def standardized_jpeg(raw: bytes) -> bytes:
    with Image.open(io.BytesIO(raw)) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB")
        scale = RESIZE_SHORT_SIDE / min(image.size)
        resized = image.resize(
            (round(image.width * scale), round(image.height * scale)),
            Image.Resampling.LANCZOS,
        )
        left = (resized.width - INPUT_SIZE) // 2
        top = (resized.height - INPUT_SIZE) // 2
        cropped = resized.crop((left, top, left + INPUT_SIZE, top + INPUT_SIZE))
        output = io.BytesIO()
        cropped.save(
            output,
            format="JPEG",
            quality=JPEG_QUALITY,
            subsampling=0,
            optimize=False,
            progressive=False,
        )
    return output.getvalue()


def source_path(source_id: str, source_key: str) -> Path:
    if source_id in {"qwen-image-2512", "flux2-klein-9b"}:
        return E32_ROOT / "ai" / source_id / source_key
    if source_id == "gpt-image-1":
        acquired = E32_ROOT / "ai" / source_id / source_key
        local = DATA_ROOT / "a3xrfgb__gpt-image-mega-4k" / source_key
        return acquired if acquired.is_file() else local
    if source_id == "nano-banana-pro-ash-local":
        return DATA_ROOT / "ash12321__nano-banana-pro-generated-1k" / source_key
    if source_id == "vision-base-native":
        return E32_ROOT / "real" / "vision" / source_key
    if source_id in {"forchheim-fodb", "csafe-mcsidb-s21"}:
        return E32_ROOT / source_key
    raise KeyError(f"no loose-file resolver for {source_id}")


def _materialize(record: Mapping[str, Any], raw: bytes) -> dict[str, Any]:
    if _sha256(raw) != record["sha256"]:
        raise ValueError(f"source SHA changed for {record['record_id']}")
    transformed = standardized_jpeg(raw)
    relative = Path(str(record["record_id"])[:2]) / f"{record['record_id']}.jpg"
    destination = OUTPUT_ROOT / relative
    if destination.is_file():
        existing = destination.read_bytes()
        if existing != transformed:
            raise ValueError(f"existing standardized input changed for {record['record_id']}")
    else:
        _write_atomic(destination, transformed)
    return {
        "record_id": record["record_id"],
        "source_id": record["source_id"],
        "label": record["label"],
        "role": record["role"],
        "input_path": relative.as_posix(),
        "input_bytes": len(transformed),
        "input_sha256": _sha256(transformed),
    }


def _parquet_raw(records: list[Mapping[str, Any]], dirname: str, column: str) -> Iterable[tuple[Mapping[str, Any], bytes]]:
    grouped: dict[str, dict[int, Mapping[str, Any]]] = defaultdict(dict)
    for record in records:
        shard, row = str(record["source_key"]).rsplit(":", 1)
        grouped[shard][int(row)] = record
    for shard, wanted in sorted(grouped.items()):
        parquet = pq.ParquetFile(DATA_ROOT / dirname / shard)
        offset = 0
        found: set[int] = set()
        for batch in parquet.iter_batches(columns=[column], batch_size=128):
            for local_index, cell in enumerate(batch.column(0).to_pylist()):
                row_index = offset + local_index
                record = wanted.get(row_index)
                if record is not None:
                    found.add(row_index)
                    yield record, raw_image(cell)
            offset += batch.num_rows
        if found != set(wanted):
            raise ValueError(f"missing selected rows in {shard}: {sorted(set(wanted) - found)[:5]}")


def _load_manifest() -> tuple[dict[str, Any], bytes]:
    compact = json.loads(MANIFEST_EVIDENCE.read_text())
    raw = MANIFEST_PATH.read_bytes()
    if len(raw) != int(compact["detailed_report_bytes"]) or _sha256(raw) != compact["detailed_report_sha256"]:
        raise ValueError("C3 manifest binding changed")
    payload = json.loads(raw)
    if payload.get("state") != "train_calibration_manifest_frozen":
        raise ValueError("C3 manifest has unexpected state")
    if any(row["role"] not in {"TRAIN", "CALIBRATION"} for row in payload["records"]):
        raise ValueError("protected role found in R0 input manifest")
    return payload, raw


def realize_inputs() -> dict[str, Any]:
    manifest, manifest_raw = _load_manifest()
    records = manifest["records"]
    by_source: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        by_source[str(record["source_id"])].append(record)
    realized: list[dict[str, Any]] = []
    for source_id in sorted(by_source):
        source_records = by_source[source_id]
        if source_id in PARQUET_SOURCES:
            dirname, column = PARQUET_SOURCES[source_id]
            for index, (record, raw) in enumerate(_parquet_raw(source_records, dirname, column), 1):
                realized.append(_materialize(record, raw))
                if index % 250 == 0:
                    print(f"{source_id} {index}/{len(source_records)}", flush=True)
        else:
            def one(record: Mapping[str, Any]) -> dict[str, Any]:
                path = source_path(source_id, str(record["source_key"]))
                return _materialize(record, path.read_bytes())

            with ThreadPoolExecutor(max_workers=WORKERS) as pool:
                futures = {pool.submit(one, record): record for record in source_records}
                for index, future in enumerate(as_completed(futures), 1):
                    realized.append(future.result())
                    if index % 250 == 0:
                        print(f"{source_id} {index}/{len(source_records)}", flush=True)
    realized.sort(key=lambda row: row["record_id"])
    if len(realized) != len(records) or {row["record_id"] for row in realized} != {row["record_id"] for row in records}:
        raise ValueError("standardized input realization lost or added records")
    report = {
        "schema_version": 1,
        "experiment": "E32/C4-R0-standardized-input",
        "state": "r0_input_realization_complete",
        "manifest_sha256": _sha256(manifest_raw),
        "preprocessing": {
            "orientation": "PIL ImageOps.exif_transpose",
            "mode": "RGB",
            "resize_short_side": RESIZE_SHORT_SIDE,
            "crop": f"center-{INPUT_SIZE}",
            "encoding": "JPEG quality=90 subsampling=0 optimize=false progressive=false",
        },
        "record_count": len(realized),
        "class_counts": dict(sorted(Counter(row["label"] for row in realized).items())),
        "role_counts": dict(sorted(Counter(row["role"] for row in realized).items())),
        "source_counts": dict(sorted(Counter(row["source_id"] for row in realized).items())),
        "total_input_bytes": sum(int(row["input_bytes"]) for row in realized),
        "records_sha256": _sha256(_json_bytes(realized)),
        "records": realized,
        "boundary": "Only TRAIN and CALIBRATION parents are materialized; no DEVELOPMENT or LOCKED bytes are read.",
    }
    detailed_raw = _json_bytes(report)
    _write_atomic(DETAILED_OUTPUT, detailed_raw)
    compact = {key: value for key, value in report.items() if key != "records"}
    compact.update(
        {
            "detailed_report_external_path": DETAILED_OUTPUT.relative_to(E32_ROOT).as_posix(),
            "detailed_report_bytes": len(detailed_raw),
            "detailed_report_sha256": _sha256(detailed_raw),
        }
    )
    _write_atomic(COMPACT_EVIDENCE, _json_bytes(compact))
    return report


def main(argv: Iterable[str] | None = None) -> int:
    del argv
    report = realize_inputs()
    print(json.dumps({key: value for key, value in report.items() if key != "records"}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
