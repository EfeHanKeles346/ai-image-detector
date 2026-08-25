"""E31/B1 — deterministic, read-only audit of the attached dataset disk.

The source root is never inferred from a personal path and is never written to.  The command
records complete physical/Parquet metadata and a bounded, shard-spread image sample for decode,
geometry, compression and overlap diagnostics.  Full content hashing belongs to B2's frozen row
selection; this audit deliberately labels sampled hash coverage as such.

Run from ``ml/``::

    PYTHONPATH=src .venv/bin/python experiments/e31_ssd_audit.py \
      --root /Volumes/LaCie/pixelproof-datasets \
      --output ../evidence/e31_ssd_audit.json
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import pyarrow.parquet as pq
from PIL import Image, ImageOps

from pixelproof.data_contract import DataRecord, DataRole, dhash_image, shortcut_audit


SAMPLE_LIMIT = 300
SAMPLE_SHARD_LIMIT = 3
APPLEDOUBLE_PREFIX = "._"
IMAGE_COLUMNS = ("image", "image_data", "img", "picture")


@dataclass(frozen=True)
class SourceSpec:
    source_id: str
    dirname: str
    role: str
    label_col: str | None = None
    label_map: Mapping[int, str] | None = None
    expected_label_names: tuple[str, str] | None = None
    generator_col: str | None = None
    known_native_risk: str | None = None
    implicit_label: str | None = None


SOURCES = (
    SourceSpec(
        "communityforensics-small",
        "OwensLab__CommunityForensics-Small",
        "train_candidate",
        "label",
        {0: "real", 1: "ai"},
        None,
        "model_name",
        "real images are 1024 px while AI images are 512 px",
    ),
    SourceSpec(
        "ai-vs-real-balanced",
        "theminji__AI-vs-Real-balanced",
        "train_candidate",
        "label",
        {0: "ai", 1: "real"},
        ("AiArtData", "RealArt"),
    ),
    SourceSpec(
        "aigc-detection-benchmark",
        "TheKernel01__AIGC-Detection-Benchmark",
        "conditional_train_candidate",
        "label",
        {0: "real", 1: "ai"},
        ("real", "fake"),
        "generator",
        "native geometry is label-separable",
    ),
    SourceSpec(
        "ai-vs-real-200k",
        "theminji__ai-vs-real-200k",
        "conditional_train_candidate",
        "label",
        {0: "ai", 1: "real"},
        ("ai", "real"),
        None,
        "native resolution is label-separable",
    ),
    SourceSpec(
        "julienlucas-modern",
        "julienlucas__midjourney-dalle-sd-nanobananapro-dataset",
        "test_only",
        "label",
        {0: "ai", 1: "real"},
        ("fake", "real"),
    ),
    SourceSpec(
        "flux-1-dev", "ash12321__flux-1-dev-generated-10k", "ai_only_candidate",
        implicit_label="ai",
    ),
    SourceSpec(
        "nano-banana", "bitmind__nano-banana", "ai_only_candidate", implicit_label="ai",
    ),
    SourceSpec(
        "nano-banana-pro", "kaupane__nano-banana-pro-gen", "ai_only_candidate",
        implicit_label="ai",
    ),
    SourceSpec(
        "communityforensics-fake", "34data__communityforensics-fake", "test_only",
        implicit_label="ai",
    ),
    SourceSpec(
        "communityforensics-real", "34data__communityforensics-real", "test_only",
        implicit_label="real",
    ),
)


def real_files(root: Path, pattern: str = "*") -> list[Path]:
    """Return stable real files, excluding exFAT AppleDouble and cache artifacts."""
    return sorted(
        path
        for path in root.rglob(pattern)
        if path.is_file()
        and not path.name.startswith(APPLEDOUBLE_PREFIX)
        and ".cache" not in path.parts
    )


def _hf_features(parquet: pq.ParquetFile) -> Mapping[str, Any]:
    raw = (parquet.schema_arrow.metadata or {}).get(b"huggingface")
    if not raw:
        return {}
    try:
        return json.loads(raw)["info"]["features"]
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return {}


def _image_bytes(value: Any) -> bytes | None:
    if isinstance(value, (bytes, bytearray)):
        return bytes(value)
    if isinstance(value, Mapping):
        raw = value.get("bytes")
        if isinstance(raw, (bytes, bytearray)):
            return bytes(raw)
    return None


def _project_label(spec: SourceSpec, raw: Any) -> str | None:
    if spec.label_map is None:
        return spec.implicit_label
    try:
        key = int(raw)
    except (TypeError, ValueError):
        return None
    return spec.label_map.get(key)


def _validate_label_metadata(spec: SourceSpec, parquet: pq.ParquetFile) -> str:
    if spec.label_col is None:
        return f"implicit_{spec.implicit_label}" if spec.implicit_label else "not_declared"
    if spec.expected_label_names is None:
        return "asserted_from_source_metadata"
    found = _hf_features(parquet).get(spec.label_col, {}).get("names")
    if list(spec.expected_label_names) != found:
        raise ValueError(
            f"{spec.source_id}: label order changed; expected "
            f"{list(spec.expected_label_names)!r}, found {found!r}"
        )
    return "verified_huggingface_classlabel"


def _spread_files(
    files: list[Path], limit: int, shard_limit: int = SAMPLE_SHARD_LIMIT
) -> list[tuple[Path, int]]:
    """Allocate a bounded sample over a bounded spread of the lexical shard range.

    Reading one embedded-image row can decompress a complete Parquet row group (4.09 GB in one
    observed CommunityForensics shard). Bounding the selection to first/middle/last prevents a
    nominal 300-row probe from reading an entire 49 GB collection.
    """
    if not files or limit <= 0:
        return []
    selected_count = min(len(files), limit, shard_limit)
    if selected_count == 1:
        selected = [files[0]]
    else:
        indices = sorted(
            {
                round(index * (len(files) - 1) / (selected_count - 1))
                for index in range(selected_count)
            }
        )
        selected = [files[index] for index in indices]
    per_file = max(1, math.ceil(limit / len(selected)))
    return [(path, per_file) for path in selected]


def _record(
    spec: SourceSpec,
    *,
    index: int,
    label: str,
    raw: bytes,
    image: Image.Image,
    standardized: bool,
) -> DataRecord:
    if standardized:
        normalized = ImageOps.fit(image.convert("RGB"), (128, 128), method=Image.Resampling.LANCZOS)
        buffer = io.BytesIO()
        normalized.save(buffer, format="JPEG", quality=90, optimize=False, progressive=False)
        payload = buffer.getvalue()
        width, height, image_format = 128, 128, "JPEG"
    else:
        payload = raw
        width, height = image.size
        image_format = image.format or "UNKNOWN"
    digest = hashlib.sha256(raw).hexdigest()
    return DataRecord(
        record_id=f"{spec.source_id}-{index:06d}-{'std' if standardized else 'native'}",
        role=DataRole.TRAIN,
        source_id=spec.source_id,
        # The contract accepts pinned commit-like revisions. This deterministic digest pins the
        # local audit namespace; B2 will replace it with the selected manifest SHA.
        source_revision=hashlib.sha256(f"local-audit:{spec.source_id}".encode()).hexdigest()[:40],
        source_key=f"sample-{index:06d}",
        label=label,
        group=spec.source_id,
        transport="fixed_128_rgb_jpeg_probe" if standardized else "native_source",
        path=f"sample/{index:06d}",
        generator=spec.source_id if label == "ai" else None,
        camera_pipeline=spec.source_id if label == "real" else None,
        content_id=digest,
        parent_id=None,
        sha256=hashlib.sha256(payload).hexdigest(),
        dhash=dhash_image(image),
        bytes=len(payload),
        width=width,
        height=height,
        image_format=image_format,
    )


def _known_e30_hashes(repo_root: Path) -> set[str]:
    hashes: set[str] = set()
    e30_root = repo_root / "ml" / "data" / "e30"
    if not e30_root.exists():
        return hashes
    for path in sorted(e30_root.rglob("*manifest.json")):
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        for row in payload.get("records", []):
            digest = row.get("sha256")
            if isinstance(digest, str) and len(digest) == 64:
                hashes.add(digest)
    return hashes


def audit_source(
    root: Path,
    spec: SourceSpec,
    *,
    sample_limit: int,
    protected_hashes: set[str],
) -> dict[str, Any]:
    folder = root / spec.dirname
    if not folder.is_dir():
        return {"source_id": spec.source_id, "dirname": spec.dirname, "state": "missing"}

    files = real_files(folder)
    parquets = [path for path in files if path.suffix.lower() == ".parquet"]
    result: dict[str, Any] = {
        "source_id": spec.source_id,
        "dirname": spec.dirname,
        "role": spec.role,
        "state": "present",
        "physical_file_count": len(files),
        "physical_bytes": sum(path.stat().st_size for path in files),
        "parquet_file_count": len(parquets),
        "known_native_risk": spec.known_native_risk,
    }
    if not parquets:
        result.update(
            {
                "parquet_rows": 0,
                "sample_state": "not_parquet_sampled",
                "sample_hash_scope": "none",
            }
        )
        return result

    raw_labels: Counter[str] = Counter()
    project_labels: Counter[str] = Counter()
    generators: Counter[str] = Counter()
    schemas: Counter[tuple[str, ...]] = Counter()
    parquet_rows = 0
    metadata_errors: list[str] = []
    label_status = None
    image_col = None
    for path in parquets:
        parquet = pq.ParquetFile(path)
        names = parquet.schema_arrow.names
        schemas[tuple(names)] += 1
        parquet_rows += parquet.metadata.num_rows
        if image_col is None:
            image_col = next((name for name in IMAGE_COLUMNS if name in names), None)
        if label_status is None:
            label_status = _validate_label_metadata(spec, parquet)
        columns = [name for name in (spec.label_col, spec.generator_col) if name and name in names]
        if spec.label_col and spec.label_col not in names:
            metadata_errors.append(f"missing label column in {path.name}")
        if columns:
            for batch in parquet.iter_batches(columns=columns, batch_size=65_536):
                values = batch.to_pydict()
                if spec.label_col in values:
                    for raw in values[spec.label_col]:
                        raw_labels[str(raw)] += 1
                        label = _project_label(spec, raw)
                        project_labels[label or "unmapped"] += 1
                if spec.generator_col in values:
                    raw_values = values.get(spec.label_col, [None] * len(values[spec.generator_col]))
                    generators.update(
                        str(generator)
                        for raw, generator in zip(raw_values, values[spec.generator_col], strict=True)
                        if _project_label(spec, raw) == "ai"
                    )
    if spec.label_col is None and spec.implicit_label:
        project_labels[spec.implicit_label] = parquet_rows

    native_records: list[DataRecord] = []
    standardized_records: list[DataRecord] = []
    decode_failures = 0
    sample_index = 0
    for path, per_file in _spread_files(parquets, sample_limit):
        parquet = pq.ParquetFile(path)
        names = parquet.schema_arrow.names
        current_image_col = next((name for name in IMAGE_COLUMNS if name in names), None)
        if current_image_col is None:
            continue
        columns = [current_image_col]
        if spec.label_col and spec.label_col in names:
            columns.append(spec.label_col)
        taken = 0
        for batch in parquet.iter_batches(columns=columns, batch_size=max(64, per_file)):
            for row in batch.to_pylist():
                raw = _image_bytes(row[current_image_col])
                label = _project_label(spec, row.get(spec.label_col) if spec.label_col else None)
                if raw is None or label not in {"real", "ai"}:
                    decode_failures += 1
                    continue
                try:
                    with Image.open(io.BytesIO(raw)) as opened:
                        opened.load()
                        image = opened.copy()
                        image.format = opened.format
                except Exception:
                    decode_failures += 1
                    continue
                native_records.append(
                    _record(spec, index=sample_index, label=label, raw=raw, image=image, standardized=False)
                )
                standardized_records.append(
                    _record(spec, index=sample_index, label=label, raw=raw, image=image, standardized=True)
                )
                sample_index += 1
                taken += 1
                if taken >= per_file or sample_index >= sample_limit:
                    break
            if taken >= per_file or sample_index >= sample_limit:
                break
        if sample_index >= sample_limit:
            break

    native_hashes = {record.sha256 for record in native_records if record.sha256}
    overlap = sorted(native_hashes & protected_hashes)
    result.update(
        {
            "parquet_rows": parquet_rows,
            "schemas": [
                {"columns": list(columns), "file_count": count}
                for columns, count in sorted(schemas.items())
            ],
            "image_column": image_col,
            "label_direction_status": label_status,
            "raw_label_counts": dict(sorted(raw_labels.items())),
            "project_label_counts": dict(sorted(project_labels.items())),
            "ai_generator_unique": len(generators),
            "ai_generator_top_counts": dict(generators.most_common(20)),
            "metadata_errors": sorted(set(metadata_errors)),
            "sample_count": len(native_records),
            "sample_decode_failures": decode_failures,
            "sample_hash_scope": "bounded_shard_spread_not_full_dataset",
            "sample_exact_overlap_with_e30": len(overlap),
        }
    )
    labels_present = {record.label for record in native_records}
    if labels_present == {"real", "ai"}:
        result["native_shortcut_audit"] = shortcut_audit(native_records)
        result["fixed_128_rgb_jpeg_probe"] = shortcut_audit(standardized_records)
    else:
        result["native_shortcut_audit"] = {
            "state": "not_applicable_single_class",
            "labels": sorted(labels_present),
        }
        result["fixed_128_rgb_jpeg_probe"] = {
            "state": "not_applicable_single_class",
            "labels": sorted(labels_present),
        }
    return result


def inventory_unregistered(root: Path, registered: set[str]) -> list[dict[str, Any]]:
    output = []
    for folder in sorted(path for path in root.iterdir() if path.is_dir()):
        if folder.name in registered or folder.name.startswith((APPLEDOUBLE_PREFIX, ".")):
            continue
        files = real_files(folder)
        output.append(
            {
                "dirname": folder.name,
                "physical_file_count": len(files),
                "physical_bytes": sum(path.stat().st_size for path in files),
                "parquet_file_count": sum(path.suffix.lower() == ".parquet" for path in files),
                "state": "inventory_only_unregistered",
            }
        )
    return output


def build_report(root: Path, repo_root: Path, sample_limit: int = SAMPLE_LIMIT) -> dict[str, Any]:
    if not root.is_dir():
        raise FileNotFoundError(f"dataset root is unavailable: {root}")
    if sample_limit < 10:
        raise ValueError("sample limit must be at least 10")
    protected_hashes = _known_e30_hashes(repo_root)
    sources = [
        audit_source(
            root,
            spec,
            sample_limit=sample_limit,
            protected_hashes=protected_hashes,
        )
        for spec in SOURCES
    ]
    return {
        "schema_version": 1,
        "experiment": "E31/B1",
        "source_root_name": root.name,
        "audit_mode": "complete_physical_and_parquet_metadata_plus_bounded_image_sample",
        "sample_limit_per_registered_source": sample_limit,
        "sample_shard_limit_per_registered_source": SAMPLE_SHARD_LIMIT,
        "protected_e30_hash_count": len(protected_hashes),
        "registered_sources": sources,
        "unregistered_inventory": inventory_unregistered(root, {spec.dirname for spec in SOURCES}),
        "boundaries": [
            "The source root was read only; no source file was created, changed or removed.",
            "Sample hashes do not prove full-dataset decontamination; B2 must hash every selected TRAIN v2 row.",
            "A fixed-size metadata probe neutralizes geometry/format metadata but does not prove pixel-level shortcut removal.",
            "Test-only and E30 rows remain forbidden for training, calibration and ensemble fitting.",
        ],
    }


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def write_report(report: Mapping[str, Any], output: Path, source_root: Path) -> None:
    if _is_within(output, source_root):
        raise ValueError("refusing to write the audit report inside the source dataset root")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".part")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    temporary.replace(output)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="read-only dataset root")
    parser.add_argument("--output", type=Path, help="compact JSON output outside the source root")
    parser.add_argument("--sample-limit", type=int, default=SAMPLE_LIMIT)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    repo_root = Path(__file__).resolve().parents[2]
    report = build_report(args.root, repo_root, args.sample_limit)
    if args.output:
        write_report(report, args.output, args.root)
        print(f"wrote {args.output}")
    else:
        print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
