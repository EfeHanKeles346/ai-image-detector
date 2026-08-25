"""Role-safe manifests and mechanical shortcut checks for E30 datasets.

The data role is part of the scientific contract, not a folder convention.  A
record cannot become training data merely because a caller points a loader at
its path, and a derived encode cannot cross the role of its underlying image.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from PIL import Image
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict


class DataRole(StrEnum):
    TRAIN = "train"
    CALIBRATION = "calibration"
    DEVELOPMENT_TEST = "development_test"
    LOCKED_FINAL_TEST = "locked_final_test"
    FUTURE_TEST = "future_test"


@dataclass(frozen=True)
class DataRecord:
    record_id: str
    role: DataRole
    source_id: str
    source_revision: str
    source_key: str
    label: str
    group: str
    transport: str
    path: str
    generator: str | None = None
    camera_pipeline: str | None = None
    content_id: str | None = None
    parent_id: str | None = None
    sha256: str | None = None
    dhash: str | None = None
    bytes: int | None = None
    width: int | None = None
    height: int | None = None
    image_format: str | None = None

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "DataRecord":
        fields = dict(value)
        fields["role"] = DataRole(fields["role"])
        return cls(**fields)


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def dhash_image(image: Image.Image, size: int = 8) -> str:
    gray = image.convert("L").resize((size + 1, size))
    values = np.asarray(gray, dtype=np.int16)
    bits = values[:, 1:] > values[:, :-1]
    number = 0
    for bit in bits.ravel():
        number = (number << 1) | int(bit)
    return f"{number:0{size * size // 4}x}"


def _valid_relative_path(value: str) -> bool:
    path = PurePosixPath(value)
    return bool(value) and not path.is_absolute() and ".." not in path.parts


def _valid_revision(value: str) -> bool:
    return len(value) == 40 and all(character in "0123456789abcdef" for character in value)


def validate_records(
    raw_records: Sequence[DataRecord | Mapping[str, Any]],
    *,
    required_role: DataRole | None = None,
    require_hashes: bool = False,
) -> list[DataRecord]:
    """Validate role, provenance, parentage and exact/underlying leakage."""
    records = [
        record if isinstance(record, DataRecord) else DataRecord.from_mapping(record)
        for record in raw_records
    ]
    if not records:
        raise ValueError("a data manifest must contain at least one record")

    by_id: dict[str, DataRecord] = {}
    source_revisions: dict[str, str] = {}
    for record in records:
        if record.record_id in by_id:
            raise ValueError(f"duplicate record id {record.record_id!r}")
        by_id[record.record_id] = record
        if required_role is not None and record.role != required_role:
            raise PermissionError(
                f"requested {required_role.value} data but {record.record_id!r} is {record.role.value}"
            )
        if record.label not in {"real", "ai"}:
            raise ValueError(f"{record.record_id!r} has ambiguous label {record.label!r}")
        if not record.source_id or not record.source_key or not record.group or not record.transport:
            raise ValueError(f"{record.record_id!r} lacks required provenance/group fields")
        if not _valid_revision(record.source_revision):
            raise ValueError(f"{record.record_id!r} has an unpinned source revision")
        previous = source_revisions.setdefault(record.source_id, record.source_revision)
        if previous != record.source_revision:
            raise ValueError(f"source {record.source_id!r} appears at multiple revisions")
        if not _valid_relative_path(record.path):
            raise ValueError(f"{record.record_id!r} has unsafe path {record.path!r}")
        if record.label == "ai" and not record.generator:
            raise ValueError(f"AI record {record.record_id!r} must name its generator")
        if record.label == "real" and not record.camera_pipeline:
            raise ValueError(f"real record {record.record_id!r} must name its camera/pipeline")
        if require_hashes:
            if record.sha256 is None or len(record.sha256) != 64 or any(
                character not in "0123456789abcdef" for character in record.sha256
            ):
                raise ValueError(f"{record.record_id!r} lacks a valid SHA-256")
            if record.bytes is None or record.bytes <= 0:
                raise ValueError(f"{record.record_id!r} lacks a positive byte count")

    for record in records:
        if record.parent_id is None:
            continue
        parent = by_id.get(record.parent_id)
        if parent is None:
            raise ValueError(f"{record.record_id!r} references missing parent {record.parent_id!r}")
        if (record.role, record.label, record.content_id) != (
            parent.role,
            parent.label,
            parent.content_id,
        ):
            raise ValueError(f"derived record {record.record_id!r} crossed its parent role/label/content")

    exact_roles: dict[str, set[DataRole]] = defaultdict(set)
    content_roles: dict[str, set[DataRole]] = defaultdict(set)
    for record in records:
        if record.sha256:
            exact_roles[record.sha256].add(record.role)
        if record.content_id:
            content_roles[record.content_id].add(record.role)
    for digest, roles in exact_roles.items():
        if len(roles) > 1:
            raise ValueError(f"exact image {digest[:12]} leaks across roles: {sorted(roles)}")
    for content_id, roles in content_roles.items():
        if len(roles) > 1:
            raise ValueError(f"underlying content {content_id!r} leaks across roles: {sorted(roles)}")
    return records


def load_manifest(
    path: Path,
    *,
    required_role: DataRole | None = None,
    require_hashes: bool = False,
) -> tuple[dict[str, Any], list[DataRecord]]:
    payload = json.loads(path.read_text())
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported data manifest schema")
    return payload, validate_records(
        payload.get("records", []), required_role=required_role, require_hashes=require_hashes
    )


def select_stratified(
    records: Sequence[Mapping[str, Any]],
    *,
    group_fields: Sequence[str],
    per_group: int,
    key_field: str = "source_key",
) -> list[Mapping[str, Any]]:
    if per_group <= 0 or not group_fields:
        raise ValueError("selection needs positive per_group and at least one group field")
    groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        groups[tuple(record[field] for field in group_fields)].append(record)
    selected: list[Mapping[str, Any]] = []
    for group, candidates in sorted(groups.items(), key=lambda item: tuple(map(str, item[0]))):
        ordered = sorted(candidates, key=lambda record: str(record[key_field]))
        if len(ordered) < per_group:
            raise ValueError(f"group {group!r} has {len(ordered)} rows; needs {per_group}")
        selected.extend(ordered[:per_group])
    return selected


def enforce_byte_ceiling(
    sizes: Iterable[int], *, total_ceiling: int, per_file_ceiling: int | None = None
) -> int:
    values = list(sizes)
    if not values or any(size <= 0 for size in values):
        raise ValueError("every selected asset must have a positive declared size")
    if per_file_ceiling is not None and any(size > per_file_ceiling for size in values):
        raise ValueError(f"an asset exceeds the {per_file_ceiling:,}-byte per-file ceiling")
    total = sum(values)
    if total > total_ceiling:
        raise ValueError(f"assets total {total:,} bytes, above {total_ceiling:,}-byte ceiling")
    return total


def content_set_sha256(records: Sequence[DataRecord]) -> str:
    if any(record.sha256 is None for record in records):
        raise ValueError("content-set hash requires every record SHA-256")
    material = "\n".join(
        f"{record.record_id}:{record.sha256}" for record in sorted(records, key=lambda item: item.record_id)
    )
    return sha256_bytes(material.encode())


def shortcut_audit(records: Sequence[DataRecord]) -> dict[str, Any]:
    """Measure metadata-only class separation on one transport regime."""
    validate_records(records)
    complete = [
        record
        for record in records
        if record.width and record.height and record.bytes and record.image_format
    ]
    if len(complete) != len(records):
        raise ValueError("shortcut audit needs width, height, bytes and image_format for every row")
    labels = np.asarray([int(record.label == "ai") for record in complete], dtype=np.int64)
    counts = np.bincount(labels, minlength=2)
    if min(counts) < 5:
        raise ValueError("shortcut probe needs at least five complete rows per class")
    formats = sorted({str(record.image_format).upper() for record in complete})
    features = []
    for record in complete:
        width, height = int(record.width), int(record.height)
        row = [
            math.log(width),
            math.log(height),
            math.log(width / height),
            float(width == height),
            math.log(int(record.bytes) / (width * height)),
        ]
        row.extend(float(str(record.image_format).upper() == fmt) for fmt in formats)
        features.append(row)
    folds = int(min(5, counts.min()))
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=2026)
    model = HistGradientBoostingClassifier(
        max_iter=80,
        max_depth=3,
        min_samples_leaf=2,
        random_state=2026,
    )
    probabilities = cross_val_predict(
        model, np.asarray(features), labels, cv=splitter, method="predict_proba"
    )[:, 1]
    auc = float(roc_auc_score(labels, probabilities))

    stats: dict[str, Any] = {}
    for label in ("real", "ai"):
        group = [record for record in complete if record.label == label]
        sides = np.asarray([max(int(record.width), int(record.height)) for record in group])
        bpp = np.asarray(
            [int(record.bytes) / (int(record.width) * int(record.height)) for record in group]
        )
        stats[label] = {
            "n": len(group),
            "formats": dict(Counter(str(record.image_format).upper() for record in group)),
            "square_fraction": float(np.mean([record.width == record.height for record in group])),
            "median_long_side": float(np.median(sides)),
            "p10_long_side": float(np.percentile(sides, 10)),
            "p90_long_side": float(np.percentile(sides, 90)),
            "mean_bytes_per_pixel": float(np.mean(bpp)),
        }
    issues = []
    if auc >= 0.65:
        issues.append(f"metadata_probe_auc={auc:.3f} exceeds 0.65")
    if set(stats["real"]["formats"]) != set(stats["ai"]["formats"]):
        issues.append("real and AI format sets differ")
    if abs(stats["real"]["square_fraction"] - stats["ai"]["square_fraction"]) > 0.60:
        issues.append("real and AI square fractions differ by more than 0.60")
    bpp_values = [stats[label]["mean_bytes_per_pixel"] for label in ("real", "ai")]
    if max(bpp_values) / max(min(bpp_values), 1e-12) > 3:
        issues.append("real and AI mean bytes-per-pixel differ by more than 3x")
    return {"metadata_probe_auc": auc, "groups": stats, "issues": issues, "pass": not issues}


def final_run_receipt(
    path: Path,
    *,
    records: Sequence[DataRecord],
    candidate_id: str,
    candidate_sha256: str,
    threshold_contract_sha256: str,
    results_sha256: str,
) -> dict[str, Any]:
    """Write one immutable receipt; an existing receipt can never be overwritten."""
    validate_records(records, required_role=DataRole.LOCKED_FINAL_TEST, require_hashes=True)
    for name, digest in {
        "candidate": candidate_sha256,
        "threshold contract": threshold_contract_sha256,
        "results": results_sha256,
    }.items():
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise ValueError(f"{name} SHA-256 is invalid")
    if path.exists():
        raise FileExistsError(f"final-run receipt already exists: {path}")
    receipt = {
        "schema_version": 1,
        "role": DataRole.LOCKED_FINAL_TEST.value,
        "candidate_id": candidate_id,
        "candidate_sha256": candidate_sha256,
        "threshold_contract_sha256": threshold_contract_sha256,
        "content_set_sha256": content_set_sha256(records),
        "results_sha256": results_sha256,
        "record_count": len(records),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(receipt, indent=2) + "\n")
    temporary.replace(path)
    return receipt


def record_to_dict(record: DataRecord) -> dict[str, Any]:
    value = asdict(record)
    value["role"] = record.role.value
    return value
