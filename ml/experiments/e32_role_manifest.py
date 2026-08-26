"""Freeze the balanced, group-aware E32 TRAIN/CALIBRATION parent manifest.

This command consumes only committed JSON metadata and never opens image bytes.  It is bound to
the final role-free eligibility overlay and its source audits.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from pixelproof.project_paths import DATA_ROOT, ML_ROOT


REPO_ROOT = ML_ROOT.parent
OUTPUT_ROOT = DATA_ROOT / "e32"
AUDIT_ROOT = OUTPUT_ROOT / "audits"
OVERLAY_PATH = OUTPUT_ROOT / "eligibility_overlay.json"
OVERLAY_EVIDENCE = REPO_ROOT / "evidence" / "e32_eligibility_overlay.json"
DETAILED_OUTPUT = OUTPUT_ROOT / "c3_role_manifest.json"
COMPACT_EVIDENCE = REPO_ROOT / "evidence" / "e32_c3_role_manifest.json"
SEED = 20260826
CALIBRATION_FRACTION = 0.20
AI_TARGETS = {
    "qwen-image-2512": 2232,
    "flux2-klein-9b": 2232,
    "nano-banana-local": 2227,
    "gpt-image-1": 2227,
    "nano-banana-pro-ash-local": 200,
    "communityforensics-ai-local": 2226,
}
REAL_SOURCES = ("vision-base-native", "forchheim-fodb", "csafe-mcsidb-s21")
FOUR_OUTPUT_SOURCES = ("qwen-image-2512", "flux2-klein-9b")


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _stable(value: str) -> tuple[str, str]:
    return _sha256(f"{SEED}:{value}".encode()), value


def _write_atomic(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_bytes(raw)
    temporary.replace(path)


def select_records(
    records: Sequence[Mapping[str, Any]], target: int, *, group_field: str | None = None
) -> list[dict[str, Any]]:
    """Select exactly ``target`` rows deterministically, optionally as indivisible groups."""
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in records:
        key = str(row[group_field]) if group_field else str(row["source_key"])
        groups[key].append(row)
    ordered = sorted(groups, key=_stable)
    chosen: list[Mapping[str, Any]] = []
    count = 0
    for key in ordered:
        group = sorted(groups[key], key=lambda row: str(row["source_key"]))
        if count + len(group) > target:
            continue
        chosen.extend(group)
        count += len(group)
        if count == target:
            break
    if count != target:
        raise ValueError(f"cannot reach exact target {target} with indivisible groups; reached {count}")
    return [dict(row) for row in sorted(chosen, key=lambda row: str(row["source_key"]))]


def calibration_groups(group_sizes: Mapping[str, int], target: int) -> set[str]:
    """Deterministic subset-sum nearest to the requested calibration row count."""
    if not group_sizes:
        raise ValueError("role assignment requires at least one group")
    ordered = sorted(group_sizes, key=_stable)
    total = sum(group_sizes.values())
    reachable = [False] * (total + 1)
    previous_sum = [-1] * (total + 1)
    previous_group = [-1] * (total + 1)
    reachable[0] = True
    for index, group in enumerate(ordered):
        size = int(group_sizes[group])
        for value in range(total - size, -1, -1):
            destination = value + size
            if reachable[value] and not reachable[destination]:
                reachable[destination] = True
                previous_sum[destination] = value
                previous_group[destination] = index
    candidates = [value for value, exists in enumerate(reachable) if exists and 0 < value < total]
    if not candidates:
        raise ValueError("source cannot populate both TRAIN and CALIBRATION")
    best = min(candidates, key=lambda value: (abs(value - target), value))
    selected: set[str] = set()
    while best:
        index = previous_group[best]
        if index < 0:
            raise RuntimeError("subset reconstruction failed")
        selected.add(ordered[index])
        best = previous_sum[best]
    return selected


def _role_group(source_id: str, row: Mapping[str, Any]) -> str:
    if source_id == "vision-base-native":
        return f"device:{row['device']}"
    if source_id == "forchheim-fodb":
        return str(row["scene_group"])
    if source_id == "csafe-mcsidb-s21":
        return f"device:{row['device']}"
    if source_id in FOUR_OUTPUT_SOURCES:
        return str(row["prompt_group"])
    if source_id == "communityforensics-ai-local":
        return f"generator:{row['model_name']}"
    return str(row.get("parent_group") or row["source_key"])


def _load_inputs() -> tuple[dict[str, Any], dict[str, Any], dict[str, dict[str, Any]]]:
    compact = json.loads(OVERLAY_EVIDENCE.read_text())
    overlay_raw = OVERLAY_PATH.read_bytes()
    if len(overlay_raw) != int(compact["detailed_report_bytes"]):
        raise ValueError("eligibility overlay byte count changed")
    if _sha256(overlay_raw) != compact["detailed_report_sha256"]:
        raise ValueError("eligibility overlay SHA-256 changed")
    overlay = json.loads(overlay_raw)
    if overlay.get("state") != "eligibility_frozen_role_free":
        raise ValueError("eligibility overlay is not role-free and frozen")
    audits: dict[str, dict[str, Any]] = {}
    for source in overlay["sources"]:
        source_id = str(source["source_id"])
        raw = (AUDIT_ROOT / f"{source_id}.json").read_bytes()
        binding = compact["audit_bindings"][source_id]
        if len(raw) != int(binding["bytes"]) or _sha256(raw) != binding["sha256"]:
            raise ValueError(f"audit binding changed for {source_id}")
        audits[source_id] = json.loads(raw)
    return compact, overlay, audits


def freeze_manifest() -> dict[str, Any]:
    compact, overlay, audits = _load_inputs()
    overlay_sources = {str(row["source_id"]): row for row in overlay["sources"]}
    selected_by_source: dict[str, list[dict[str, Any]]] = {}
    for source_id, source in overlay_sources.items():
        eligible = set(map(str, source["eligible_source_keys"]))
        candidates = [
            dict(row) for row in audits[source_id]["records"] if str(row["source_key"]) in eligible
        ]
        if len(candidates) != int(source["eligible_rows"]):
            raise ValueError(f"eligible/audit row mismatch for {source_id}")
        if source_id in AI_TARGETS:
            group_field = "prompt_group" if source_id in FOUR_OUTPUT_SOURCES else None
            candidates = select_records(candidates, AI_TARGETS[source_id], group_field=group_field)
        elif source_id not in REAL_SOURCES:
            raise ValueError(f"unexpected source {source_id}")
        selected_by_source[source_id] = candidates

    records: list[dict[str, Any]] = []
    source_summaries: list[dict[str, Any]] = []
    leakage_failures: list[str] = []
    for source_id in sorted(selected_by_source):
        source = overlay_sources[source_id]
        chosen = selected_by_source[source_id]
        group_sizes = Counter(_role_group(source_id, row) for row in chosen)
        calibration = calibration_groups(
            group_sizes, round(CALIBRATION_FRACTION * len(chosen))
        )
        role_groups: dict[str, set[str]] = defaultdict(set)
        role_counts = Counter()
        for row in chosen:
            group = _role_group(source_id, row)
            role = "CALIBRATION" if group in calibration else "TRAIN"
            role_groups[role].add(group)
            role_counts[role] += 1
            source_key = str(row["source_key"])
            records.append(
                {
                    "record_id": _sha256(f"{source_id}:{source_key}".encode()),
                    "source_id": source_id,
                    "source_key": source_key,
                    "label": str(source["label"]),
                    "role": role,
                    "role_group": group,
                    "parent_group": str(row.get("parent_group") or source_key),
                    "sha256": str(row["sha256"]),
                    "decoded_format": str(row["decoded_format"]),
                    "width": int(row["width"]),
                    "height": int(row["height"]),
                    "device": row.get("device"),
                    "scene_group": row.get("scene_group"),
                    "model_name": row.get("model_name"),
                }
            )
        overlap = role_groups["TRAIN"] & role_groups["CALIBRATION"]
        if overlap:
            leakage_failures.append(f"{source_id}:role_group_overlap:{len(overlap)}")
        source_summaries.append(
            {
                "source_id": source_id,
                "label": source["label"],
                "selected_rows": len(chosen),
                "role_counts": dict(sorted(role_counts.items())),
                "role_group_counts": {
                    role: len(role_groups[role]) for role in ("TRAIN", "CALIBRATION")
                },
                "role_group_kind": (
                    "scene" if source_id == "forchheim-fodb" else
                    "device" if source_id in {"vision-base-native", "csafe-mcsidb-s21"} else
                    "prompt" if source_id in FOUR_OUTPUT_SOURCES else
                    "generator_identity" if source_id == "communityforensics-ai-local" else
                    "parent"
                ),
            }
        )

    records.sort(key=lambda row: (row["source_id"], row["source_key"]))
    if len(records) != 22_688 or len({row["record_id"] for row in records}) != len(records):
        raise ValueError("balanced manifest count or record identity failed")
    class_counts = Counter(row["label"] for row in records)
    if class_counts != {"ai": 11_344, "real": 11_344}:
        raise ValueError(f"manifest is not exactly balanced: {class_counts}")
    if leakage_failures:
        raise ValueError(f"role leakage: {leakage_failures}")
    role_counts = Counter(row["role"] for row in records)
    role_class_counts = {
        role: dict(sorted(Counter(row["label"] for row in records if row["role"] == role).items()))
        for role in ("TRAIN", "CALIBRATION")
    }
    report = {
        "schema_version": 1,
        "experiment": "E32/C3-balanced-role-manifest",
        "state": "train_calibration_manifest_frozen",
        "seed": SEED,
        "calibration_fraction_target": CALIBRATION_FRACTION,
        "eligibility_overlay_sha256": compact["detailed_report_sha256"],
        "class_counts": dict(sorted(class_counts.items())),
        "role_counts": dict(sorted(role_counts.items())),
        "role_class_counts": role_class_counts,
        "sources": source_summaries,
        "records_sha256": _sha256(_json_bytes(records)),
        "records": records,
        "checks": {
            "balanced_classes": True,
            "unique_record_ids": True,
            "source_role_cells_nonempty": all(
                all(summary["role_counts"].get(role, 0) > 0 for role in ("TRAIN", "CALIBRATION"))
                for summary in source_summaries
            ),
            "role_group_overlap_count": 0,
            "development_or_locked_rows": 0,
        },
        "limitations": [
            "FODB roles are scene-disjoint but not camera-disjoint because its crossed design connects every device through shared scenes.",
            "This manifest contains no DEVELOPMENT or LOCKED FINAL row and cannot support final-performance claims.",
        ],
    }
    if not report["checks"]["source_role_cells_nonempty"]:
        raise ValueError("at least one source has an empty role cell")
    detailed_raw = _json_bytes(report)
    _write_atomic(DETAILED_OUTPUT, detailed_raw)
    compact_report = {key: value for key, value in report.items() if key != "records"}
    compact_report.update(
        {
            "detailed_report_external_path": DETAILED_OUTPUT.relative_to(OUTPUT_ROOT).as_posix(),
            "detailed_report_bytes": len(detailed_raw),
            "detailed_report_sha256": _sha256(detailed_raw),
        }
    )
    _write_atomic(COMPACT_EVIDENCE, _json_bytes(compact_report))
    return report


def main(argv: Iterable[str] | None = None) -> int:
    del argv
    report = freeze_manifest()
    print(json.dumps({key: value for key, value in report.items() if key != "records"}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
