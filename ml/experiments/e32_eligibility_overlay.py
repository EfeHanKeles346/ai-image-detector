"""Freeze a role-free, globally decontaminated E32 eligibility overlay.

The overlay may only remove rows from the immutable E32 selections. It recomputes duplicate
components across every realized AI and VISION record, preserves prompt-parent units, applies the
AI source-share cap, and binds its output to every input byte receipt. It assigns no model role.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence

import e32_ai_pool_selection as pool_selection
import e32_data_system as real_acquisition
from pixelproof.project_paths import ML_ROOT


REPO_ROOT = ML_ROOT.parent
OUTPUT_ROOT = real_acquisition.OUTPUT_ROOT
AUDIT_ROOT = OUTPUT_ROOT / "audits"
DETAILED_OUTPUT = OUTPUT_ROOT / "eligibility_overlay.json"
COMPACT_EVIDENCE = REPO_ROOT / "evidence" / "e32_eligibility_overlay.json"
AI_SOURCE_CAP_DENOMINATOR = 5
PHASH_DUPLICATE_MAX_DISTANCE = 5
AI_SOURCE_IDS = (
    "qwen-image-2512",
    "flux2-klein-9b",
    "nano-banana-local",
    "gpt-image-1",
    "nano-banana-pro-ash-local",
    "communityforensics-ai-local",
)
REAL_SOURCE_IDS = ("vision-base-native", "forchheim-fodb")
ALLOWED_AUDIT_FAILURES = {
    "within_source_exact_duplicates",
    "within_source_confirmed_perceptual_duplicates",
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


def _hamming(left: str, right: str) -> int:
    return (int(left, 16) ^ int(right, 16)).bit_count()


def _row_duplicate_components(records: Sequence[Mapping[str, Any]]) -> list[list[int]]:
    parent = list(range(len(records)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    by_sha: dict[str, list[int]] = defaultdict(list)
    by_dhash: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(records):
        by_sha[str(row["sha256"])].append(index)
        by_dhash[str(row["dhash"])].append(index)
    for indices in by_sha.values():
        for index in indices[1:]:
            union(indices[0], index)
    for indices in by_dhash.values():
        for offset, left in enumerate(indices):
            for right in indices[offset + 1 :]:
                if _hamming(str(records[left]["phash"]), str(records[right]["phash"])) <= PHASH_DUPLICATE_MAX_DISTANCE:
                    union(left, right)
    grouped: dict[int, list[int]] = defaultdict(list)
    for index in range(len(records)):
        grouped[find(index)].append(index)
    return sorted((sorted(group) for group in grouped.values() if len(group) > 1), key=lambda group: group[0])


def _duplicate_unit_exclusions(
    records: Sequence[Mapping[str, Any]], components: Sequence[Sequence[int]]
) -> tuple[dict[tuple[str, str], str], dict[str, int]]:
    unit_parent: dict[tuple[str, str], tuple[str, str]] = {}

    def find(unit: tuple[str, str]) -> tuple[str, str]:
        unit_parent.setdefault(unit, unit)
        while unit_parent[unit] != unit:
            unit_parent[unit] = unit_parent[unit_parent[unit]]
            unit = unit_parent[unit]
        return unit

    def union(left: tuple[str, str], right: tuple[str, str]) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            unit_parent[right_root] = left_root

    hard_excluded: dict[tuple[str, str], str] = {}
    for component in components:
        rows = [records[index] for index in component]
        units = [(str(row["source_id"]), str(row["unit_id"])) for row in rows]
        unique_units = sorted(set(units))
        for unit in unique_units:
            find(unit)
        for unit in unique_units[1:]:
            union(unique_units[0], unit)
        labels = {str(row["label"]) for row in rows}
        if len(labels) > 1:
            for unit in unique_units:
                hard_excluded[unit] = "cross_label_duplicate"
        counts = Counter(units)
        for unit, count in counts.items():
            if count > 1 and unit not in hard_excluded:
                hard_excluded[unit] = "within_parent_duplicate"

    connected: dict[tuple[str, str], list[tuple[str, str]]] = defaultdict(list)
    for unit in unit_parent:
        connected[find(unit)].append(unit)
    excluded = dict(hard_excluded)
    for units in connected.values():
        available = sorted(unit for unit in units if unit not in hard_excluded)
        if not available:
            continue
        canonical = available[0]
        for unit in available:
            if unit != canonical:
                excluded[unit] = "same_label_duplicate_noncanonical"
    return excluded, dict(sorted(Counter(excluded.values()).items()))


def _max_source_cap_targets(unit_sizes: Mapping[str, Sequence[int]]) -> dict[str, int]:
    """Maximize retained rows under exact <=20% share and indivisible equal-size units."""
    source_shape: dict[str, tuple[int, int]] = {}
    for source_id, sizes in unit_sizes.items():
        unique = set(sizes)
        if len(unique) != 1:
            raise ValueError(f"mixed parent-unit sizes for {source_id}: {sorted(unique)}")
        size = next(iter(unique))
        source_shape[source_id] = (size, len(sizes))
    best_total = -1
    best: dict[str, int] | None = None
    maximum = max(size * count for size, count in source_shape.values())
    for limit in range(maximum + 1):
        targets = {
            source_id: min(count, limit // size) * size
            for source_id, (size, count) in source_shape.items()
        }
        total = sum(targets.values())
        if total <= best_total:
            continue
        if all(AI_SOURCE_CAP_DENOMINATOR * count <= total for count in targets.values()):
            best_total, best = total, targets
    if best is None or best_total <= 0:
        raise ValueError("no non-empty source-cap solution")
    return best


def _stable_unit_order(selection_sha256: str, source_id: str, unit_id: str) -> tuple[str, str]:
    material = f"{selection_sha256}:{source_id}:{unit_id}".encode()
    return _sha256(material), unit_id


def _selection_records() -> tuple[list[dict[str, Any]], dict[str, str], dict[str, bytes]]:
    ai_raw = pool_selection.DETAILED_SELECTION.read_bytes()
    real_raw = real_acquisition.DETAILED_SELECTION.read_bytes()
    fodb_raw = (OUTPUT_ROOT / "fodb_orig_extraction.json").read_bytes()
    ai = json.loads(ai_raw)
    real = json.loads(real_raw)
    fodb = json.loads(fodb_raw)
    if fodb.get("state") != "orig_extraction_complete_role_free":
        raise ValueError("FODB extraction receipt has unexpected state")
    records: list[dict[str, Any]] = []
    for source_id in AI_SOURCE_IDS:
        source = next(item for item in ai["sources"] if item["source_id"] == source_id)
        for row in source["records"]:
            records.append(
                {
                    "source_id": source_id,
                    "source_key": str(row["source_key"]),
                    "unit_id": str(row["parent_group"]),
                    "label": "ai",
                }
            )
    for source_id in REAL_SOURCE_IDS:
        if source_id == "forchheim-fodb":
            for row in fodb["records"]:
                key = str(row["source_key"])
                records.append(
                    {
                        "source_id": source_id,
                        "source_key": key,
                        "unit_id": f"fodb:{row['camera_pipeline']}:{PurePosixPath(key).stem}",
                        "label": "real",
                    }
                )
            continue
        source = next(item for item in real["sources"] if item["source_id"] == source_id)
        for row in source["assets"]:
            key = str(row["source_key"])
            records.append(
                {
                    "source_id": source_id,
                    "source_key": key,
                    "unit_id": f"vision:{key}",
                    "label": "real",
                }
            )
    hashes = {
        "ai": _sha256(ai_raw),
        "real": _sha256(real_raw),
        "fodb_extraction": _sha256(fodb_raw),
    }
    return records, hashes, {"ai": ai_raw, "real": real_raw, "fodb_extraction": fodb_raw}


def freeze_overlay() -> dict[str, Any]:
    selected, selection_hashes, _ = _selection_records()
    selected_by_source: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in selected:
        selected_by_source[row["source_id"]][row["source_key"]] = row

    audit_bindings: dict[str, dict[str, Any]] = {}
    realized: list[dict[str, Any]] = []
    for source_id in (*AI_SOURCE_IDS, *REAL_SOURCE_IDS):
        path = AUDIT_ROOT / f"{source_id}.json"
        raw = path.read_bytes()
        report = json.loads(raw)
        if int(report.get("schema_version", -1)) != 2:
            raise ValueError(f"{source_id} is not a schema-v2 audit")
        reasons = {str(item["reason"]) for item in report.get("failures", [])}
        unexpected = reasons - ALLOWED_AUDIT_FAILURES
        if unexpected:
            raise ValueError(f"{source_id} has non-repairable audit failures: {sorted(unexpected)}")
        audit_rows = {str(row["source_key"]): row for row in report["records"]}
        if set(audit_rows) != set(selected_by_source[source_id]):
            raise ValueError(f"{source_id} audit rows do not equal frozen selection")
        for key, selection_row in selected_by_source[source_id].items():
            audit_row = audit_rows[key]
            realized.append(
                {
                    **selection_row,
                    "sha256": str(audit_row["sha256"]),
                    "dhash": str(audit_row["dhash"]),
                    "phash": str(audit_row["phash"]),
                }
            )
        audit_bindings[source_id] = {
            "sha256": _sha256(raw),
            "bytes": len(raw),
            "state": report["state"],
        }

    components = _row_duplicate_components(realized)
    duplicate_excluded, duplicate_reason_counts = _duplicate_unit_exclusions(realized, components)
    units: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for row in realized:
        units[row["source_id"]][row["unit_id"]].append(row["source_key"])

    eligible_ai_units = {
        source_id: {
            unit_id: sorted(keys)
            for unit_id, keys in source_units.items()
            if (source_id, unit_id) not in duplicate_excluded
        }
        for source_id, source_units in units.items()
        if source_id in AI_SOURCE_IDS
    }
    cap_targets = _max_source_cap_targets(
        {
            source_id: [len(keys) for keys in source_units.values()]
            for source_id, source_units in eligible_ai_units.items()
        }
    )
    cap_excluded: set[tuple[str, str]] = set()
    for source_id, source_units in eligible_ai_units.items():
        ordered = sorted(
            source_units,
            key=lambda unit_id: _stable_unit_order(selection_hashes["ai"], source_id, unit_id),
        )
        unit_size = len(source_units[ordered[0]])
        keep_units = cap_targets[source_id] // unit_size
        cap_excluded.update((source_id, unit_id) for unit_id in ordered[keep_units:])

    source_results = []
    for source_id in (*AI_SOURCE_IDS, *REAL_SOURCE_IDS):
        label = "ai" if source_id in AI_SOURCE_IDS else "real"
        eligible_keys: list[str] = []
        exclusions: list[dict[str, Any]] = []
        for unit_id, keys in sorted(units[source_id].items()):
            unit = (source_id, unit_id)
            reason = duplicate_excluded.get(unit)
            if reason is None and unit in cap_excluded:
                reason = "source_cap_trim"
            if reason is None:
                eligible_keys.extend(keys)
            else:
                exclusions.append(
                    {"unit_id": unit_id, "reason": reason, "source_keys": sorted(keys)}
                )
        source_results.append(
            {
                "source_id": source_id,
                "label": label,
                "selected_rows": sum(len(keys) for keys in units[source_id].values()),
                "selected_units": len(units[source_id]),
                "eligible_rows": len(eligible_keys),
                "eligible_units": len(units[source_id]) - len(exclusions),
                "eligible_source_keys": sorted(eligible_keys),
                "exclusions": exclusions,
                "exclusion_reason_counts": dict(
                    sorted(Counter(item["reason"] for item in exclusions).items())
                ),
            }
        )

    ai_total = sum(item["eligible_rows"] for item in source_results if item["label"] == "ai")
    real_total = sum(item["eligible_rows"] for item in source_results if item["label"] == "real")
    for item in source_results:
        denominator = ai_total if item["label"] == "ai" else real_total
        item["eligible_share"] = round(item["eligible_rows"] / denominator, 8)
        if item["label"] == "ai" and AI_SOURCE_CAP_DENOMINATOR * item["eligible_rows"] > ai_total:
            raise AssertionError(f"source cap failed for {item['source_id']}")

    detailed = {
        "schema_version": 1,
        "experiment": "E32/C2-eligibility-overlay",
        "state": "eligibility_frozen_role_free",
        "selection_sha256": selection_hashes,
        "audit_bindings": audit_bindings,
        "global_realized_rows": len(realized),
        "global_duplicate_row_components": len(components),
        "duplicate_excluded_unit_reason_counts": duplicate_reason_counts,
        "ai_source_cap": "<=20% of eligible AI rows",
        "ai_eligible_rows": ai_total,
        "real_eligible_rows": real_total,
        "sources": source_results,
        "boundaries": [
            "The overlay only excludes immutable selected rows and never adds replacements.",
            "No eligible row has TRAIN, CALIBRATION, DEVELOPMENT or LOCKED FINAL role.",
            "Same-label duplicates keep a deterministic parent unit; cross-label duplicates lose every affected unit.",
            "Four-variant Qwen and FLUX prompt groups remain indivisible.",
        ],
    }
    detailed_raw = _json_bytes(detailed)
    _write_atomic(DETAILED_OUTPUT, detailed_raw)
    compact_sources = [
        {key: value for key, value in source.items() if key not in {"eligible_source_keys", "exclusions"}}
        for source in source_results
    ]
    compact = {
        **{key: value for key, value in detailed.items() if key != "sources"},
        "sources": compact_sources,
        "detailed_report_sha256": _sha256(detailed_raw),
        "detailed_report_bytes": len(detailed_raw),
        "detailed_report_external_path": DETAILED_OUTPUT.relative_to(OUTPUT_ROOT).as_posix(),
    }
    _write_atomic(COMPACT_EVIDENCE, _json_bytes(compact))
    return compact


def main(argv: Iterable[str] | None = None) -> int:
    del argv
    print(json.dumps(freeze_overlay(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
