"""Freeze score-blind parent roles for E43 RR adaptation."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from pixelproof.project_paths import DATA_ROOT, ML_ROOT


RR_ROOT = DATA_ROOT / "e33_rrdataset"
RR_MANIFEST = RR_ROOT / "e42_rr_unscored_manifest.json"
RR_MANIFEST_SHA256 = "b2d815afab0bbafa339baf70eac19afbaf955e545041c550193340763ac30c98"
ROOT = DATA_ROOT / "e43"
MANIFEST = ROOT / "rr_roles.json"
EVIDENCE = ML_ROOT.parent / "evidence" / "e43_rr_roles.json"
CONDITIONS = {"original", "transfer", "redigital"}
AI_SOURCES = {
    "culture_and_religion",
    "everyday_life",
    "labor_and_production",
    "medical_and_public_health",
    "natural_disasters_and_accidents",
    "political_and_social_events",
    "war_and_conflict",
}
STRATUM_CAPS = {"real": 1_960, **{source: 280 for source in AI_SOURCES}}
ROLE_COUNTS = {
    "real": {"train": 980, "calibration": 490, "development": 490},
    **{
        source: {"train": 140, "calibration": 70, "development": 70}
        for source in AI_SOURCES
    },
}


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024**2):
            digest.update(chunk)
    return digest.hexdigest()


def _rank(namespace: str, parent: str) -> bytes:
    return hashlib.sha256(f"{namespace}|{parent}".encode()).digest()


def assign_roles(
    rows: Sequence[Mapping[str, Any]],
    *,
    caps: Mapping[str, int] = STRATUM_CAPS,
    role_counts: Mapping[str, Mapping[str, int]] = ROLE_COUNTS,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_parent: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_parent[str(row["parent_id"])].append(row)

    strata: dict[str, list[str]] = defaultdict(list)
    parent_rows: dict[str, list[Mapping[str, Any]]] = {}
    for parent, versions in by_parent.items():
        if {str(row["condition"]) for row in versions} != CONDITIONS or len(versions) != 3:
            continue
        labels = {int(row["label"]) for row in versions}
        sources = {str(row["source"]) for row in versions}
        if len(labels) != 1 or len(sources) != 1:
            raise ValueError(f"E43 RR parent label/source changed: {parent}")
        label = next(iter(labels))
        source = next(iter(sources))
        if label == 0:
            stratum = "real"
        elif label == 1 and source in AI_SOURCES:
            stratum = source
        else:
            raise ValueError(f"E43 RR unexpected label/source: {label}/{source}")
        strata[stratum].append(parent)
        parent_rows[parent] = versions

    selected_roles: dict[str, str] = {}
    available = {stratum: len(parents) for stratum, parents in sorted(strata.items())}
    for stratum, cap in sorted(caps.items()):
        candidates = sorted(strata.get(stratum, []), key=lambda parent: (_rank("E43_SELECT", parent), parent))
        if len(candidates) < cap:
            raise ValueError(f"E43 RR stratum too small: {stratum}={len(candidates)} < {cap}")
        selected = candidates[:cap]
        counts = role_counts[stratum]
        if sum(int(value) for value in counts.values()) != cap:
            raise ValueError(f"E43 role counts do not sum to cap: {stratum}")
        ordered = sorted(selected, key=lambda parent: (_rank("E43_ROLE", parent), parent))
        start = 0
        for role in ("train", "calibration", "development"):
            stop = start + int(counts[role])
            for parent in ordered[start:stop]:
                selected_roles[parent] = role
            start = stop

    output = []
    for parent, role in selected_roles.items():
        for row in parent_rows[parent]:
            output.append({**dict(row), "e43_role": role})
    output.sort(key=lambda row: str(row["record_id"]))
    summary = {
        "available_complete_parents_by_stratum": available,
        "selected_parents": len(selected_roles),
        "selected_rows": len(output),
        "parents_by_role": dict(sorted(Counter(selected_roles.values()).items())),
        "rows_by_role": dict(sorted(Counter(str(row["e43_role"]) for row in output).items())),
        "rows_by_role_label": {
            f"{role}/{label}": sum(
                str(row["e43_role"]) == role and int(row["label"]) == label for row in output
            )
            for role in ("train", "calibration", "development")
            for label in (0, 1)
        },
        "rows_by_condition": dict(sorted(Counter(str(row["condition"]) for row in output).items())),
    }
    return output, summary


def freeze() -> dict[str, Any]:
    if MANIFEST.exists() or EVIDENCE.exists():
        raise FileExistsError("E43 RR role manifest already exists; no silent replacement")
    if _digest(RR_MANIFEST) != RR_MANIFEST_SHA256:
        raise ValueError("E42 RR unscored manifest changed")
    source = json.loads(RR_MANIFEST.read_text())
    if source.get("state") != "rr_final_manifest_frozen_unscored":
        raise ValueError("E42 RR source manifest state changed")
    rows, summary = assign_roles(source["rows"])
    expected = {
        "selected_parents": 3_920,
        "selected_rows": 11_760,
        "parents_by_role": {"calibration": 980, "development": 980, "train": 1_960},
        "rows_by_role": {"calibration": 2_940, "development": 2_940, "train": 5_880},
        "rows_by_role_label": {
            "train/0": 2_940, "train/1": 2_940,
            "calibration/0": 1_470, "calibration/1": 1_470,
            "development/0": 1_470, "development/1": 1_470,
        },
        "rows_by_condition": {"original": 3_920, "redigital": 3_920, "transfer": 3_920},
    }
    for key, value in expected.items():
        if summary[key] != value:
            raise ValueError(f"E43 RR role result changed: {key}={summary[key]!r}")
    payload = {
        "schema_version": 1,
        "state": "e43_rr_roles_frozen_before_features",
        "source_manifest_sha256": RR_MANIFEST_SHA256,
        "selection": {
            "rule": "complete triplets; independent SHA256 selection and role namespaces",
            "stratum_caps": dict(sorted(STRATUM_CAPS.items())),
            "role_counts": {key: dict(value) for key, value in sorted(ROLE_COUNTS.items())},
            "score_files_read": 0,
        },
        "summary": summary,
        "rows": rows,
        "boundary": "Consumed RR development only; no score/model access and no image copy.",
    }
    raw = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    temporary = MANIFEST.with_suffix(".json.part")
    temporary.write_bytes(raw)
    temporary.replace(MANIFEST)
    evidence = {
        "schema_version": 1,
        "state": payload["state"],
        "source_manifest_sha256": RR_MANIFEST_SHA256,
        "summary": summary,
        "detailed_manifest_bytes": len(raw),
        "detailed_manifest_sha256": hashlib.sha256(raw).hexdigest(),
        "score_files_read": 0,
        "model_scores_created": 0,
    }
    EVIDENCE.parent.mkdir(parents=True, exist_ok=True)
    EVIDENCE.write_bytes((json.dumps(evidence, indent=2, sort_keys=True) + "\n").encode())
    return evidence


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze",))
    args = parser.parse_args(argv)
    result = freeze() if args.command == "freeze" else None
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
