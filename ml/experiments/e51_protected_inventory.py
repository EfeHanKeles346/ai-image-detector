"""Verify protected manifests and E49 reserve identities; never read model scores."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from experiments.e51_prefit_audit import INPUTS, ROOT, load_inputs
from experiments.e51_train_cal_realize import (
    PROTECTED_ROLE_PATHS, EXTRA_PROTECTED, _write_atomic, require_protected_paths,
)
from pixelproof.project_paths import DATA_ROOT, ML_ROOT

REPORT = ROOT / "audit/protected_inventory_v1.json"
EVIDENCE = ML_ROOT.parent / "evidence/e51_protected_inventory.json"
ADDITIONAL = {
    DATA_ROOT / "e49_d1_dotting/manifest_unscored.json":
        "048572a41b47b65d3d09bd39bee45a40745a40c2e15c50444b41f8384fdccc9d",
    DATA_ROOT / "e49/open_components_contract_v2.json":
        "1d4e184c27cb87cf832045a23b6966f382673c3bcd8342a900c07130bd9182aa",
    DATA_ROOT / "e49/openfake/asset_contract_untransferred_unscored.json":
        "7b71449e0e7d9ea22973f021af2d4ec49cc395e3fffe6816ffc123274a571415",
}


def identity_rows(value: Any):
    """Nested contracts contain reserve rows below component dictionaries."""
    if isinstance(value, dict):
        if any(value.get(k) for k in ("parent_id", "identity", "record_id")):
            yield value
        for child in value.values():
            if isinstance(child, (dict, list)):
                yield from identity_rows(child)
    elif isinstance(value, list):
        for child in value:
            yield from identity_rows(child)


def keys(row: dict) -> set[tuple[str, str]]:
    result = {("identity", str(row[k])) for k in ("parent_id", "identity", "record_id") if row.get(k)}
    for field in ("sha256", "sha1"):
        if row.get(field):
            result.add((field, str(row[field])))
    return result


def inventory() -> dict:
    if REPORT.exists() or EVIDENCE.exists():
        raise FileExistsError("protected inventory already exists")
    train, cal = load_inputs()
    train_payload = json.loads(INPUTS["TRAIN"][0].read_text())
    pins = {Path(r['path']): r['sha256'] for r in train_payload['protected_role_manifests']}
    pins.update(ADDITIONAL)
    required = tuple(PROTECTED_ROLE_PATHS) + EXTRA_PROTECTED + tuple(ADDITIONAL)
    require_protected_paths(required)
    if not set(required).issubset(pins):
        raise ValueError("required protected manifest has no frozen hash")
    all_keys, e49_keys = set(), set()
    sources = []
    for path in sorted(set(required)):
        raw = path.read_bytes()
        sha = hashlib.sha256(raw).hexdigest()
        if sha != pins[path]:
            raise ValueError(f"protected manifest changed: {path}")
        rows = list(identity_rows(json.loads(raw)))
        source_keys = set().union(*(keys(r) for r in rows)) if rows else set()
        all_keys.update(source_keys)
        if path.relative_to(DATA_ROOT).parts[0].startswith("e49"):
            e49_keys.update(source_keys)
        sources.append({"path": str(path), "sha256": sha, "identity_rows": len(rows),
                        "exact_keys": len(source_keys)})
    hits = []
    for row in train + cal:
        # Historical TRAIN/CAL reuse is deliberate. Only new source parents must be
        # disjoint from all historical roles; EVERY E51 row must be disjoint from E49.
        new_source = row.get("source") in {"SCIMD-17", "SCMI30-IITRPR"}
        matched = keys(row) & (all_keys if new_source else e49_keys)
        if matched:
            hits.append({"parent_id": row['parent_id'], "role": row['audit_role'],
                         "matched_keys": sorted(matched)})
    report = {
        "schema_version": 1, "state": "protected_metadata_overlap" if hits else "protected_metadata_check_passed",
        "source_count": len(sources), "sources": sources, "overlaps": hits,
        "e49_protected_exact_keys": len(e49_keys),
        "new_source_parents": len({r['parent_id'] for r in train + cal
                                    if r.get('source') in {'SCIMD-17', 'SCMI30-IITRPR'}}),
        "reserved_e49_components": {"commons": 1100, "stylegan2": 240, "openfake": 960},
        "model_scores_created": 0, "training_authorized": False,
        "limitations": [
            "Metadata identity/available encoded hashes only; no canonical protected-pixel audit yet",
            "Unfetched reserves can be excluded by identity, not certified byte/perceptual disjoint",
            "Only the listed E49 v2/asset reserves are enumerated; superseded reserve contracts still need review",
        ],
    }
    _write_atomic(REPORT, report)
    _write_atomic(EVIDENCE, report)
    return report


if __name__ == "__main__":
    result = inventory()
    print(json.dumps({k: v for k, v in result.items() if k not in {"sources", "overlaps"}}, indent=2))
    print("overlap observations:", len(result['overlaps']))
