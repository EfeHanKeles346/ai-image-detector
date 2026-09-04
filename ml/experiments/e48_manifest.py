"""Bind the score-blind E48 FIT/CAL/DEVELOPMENT candidate population."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import tarfile
from typing import Any, Iterable, Mapping, Sequence

from experiments.e46_manifests import (
    SYNTH_AUDITED,
    TRUE_ARCHIVE,
    TRUE_CONTRACT,
    TRUE_MANIFEST,
    TRUEFAKE_SHA256,
    _decode,
    _digest,
    parse_truefake_member,
)
from experiments.e47_caldev_manifest import (
    CONTRACT as E47_CONTRACT,
    MANIFEST as E47_MANIFEST,
    identity_exclusions,
)
from pixelproof.project_paths import DATA_ROOT, ML_ROOT


ROOT = DATA_ROOT / "e48"
NAMESPACE = "E48_MONOTONE_NONVETO_V1"
CONTRACT = ROOT / "selection_contract.json"
CONTRACT_EVIDENCE = ML_ROOT.parent / "evidence" / "e48_selection_contract.json"
POOL = ROOT / "candidate_pool"
MANIFEST = ROOT / "manifest_unscored.json"
MANIFEST_EVIDENCE = ML_ROOT.parent / "evidence" / "e48_manifest.json"
E32 = DATA_ROOT / "e32"
E42_MANIFEST = DATA_ROOT / "e42" / "parent_manifest.json"
C3_MANIFEST = E32 / "c3_role_manifest.json"
REAL_AUDITS = {
    "vision-base-native": E32 / "audits" / "vision-base-native.json",
    "csafe-mcsidb-s21": E32 / "audits" / "csafe-mcsidb-s21.json",
    "forchheim-fodb": E32 / "audits" / "forchheim-fodb.json",
}
REAL_QUOTAS = {
    ("FIT", "vision-base-native"): 150,
    ("FIT", "csafe-mcsidb-s21"): 150,
    ("CAL", "vision-base-native"): 150,
    ("CAL", "csafe-mcsidb-s21"): 150,
    ("DEVELOPMENT", "forchheim-fodb"): 600,
}
AI_QUOTAS = {
    ("FIT", "FLUX.1"): 100,
    ("FIT", "StyleGAN2"): 100,
    ("FIT", "StableDiffusion1.5"): 50,
    ("FIT", "StableDiffusionXL"): 50,
    ("CAL", "FLUX.1"): 100,
    ("CAL", "StyleGAN2"): 100,
    ("CAL", "StableDiffusion1.5"): 50,
    ("CAL", "StableDiffusionXL"): 50,
    ("DEVELOPMENT", "FLUX.1"): 150,
    ("DEVELOPMENT", "StableDiffusion3"): 150,
    ("DEVELOPMENT", "StyleGAN"): 150,
    ("DEVELOPMENT", "StyleGAN3"): 150,
}
RESERVE_FACTOR = 1.20
PROTECTED_ROLE_PATHS = (
    DATA_ROOT / "e33_rrdataset" / "r1c_cal_manifest.json",
    DATA_ROOT / "e33_rrdataset" / "e42_rr_unscored_manifest.json",
    DATA_ROOT / "e36" / "cal_manifest.json",
    DATA_ROOT / "e36" / "final_manifest.json",
    DATA_ROOT / "e39" / "final_manifest.json",
    E42_MANIFEST,
    DATA_ROOT / "e42_external" / "bfree_viral" / "unscored_manifest.json",
    DATA_ROOT / "e43_dda_coco" / "unscored_manifest.json",
    DATA_ROOT / "e44" / "fusion_contract.json",
    DATA_ROOT / "e44" / "successor_contract.json",
    DATA_ROOT / "e45_mediaeval_itwsm" / "unscored_manifest.json",
    SYNTH_AUDITED,
    TRUE_MANIFEST,
    E47_MANIFEST,
)


def _write(path: Path, value: Any) -> bytes:
    raw = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_bytes(raw)
    temporary.replace(path)
    return raw


def _rank(role: str, source: str, identity: str) -> str:
    return hashlib.sha256(f"{NAMESPACE}|{role}|{source}|{identity}".encode()).hexdigest()


def _device_split(source: str, devices: Sequence[str]) -> dict[str, str]:
    ordered = sorted(set(devices), key=lambda value: (_rank("DEVICE", source, value), value))
    if len(ordered) < 4:
        raise ValueError(f"E48 needs multiple devices for {source}")
    return {device: ("FIT" if index % 2 == 0 else "CAL") for index, device in enumerate(ordered)}


def balanced_select(
    rows: Sequence[Mapping[str, Any]], count: int, *, max_per_scene: int | None = None
) -> list[dict[str, Any]]:
    """Deterministic device round-robin with an optional repeated-scene cap."""
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row["device"])].append(dict(row))
    for values in groups.values():
        values.sort(key=lambda row: (str(row["rank"]), str(row["identity"])))
    devices = sorted(groups, key=lambda value: (_rank("DEVICE_ORDER", "real", value), value))
    selected: list[dict[str, Any]] = []
    scene_counts: Counter[str] = Counter()
    cursors = {device: 0 for device in devices}
    while len(selected) < count:
        progressed = False
        for device in devices:
            values = groups[device]
            while cursors[device] < len(values):
                row = values[cursors[device]]
                cursors[device] += 1
                scene = str(row.get("scene_group") or row["identity"])
                if max_per_scene is not None and scene_counts[scene] >= max_per_scene:
                    continue
                selected.append(row)
                scene_counts[scene] += 1
                progressed = True
                break
            if len(selected) == count:
                break
        if not progressed:
            raise ValueError(f"cannot fill E48 balanced selection: {len(selected)}/{count}")
    return selected


def _current_training_keys() -> set[tuple[str, str]]:
    e42 = json.loads(E42_MANIFEST.read_text())
    trained_ids = {
        str(row["parent_id"]).split(":", 1)[1]
        for row in e42["rows"]
        if row.get("role") == "train" and str(row.get("parent_id", "")).startswith("e32:")
    }
    c3 = json.loads(C3_MANIFEST.read_text())
    return {
        (str(row["source_id"]), str(row["source_key"]))
        for row in c3["records"] if str(row["record_id"]) in trained_ids
    }


def _real_path(source: str, source_key: str) -> Path:
    roots = {
        "vision-base-native": E32 / "real" / "vision",
        "csafe-mcsidb-s21": E32,
        "forchheim-fodb": E32,
    }
    return roots[source] / source_key


def _real_candidates() -> tuple[list[dict[str, Any]], dict[str, str]]:
    training_keys = _current_training_keys()
    all_rows: dict[str, list[dict[str, Any]]] = {}
    device_roles: dict[str, str] = {}
    for source, path in REAL_AUDITS.items():
        payload = json.loads(path.read_text())
        records = []
        for row in payload["records"]:
            key = (source, str(row["source_key"]))
            if key in training_keys or not row.get("sha256") or not row.get("dhash"):
                continue
            resolved = _real_path(source, str(row["source_key"]))
            records.append({
                "identity": f"{source}:{row['source_key']}", "source": source,
                "source_key": row["source_key"], "path": str(resolved),
                "sha256": row["sha256"], "dhash": row["dhash"], "label": 0,
                "device": str(row["device"]), "camera_pipeline": str(row["camera_pipeline"]),
                "scene_group": row.get("scene_group"),
            })
        all_rows[source] = records
        if source != "forchheim-fodb":
            split = _device_split(source, [row["device"] for row in records])
            device_roles.update({f"{source}:{device}": role for device, role in split.items()})

    selected = []
    for (role, source), quota in REAL_QUOTAS.items():
        reserve = math.ceil(quota * RESERVE_FACTOR)
        rows = all_rows[source]
        if source != "forchheim-fodb":
            rows = [row for row in rows if device_roles[f"{source}:{row['device']}"] == role]
        ranked = [{**row, "role": role, "rank": _rank(role, source, row["identity"]),
                   "target_quota": quota} for row in rows]
        selected.extend(balanced_select(ranked, reserve,
                                        max_per_scene=6 if source == "forchheim-fodb" else None))
    return selected, device_roles


def _used_truefake_members() -> tuple[set[str], dict[str, str]]:
    e46_raw = TRUE_CONTRACT.read_bytes()
    e47_raw = E47_CONTRACT.read_bytes()
    e46 = json.loads(e46_raw)
    e47 = json.loads(e47_raw)
    members = {str(row["member"]) for row in e46["selection"]["candidate_rows"]}
    members.update(str(row["member"]) for row in e47["candidate_rows"])
    if len(members) != 6_380:
        raise ValueError(f"E48 prior TrueFake identity count changed: {len(members)}")
    return members, {
        "e46_reserve_contract_sha256": hashlib.sha256(e46_raw).hexdigest(),
        "e47_candidate_contract_sha256": hashlib.sha256(e47_raw).hexdigest(),
    }


def _ai_candidates(excluded: set[str]) -> list[dict[str, Any]]:
    available: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    with tarfile.open(TRUE_ARCHIVE, "r:gz") as bundle:
        for member in bundle.getmembers():
            if member.isdir() or member.name in excluded:
                continue
            label, source, content = parse_truefake_member(member.name)
            if label != 1:
                continue
            for role in ("FIT", "CAL", "DEVELOPMENT"):
                if (role, source) in AI_QUOTAS:
                    available[(role, source)].append({
                        "identity": member.name, "member": member.name, "bytes": member.size,
                        "source": source, "content": content, "label": 1, "role": role,
                        "rank": _rank(role, source, member.name),
                        "target_quota": AI_QUOTAS[(role, source)],
                    })
    selected, used = [], set()
    for key, quota in AI_QUOTAS.items():
        reserve = math.ceil(quota * RESERVE_FACTOR)
        ordered = sorted(available[key], key=lambda row: (row["rank"], row["member"]))
        chosen = [row for row in ordered if row["member"] not in used][:reserve]
        if len(chosen) != reserve:
            raise ValueError(f"insufficient E48 TrueFake candidates for {key}")
        selected.extend(chosen)
        used.update(row["member"] for row in chosen)
    return selected


def bind() -> dict[str, Any]:
    if CONTRACT.exists() or CONTRACT_EVIDENCE.exists():
        raise FileExistsError("E48 selection contract already exists")
    if _digest(TRUE_ARCHIVE) != TRUEFAKE_SHA256:
        raise ValueError("E48 TrueFake archive changed")
    excluded, prior_contracts = _used_truefake_members()
    real, device_roles = _real_candidates()
    ai = _ai_candidates(excluded)
    rows = sorted(real + ai, key=lambda row: (row["role"], row["label"], row["source"], row["rank"]))
    payload = {
        "schema_version": 1, "state": "e48_selection_frozen_before_new_payload_access",
        "namespace": NAMESPACE, "truefake_archive_sha256": TRUEFAKE_SHA256,
        "prior_contracts": prior_contracts, "excluded_prior_truefake_candidates": len(excluded),
        "current_training_manifest_sha256": _digest(E42_MANIFEST),
        "real_audit_sha256": {name: _digest(path) for name, path in REAL_AUDITS.items()},
        "selection": "namespace SHA rank; camera-device round robin; 20% identity-audit headroom",
        "device_roles": device_roles,
        "quotas": {
            **{f"{role}:REAL:{source}": {"target": quota, "reserve": math.ceil(quota * RESERVE_FACTOR)}
               for (role, source), quota in REAL_QUOTAS.items()},
            **{f"{role}:AI:{source}": {"target": quota, "reserve": math.ceil(quota * RESERVE_FACTOR)}
               for (role, source), quota in AI_QUOTAS.items()},
        },
        "candidate_rows": rows,
        "forbidden": ["model access before identity audit", "score-dependent replacement",
                      "E46/E47 candidate reuse", "current-candidate training identity reuse"],
        "model_scores_created": 0,
    }
    raw = _write(CONTRACT, payload)
    evidence = {
        "schema_version": 1, "state": payload["state"], "namespace": NAMESPACE,
        "candidate_rows": len(rows), "target_rows": sum(REAL_QUOTAS.values()) + sum(AI_QUOTAS.values()),
        "by_role_label": {f"{role}:{label}": count for (role, label), count in sorted(
            Counter((row["role"], row["label"]) for row in rows).items())},
        "excluded_prior_truefake_candidates": len(excluded),
        "contract_bytes": len(raw), "contract_sha256": hashlib.sha256(raw).hexdigest(),
        "model_scores_created": 0,
    }
    _write(CONTRACT_EVIDENCE, evidence)
    return evidence


def _row_collections(payload: Mapping[str, Any]) -> Iterable[Mapping[str, Any]]:
    for key in ("rows", "records", "e35_rows"):
        value = payload.get(key)
        if isinstance(value, list):
            yield from (row for row in value if isinstance(row, Mapping))


def _protected_role_hashes() -> tuple[set[str], set[str], list[dict[str, str]]]:
    exact: set[str] = set()
    perceptual: set[str] = set()
    sources = []
    for path in PROTECTED_ROLE_PATHS:
        if not path.is_file():
            continue
        raw = path.read_bytes()
        for row in _row_collections(json.loads(raw)):
            if row.get("sha256"):
                exact.add(str(row["sha256"]))
            if row.get("dhash"):
                perceptual.add(str(row["dhash"]))
        sources.append({"path": str(path), "sha256": hashlib.sha256(raw).hexdigest()})
    if not exact or not perceptual:
        raise ValueError("E48 protected role hashes unavailable")
    return exact, perceptual, sources


def extract() -> dict[str, Any]:
    if MANIFEST.exists() or MANIFEST_EVIDENCE.exists():
        raise FileExistsError("E48 manifest already exists")
    contract_raw = CONTRACT.read_bytes()
    receipt = json.loads(CONTRACT_EVIDENCE.read_text())
    if hashlib.sha256(contract_raw).hexdigest() != receipt.get("contract_sha256"):
        raise ValueError("E48 selection contract changed")
    contract = json.loads(contract_raw)
    candidate_rows = contract["candidate_rows"]
    real_candidates = [row for row in candidate_rows if int(row["label"]) == 0]
    ai_by_member = {str(row["member"]): row for row in candidate_rows if int(row["label"]) == 1}

    decoded = []
    failures = []
    for row in real_candidates:
        path = Path(str(row["path"]))
        try:
            if not path.is_file() or _digest(path) != row["sha256"]:
                raise ValueError("camera payload hash changed")
            decoded.append({**row, "record_id": f"e48:real:{row['identity']}",
                            "condition": "camera-original"})
        except (OSError, ValueError) as error:
            failures.append({"identity": row["identity"], "error": f"{type(error).__name__}: {error}"})

    if _digest(TRUE_ARCHIVE) != TRUEFAKE_SHA256:
        raise ValueError("E48 TrueFake archive changed before extraction")
    with tarfile.open(TRUE_ARCHIVE, "r|gz") as bundle:
        for member in bundle:
            row = ai_by_member.get(member.name)
            if row is None:
                continue
            try:
                stream = bundle.extractfile(member)
                if stream is None:
                    raise ValueError("TrueFake member payload unavailable")
                raw = stream.read()
                image = _decode(raw, member.name)
                target = POOL / member.name
                target.parent.mkdir(parents=True, exist_ok=True)
                temporary = target.with_suffix(target.suffix + ".part")
                temporary.write_bytes(raw)
                temporary.replace(target)
                decoded.append({**row, **image, "record_id": f"e48:ai:{member.name}",
                                "path": str(target), "condition": "Facebook"})
            except (OSError, ValueError, tarfile.TarError) as error:
                failures.append({"identity": member.name, "error": f"{type(error).__name__}: {error}"})
    if len(decoded) + len(failures) != len(candidate_rows):
        raise ValueError("E48 candidate decode coverage changed")

    prior_exact, prior_dhash, protected = _protected_role_hashes()
    reasons = identity_exclusions(decoded, prior_exact, prior_dhash)
    excluded = set(reasons)
    selected = []
    for (role, source), quota in REAL_QUOTAS.items():
        eligible = [row for row in decoded if row["role"] == role and row["source"] == source
                    and row["record_id"] not in excluded]
        selected.extend(balanced_select(eligible, quota,
                                        max_per_scene=5 if source == "forchheim-fodb" else None))
    for (role, source), quota in AI_QUOTAS.items():
        eligible = sorted((row for row in decoded if row["role"] == role and row["source"] == source
                           and row["record_id"] not in excluded), key=lambda row: (row["rank"], row["identity"]))
        if len(eligible) < quota:
            raise ValueError(f"insufficient clean E48 AI candidates for {(role, source)}")
        selected.extend(eligible[:quota])
    selected.sort(key=lambda row: (row["role"], row["label"], row["source"], row["rank"]))
    expected = sum(REAL_QUOTAS.values()) + sum(AI_QUOTAS.values())
    if len(selected) != expected or len({row["record_id"] for row in selected}) != expected:
        raise ValueError("E48 selected identity count changed")

    payload = {
        "schema_version": 1, "state": "e48_decontaminated_frozen_unscored",
        "selection_contract_sha256": receipt["contract_sha256"],
        "counts": {
            "candidates": len(candidate_rows), "decoded": len(decoded),
            "decode_failures": len(failures), "identity_exclusions": len(excluded),
            "selected": len(selected),
            "by_role_label": {f"{role}:{label}": count for (role, label), count in sorted(
                Counter((row["role"], row["label"]) for row in selected).items())},
            "by_role_source": {f"{role}:{source}": count for (role, source), count in sorted(
                Counter((row["role"], row["source"]) for row in selected).items())},
        },
        "decode_failures": failures, "identity_exclusion_reasons": reasons,
        "protected_role_manifests": protected, "rows": selected,
        "model_scores_created": 0,
        "boundary": "All rows verified/decontaminated before model access; DEVELOPMENT remains unopened.",
    }
    raw = _write(MANIFEST, payload)
    evidence = {
        "schema_version": 1, "state": payload["state"], "counts": payload["counts"],
        "protected_role_manifest_count": len(protected), "manifest_bytes": len(raw),
        "manifest_sha256": hashlib.sha256(raw).hexdigest(), "model_scores_created": 0,
    }
    _write(MANIFEST_EVIDENCE, evidence)
    return evidence


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("bind", "extract"))
    args = parser.parse_args(argv)
    print(json.dumps(bind() if args.command == "bind" else extract(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
