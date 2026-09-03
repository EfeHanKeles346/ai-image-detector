"""Bind and extract the score-blind E47 TrueFake CAL/DEVELOPMENT population."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import tarfile
from typing import Any, Iterable, Mapping, Sequence

from experiments.e46_manifests import (
    SYNTH_AUDITED,
    TRUE_ARCHIVE,
    TRUE_CONTRACT,
    TRUEFAKE_SHA256,
    _all_protected,
    _decode,
    _digest,
    _rows,
    _write_atomic,
    parse_truefake_member,
)
from pixelproof.project_paths import DATA_ROOT, ML_ROOT


ROOT = DATA_ROOT / "e47"
NAMESPACE = "E47_TRUEFAKE_CALDEV_V1"
CONTRACT = ROOT / "caldev_selection_contract.json"
POOL = ROOT / "caldev_candidate_pool"
MANIFEST = ROOT / "caldev_manifest_unscored.json"
CONTRACT_EVIDENCE = ML_ROOT.parent / "evidence" / "e47_caldev_contract.json"
MANIFEST_EVIDENCE = ML_ROOT.parent / "evidence" / "e47_caldev_manifest.json"
OLD_FINAL = DATA_ROOT / "e46" / "truefake_facebook" / "final_manifest_unscored.json"

QUOTAS = {
    ("CAL", "FFHQ"): (0, 600),
    ("CAL", "StableDiffusion1.5"): (1, 200),
    ("CAL", "StableDiffusionXL"): (1, 200),
    ("CAL", "StyleGAN2"): (1, 200),
    ("DEVELOPMENT", "FORLAB"): (0, 600),
    ("DEVELOPMENT", "FLUX.1"): (1, 100),
    ("DEVELOPMENT", "StableDiffusion3"): (1, 100),
    ("DEVELOPMENT", "StyleGAN"): (1, 200),
    ("DEVELOPMENT", "StyleGAN3"): (1, 200),
}
RESERVE_FACTOR = 1.20


def _rank(role: str, source: str, member: str) -> str:
    return hashlib.sha256(f"{NAMESPACE}|{role}|{source}|{member}".encode()).hexdigest()


def _old_reserve_members() -> set[str]:
    payload = json.loads(TRUE_CONTRACT.read_text())
    rows = payload.get("selection", {}).get("candidate_rows", [])
    members = {str(row["member"]) for row in rows}
    if len(members) != 3_500:
        raise ValueError("E46 reserve identity changed")
    return members


def _candidate_rows() -> list[dict[str, Any]]:
    excluded = _old_reserve_members()
    by_key: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    with tarfile.open(TRUE_ARCHIVE, "r:gz") as bundle:
        for member in bundle.getmembers():
            if member.isdir():
                continue
            label, source, content = parse_truefake_member(member.name)
            for role in ("CAL", "DEVELOPMENT"):
                key = (role, source)
                if key not in QUOTAS or member.name in excluded:
                    continue
                expected_label, _ = QUOTAS[key]
                if label != expected_label:
                    raise ValueError(f"E47 label contract changed: {member.name}")
                by_key[key].append({
                    "member": member.name,
                    "bytes": member.size,
                    "label": label,
                    "source": source,
                    "content": content,
                    "role": role,
                    "rank": _rank(role, source, member.name),
                })
    chosen = []
    for key, (_, quota) in QUOTAS.items():
        reserve = int(quota * RESERVE_FACTOR)
        ordered = sorted(by_key[key], key=lambda row: row["rank"])
        if len(ordered) < reserve:
            raise ValueError(f"insufficient E47 candidates for {key}")
        chosen.extend({**row, "quota": quota, "rank_index": index}
                      for index, row in enumerate(ordered[:reserve]))
    return sorted(chosen, key=lambda row: (row["role"], row["source"], row["rank_index"]))


def bind() -> dict[str, Any]:
    if CONTRACT.exists() or CONTRACT_EVIDENCE.exists():
        raise FileExistsError("E47 CAL/DEVELOPMENT contract already exists")
    if _digest(TRUE_ARCHIVE) != TRUEFAKE_SHA256:
        raise ValueError("TrueFake archive changed before E47 CAL/DEVELOPMENT binding")
    candidates = _candidate_rows()
    old_contract_sha = _digest(TRUE_CONTRACT)
    payload = {
        "schema_version": 1,
        "state": "e47_caldev_selection_frozen_unscored",
        "namespace": NAMESPACE,
        "archive_sha256": TRUEFAKE_SHA256,
        "excluded_e46_reserve_contract_sha256": old_contract_sha,
        "excluded_e46_reserve_members": 3_500,
        "selection_rule": "lowest namespace SHA-256 rank outside E46 reserve; 20% audit headroom",
        "quotas": {f"{role}:{source}": {"label": label, "target": quota,
                                        "reserve": int(quota * RESERVE_FACTOR)}
                   for (role, source), (label, quota) in sorted(QUOTAS.items())},
        "candidate_rows": candidates,
        "model_scores_created": 0,
        "boundary": "Archive inventory and member names only; no selected payload decoded or scored.",
    }
    raw = _write_atomic(CONTRACT, payload)
    evidence = {
        "schema_version": 1,
        "state": payload["state"],
        "namespace": NAMESPACE,
        "archive_sha256": TRUEFAKE_SHA256,
        "excluded_e46_reserve_members": 3_500,
        "candidate_rows": len(candidates),
        "target_rows": sum(quota for _, quota in QUOTAS.values()),
        "by_role": dict(sorted(Counter(row["role"] for row in candidates).items())),
        "contract_bytes": len(raw),
        "contract_sha256": hashlib.sha256(raw).hexdigest(),
        "model_scores_created": 0,
    }
    _write_atomic(CONTRACT_EVIDENCE, evidence)
    return evidence


def identity_exclusions(
    rows: Sequence[Mapping[str, Any]], prior_exact: set[str], prior_dhash: set[str]
) -> dict[str, list[str]]:
    reasons: dict[str, list[str]] = defaultdict(list)
    seen_exact: dict[str, str] = {}
    seen_dhash: dict[str, str] = {}
    for row in sorted(rows, key=lambda item: str(item["rank"])):
        record_id = str(row["record_id"])
        exact, dhash = str(row["sha256"]), str(row["dhash"])
        if exact in prior_exact:
            reasons[record_id].append("protected_exact_overlap")
        if dhash in prior_dhash:
            reasons[record_id].append("protected_dhash_overlap")
        if exact in seen_exact:
            reasons[record_id].append(f"internal_exact_duplicate_of:{seen_exact[exact]}")
        else:
            seen_exact[exact] = record_id
        if dhash in seen_dhash:
            reasons[record_id].append(f"internal_dhash_duplicate_of:{seen_dhash[dhash]}")
        else:
            seen_dhash[dhash] = record_id
    return {key: value for key, value in sorted(reasons.items())}


def extract() -> dict[str, Any]:
    if MANIFEST.exists() or MANIFEST_EVIDENCE.exists():
        raise FileExistsError("E47 CAL/DEVELOPMENT manifest already exists")
    receipt = json.loads(CONTRACT_EVIDENCE.read_text())
    contract_raw = CONTRACT.read_bytes()
    if hashlib.sha256(contract_raw).hexdigest() != receipt["contract_sha256"]:
        raise ValueError("E47 CAL/DEVELOPMENT contract changed")
    contract = json.loads(contract_raw)
    candidates = {row["member"]: row for row in contract["candidate_rows"]}
    decoded = []
    failures = []
    with tarfile.open(TRUE_ARCHIVE, "r|gz") as bundle:
        for member in bundle:
            candidate = candidates.get(member.name)
            if candidate is None:
                continue
            try:
                stream = bundle.extractfile(member)
                if stream is None:
                    raise ValueError("member payload unavailable")
                raw = stream.read()
                image = _decode(raw, member.name)
                target = POOL / member.name
                target.parent.mkdir(parents=True, exist_ok=True)
                temporary = target.with_suffix(target.suffix + ".part")
                temporary.write_bytes(raw)
                temporary.replace(target)
                decoded.append({**candidate, **image, "record_id": f"e47:caldev:{member.name}",
                                "path": str(target), "condition": "Facebook"})
            except (OSError, ValueError, tarfile.TarError) as error:
                failures.append({"member": member.name, "error": f"{type(error).__name__}: {error}"})
    if len(decoded) + len(failures) != len(candidates):
        raise ValueError("E47 candidate extraction coverage changed")

    prior_exact, prior_dhash, protected = _all_protected()
    for manifest_path in (SYNTH_AUDITED, OLD_FINAL):
        raw = manifest_path.read_bytes()
        for row in _rows(json.loads(raw)):
            prior_exact.add(str(row["sha256"]))
            prior_dhash.add(str(row["dhash"]))
        protected.append({"path": str(manifest_path), "sha256": hashlib.sha256(raw).hexdigest()})
    reasons = identity_exclusions(decoded, prior_exact, prior_dhash)
    excluded = set(reasons)
    selected = []
    for key, (_, quota) in QUOTAS.items():
        role, source = key
        eligible = sorted((row for row in decoded if row["role"] == role and row["source"] == source
                           and row["record_id"] not in excluded), key=lambda row: row["rank_index"])
        if len(eligible) < quota:
            raise ValueError(f"insufficient clean E47 candidates for {key}")
        selected.extend(eligible[:quota])
    selected.sort(key=lambda row: (row["role"], row["label"], row["source"], row["rank_index"]))
    payload = {
        "schema_version": 1,
        "state": "e47_caldev_decontaminated_frozen_unscored",
        "archive_sha256": TRUEFAKE_SHA256,
        "selection_contract_sha256": receipt["contract_sha256"],
        "counts": {
            "candidate_rows": len(candidates),
            "decoded": len(decoded),
            "decode_failures": len(failures),
            "identity_exclusions": len(excluded),
            "selected_rows": len(selected),
            "by_role": dict(sorted(Counter(row["role"] for row in selected).items())),
            "by_role_label": {f"{role}:{label}": count for (role, label), count in sorted(
                Counter((row["role"], row["label"]) for row in selected).items())},
            "by_role_source": {f"{role}:{source}": count for (role, source), count in sorted(
                Counter((row["role"], row["source"]) for row in selected).items())},
        },
        "decode_failures": failures,
        "identity_exclusion_reasons": reasons,
        "protected_manifests": protected,
        "rows": selected,
        "model_scores_created": 0,
        "boundary": "All CAL/DEVELOPMENT rows decoded and identity-audited before model access.",
    }
    raw = _write_atomic(MANIFEST, payload)
    evidence = {
        "schema_version": 1,
        "state": payload["state"],
        "archive_sha256": TRUEFAKE_SHA256,
        "counts": payload["counts"],
        "protected_manifest_count": len(protected),
        "manifest_bytes": len(raw),
        "manifest_sha256": hashlib.sha256(raw).hexdigest(),
        "model_scores_created": 0,
    }
    _write_atomic(MANIFEST_EVIDENCE, evidence)
    return evidence


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("bind", "extract"))
    args = parser.parse_args(argv)
    result = bind() if args.command == "bind" else extract()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
