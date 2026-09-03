"""Build score-blind E46 identity audits and the TrueFake final manifest."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
from io import BytesIO
import json
from pathlib import Path, PurePosixPath
import subprocess
import tarfile
from typing import Any, Iterable, Mapping, Sequence

from PIL import Image

from experiments.e45_mediaeval_manifest import protected_hashes
from pixelproof.data_contract import dhash_image
from pixelproof.project_paths import DATA_ROOT, ML_ROOT


TRUEFAKE_BYTES = 4_207_525_545
TRUEFAKE_SHA256 = "413cb7f9664cf5f4e37a2ae0bea5d1a999c47398ca1c267a2173e88c2cda0d63"
TRUEFAKE_SOURCES = {
    "FLUX.1": (1, 5_000, 125, 250),
    "StableDiffusion1.5": (1, 5_000, 125, 250),
    "StableDiffusion2": (1, 5_000, 125, 250),
    "StableDiffusion3": (1, 5_000, 125, 250),
    "StableDiffusionXL": (1, 5_000, 125, 250),
    "StyleGAN": (1, 5_000, 125, 250),
    "StyleGAN2": (1, 5_000, 125, 250),
    "StyleGAN3": (1, 5_000, 125, 250),
    "FFHQ": (0, 10_000, 500, 750),
    "FORLAB": (0, 10_000, 500, 750),
}
TRUEFAKE_NAMESPACE = "E46_TRUEFAKE_FACEBOOK_FINAL_V1"
MAX_MEMBER_BYTES = 100 * 1024**2
MAX_PIXELS = 50_000_000

ROOT = DATA_ROOT / "e46"
SYNTH_ROOT = ROOT / "synthwildx"
SYNTH_RAW = SYNTH_ROOT / "manifest_unscored.json"
SYNTH_RECEIPT = SYNTH_ROOT / "acquisition_receipt.json"
SYNTH_AUDITED = SYNTH_ROOT / "audited_manifest_unscored.json"
TRUE_ROOT = ROOT / "truefake_facebook"
TRUE_ARCHIVE = TRUE_ROOT / "Facebook.tar.gz"
TRUE_RECEIPT = TRUE_ROOT / "acquisition_receipt.json"
TRUE_CONTRACT = TRUE_ROOT / "final_selection_contract.json"
TRUE_POOL = TRUE_ROOT / "candidate_pool"
TRUE_MANIFEST = TRUE_ROOT / "final_manifest_unscored.json"
SYNTH_EVIDENCE = ML_ROOT.parent / "evidence" / "e46_synthwildx_manifest.json"
TRUE_CONTRACT_EVIDENCE = ML_ROOT.parent / "evidence" / "e46_truefake_contract.json"
TRUE_MANIFEST_EVIDENCE = ML_ROOT.parent / "evidence" / "e46_truefake_manifest.json"


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def _write_atomic(path: Path, value: Any) -> bytes:
    raw = _json_bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_bytes(raw)
    temporary.replace(path)
    return raw


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024**2):
            digest.update(chunk)
    return digest.hexdigest()


def _rows(payload: Mapping[str, Any]) -> Sequence[Mapping[str, Any]]:
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise ValueError("manifest rows missing")
    return rows


def _all_protected() -> tuple[set[str], set[str], list[dict[str, Any]]]:
    exact, dhashes, sources = protected_hashes()
    e45 = DATA_ROOT / "e45_mediaeval_itwsm" / "unscored_manifest.json"
    if not e45.is_file():
        raise FileNotFoundError("consumed E45 manifest missing")
    raw = e45.read_bytes()
    for row in _rows(json.loads(raw)):
        exact.add(str(row["sha256"]))
        dhashes.add(str(row["dhash"]))
    sources.append({"path": str(e45), "sha256": hashlib.sha256(raw).hexdigest()})
    return exact, dhashes, sources


def _identity_audit(rows: Sequence[Mapping[str, Any]], prior_exact: set[str], prior_dhash: set[str]) -> dict[str, Any]:
    reasons: dict[str, set[str]] = defaultdict(set)
    exact_groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    dhash_groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        record_id = str(row["record_id"])
        exact = str(row["sha256"])
        dhash = str(row["dhash"])
        exact_groups[exact].append(row)
        dhash_groups[dhash].append(row)
        if exact in prior_exact:
            reasons[record_id].add("protected_exact_overlap")
        if dhash in prior_dhash:
            reasons[record_id].add("protected_dhash_overlap")
    exact_duplicates = []
    for digest, versions in exact_groups.items():
        if len(versions) < 2:
            continue
        ordered = sorted(versions, key=lambda row: str(row["record_id"]))
        labels = sorted({int(row["label"]) for row in versions})
        exact_duplicates.append({"sha256": digest, "record_ids": [row["record_id"] for row in ordered],
                                 "labels": labels})
        if len(labels) > 1:
            for row in ordered:
                reasons[str(row["record_id"])].add("cross_label_exact_duplicate")
        else:
            for row in ordered[1:]:
                reasons[str(row["record_id"])].add(f"same_label_exact_duplicate_of:{ordered[0]['record_id']}")
    dhash_diagnostic = [
        {"dhash": digest, "record_ids": sorted(str(row["record_id"]) for row in versions),
         "labels": sorted({int(row["label"]) for row in versions})}
        for digest, versions in dhash_groups.items() if len(versions) > 1
    ]
    return {
        "excluded_record_ids": sorted(reasons),
        "exclusion_reasons": {key: sorted(value) for key, value in sorted(reasons.items())},
        "exact_duplicate_groups": sorted(exact_duplicates, key=lambda item: item["sha256"]),
        "dhash_duplicate_diagnostic": sorted(dhash_diagnostic, key=lambda item: item["dhash"]),
    }


def audit_synthwildx() -> dict[str, Any]:
    if SYNTH_AUDITED.exists() or SYNTH_EVIDENCE.exists():
        raise FileExistsError("SynthWildX audited manifest already exists")
    receipt = json.loads(SYNTH_RECEIPT.read_text())
    raw = SYNTH_RAW.read_bytes()
    if hashlib.sha256(raw).hexdigest() != receipt["manifest_sha256"]:
        raise ValueError("SynthWildX manifest changed")
    candidates = []
    for row in _rows(json.loads(raw)):
        if row["status"] != "ok":
            continue
        path = Path(str(row["path"]))
        if not path.is_file() or _digest(path) != row["sha256"]:
            raise ValueError(f"SynthWildX payload changed: {path}")
        candidates.append({**row, "record_id": f"e46:synthwildx:{row['filename']}"})
    prior_exact, prior_dhash, protected = _all_protected()
    audit = _identity_audit(candidates, prior_exact, prior_dhash)
    excluded = set(audit["excluded_record_ids"])
    selected = [row for row in candidates if row["record_id"] not in excluded]
    payload = {
        "schema_version": 1,
        "state": "e46_synthwildx_audited_unscored",
        "role": "E46_CAL_DEV",
        "source_manifest_sha256": receipt["manifest_sha256"],
        "counts": {
            "declared": receipt["declared_rows"], "downloaded": len(candidates),
            "selected": len(selected), "excluded": len(candidates) - len(selected),
            "by_role": dict(sorted(Counter(row["role"] for row in selected).items())),
            "by_type": dict(sorted(Counter(row["typ"] for row in selected).items())),
            "by_type_role": {f"{typ}:{role}": count for (typ, role), count in sorted(
                Counter((row["typ"], row["role"]) for row in selected).items())},
        },
        "protected_manifests": protected,
        "audit": audit,
        "rows": selected,
        "model_scores_created": 0,
        "boundary": "Identity and payload audit only; CAL/DEVELOPMENT roles unchanged; no score exists.",
    }
    manifest_raw = _write_atomic(SYNTH_AUDITED, payload)
    evidence = {
        "schema_version": 1, "state": payload["state"], "role": payload["role"],
        "counts": payload["counts"],
        "audit_counts": {"exact_groups": len(audit["exact_duplicate_groups"]),
                         "dhash_groups": len(audit["dhash_duplicate_diagnostic"]),
                         "excluded": len(excluded)},
        "protected_manifest_count": len(protected),
        "manifest_bytes": len(manifest_raw),
        "manifest_sha256": hashlib.sha256(manifest_raw).hexdigest(),
        "model_scores_created": 0,
    }
    _write_atomic(SYNTH_EVIDENCE, evidence)
    return evidence


def parse_truefake_member(name: str) -> tuple[int, str, str]:
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or path.suffix.lower() != ".jpg":
        raise ValueError(f"unsafe TrueFake member: {name!r}")
    parts = path.parts
    if len(parts) == 4 and parts[:2] == ("Facebook", "Real") and parts[2] in {"FFHQ", "FORLAB"}:
        return 0, parts[2], "real"
    if len(parts) in {4, 5} and parts[:2] == ("Facebook", "Fake") and parts[2] in TRUEFAKE_SOURCES:
        return 1, parts[2], parts[3] if len(parts) == 5 else "faces"
    raise ValueError(f"unexpected TrueFake member: {name!r}")


def _rank(name: str, source: str) -> str:
    return hashlib.sha256(f"{TRUEFAKE_NAMESPACE}|{source}|{name}".encode()).hexdigest()


def bind_truefake() -> dict[str, Any]:
    if TRUE_CONTRACT.exists() or TRUE_CONTRACT_EVIDENCE.exists():
        raise FileExistsError("TrueFake selection contract already exists")
    receipt_raw = TRUE_RECEIPT.read_bytes()
    receipt = json.loads(receipt_raw)
    if TRUE_ARCHIVE.stat().st_size != TRUEFAKE_BYTES or receipt["sha256"] != TRUEFAKE_SHA256:
        raise ValueError("TrueFake receipt identity changed")
    if _digest(TRUE_ARCHIVE) != TRUEFAKE_SHA256:
        raise ValueError("TrueFake archive changed")
    subprocess.run(["/usr/bin/gzip", "-t", str(TRUE_ARCHIVE)], check=True)
    facts = []
    counts: Counter[str] = Counter()
    with tarfile.open(TRUE_ARCHIVE, "r:gz") as bundle:
        for member in bundle.getmembers():
            if member.isdir():
                continue
            if not member.isfile() or member.issym() or member.islnk():
                raise ValueError(f"unsafe TrueFake TAR entry: {member.name!r}")
            label, source, content = parse_truefake_member(member.name)
            if member.size <= 0 or member.size > MAX_MEMBER_BYTES:
                raise ValueError(f"unsafe TrueFake member size: {member.name!r}")
            counts[source] += 1
            facts.append({"member": member.name, "bytes": member.size, "label": label,
                          "source": source, "content": content, "rank": _rank(member.name, source)})
    for source, (_, expected, _, _) in TRUEFAKE_SOURCES.items():
        if counts[source] != expected:
            raise ValueError(f"TrueFake count changed for {source}: {counts[source]}")
    inventory_sha = hashlib.sha256(_json_bytes(facts)).hexdigest()
    candidates = []
    for source, (_, _, quota, reserve) in TRUEFAKE_SOURCES.items():
        ordered = sorted((row for row in facts if row["source"] == source), key=lambda row: row["rank"])
        candidates.extend({**row, "quota": quota, "rank_index": index}
                          for index, row in enumerate(ordered[:reserve]))
    candidates.sort(key=lambda row: (row["source"], row["rank_index"]))
    payload = {
        "schema_version": 1,
        "state": "e46_truefake_final_selection_frozen_unscored",
        "role": "E46_UNTOUCHED_FINAL",
        "archive_sha256": TRUEFAKE_SHA256,
        "receipt_sha256": hashlib.sha256(receipt_raw).hexdigest(),
        "gzip_integrity_passed": True,
        "inventory": {"files": len(facts), "by_source": dict(sorted(counts.items())),
                      "facts_sha256": inventory_sha},
        "selection": {
            "namespace": TRUEFAKE_NAMESPACE,
            "rule": "lowest SHA-256 ranks that decode and avoid protected/exact overlap",
            "target_rows": 2_000,
            "target_real": 1_000,
            "target_ai": 1_000,
            "candidate_pool_rows": len(candidates),
            "source_quotas": {source: quota for source, (_, _, quota, _) in TRUEFAKE_SOURCES.items()},
            "candidate_rows": candidates,
        },
        "model_scores_created": 0,
        "boundary": "Archive integrity, inventory and score-blind rank contract only; no image extracted or scored.",
    }
    contract_raw = _write_atomic(TRUE_CONTRACT, payload)
    evidence = {
        "schema_version": 1, "state": payload["state"], "role": payload["role"],
        "archive_sha256": TRUEFAKE_SHA256, "gzip_integrity_passed": True,
        "inventory": payload["inventory"],
        "selection": {key: value for key, value in payload["selection"].items() if key != "candidate_rows"},
        "contract_bytes": len(contract_raw),
        "contract_sha256": hashlib.sha256(contract_raw).hexdigest(),
        "model_scores_created": 0,
    }
    _write_atomic(TRUE_CONTRACT_EVIDENCE, evidence)
    return evidence


def _decode(raw: bytes, member: str) -> dict[str, Any]:
    if not raw or len(raw) > MAX_MEMBER_BYTES:
        raise ValueError(f"unsafe TrueFake payload: {member}")
    with Image.open(BytesIO(raw)) as opened:
        opened.verify()
    with Image.open(BytesIO(raw)) as opened:
        width, height = opened.size
        if width <= 0 or height <= 0 or width * height > MAX_PIXELS:
            raise ValueError(f"unsafe TrueFake geometry: {member}")
        decoded_format = str(opened.format)
        rgb = opened.convert("RGB")
        rgb.load()
        dhash = dhash_image(rgb)
    return {"bytes": len(raw), "width": width, "height": height, "decoded_format": decoded_format,
            "sha256": hashlib.sha256(raw).hexdigest(), "dhash": dhash}


def extract_truefake() -> dict[str, Any]:
    if TRUE_MANIFEST.exists() or TRUE_MANIFEST_EVIDENCE.exists():
        raise FileExistsError("TrueFake final manifest already exists")
    evidence = json.loads(TRUE_CONTRACT_EVIDENCE.read_text())
    contract_raw = TRUE_CONTRACT.read_bytes()
    if hashlib.sha256(contract_raw).hexdigest() != evidence["contract_sha256"]:
        raise ValueError("TrueFake selection contract changed")
    contract = json.loads(contract_raw)
    candidates = {row["member"]: row for row in contract["selection"]["candidate_rows"]}
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
                target = TRUE_POOL / member.name
                target.parent.mkdir(parents=True, exist_ok=True)
                partial = target.with_suffix(target.suffix + ".part")
                partial.write_bytes(raw)
                partial.replace(target)
                decoded.append({**candidate, **image, "record_id": f"e46:truefake:{member.name}",
                                "path": str(target), "condition": "Facebook"})
            except (OSError, ValueError, tarfile.TarError) as error:
                failures.append({"member": member.name, "error": f"{type(error).__name__}: {error}"})
    if len(decoded) + len(failures) != len(candidates):
        raise ValueError("TrueFake candidate extraction coverage changed")

    prior_exact, prior_dhash, protected = _all_protected()
    synth_raw = SYNTH_AUDITED.read_bytes()
    for row in _rows(json.loads(synth_raw)):
        prior_exact.add(str(row["sha256"]))
        prior_dhash.add(str(row["dhash"]))
    protected.append({"path": str(SYNTH_AUDITED), "sha256": hashlib.sha256(synth_raw).hexdigest()})
    audit = _identity_audit(decoded, prior_exact, prior_dhash)
    excluded = set(audit["excluded_record_ids"])
    selected = []
    for source, (_, _, quota, _) in TRUEFAKE_SOURCES.items():
        eligible = sorted((row for row in decoded if row["source"] == source and row["record_id"] not in excluded),
                          key=lambda row: row["rank_index"])
        if len(eligible) < quota:
            raise ValueError(f"insufficient clean TrueFake candidates for {source}")
        selected.extend(eligible[:quota])
    selected.sort(key=lambda row: (row["label"], row["source"], row["rank_index"]))
    payload = {
        "schema_version": 1,
        "state": "e46_truefake_decontaminated_final_frozen_unscored",
        "role": "E46_UNTOUCHED_FINAL",
        "archive_sha256": TRUEFAKE_SHA256,
        "selection_contract_sha256": evidence["contract_sha256"],
        "synthwildx_manifest_sha256": hashlib.sha256(synth_raw).hexdigest(),
        "counts": {
            "candidate_rows": len(candidates), "decoded_candidates": len(decoded),
            "decode_failures": len(failures), "audit_exclusions": len(excluded),
            "selected_rows": len(selected),
            "selected_by_label": dict(sorted(Counter(str(row["label"]) for row in selected).items())),
            "selected_by_source": dict(sorted(Counter(row["source"] for row in selected).items())),
        },
        "decode_failures": failures,
        "protected_manifests": protected,
        "audit": audit,
        "rows": selected,
        "model_scores_created": 0,
        "boundary": "All final rows decoded and decontaminated before model access; no score exists.",
    }
    manifest_raw = _write_atomic(TRUE_MANIFEST, payload)
    result = {
        "schema_version": 1, "state": payload["state"], "role": payload["role"],
        "archive_sha256": TRUEFAKE_SHA256, "counts": payload["counts"],
        "audit_counts": {"exact_groups": len(audit["exact_duplicate_groups"]),
                         "dhash_groups": len(audit["dhash_duplicate_diagnostic"]),
                         "excluded": len(excluded)},
        "protected_manifest_count": len(protected),
        "manifest_bytes": len(manifest_raw),
        "manifest_sha256": hashlib.sha256(manifest_raw).hexdigest(),
        "model_scores_created": 0,
    }
    _write_atomic(TRUE_MANIFEST_EVIDENCE, result)
    return result


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("audit-synthwildx", "bind-truefake", "extract-truefake"))
    args = parser.parse_args(argv)
    result = {"audit-synthwildx": audit_synthwildx, "bind-truefake": bind_truefake,
              "extract-truefake": extract_truefake}[args.command]()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
