"""Freeze and score the one-shot E42 RRDataset external robustness test."""

from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import joblib
import numpy as np
from PIL import Image

from experiments.e33_r1c import scenario_from_filename
from experiments.e42_features import BLOCKS, MODEL_IDS, aggregate_tokens, texture_crops, transport_image
from pixelproof.benchmark_metrics import evaluate_binary_scores
from pixelproof.data_contract import dhash_image
from pixelproof.e32_candidate import DINO_REPO_ID, DINO_WEIGHT_SHA256
from pixelproof.project_paths import DATA_ROOT, ML_ROOT


REPO_ROOT = ML_ROOT.parent
RR_ROOT = DATA_ROOT / "e33_rrdataset"
TEST_ROOT = RR_ROOT / "test"
E42_ROOT = DATA_ROOT / "e42"
CANDIDATE = E42_ROOT / "e42_small.joblib"
CANDIDATE_SHA256 = "6768466ab6eaaae6ae3ee00c784b13ff337955ccb38a5c2db09662939f9062e7"
THRESHOLD = 0.6600460410118104
ROOT_CONTRACT = REPO_ROOT / "evidence" / "e42_rr_final_contract.json"
ROOT_CONTRACT_SHA256 = "bc986c83bb4b1f04c5f2f6b2097682999cdcf8ee77bfc22780f5ffd7699e080b"
MANIFEST = RR_ROOT / "e42_rr_unscored_manifest.json"
MANIFEST_EVIDENCE = REPO_ROOT / "evidence" / "e42_rr_manifest.json"
SCORE_CONTRACT = RR_ROOT / "e42_rr_score_contract.json"
SCORE_CONTRACT_EVIDENCE = REPO_ROOT / "evidence" / "e42_rr_score_contract.json"
SCORES = RR_ROOT / "e42_rr_scores.jsonl"
REPORT = RR_ROOT / "e42_rr_report.json"
RESULT_EVIDENCE = REPO_ROOT / "evidence" / "e42_rr_result.json"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
CONDITIONS = ("original", "transfer", "redigital")
CLASS_LABELS = {"real": 0, "ai": 1}
MIN_CONDITION_CLASS = 20


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


def canonical_stem(filename: str, condition: str) -> str:
    """Map a declared RR derivative filename back to its original parent stem."""
    if condition not in CONDITIONS:
        raise ValueError(f"unknown RR condition: {condition!r}")
    stem = Path(filename).stem
    for prefix in ("original_", "transfer_", "redigital_"):
        if stem.startswith(prefix):
            stem = stem[len(prefix):]
            break
    if not stem:
        raise ValueError(f"empty RR parent stem: {filename!r}")
    return stem


def source_from_row(class_name: str, stem: str) -> str:
    if class_name not in CLASS_LABELS:
        raise ValueError(f"unknown RR class: {class_name!r}")
    return scenario_from_filename(f"{stem}.jpg")


def _protected_hashes() -> tuple[set[str], set[str], list[str]]:
    exact: set[str] = set()
    perceptual: set[str] = set()
    paths = [E42_ROOT / "parent_manifest.json"]
    bfree = DATA_ROOT / "e42_external" / "bfree_viral" / "unscored_manifest.json"
    if bfree.is_file():
        paths.append(bfree)
    consumed = []
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"protected E42 manifest missing: {path}")
        payload = json.loads(path.read_text())
        for row in payload.get("rows", payload.get("records", [])):
            if row.get("sha256"):
                exact.add(str(row["sha256"]))
            if row.get("dhash"):
                perceptual.add(str(row["dhash"]))
        consumed.append(str(path))
    return exact, perceptual, consumed


def audit_rows(
    rows: Sequence[Mapping[str, Any]], prior_exact: set[str], prior_dhash: set[str]
) -> dict[str, Any]:
    by_parent: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    by_exact: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    by_dhash: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_parent[str(row["parent_id"])].append(row)
        by_exact[str(row["sha256"])].append(row)
        by_dhash[str(row["dhash"])].append(row)

    duplicate_parent_conditions = []
    label_crossing_parents = []
    condition_sets: dict[str, int] = defaultdict(int)
    for parent, versions in by_parent.items():
        conditions = [str(row["condition"]) for row in versions]
        if len(conditions) != len(set(conditions)):
            duplicate_parent_conditions.append(parent)
        if len({int(row["label"]) for row in versions}) != 1:
            label_crossing_parents.append(parent)
        condition_sets["+".join(sorted(set(conditions)))] += 1

    def cross_parent(groups: Mapping[str, Sequence[Mapping[str, Any]]]) -> list[dict[str, Any]]:
        result = []
        for value, versions in groups.items():
            parents = sorted({str(row["parent_id"]) for row in versions})
            labels = sorted({int(row["label"]) for row in versions})
            if len(parents) > 1:
                result.append({"hash": value, "parents": parents, "labels": labels})
        return result

    exact_cross = cross_parent(by_exact)
    dhash_cross = cross_parent(by_dhash)
    prior_exact_rows = sorted(str(row["record_id"]) for row in rows if str(row["sha256"]) in prior_exact)
    prior_dhash_rows = sorted(str(row["record_id"]) for row in rows if str(row["dhash"]) in prior_dhash)
    prior_parents = {
        str(row["parent_id"])
        for row in rows
        if str(row["sha256"]) in prior_exact or str(row["dhash"]) in prior_dhash
    }

    adjacency: dict[str, set[str]] = {parent: set() for parent in by_parent}
    cross_label_exact = []
    for group in exact_cross:
        parents = [str(parent) for parent in group["parents"]]
        if len(group["labels"]) > 1:
            cross_label_exact.append(group)
        for parent in parents:
            adjacency[parent].update(other for other in parents if other != parent)

    components = []
    seen: set[str] = set()
    for parent in sorted(adjacency):
        if parent in seen or not adjacency[parent]:
            continue
        stack = [parent]
        component: set[str] = set()
        while stack:
            current = stack.pop()
            if current in component:
                continue
            component.add(current)
            stack.extend(adjacency[current] - component)
        seen.update(component)
        components.append(sorted(component))

    exclusions: dict[str, list[str]] = defaultdict(list)
    for parent in sorted(prior_parents):
        exclusions[parent].append("protected_exact_or_dhash_overlap")
    for component in components:
        if set(component) & prior_parents:
            for parent in component:
                exclusions[parent].append("exact_duplicate_component_touches_protected_role")
        else:
            for parent in component[1:]:
                exclusions[parent].append(f"same_label_exact_duplicate_of:{component[0]}")

    failures = {
        "duplicate_parent_condition": duplicate_parent_conditions,
        "label_crossing_parent": label_crossing_parents,
        "cross_label_exact": cross_label_exact,
    }
    return {
        "passed": not any(failures.values()),
        "failures": failures,
        "prior_exact_overlap_records": prior_exact_rows,
        "prior_dhash_overlap_records": prior_dhash_rows,
        "exact_duplicate_components": components,
        "excluded_parent_ids": sorted(exclusions),
        "exclusion_reasons": {parent: sorted(set(reasons)) for parent, reasons in sorted(exclusions.items())},
        "condition_sets_by_parent": dict(sorted(condition_sets.items())),
        "cross_parent_exact_diagnostic_count": len(exact_cross),
        "cross_parent_exact_diagnostic": exact_cross,
        "cross_parent_dhash_diagnostic_count": len(dhash_cross),
        "cross_parent_dhash_diagnostic": dhash_cross,
    }


def build_manifest() -> dict[str, Any]:
    if MANIFEST.exists() or MANIFEST_EVIDENCE.exists():
        raise FileExistsError("RR unscored manifest already exists; no silent remanifest")
    if _digest(ROOT_CONTRACT) != ROOT_CONTRACT_SHA256:
        raise ValueError("E42 RR root contract changed")
    extraction_path = RR_ROOT / "test_extraction_receipt.json"
    extraction = json.loads(extraction_path.read_text())
    if extraction.get("state") != "test_extraction_complete_unscored":
        raise ValueError("RR test extraction is not complete and unscored")

    rows: list[dict[str, Any]] = []
    for condition in CONDITIONS:
        for class_name, label in CLASS_LABELS.items():
            class_root = TEST_ROOT / condition / class_name
            if not class_root.is_dir():
                raise FileNotFoundError(f"missing RR test class directory: {class_root}")
            for path in sorted(class_root.rglob("*")):
                if not path.is_file() or path.name.startswith("._") or path.suffix.lower() not in IMAGE_SUFFIXES:
                    continue
                relative = path.relative_to(TEST_ROOT)
                stem = canonical_stem(path.name, condition)
                source = source_from_row(class_name, stem)
                sha256 = _digest(path)
                with Image.open(path) as opened:
                    opened.load()
                    rgb = opened.convert("RGB")
                    width, height = rgb.size
                    perceptual = dhash_image(rgb)
                rows.append({
                    "record_id": f"rrtest:{relative.as_posix()}",
                    "parent_id": f"rrtest:{class_name}:{stem}",
                    "path": str(path),
                    "relative_path": relative.as_posix(),
                    "label": label,
                    "class_name": class_name,
                    "source": source,
                    "condition": condition,
                    "bytes": path.stat().st_size,
                    "width": width,
                    "height": height,
                    "sha256": sha256,
                    "dhash": perceptual,
                })
                if len(rows) % 1_000 == 0:
                    print(f"E42 RR manifest {len(rows)}/{extraction['image_count']}", flush=True)
    rows.sort(key=lambda row: row["record_id"])
    if len(rows) != int(extraction["image_count"]):
        raise ValueError(f"RR decoded row count changed: {len(rows)} != {extraction['image_count']}")
    if sum(int(row["bytes"]) for row in rows) != int(extraction["image_bytes"]):
        raise ValueError("RR decoded byte count changed")
    if len({str(row["record_id"]) for row in rows}) != len(rows):
        raise ValueError("RR record identifiers are not unique")

    prior_exact, prior_dhash, prior_manifests = _protected_hashes()
    audit = audit_rows(rows, prior_exact, prior_dhash)
    if not audit["passed"]:
        raise ValueError(f"RR final structural audit failed: {audit['failures']}")
    excluded_parents = set(audit["excluded_parent_ids"])
    selected = [row for row in rows if str(row["parent_id"]) not in excluded_parents]
    if any(
        str(row["sha256"]) in prior_exact or str(row["dhash"]) in prior_dhash
        for row in selected
    ):
        raise ValueError("RR decontamination left a protected-role overlap")
    by_condition_class = {
        f"{condition}/{class_name}": sum(
            row["condition"] == condition and row["class_name"] == class_name for row in selected
        )
        for condition in CONDITIONS
        for class_name in CLASS_LABELS
    }
    if min(by_condition_class.values()) < MIN_CONDITION_CLASS:
        raise ValueError(f"RR decontamination emptied a required condition/class: {by_condition_class}")
    official_counts = {
        "rows": len(rows),
        "parents": len({str(row["parent_id"]) for row in rows}),
        "image_bytes": sum(int(row["bytes"]) for row in rows),
    }
    payload = {
        "schema_version": 1,
        "experiment": "E42/RRDataset-one-shot-external-robustness",
        "state": "rr_final_manifest_frozen_unscored",
        "root_contract_sha256": ROOT_CONTRACT_SHA256,
        "candidate_sha256": CANDIDATE_SHA256,
        "threshold": THRESHOLD,
        "extraction_receipt_sha256": _digest(extraction_path),
        "official_archive_counts": official_counts,
        "counts": {
            "rows": len(selected),
            "parents": len({str(row["parent_id"]) for row in selected}),
            "image_bytes": sum(int(row["bytes"]) for row in selected),
            "by_condition_class": by_condition_class,
            "official_row_coverage_after_decontamination": len(selected) / len(rows),
            "excluded_parents": len(excluded_parents),
            "excluded_rows": len(rows) - len(selected),
        },
        "protected_manifests": prior_manifests,
        "protected_exact_hashes": len(prior_exact),
        "protected_dhashes": len(prior_dhash),
        "audit": audit,
        "rows": selected,
        "boundary": "Every official image was decoded and audited before model access; protected/duplicate parents were excluded as whole events before the scored manifest was declared.",
    }
    raw = _write_atomic(MANIFEST, payload)
    compact = {
        "schema_version": 1,
        "state": payload["state"],
        "candidate_sha256": CANDIDATE_SHA256,
        "threshold": THRESHOLD,
        "counts": payload["counts"],
        "audit": {
            "passed": audit["passed"],
            "condition_sets_by_parent": audit["condition_sets_by_parent"],
            "cross_parent_exact_diagnostic_count": audit["cross_parent_exact_diagnostic_count"],
            "cross_parent_dhash_diagnostic_count": audit["cross_parent_dhash_diagnostic_count"],
            "excluded_parent_count": len(excluded_parents),
        },
        "detailed_manifest_bytes": len(raw),
        "detailed_manifest_sha256": hashlib.sha256(raw).hexdigest(),
        "production_scores_created": 0,
    }
    _write_atomic(MANIFEST_EVIDENCE, compact)
    return compact


def bind_score_contract() -> dict[str, Any]:
    if SCORE_CONTRACT.exists() or SCORE_CONTRACT_EVIDENCE.exists():
        raise FileExistsError("RR score contract already exists; no silent rebinding")
    manifest = json.loads(MANIFEST.read_text())
    if manifest.get("state") != "rr_final_manifest_frozen_unscored" or not manifest.get("audit", {}).get("passed"):
        raise ValueError("RR unscored manifest is not eligible")
    if _digest(CANDIDATE) != CANDIDATE_SHA256:
        raise ValueError("E42 candidate changed before RR score binding")
    payload = {
        "schema_version": 1,
        "experiment": "E42/RRDataset-one-shot-external-robustness",
        "state": "rr_final_score_contract_frozen",
        "root_contract_sha256": ROOT_CONTRACT_SHA256,
        "manifest_sha256": _digest(MANIFEST),
        "candidate_sha256": CANDIDATE_SHA256,
        "threshold": THRESHOLD,
        "declared_rows": int(manifest["counts"]["rows"]),
        "decision": "one score per manifest row; original full gate; transfer and redigital AUC >=0.85 and balanced accuracy >=0.80; coverage 1.0",
        "forbidden": ["threshold change", "row removal", "retry after a completed score stream", "test-informed fitting"],
    }
    raw = _write_atomic(SCORE_CONTRACT, payload)
    evidence = {**payload, "detailed_contract_bytes": len(raw), "detailed_contract_sha256": hashlib.sha256(raw).hexdigest()}
    _write_atomic(SCORE_CONTRACT_EVIDENCE, evidence)
    return evidence


def source_rates(rows: Sequence[Mapping[str, Any]], threshold: float) -> dict[str, Any]:
    grouped: dict[tuple[int, str], list[bool]] = defaultdict(list)
    for row in rows:
        grouped[(int(row["label"]), str(row["source"]))].append(float(row["score"]) >= threshold)
    real = {source: float(np.mean(values)) for (label, source), values in sorted(grouped.items()) if label == 0}
    ai = {source: float(np.mean(values)) for (label, source), values in sorted(grouped.items()) if label == 1}
    if not real or not ai:
        raise ValueError("RR source rates require both classes")
    return {
        "real_fp_by_source": real,
        "real_macro_fp": float(np.mean(list(real.values()))),
        "real_worst_source_fp": max(real.values()),
        "ai_recall_by_source": ai,
        "ai_macro_recall": float(np.mean(list(ai.values()))),
        "ai_worst_source_recall": min(ai.values()),
    }


def final_gate(by_condition: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    clean = by_condition["original"]
    metrics = clean["metrics"]
    rates = clean["rates"]
    checks: dict[str, bool] = {
        "original_auc_gte_0_90": float(metrics["roc_auc"]) >= 0.90,
        "original_tpr_at_fpr10_gte_0_80": float(metrics["tpr_at_fpr"]["tpr"]) >= 0.80,
        "original_eer_lte_0_15": float(metrics["eer"]) <= 0.15,
        "original_balanced_accuracy_gte_0_85": float(metrics["balanced_accuracy"]) >= 0.85,
        "original_real_macro_fp_lte_0_10": float(rates["real_macro_fp"]) <= 0.10 + 1e-12,
        "original_real_worst_source_fp_lte_0_20": float(rates["real_worst_source_fp"]) <= 0.20 + 1e-12,
        "original_ai_macro_recall_gte_0_80": float(rates["ai_macro_recall"]) >= 0.80,
        "original_ai_worst_source_recall_gte_0_60": float(rates["ai_worst_source_recall"]) >= 0.60,
        "original_coverage_eq_1": float(metrics["coverage"]) == 1.0,
    }
    for condition in ("transfer", "redigital"):
        item = by_condition[condition]
        condition_metrics = item["metrics"]
        sufficiently_populated = min(
            int(condition_metrics["counts"]["real_succeeded"]),
            int(condition_metrics["counts"]["ai_succeeded"]),
        ) >= MIN_CONDITION_CLASS
        checks[f"{condition}_sufficient_population"] = sufficiently_populated
        checks[f"{condition}_auc_gte_0_85"] = sufficiently_populated and float(condition_metrics["roc_auc"]) >= 0.85
        checks[f"{condition}_balanced_accuracy_gte_0_80"] = sufficiently_populated and float(condition_metrics["balanced_accuracy"]) >= 0.80
        checks[f"{condition}_coverage_eq_1"] = float(condition_metrics["coverage"]) == 1.0
    return {"passed": all(checks.values()), "checks": checks}


def _prepare(row: Mapping[str, Any]) -> list[np.ndarray]:
    path = Path(str(row["path"]))
    if _digest(path) != str(row["sha256"]):
        raise ValueError(f"RR final image changed: {row['relative_path']}")
    with Image.open(path) as opened:
        return texture_crops(transport_image(opened, "clean"))


def _score(rows: Sequence[Mapping[str, Any]], head: Any, batch_views: int) -> list[dict[str, Any]]:
    from huggingface_hub import snapshot_download
    import timm
    import torch

    snapshot = Path(snapshot_download(DINO_REPO_ID, local_files_only=True))
    if _digest(snapshot / "model.safetensors") != DINO_WEIGHT_SHA256:
        raise ValueError("cached DINOv2-S weights changed")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = timm.create_model(MODEL_IDS["small"], pretrained=True, num_classes=0, img_size=224).to(device).eval()
    config = timm.data.resolve_data_config({}, model=model)
    mean = torch.tensor(config["mean"], device=device).view(1, 3, 1, 1)
    std = torch.tensor(config["std"], device=device).view(1, 3, 1, 1)
    output: list[dict[str, Any]] = []
    with torch.inference_mode(), ThreadPoolExecutor(max_workers=6) as pool:
        for start in range(0, len(rows), batch_views):
            group = rows[start : start + batch_views]
            arrays = [array for pack in pool.map(_prepare, group) for array in pack]
            tensor = torch.from_numpy(np.stack(arrays)).to(device)
            tensor = tensor.permute(0, 3, 1, 2).float().div_(255.0)
            intermediates = model.forward_intermediates(
                (tensor - mean) / std,
                indices=list(BLOCKS["small"]),
                return_prefix_tokens=True,
                norm=True,
                intermediates_only=True,
            )
            tokens = torch.stack([item[1][:, 0, :] for item in intermediates], dim=1)
            features = aggregate_tokens(tokens.float().cpu().numpy(), len(group))
            scores = head.predict_proba(features)[:, 1]
            for row, value in zip(group, scores, strict=True):
                score = float(value)
                if not np.isfinite(score):
                    raise ValueError(f"non-finite RR score: {row['relative_path']}")
                output.append({
                    "record_id": row["record_id"],
                    "parent_id": row["parent_id"],
                    "relative_path": row["relative_path"],
                    "label": int(row["label"]),
                    "source": row["source"],
                    "condition": row["condition"],
                    "status": "ok",
                    "score": score,
                })
            print(f"E42 RR score {min(start + batch_views, len(rows))}/{len(rows)}", flush=True)
    return output


def run(batch_views: int = 24) -> dict[str, Any]:
    if any(path.exists() for path in (SCORES, REPORT, RESULT_EVIDENCE)):
        raise FileExistsError("RR E42 result already exists; retry is forbidden")
    score_contract = json.loads(SCORE_CONTRACT.read_text())
    manifest = json.loads(MANIFEST.read_text())
    if (
        score_contract.get("state") != "rr_final_score_contract_frozen"
        or score_contract.get("manifest_sha256") != _digest(MANIFEST)
        or score_contract.get("candidate_sha256") != CANDIDATE_SHA256
        or int(score_contract.get("declared_rows", -1)) != len(manifest.get("rows", []))
        or float(score_contract.get("threshold", -1)) != THRESHOLD
    ):
        raise ValueError("RR score contract changed")
    if _digest(CANDIDATE) != CANDIDATE_SHA256:
        raise ValueError("E42 candidate changed")
    artifact = joblib.load(CANDIDATE)
    if (
        artifact.get("positive_label") != "ai"
        or artifact.get("model_id") != MODEL_IDS["small"]
        or tuple(artifact.get("block_indices", ())) != BLOCKS["small"]
        or float(artifact.get("threshold", -1)) != THRESHOLD
    ):
        raise ValueError("E42 artifact inference contract changed")

    scored = _score(manifest["rows"], artifact["head"], batch_views)
    raw_scores = b"".join((json.dumps(row, sort_keys=True) + "\n").encode() for row in scored)
    temporary = SCORES.with_suffix(SCORES.suffix + ".part")
    temporary.write_bytes(raw_scores)
    temporary.replace(SCORES)
    by_condition = {}
    for condition in CONDITIONS:
        selected = [row for row in scored if row["condition"] == condition]
        by_condition[condition] = {
            "metrics": evaluate_binary_scores(selected, threshold=THRESHOLD),
            "rates": source_rates(selected, THRESHOLD),
        }
    gate = final_gate(by_condition)
    report = {
        "schema_version": 1,
        "experiment": "E42/RRDataset-one-shot-external-robustness",
        "state": "external_final_passed" if gate["passed"] else "external_final_failed",
        "candidate_sha256": CANDIDATE_SHA256,
        "manifest_sha256": _digest(MANIFEST),
        "score_contract_sha256": _digest(SCORE_CONTRACT),
        "scores_sha256": hashlib.sha256(raw_scores).hexdigest(),
        "scores_bytes": len(raw_scores),
        "threshold": THRESHOLD,
        "counts": manifest["counts"],
        "by_condition": by_condition,
        "gate": gate,
        "limitations": [
            "RR REAL filenames expose one pooled source rather than camera identities.",
            "Transfer and redigital rows are derivatives and are reported by condition, not as independent parents.",
            "RR validation/train data were consumed earlier; this is external robustness transfer, not the sole universal final claim.",
        ],
        "boundary": "First and only completed E42 score stream on the frozen RR manifest; no retry or retuning.",
    }
    _write_atomic(REPORT, report)
    _write_atomic(RESULT_EVIDENCE, report)
    return report


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("build-manifest", "bind-score-contract", "score"))
    parser.add_argument("--batch-views", type=int, default=24)
    args = parser.parse_args(argv)
    if args.command == "build-manifest":
        result = build_manifest()
    elif args.command == "bind-score-contract":
        result = bind_score_contract()
    else:
        result = run(args.batch_views)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
