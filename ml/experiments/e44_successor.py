"""Validate the frozen E44-C successor on a newly DDA-scored local population."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import joblib
import numpy as np
from PIL import Image

from experiments.e42_features import MANIFEST as E42_MANIFEST, assigned_transport, transport_image
from experiments.e43_development import SCORES as GENERALIST_SCORES
from experiments.e43_features import MANIFEST as RR_MANIFEST
from experiments.e43_train import _digest
from experiments.e44_fusion import (
    CANDIDATE,
    CONTRACT as FUSION_CONTRACT,
    _feature,
    _rates,
)
from pixelproof.benchmark_metrics import evaluate_binary_scores
from pixelproof.dda_candidate import CHECKPOINT_SHA256, OfficialDDACandidate
from pixelproof.project_paths import DATA_ROOT, ML_ROOT


ROOT = DATA_ROOT / "e44"
CONTRACT = ROOT / "successor_contract.json"
CONTRACT_EVIDENCE = ML_ROOT.parent / "evidence" / "e44_successor_contract.json"
DDA_SCORES = ROOT / "successor_dda_scores.jsonl"
DDA_EVIDENCE = ML_ROOT.parent / "evidence" / "e44_successor_dda_scores.json"
REPORT = ROOT / "successor_development.json"
SCORES = ROOT / "successor_development_scores.jsonl"
RESULT_EVIDENCE = ML_ROOT.parent / "evidence" / "e44_successor_development.json"
FUSION_SHA256 = "19fd7bbcfed6ea85b9aa0c620663880f9fed24fbdbb084b09057283ea38bb100"
SUCCESSOR_THRESHOLD = 0.34779336534869326
EXPECTED_RR = 2_940
EXPECTED_E42 = 2_160
EXPECTED_ROWS = EXPECTED_RR + EXPECTED_E42


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def _write(path: Path, value: Any) -> bytes:
    raw = _json_bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_bytes(raw)
    temporary.replace(path)
    return raw


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> tuple[int, str]:
    raw = b"".join((json.dumps(row, sort_keys=True) + "\n").encode() for row in rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_bytes(raw)
    temporary.replace(path)
    return len(raw), hashlib.sha256(raw).hexdigest()


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def population_rows() -> list[dict[str, Any]]:
    rr_manifest = json.loads(RR_MANIFEST.read_text())
    rr = [
        {
            "record_id": str(row["record_id"]),
            "parent_id": str(row["parent_id"]),
            "path": str(row["path"]),
            "sha256": str(row["sha256"]),
            "label": int(row["label"]),
            "source": str(row["source"]),
            "condition": str(row["condition"]),
            "population": "rr",
            "transport": "pre_rendered",
        }
        for row in rr_manifest["rows"]
        if str(row["e43_role"]) == "development"
    ]
    e42_manifest = json.loads(E42_MANIFEST.read_text())
    parents = [
        row for row in e42_manifest["rows"]
        if str(row["role"]) == "development"
        and str(row["provenance"]) in {"e36_former_final_consumed", "e39_consumed"}
    ]
    e42 = []
    for row in parents:
        for condition in ("clean", assigned_transport(str(row["parent_id"]))):
            e42.append({
                "record_id": f"{row['parent_id']}|{condition}",
                "parent_id": str(row["parent_id"]),
                "path": str(row["path"]),
                "sha256": str(row["sha256"]),
                "label": int(row["label"]),
                "source": str(row["source"]),
                "condition": condition,
                "population": "e42",
                "transport": condition,
            })
    output = sorted(rr + e42, key=lambda row: row["record_id"])
    if len(rr) != EXPECTED_RR or len(e42) != EXPECTED_E42 or len(output) != EXPECTED_ROWS:
        raise ValueError("E44-C successor population count changed")
    if len({row["record_id"] for row in output}) != len(output):
        raise ValueError("E44-C record IDs are not unique")
    return output


def bind() -> dict[str, Any]:
    if CONTRACT.exists() or CONTRACT_EVIDENCE.exists():
        raise FileExistsError("E44-C contract already exists")
    if _digest(CANDIDATE) != FUSION_SHA256:
        raise ValueError("E44-B fusion artifact changed")
    fusion = joblib.load(CANDIDATE)
    if float(fusion["threshold"]) >= SUCCESSOR_THRESHOLD:
        raise ValueError("E44-C successor cut no longer exceeds E44-B cut")
    generalist_rows = _load_jsonl(GENERALIST_SCORES)
    generalist = {str(row["record_id"]): row for row in generalist_rows}
    rows = population_rows()
    if any(row["record_id"] not in generalist for row in rows):
        raise ValueError("E44-C rows do not join the frozen E43-S stream")
    prior = json.loads(FUSION_CONTRACT.read_text())
    prior_sha = {str(row["sha256"]) for row in prior["e35_rows"]}
    if any(row["sha256"] in prior_sha for row in rows):
        raise ValueError("E44-C exact-overlaps the E35 fusion population")
    verified: dict[str, str] = {}
    for row in rows:
        path = Path(row["path"])
        if row["path"] not in verified:
            verified[row["path"]] = _digest(path)
        if verified[row["path"]] != row["sha256"]:
            raise ValueError(f"E44-C input changed: {row['record_id']}")
    bound_rows = [{**row, "generalist_score": float(generalist[row["record_id"]]["score"])} for row in rows]
    payload = {
        "schema_version": 1,
        "state": "e44_successor_contract_frozen_before_dda_scores",
        "role": "new_comparative_development_not_final",
        "fusion_sha256": FUSION_SHA256,
        "successor_threshold": SUCCESSOR_THRESHOLD,
        "specialist_checkpoint_sha256": CHECKPOINT_SHA256,
        "inputs": {
            "rr_manifest_sha256": _digest(RR_MANIFEST),
            "e42_manifest_sha256": _digest(E42_MANIFEST),
            "generalist_scores_sha256": _digest(GENERALIST_SCORES),
        },
        "rows": bound_rows,
        "gates": {
            "coverage": 1.0,
            "pooled_auc_min": 0.90,
            "pooled_balanced_accuracy_min": 0.85,
            "rr_original_real_fp_max": 0.10,
            "rr_original_ai_macro_min": 0.80,
            "rr_original_ai_worst_min": 0.60,
            "e42_real_macro_fp_max": 0.10,
            "e42_real_worst_fp_max": 0.20,
            "e42_ai_macro_min": 0.80,
            "e42_ai_worst_min": 0.60,
        },
        "forbidden": ["threshold change", "row removal", "test-informed refit", "final claim"],
        "dda_scores_created": 0,
    }
    raw = _write(CONTRACT, payload)
    evidence = {
        "schema_version": 1,
        "state": payload["state"],
        "role": payload["role"],
        "fusion_sha256": FUSION_SHA256,
        "successor_threshold": SUCCESSOR_THRESHOLD,
        "specialist_checkpoint_sha256": CHECKPOINT_SHA256,
        "inputs": payload["inputs"],
        "counts": {"rr": EXPECTED_RR, "e42": EXPECTED_E42, "rows": EXPECTED_ROWS,
                   "unique_paths": len(verified)},
        "population_sha256": hashlib.sha256(_json_bytes(bound_rows)).hexdigest(),
        "detailed_contract_bytes": len(raw),
        "detailed_contract_sha256": hashlib.sha256(raw).hexdigest(),
        "dda_scores_created": 0,
        "forbidden": payload["forbidden"],
    }
    _write(CONTRACT_EVIDENCE, evidence)
    return evidence


def _prepared(row: Mapping[str, Any], candidate: OfficialDDACandidate):
    with Image.open(Path(str(row["path"]))) as opened:
        image = opened.convert("RGB")
        if str(row["population"]) == "e42":
            image = transport_image(image, str(row["transport"]))
        return candidate.transform(image)


def score_dda(batch_size: int = 2) -> dict[str, Any]:
    if DDA_SCORES.exists() or DDA_EVIDENCE.exists():
        raise FileExistsError("E44-C DDA scores already exist")
    contract = json.loads(CONTRACT.read_text())
    evidence = json.loads(CONTRACT_EVIDENCE.read_text())
    if _digest(CONTRACT) != evidence["detailed_contract_sha256"] or contract["dda_scores_created"] != 0:
        raise ValueError("E44-C contract changed")
    candidate = OfficialDDACandidate()
    import torch

    output = []
    rows = contract["rows"]
    with torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            group = rows[start:start + batch_size]
            tensors = [_prepared(row, candidate) for row in group]
            values = candidate.model(torch.stack(tensors).to(candidate.device)).sigmoid().flatten().cpu().numpy()
            for row, score in zip(group, values, strict=True):
                output.append({
                    **{key: row[key] for key in (
                        "record_id", "parent_id", "label", "source", "condition", "population",
                        "generalist_score")},
                    "specialist_score": float(score),
                    "status": "ok",
                })
            completed = min(start + batch_size, len(rows))
            if completed == len(rows) or completed // 100 != start // 100:
                print(f"E44-C DDA {completed}/{len(rows)}", flush=True)
    size, digest = _write_jsonl(DDA_SCORES, output)
    result = {"schema_version": 1, "state": "e44_successor_dda_scores_complete",
              "rows": len(output), "bytes": size, "sha256": digest, "coverage": 1.0}
    _write(DDA_EVIDENCE, result)
    return result


def _summarize(rows: Sequence[Mapping[str, Any]], threshold: float) -> dict[str, Any]:
    return {"metrics": evaluate_binary_scores(rows, threshold=threshold), "rates": _rates(rows, threshold)}


def gate(rows: Sequence[Mapping[str, Any]], threshold: float) -> dict[str, Any]:
    pooled = evaluate_binary_scores(rows, threshold=threshold)
    rr = [row for row in rows if row["population"] == "rr"]
    rr_conditions = {condition: _summarize([row for row in rr if row["condition"] == condition], threshold)
                     for condition in ("original", "transfer", "redigital")}
    e42 = [row for row in rows if row["population"] == "e42"]
    e42_clean = _summarize([row for row in e42 if row["condition"] == "clean"], threshold)
    e42_robust = _summarize([row for row in e42 if row["condition"] != "clean"], threshold)
    rr_original = rr_conditions["original"]
    checks = {
        "coverage_eq_1": float(pooled["coverage"]) == 1.0,
        "pooled_auc_gte_0_90": float(pooled["roc_auc"]) >= 0.90,
        "pooled_balanced_accuracy_gte_0_85": float(pooled["balanced_accuracy"]) >= 0.85,
        "rr_original_auc_gte_0_90": float(rr_original["metrics"]["roc_auc"]) >= 0.90,
        "rr_original_balanced_accuracy_gte_0_85": float(rr_original["metrics"]["balanced_accuracy"]) >= 0.85,
        "rr_original_real_fp_lte_0_10": rr_original["rates"]["real_macro_fp"] <= 0.10,
        "rr_original_ai_macro_gte_0_80": rr_original["rates"]["ai_macro_recall"] >= 0.80,
        "rr_original_ai_worst_gte_0_60": rr_original["rates"]["ai_worst_recall"] >= 0.60,
        "rr_transfer_balanced_accuracy_gte_0_80": float(rr_conditions["transfer"]["metrics"]["balanced_accuracy"]) >= 0.80,
        "rr_redigital_balanced_accuracy_gte_0_80": float(rr_conditions["redigital"]["metrics"]["balanced_accuracy"]) >= 0.80,
        "e42_clean_auc_gte_0_90": float(e42_clean["metrics"]["roc_auc"]) >= 0.90,
        "e42_clean_balanced_accuracy_gte_0_85": float(e42_clean["metrics"]["balanced_accuracy"]) >= 0.85,
        "e42_clean_real_macro_fp_lte_0_10": e42_clean["rates"]["real_macro_fp"] <= 0.10,
        "e42_clean_real_worst_fp_lte_0_20": e42_clean["rates"]["real_worst_fp"] <= 0.20,
        "e42_clean_ai_macro_gte_0_80": e42_clean["rates"]["ai_macro_recall"] >= 0.80,
        "e42_clean_ai_worst_gte_0_60": e42_clean["rates"]["ai_worst_recall"] >= 0.60,
        "e42_robust_auc_gte_0_85": float(e42_robust["metrics"]["roc_auc"]) >= 0.85,
        "e42_robust_balanced_accuracy_gte_0_80": float(e42_robust["metrics"]["balanced_accuracy"]) >= 0.80,
        "e42_robust_real_macro_fp_lte_0_10": e42_robust["rates"]["real_macro_fp"] <= 0.10,
        "e42_robust_real_worst_fp_lte_0_20": e42_robust["rates"]["real_worst_fp"] <= 0.20,
        "e42_robust_ai_macro_gte_0_80": e42_robust["rates"]["ai_macro_recall"] >= 0.80,
        "e42_robust_ai_worst_gte_0_60": e42_robust["rates"]["ai_worst_recall"] >= 0.60,
    }
    return {"passed": all(checks.values()), "checks": checks, "pooled": pooled,
            "rr_by_condition": rr_conditions, "e42_clean": e42_clean, "e42_robust": e42_robust}


def evaluate() -> dict[str, Any]:
    if any(path.exists() for path in (SCORES, REPORT, RESULT_EVIDENCE)):
        raise FileExistsError("E44-C result already exists")
    if _digest(CANDIDATE) != FUSION_SHA256:
        raise ValueError("E44-C fusion artifact changed")
    artifact = joblib.load(CANDIDATE)
    rows = _load_jsonl(DDA_SCORES)
    features = np.asarray([_feature(row["generalist_score"], row["specialist_score"]) for row in rows])
    values = artifact["head"].predict_proba(features)[:, 1]
    scored = [{**row, "score": float(value)} for row, value in zip(rows, values, strict=True)]
    size, digest = _write_jsonl(SCORES, scored)
    found = gate(scored, SUCCESSOR_THRESHOLD)
    report = {"schema_version": 1,
              "state": "e44_successor_development_passed" if found["passed"] else "e44_successor_development_failed",
              "fusion_sha256": FUSION_SHA256, "threshold": SUCCESSOR_THRESHOLD,
              "counts": {"rows": len(scored)}, "score_stream": {"bytes": size, "sha256": digest},
              "gate": found, "boundary": "New comparative DEVELOPMENT only; not independent final evidence."}
    raw = _write(REPORT, report)
    _write(RESULT_EVIDENCE, {**report, "detailed_report_sha256": hashlib.sha256(raw).hexdigest()})
    return report


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("bind", "score-dda", "evaluate"))
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args(argv)
    if args.command == "bind":
        result = bind()
    elif args.command == "score-dda":
        result = score_dda(args.batch_size)
    else:
        result = evaluate()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
