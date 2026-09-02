"""Score the frozen E43-S candidate once on consumed DEVELOPMENT populations."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import joblib
import numpy as np

from experiments.e42_train import source_rates
from experiments.e43_train import (
    E42_FEATURES,
    E42_FEATURES_SHA256,
    RR_FEATURES,
    RR_FEATURES_SHA256,
    _load,
)
from pixelproof.benchmark_metrics import evaluate_binary_scores
from pixelproof.project_paths import DATA_ROOT, ML_ROOT


ROOT = DATA_ROOT / "e43"
CANDIDATE = ROOT / "e43_small_predev.joblib"
CANDIDATE_SHA256 = "a3aec445926bcc8707b3775f01d2cdd9491ba8495ad8a8ec306840556ca47390"
THRESHOLD = 0.8712875247001649
SCORES = ROOT / "e43_small_development_scores.jsonl"
REPORT = ROOT / "e43_small_development.json"
EVIDENCE = ML_ROOT.parent / "evidence" / "e43_small_development.json"

# Frozen E42-S source-held-out baselines. The E43 no-material-regression floors below allow a
# maximum five-point balanced-accuracy decline while retaining the original source safety gates.
E42_BASELINE = {
    "clean_auc": 0.9928736964078795,
    "clean_balanced_accuracy": 0.9547709243248061,
    "robust_auc": 0.9933822505348071,
    "robust_balanced_accuracy": 0.9392306021035743,
}


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024**2):
            digest.update(chunk)
    return digest.hexdigest()


def _rows(
    values: Mapping[str, np.ndarray], mask: np.ndarray, scores: np.ndarray, scope: str
) -> list[dict[str, Any]]:
    indices = np.flatnonzero(mask)
    if len(indices) != len(scores):
        raise ValueError("E43 score alignment changed")
    return [
        {
            "scope": scope,
            "record_id": str(values["record_ids"][index]),
            "parent_id": str(values["parent_ids"][index]),
            "label": int(values["labels"][index]),
            "source": str(values["sources"][index]),
            "condition": str(values["conditions"][index]),
            "score": float(score),
            "status": "ok",
        }
        for index, score in zip(indices, scores, strict=True)
    ]


def rr_gate(by_condition: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    original = by_condition["original"]
    checks = {
        "original_auc_gte_0_90": float(original["metrics"]["roc_auc"]) >= 0.90,
        "original_tpr_at_fpr10_gte_0_80": float(original["metrics"]["tpr_at_fpr"]["tpr"]) >= 0.80,
        "original_eer_lte_0_15": float(original["metrics"]["eer"]) <= 0.15,
        "original_balanced_accuracy_gte_0_85": float(original["metrics"]["balanced_accuracy"]) >= 0.85,
        "original_real_macro_fp_lte_0_10": float(original["rates"]["real_macro_fp"]) <= 0.10 + 1e-12,
        "original_real_worst_fp_lte_0_20": float(original["rates"]["real_worst_source_fp"]) <= 0.20 + 1e-12,
        "original_ai_macro_recall_gte_0_80": float(original["rates"]["ai_macro_recall"]) >= 0.80,
        "original_ai_worst_recall_gte_0_60": float(original["rates"]["ai_worst_source_recall"]) >= 0.60,
    }
    for condition in ("original", "transfer", "redigital"):
        metrics = by_condition[condition]["metrics"]
        checks[f"{condition}_coverage_eq_1"] = float(metrics["coverage"]) == 1.0
    for condition in ("transfer", "redigital"):
        metrics = by_condition[condition]["metrics"]
        checks[f"{condition}_auc_gte_0_85"] = float(metrics["roc_auc"]) >= 0.85
        checks[f"{condition}_balanced_accuracy_gte_0_80"] = (
            float(metrics["balanced_accuracy"]) >= 0.80
        )
    return {"passed": all(checks.values()), "checks": checks}


def historical_gate(clean: Mapping[str, Any], robust: Mapping[str, Any]) -> dict[str, Any]:
    checks = {
        "clean_auc_within_0_02": float(clean["metrics"]["roc_auc"]) >= E42_BASELINE["clean_auc"] - 0.02,
        "clean_balanced_accuracy_within_0_05": (
            float(clean["metrics"]["balanced_accuracy"])
            >= E42_BASELINE["clean_balanced_accuracy"] - 0.05
        ),
        "clean_real_macro_fp_lte_0_10": float(clean["rates"]["real_macro_fp"]) <= 0.10 + 1e-12,
        "clean_real_worst_fp_lte_0_20": float(clean["rates"]["real_worst_source_fp"]) <= 0.20 + 1e-12,
        "clean_ai_macro_recall_gte_0_85": float(clean["rates"]["ai_macro_recall"]) >= 0.85,
        "clean_ai_worst_recall_gte_0_60": float(clean["rates"]["ai_worst_source_recall"]) >= 0.60,
        "clean_coverage_eq_1": float(clean["metrics"]["coverage"]) == 1.0,
        "robust_auc_within_0_02": float(robust["metrics"]["roc_auc"]) >= E42_BASELINE["robust_auc"] - 0.02,
        "robust_balanced_accuracy_within_0_05": (
            float(robust["metrics"]["balanced_accuracy"])
            >= E42_BASELINE["robust_balanced_accuracy"] - 0.05
        ),
        "robust_coverage_eq_1": float(robust["metrics"]["coverage"]) == 1.0,
    }
    return {"passed": all(checks.values()), "checks": checks}


def _summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "metrics": evaluate_binary_scores(rows, threshold=THRESHOLD),
        "rates": source_rates(rows, THRESHOLD),
    }


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> tuple[int, str]:
    raw = b"".join((json.dumps(row, sort_keys=True) + "\n").encode() for row in rows)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_bytes(raw)
    temporary.replace(path)
    return len(raw), hashlib.sha256(raw).hexdigest()


def evaluate() -> dict[str, Any]:
    if any(path.exists() for path in (SCORES, REPORT, EVIDENCE)):
        raise FileExistsError("E43-S DEVELOPMENT result already exists; no retry")
    if _digest(CANDIDATE) != CANDIDATE_SHA256:
        raise ValueError("E43-S candidate changed")
    artifact = joblib.load(CANDIDATE)
    if float(artifact["threshold"]) != THRESHOLD:
        raise ValueError("E43-S frozen threshold changed")
    head = artifact["head"]

    rr = _load(RR_FEATURES, RR_FEATURES_SHA256)
    rr_mask = rr["roles"].astype(str) == "development"
    if int(rr_mask.sum()) != 2_940:
        raise ValueError("E43 RR DEVELOPMENT population changed")
    rr_rows = _rows(rr, rr_mask, head.predict_proba(rr["features"][rr_mask])[:, 1], "rr_development")
    by_condition = {
        condition: _summary([row for row in rr_rows if row["condition"] == condition])
        for condition in ("original", "transfer", "redigital")
    }
    local_gate = rr_gate(by_condition)

    e42 = _load(E42_FEATURES, E42_FEATURES_SHA256)
    historical_mask = e42["roles"].astype(str) == "development"
    if int(historical_mask.sum()) != 11_230:
        raise ValueError("E43 historical DEVELOPMENT population changed")
    historical_rows = _rows(
        e42,
        historical_mask,
        head.predict_proba(e42["features"][historical_mask])[:, 1],
        "historical_regression",
    )
    historical_clean_rows = [row for row in historical_rows if row["condition"] == "clean"]
    historical_robust_rows = [row for row in historical_rows if row["condition"] != "clean"]
    historical_clean = _summary(historical_clean_rows)
    historical_robust = _summary(historical_robust_rows)
    regression_gate = historical_gate(historical_clean, historical_robust)

    rows = rr_rows + historical_rows
    score_bytes, score_sha256 = _write_jsonl(SCORES, rows)
    passed = bool(local_gate["passed"] and regression_gate["passed"])
    report = {
        "schema_version": 1,
        "state": "e43_small_development_passed" if passed else "e43_small_development_failed",
        "candidate_sha256": CANDIDATE_SHA256,
        "threshold": THRESHOLD,
        "rr": {"by_condition": by_condition, "gate": local_gate},
        "historical_e42_regression": {
            "baseline": E42_BASELINE,
            "clean": historical_clean,
            "robust": historical_robust,
            "gate": regression_gate,
            "disclosure": "Diagnostic is consumed and partially replayed in E43 fit; it is only a regression check.",
        },
        "gate": {"passed": passed, "rr_passed": local_gate["passed"], "historical_passed": regression_gate["passed"]},
        "score_stream": {"rows": len(rows), "bytes": score_bytes, "sha256": score_sha256},
        "itwsm_scores_created": 0,
        "next_action": "package E43-S and await untouched ITW-SM" if passed else "E43-S stops; DINOv2-L arm is unlocked",
        "boundary": "Consumed local DEVELOPMENT only. ITW-SM remains unopened and mandatory for final evidence.",
    }
    raw = (json.dumps(report, indent=2, sort_keys=True) + "\n").encode()
    REPORT.write_bytes(raw)
    EVIDENCE.write_bytes(raw)
    return report


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("evaluate",))
    args = parser.parse_args(argv)
    result = evaluate() if args.command == "evaluate" else None
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
