"""Freeze one E39 threshold from consumed E38 scores without changing the model."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import joblib

from pixelproof.benchmark_metrics import evaluate_binary_scores, select_source_robust_threshold
from pixelproof.project_paths import DATA_ROOT, ML_ROOT


E38_ROOT = DATA_ROOT / "e38"
E39_ROOT = DATA_ROOT / "e39"
E38_CANDIDATE = E38_ROOT / "e38_dinov2s.joblib"
E38_CANDIDATE_SHA256 = "fddbe475adb807bcf523127095b2a8221443761ad4afa71d8c163829afc44067"
E38_SCORES = E38_ROOT / "final_scores.jsonl"
E38_SCORES_SHA256 = "dd4f181de50d56ee5862db8b623053d0eb1f8e6471ff84c3e3506d8fe728dc2d"
E38_EVIDENCE = ML_ROOT.parent / "evidence" / "e38_final_result.json"
E38_EVIDENCE_SHA256 = "c0485a9e9b1f5d0901a0eb3904a885c77d414cdf720ec5d0113af7e386e220ff"
ROLE_AMENDMENT = ML_ROOT.parent / "evidence" / "e39_role_amendment.json"
CANDIDATE = E39_ROOT / "e39_threshold_candidate.json"
EVIDENCE = ML_ROOT.parent / "evidence" / "e39_calibration.json"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def calibration_gate(metrics: Mapping[str, Any], frontier: Mapping[str, Any]) -> dict[str, Any]:
    checks = {
        "roc_auc_gte_0.90": float(metrics["roc_auc"]) >= 0.90,
        "tpr_at_fpr10_gte_0.80": float(metrics["tpr_at_fpr"]["tpr"]) >= 0.80,
        "eer_lte_0.15": float(metrics["eer"]) <= 0.15,
        "balanced_accuracy_gte_0.85": float(metrics["balanced_accuracy"]) >= 0.85,
        "real_macro_fp_lte_0.10": float(frontier["real_macro_false_positive_rate"]) <= 0.10 + 1e-12,
        "real_worst_device_fp_lte_0.20": float(frontier["real_worst_false_positive_rate"]) <= 0.20 + 1e-12,
        "ai_macro_recall_gte_0.80": float(frontier["ai_macro_recall"]) >= 0.80,
        "ai_worst_family_recall_gte_0.60": float(frontier["ai_worst_recall"]) >= 0.60,
        "coverage_eq_1": float(metrics["coverage"]) == 1.0,
    }
    return {"passed": all(checks.values()), "checks": checks}


def select_e39_threshold(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Select the lowest threshold satisfying every frozen source-level budget."""
    return select_source_robust_threshold(
        rows,
        macro_real_fpr_budget=0.10,
        worst_real_fpr_budget=0.20,
        macro_ai_recall_floor=0.80,
        worst_ai_recall_floor=0.60,
        min_group_size=20,
    )


def _load_bound_scores() -> list[dict[str, Any]]:
    bindings = (
        (E38_CANDIDATE, E38_CANDIDATE_SHA256, "E38 candidate"),
        (E38_SCORES, E38_SCORES_SHA256, "E38 scores"),
        (E38_EVIDENCE, E38_EVIDENCE_SHA256, "E38 evidence"),
    )
    for path, expected, label in bindings:
        if not path.is_file() or _sha256_file(path) != expected:
            raise ValueError(f"{label} binding changed")
    artifact = joblib.load(E38_CANDIDATE)
    if artifact.get("positive_label") != "ai" or float(artifact.get("threshold", -1)) != 0.8961896300315858:
        raise ValueError("E38 candidate decision contract changed")
    amendment = json.loads(ROLE_AMENDMENT.read_text())
    if amendment.get("state") != "role_amended_before_e39_candidate" or amendment.get("counts", {}).get("total") != 640:
        raise ValueError("E39 role amendment changed")
    rows = [json.loads(line) for line in E38_SCORES.read_text().splitlines() if line.strip()]
    if len(rows) != 640 or sum(int(row["label"]) == 0 for row in rows) != 400:
        raise ValueError("E38 score population changed")
    return rows


def run() -> dict[str, Any]:
    if CANDIDATE.exists() or EVIDENCE.exists():
        raise FileExistsError("E39 calibration output already exists; no silent rerun")
    rows = _load_bound_scores()
    frontier = select_e39_threshold(rows)
    threshold = float(frontier["threshold"])
    metrics = evaluate_binary_scores(rows, threshold=threshold)
    gate = calibration_gate(metrics, frontier)
    if not gate["passed"]:
        raise ValueError("E39 threshold failed the frozen calibration gate")
    role_sha = _sha256_file(ROLE_AMENDMENT)
    report = {
        "schema_version": 1,
        "experiment": "E39/calibration-transfer-correction",
        "state": "calibration_passed_candidate_frozen",
        "counts": {"real": 400, "ai": 240, "total": 640},
        "underlying_candidate_sha256": E38_CANDIDATE_SHA256,
        "consumed_scores_sha256": E38_SCORES_SHA256,
        "role_amendment_sha256": role_sha,
        "threshold": threshold,
        "frontier": frontier,
        "metrics": metrics,
        "gate": gate,
        "boundary": "Development-selected on consumed E38 FINAL scores. Eligible only for one new independent E39 FINAL; not evidence that E38 passed.",
    }
    _write_atomic(EVIDENCE, report)
    _write_atomic(CANDIDATE, {
        "schema_version": 1,
        "state": "research_candidate_frozen_final_eligible",
        "detector": "E38 DINOv2-S representation plus byte-identical fitted logistic head",
        "underlying_artifact_path": str(E38_CANDIDATE),
        "underlying_artifact_sha256": E38_CANDIDATE_SHA256,
        "positive_label": "ai",
        "threshold": threshold,
        "calibration_scores_sha256": E38_SCORES_SHA256,
        "role_amendment_sha256": role_sha,
        "gate": gate,
        "boundary": report["boundary"],
    })
    return report


def main(argv: Iterable[str] | None = None) -> int:
    argparse.ArgumentParser(description=__doc__).parse_args(argv)
    print(json.dumps(run(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
