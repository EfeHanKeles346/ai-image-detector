"""One-shot DDA calibration on the frozen E36 CAL manifest."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from pixelproof.benchmark_metrics import evaluate_binary_scores
from pixelproof.dda_candidate import CHECKPOINT_SHA256, OfficialDDACandidate, THRESHOLD as PUBLISHED_THRESHOLD
from pixelproof.project_paths import DATA_ROOT, ML_ROOT


ROOT = DATA_ROOT / "e36"
MANIFEST = ROOT / "cal_manifest.json"
EXPECTED_MANIFEST_SHA256 = "4ed1b7341df39fed1f1fc19a20424f15362c0df224bcc77189045ba4a4af2e03"
SCORES = ROOT / "cal_scores.jsonl"
REPORT = ROOT / "calibration_report.json"
CANDIDATE = ROOT / "dda_calibrated_candidate.json"
EVIDENCE = ML_ROOT.parent / "evidence" / "e36_calibration.json"


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def _write_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_bytes(_json_bytes(value))
    temporary.replace(path)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rates(rows: Sequence[dict[str, Any]], threshold: float) -> dict[str, Any]:
    by_real: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_ai: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        (by_ai if row["label"] == 1 else by_real)[str(row["source"])].append(row)
    real_fp = {
        source: float(np.mean([float(row["score"]) >= threshold for row in group]))
        for source, group in sorted(by_real.items())
    }
    ai_recall = {
        source: float(np.mean([float(row["score"]) >= threshold for row in group]))
        for source, group in sorted(by_ai.items())
    }
    return {
        "threshold": threshold,
        "real_fp_by_device": real_fp,
        "real_macro_fp": float(np.mean(list(real_fp.values()))),
        "real_worst_device_fp": max(real_fp.values()),
        "ai_recall_by_family": ai_recall,
        "ai_macro_recall": float(np.mean(list(ai_recall.values()))),
        "ai_worst_family_recall": min(ai_recall.values()),
    }


def select_threshold(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    for threshold in sorted({float(row["score"]) for row in rows}):
        summary = _rates(rows, threshold)
        if summary["real_macro_fp"] <= 0.10 and summary["real_worst_device_fp"] <= 0.20:
            return summary
    raise ValueError("no E36 threshold satisfies authentic CAL budgets")


def _bootstrap_macro(
    rows: Sequence[dict[str, Any]], *, threshold: float, label: int, iterations: int = 2_000
) -> dict[str, float]:
    groups: dict[str, np.ndarray] = {}
    for source in sorted({str(row["source"]) for row in rows if row["label"] == label}):
        groups[source] = np.asarray(
            [float(row["score"]) >= threshold for row in rows if row["label"] == label and row["source"] == source],
            dtype=np.float64,
        )
    generator = np.random.default_rng(20_260_827 + label)
    samples = []
    for _ in range(iterations):
        rates = [float(generator.choice(values, size=len(values), replace=True).mean()) for values in groups.values()]
        samples.append(float(np.mean(rates)))
    return {
        "estimate": float(np.mean([values.mean() for values in groups.values()])),
        "low_95": float(np.quantile(samples, 0.025)),
        "high_95": float(np.quantile(samples, 0.975)),
        "iterations": iterations,
    }


def calibration_gate(metrics: MappingLike, frontier: MappingLike) -> dict[str, Any]:
    checks = {
        "roc_auc_gte_0.90": float(metrics["roc_auc"]) >= 0.90,
        "tpr_at_fpr10_gte_0.80": float(metrics["tpr_at_fpr"]["tpr"]) >= 0.80,
        "eer_lte_0.15": float(metrics["eer"]) <= 0.15,
        "balanced_accuracy_gte_0.85": float(metrics["balanced_accuracy"]) >= 0.85,
        "real_macro_fp_lte_0.10": float(frontier["real_macro_fp"]) <= 0.10,
        "real_worst_device_fp_lte_0.20": float(frontier["real_worst_device_fp"]) <= 0.20,
        "ai_macro_recall_gte_0.80": float(frontier["ai_macro_recall"]) >= 0.80,
        "ai_worst_family_recall_gte_0.60": float(frontier["ai_worst_family_recall"]) >= 0.60,
    }
    return {"passed": all(checks.values()), "checks": checks}


MappingLike = dict[str, Any]


def run(batch_size: int = 2) -> dict[str, Any]:
    if any(path.exists() for path in (SCORES, REPORT, CANDIDATE, EVIDENCE)):
        raise FileExistsError("E36 calibration output already exists; no silent rerun")
    if _sha256(MANIFEST) != EXPECTED_MANIFEST_SHA256:
        raise ValueError("E36 CAL manifest binding changed")
    manifest = json.loads(MANIFEST.read_text())
    if manifest.get("state") != "cal_manifest_frozen_unscored":
        raise ValueError("E36 CAL manifest is not frozen/unscored")
    candidate = OfficialDDACandidate()
    rows = []
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in manifest["rows"]:
        by_source[str(row["source"])].append(row)
    for source, source_rows in sorted(by_source.items()):
        paths = [ROOT / row["path"] for row in source_rows]
        scores = candidate.score_paths(paths, batch_size=batch_size)
        for row, score in zip(source_rows, scores, strict=True):
            rows.append({
                "path": row["path"],
                "parent_id": row["parent_id"],
                "label": row["label"],
                "source": row["source"],
                "condition": row["condition"],
                "status": "ok",
                "score": score.score,
            })
        print(f"E36 CAL {source} {len(source_rows)}/{len(source_rows)}", flush=True)
    score_raw = b"".join((json.dumps(row, sort_keys=True) + "\n").encode() for row in rows)
    score_part = SCORES.with_suffix(SCORES.suffix + ".part")
    score_part.write_bytes(score_raw)
    score_part.replace(SCORES)
    selected = select_threshold(rows)
    metrics = evaluate_binary_scores(rows, threshold=float(selected["threshold"]))
    published = _rates(rows, PUBLISHED_THRESHOLD)
    gate = calibration_gate(metrics, selected)
    report = {
        "schema_version": 1,
        "experiment": "E36/DDA-clean-calibration",
        "state": "calibration_passed" if gate["passed"] else "calibration_failed",
        "manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "scores_sha256": hashlib.sha256(score_raw).hexdigest(),
        "scores_bytes": len(score_raw),
        "counts": manifest["counts"],
        "published_threshold_diagnostic": published,
        "selected_frontier": selected,
        "selected_metrics": metrics,
        "bootstrap": {
            "real_macro_fp": _bootstrap_macro(rows, threshold=float(selected["threshold"]), label=0),
            "ai_macro_recall": _bootstrap_macro(rows, threshold=float(selected["threshold"]), label=1),
        },
        "calibration_gate": gate,
        "boundary": "One threshold selected on frozen CAL; no FINAL archive/blob downloaded or scored.",
    }
    _write_atomic(REPORT, report)
    _write_atomic(EVIDENCE, report)
    if gate["passed"]:
        _write_atomic(CANDIDATE, {
            "schema_version": 1,
            "state": "candidate_frozen_final_eligible",
            "detector": candidate.contract(),
            "checkpoint_sha256": CHECKPOINT_SHA256,
            "cal_manifest_sha256": EXPECTED_MANIFEST_SHA256,
            "cal_scores_sha256": report["scores_sha256"],
            "threshold": selected["threshold"],
            "gate": gate,
            "boundary": "Immutable candidate; FINAL may be acquired but cannot change this threshold.",
        })
    return report


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args(argv)
    print(json.dumps(run(args.batch_size), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
