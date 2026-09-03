"""Diagnose the consumed E45 result without repairing or reclassifying it."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from experiments.e45_mediaeval_score import (
    BINARY_THRESHOLD,
    GENERALIST_SCORES,
    GENERALIST_SHA256,
    PLATFORMS,
    SPECIALIST_SCORES,
    FUSED_SCORES,
    binary_platform_rates,
)
from experiments.e43_train import CANDIDATE as GENERALIST
from pixelproof.benchmark_metrics import evaluate_binary_scores
from pixelproof.dda_candidate import THRESHOLD as SPECIALIST_THRESHOLD
from pixelproof.project_paths import DATA_ROOT, ML_ROOT


ROOT = DATA_ROOT / "e45_mediaeval_itwsm"
FINAL_EVIDENCE = ML_ROOT.parent / "evidence" / "e45_final_result.json"
REPORT = ROOT / "e46_posthoc_diagnostic.json"
EVIDENCE = ML_ROOT.parent / "evidence" / "e46_e45_posthoc_diagnostic.json"
GENERALIST_THRESHOLD = 0.8712875247001649
EXPECTED = {
    GENERALIST_SCORES: "43ecaa3f436a58c05045e978c97253e559797716a7ecf8c68710616c3b6fc171",
    SPECIALIST_SCORES: "889469866e3c5167d3430f088e038cb1688470ad6388ce55e41d451b67669bb7",
    FUSED_SCORES: "b84f8c406e14c2d3e01edf4d9e8660ec9aa2642802b786b4485fa8da3243c67e",
}


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024**2):
            digest.update(chunk)
    return digest.hexdigest()


def _write(path: Path, value: Any) -> bytes:
    raw = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_bytes(raw)
    temporary.replace(path)
    return raw


def _load(path: Path) -> list[dict[str, Any]]:
    if _digest(path) != EXPECTED[path]:
        raise ValueError(f"consumed E45 score stream changed: {path}")
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def summarize(rows: Sequence[Mapping[str, Any]], threshold: float) -> dict[str, Any]:
    metric_rows = [{**row, "source": row["platform"], "condition": "original"} for row in rows]
    return {
        "threshold": threshold,
        "metrics": evaluate_binary_scores(metric_rows, threshold=threshold),
        "rates": binary_platform_rates(rows, threshold),
        "real_score_quantiles": {
            str(q): float(np.quantile(
                [float(row["score"]) for row in rows if int(row["label"]) == 0], q
            ))
            for q in (0.5, 0.9, 0.95, 0.99)
        },
        "ai_score_quantiles": {
            str(q): float(np.quantile(
                [float(row["score"]) for row in rows if int(row["label"]) == 1], q
            ))
            for q in (0.1, 0.5)
        },
    }


def threshold_at_real_budget(rows: Sequence[Mapping[str, Any]], budget: float = 0.10) -> float:
    if not 0 < budget < 1:
        raise ValueError("post-hoc REAL budget must be in (0, 1)")
    real = np.asarray([float(row["score"]) for row in rows if int(row["label"]) == 0])
    if not len(real):
        raise ValueError("post-hoc threshold requires REAL rows")
    return float(np.quantile(real, 1 - budget, method="higher"))


def pattern_counts(
    generalist: Sequence[Mapping[str, Any]], specialist: Sequence[Mapping[str, Any]],
    fused: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    output: dict[str, Counter[str]] = {
        "all_real": Counter(), "fusion_false_ai": Counter(),
        "all_ai": Counter(), "fusion_false_real": Counter(),
    }
    for g_row, d_row, f_row in zip(generalist, specialist, fused, strict=True):
        if not (g_row["record_id"] == d_row["record_id"] == f_row["record_id"]):
            raise ValueError("E45 consumed arms changed order")
        g_flag = float(g_row["score"]) >= GENERALIST_THRESHOLD
        d_flag = float(d_row["score"]) >= SPECIALIST_THRESHOLD
        f_flag = float(f_row["score"]) >= BINARY_THRESHOLD
        pattern = f"generalist_{int(g_flag)}_specialist_{int(d_flag)}"
        label = int(f_row["label"])
        output["all_ai" if label else "all_real"][pattern] += 1
        if label == 0 and f_flag:
            output["fusion_false_ai"][pattern] += 1
        if label == 1 and not f_flag:
            output["fusion_false_real"][pattern] += 1
    return {key: dict(sorted(value.items())) for key, value in output.items()}


def diagnose() -> dict[str, Any]:
    if REPORT.exists() or EVIDENCE.exists():
        raise FileExistsError("E46 post-hoc diagnostic already exists")
    final = json.loads(FINAL_EVIDENCE.read_text())
    if final.get("state") != "e45_independent_final_failed":
        raise ValueError("E46 diagnosis requires the preserved failed E45 result")
    if _digest(GENERALIST) != GENERALIST_SHA256:
        raise ValueError("E43-S candidate changed")
    generalist = _load(GENERALIST_SCORES)
    specialist = _load(SPECIALIST_SCORES)
    fused = _load(FUSED_SCORES)
    if not (len(generalist) == len(specialist) == len(fused) == 9_978):
        raise ValueError("E45 consumed arm counts changed")

    posthoc_threshold = threshold_at_real_budget(fused)
    posthoc = summarize(fused, posthoc_threshold)
    checks = {
        "auc_gte_0_90": posthoc["metrics"]["roc_auc"] >= 0.90,
        "balanced_accuracy_gte_0_85": posthoc["metrics"]["balanced_accuracy"] >= 0.85,
        "pooled_real_false_ai_lte_0_10_approx": posthoc["rates"]["pooled_real_false_ai"] <= 0.101,
        "pooled_ai_recall_gte_0_80": posthoc["rates"]["pooled_ai_recall"] >= 0.80,
        "worst_platform_real_false_ai_lte_0_20": posthoc["rates"]["worst_real_false_ai"] <= 0.20,
        "worst_platform_ai_recall_gte_0_60": posthoc["rates"]["worst_ai_recall"] >= 0.60,
    }
    payload = {
        "schema_version": 1,
        "state": "e46_consumed_e45_posthoc_diagnostic_complete",
        "input_state": final["state"],
        "input_score_sha256": {path.name: digest for path, digest in EXPECTED.items()},
        "arms": {
            "generalist": summarize(generalist, GENERALIST_THRESHOLD),
            "specialist": summarize(specialist, SPECIALIST_THRESHOLD),
            "fusion_frozen": summarize(fused, BINARY_THRESHOLD),
        },
        "arm_decision_patterns": pattern_counts(generalist, specialist, fused),
        "posthoc_real_10pct_operating_point": {
            **posthoc,
            "checks": checks,
            "all_six_binary_checks_would_pass": all(checks.values()),
            "status": "diagnostic_only_selected_on_consumed_final_forbidden_for_deployment",
        },
        "conclusion": (
            "E45 failure is dominated by fusion/threshold calibration transfer. Ranking remains "
            "strong and a much higher consumed-data cut separates the classes, but E45 cannot "
            "supply that cut. E46 needs separate in-the-wild CAL plus a distinct untouched FINAL."
        ),
        "forbidden": [
            "repairing E45", "deploying the post-hoc threshold", "training or calibrating on E45",
            "claiming a pass from this diagnostic",
        ],
    }
    raw = _write(REPORT, payload)
    evidence = {
        "schema_version": 1,
        "state": payload["state"],
        "input_state": payload["input_state"],
        "input_score_sha256": payload["input_score_sha256"],
        "arm_summary": {
            name: {
                "auc": value["metrics"]["roc_auc"],
                "balanced_accuracy": value["metrics"]["balanced_accuracy"],
                "real_false_ai": value["rates"]["pooled_real_false_ai"],
                "ai_recall": value["rates"]["pooled_ai_recall"],
            }
            for name, value in payload["arms"].items()
        },
        "posthoc_real_10pct_operating_point": {
            "threshold": posthoc_threshold,
            "balanced_accuracy": posthoc["metrics"]["balanced_accuracy"],
            "real_false_ai": posthoc["rates"]["pooled_real_false_ai"],
            "ai_recall": posthoc["rates"]["pooled_ai_recall"],
            "worst_platform_real_false_ai": posthoc["rates"]["worst_real_false_ai"],
            "worst_platform_ai_recall": posthoc["rates"]["worst_ai_recall"],
            "all_six_binary_checks_would_pass": all(checks.values()),
            "status": payload["posthoc_real_10pct_operating_point"]["status"],
        },
        "conclusion": payload["conclusion"],
        "forbidden": payload["forbidden"],
        "detailed_report_bytes": len(raw),
        "detailed_report_sha256": hashlib.sha256(raw).hexdigest(),
    }
    _write(EVIDENCE, evidence)
    return evidence


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("diagnose",))
    args = parser.parse_args(argv)
    result = diagnose() if args.command == "diagnose" else None
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
