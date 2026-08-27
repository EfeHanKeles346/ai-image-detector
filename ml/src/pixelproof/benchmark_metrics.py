"""NIST-style metrics and source-aware threshold selection for binary detectors."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, roc_curve


class BenchmarkContractError(ValueError):
    """Raised when a benchmark row or role would make the result ambiguous."""


def _successful(records: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    successful = []
    for index, record in enumerate(records):
        label = record.get("label")
        if label not in (0, 1):
            raise BenchmarkContractError(f"row {index} label must be explicit 0=REAL or 1=AI")
        if not str(record.get("source", "")).strip():
            raise BenchmarkContractError(f"row {index} has no source")
        status = record.get("status", "ok")
        if status == "ok":
            score = float(record.get("score", float("nan")))
            if not np.isfinite(score) or not 0.0 <= score <= 1.0:
                raise BenchmarkContractError(f"row {index} has invalid successful score {score!r}")
            successful.append(record)
        elif status != "error":
            raise BenchmarkContractError(f"row {index} has invalid status {status!r}")
    return successful


def _eer(labels: np.ndarray, scores: np.ndarray) -> float:
    fpr, tpr, _ = roc_curve(labels, scores, drop_intermediate=False)
    fnr = 1.0 - tpr
    index = int(np.argmin(np.abs(fpr - fnr)))
    return float((fpr[index] + fnr[index]) / 2.0)


def _tpr_at_fpr(labels: np.ndarray, scores: np.ndarray, target: float) -> float:
    fpr, tpr, _ = roc_curve(labels, scores, drop_intermediate=False)
    eligible = tpr[fpr <= target + 1e-12]
    return float(eligible.max()) if len(eligible) else 0.0


def _threshold_rates(
    records: Sequence[Mapping[str, Any]], threshold: float
) -> dict[str, Any]:
    labels = np.asarray([int(row["label"]) for row in records], dtype=np.int64)
    scores = np.asarray([float(row["score"]) for row in records], dtype=np.float64)
    predictions = scores >= threshold
    tp = int(np.sum((labels == 1) & predictions))
    fn = int(np.sum((labels == 1) & ~predictions))
    fp = int(np.sum((labels == 0) & predictions))
    tn = int(np.sum((labels == 0) & ~predictions))
    return {
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions))
        if set(labels.tolist()) == {0, 1}
        else None,
        "ai_recall": tp / (tp + fn) if tp + fn else None,
        "real_false_positive_rate": fp / (fp + tn) if fp + tn else None,
        "confusion": {"tp": tp, "fn": fn, "fp": fp, "tn": tn},
    }


def evaluate_binary_scores(
    records: Sequence[Mapping[str, Any]],
    *,
    threshold: float,
    fpr_target: float = 0.10,
) -> dict[str, Any]:
    """Measure all rows while keeping inference failures visible and gateable."""
    if not 0.0 <= threshold <= 1.0:
        raise BenchmarkContractError("threshold must be in [0, 1]")
    if not 0.0 < fpr_target < 1.0:
        raise BenchmarkContractError("fpr_target must be in (0, 1)")
    successful = _successful(records)
    if not successful:
        raise BenchmarkContractError("benchmark has no successful score")
    labels = np.asarray([int(row["label"]) for row in successful], dtype=np.int64)
    scores = np.asarray([float(row["score"]) for row in successful], dtype=np.float64)
    if set(labels.tolist()) != {0, 1}:
        raise BenchmarkContractError("benchmark needs both REAL and AI successful rows")

    threshold_rates = _threshold_rates(successful, threshold)
    correct = threshold_rates["confusion"]["tp"] + threshold_rates["confusion"]["tn"]
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in successful:
        grouped[(str(row["source"]), str(row.get("condition", "original")))].append(row)
    per_group = []
    for (source, condition), rows in sorted(grouped.items()):
        rates = _threshold_rates(rows, threshold)
        per_group.append({
            "source": source,
            "condition": condition,
            "label": int(rows[0]["label"]) if len({int(row["label"]) for row in rows}) == 1 else None,
            "count": len(rows),
            **rates,
        })

    real_scores = scores[labels == 0]
    ai_scores = scores[labels == 1]
    return {
        "counts": {
            "total": len(records),
            "succeeded": len(successful),
            "failed": len(records) - len(successful),
            "real_succeeded": int(np.sum(labels == 0)),
            "ai_succeeded": int(np.sum(labels == 1)),
        },
        "coverage": len(successful) / len(records),
        "accuracy_counting_failures_wrong": correct / len(records),
        "roc_auc": float(roc_auc_score(labels, scores)),
        "eer": _eer(labels, scores),
        "tpr_at_fpr": {"fpr": fpr_target, "tpr": _tpr_at_fpr(labels, scores, fpr_target)},
        "brier_target_ai": float(np.mean((1.0 - ai_scores) ** 2)),
        "brier_nontarget_real": float(np.mean(real_scores**2)),
        "stored_threshold": threshold,
        **threshold_rates,
        "per_group": per_group,
    }


def select_source_robust_threshold(
    records: Sequence[Mapping[str, Any]],
    *,
    macro_real_fpr_budget: float = 0.10,
    worst_real_fpr_budget: float = 0.20,
    macro_ai_recall_floor: float = 0.80,
    worst_ai_recall_floor: float = 0.60,
    min_group_size: int = 20,
) -> dict[str, Any]:
    """Choose the lowest CAL-only threshold satisfying frozen source-level gates."""
    successful = _successful(records)
    if len(successful) != len(records):
        raise BenchmarkContractError("threshold calibration cannot silently discard failed rows")
    if min_group_size <= 0:
        raise BenchmarkContractError("min_group_size must be positive")
    by_class_source: dict[tuple[int, str], list[float]] = defaultdict(list)
    for row in successful:
        by_class_source[(int(row["label"]), str(row["source"]))].append(float(row["score"]))
    real_groups = {key[1]: values for key, values in by_class_source.items() if key[0] == 0 and len(values) >= min_group_size}
    ai_groups = {key[1]: values for key, values in by_class_source.items() if key[0] == 1 and len(values) >= min_group_size}
    if not real_groups or not ai_groups:
        raise BenchmarkContractError("calibration needs sufficiently sized REAL and AI source groups")

    scores = np.asarray([float(row["score"]) for row in successful], dtype=np.float64)
    candidates = sorted(
        {float(value) for value in scores}
        | {float(np.nextafter(value, np.inf)) for value in scores}
    )
    chosen: tuple[float, dict[str, float], dict[str, float]] | None = None
    for threshold in candidates:
        real_rates = {
            source: float(np.mean(np.asarray(values) >= threshold))
            for source, values in real_groups.items()
        }
        if np.mean(list(real_rates.values())) > macro_real_fpr_budget + 1e-12:
            continue
        if max(real_rates.values()) > worst_real_fpr_budget + 1e-12:
            continue
        ai_rates = {
            source: float(np.mean(np.asarray(values) >= threshold))
            for source, values in ai_groups.items()
        }
        chosen = (threshold, real_rates, ai_rates)
        break
    if chosen is None:
        raise BenchmarkContractError("no threshold satisfies the REAL false-positive gates")
    threshold, real_rates, ai_rates = chosen
    macro_ai = float(np.mean(list(ai_rates.values())))
    worst_ai = min(ai_rates.values())
    if macro_ai < macro_ai_recall_floor or worst_ai < worst_ai_recall_floor:
        raise BenchmarkContractError(
            "first REAL-safe threshold fails the frozen AI recall gates: "
            f"macro={macro_ai:.4f}, worst={worst_ai:.4f}"
        )
    return {
        "threshold": threshold,
        "real_false_positive_rate_by_source": real_rates,
        "real_macro_false_positive_rate": float(np.mean(list(real_rates.values()))),
        "real_worst_false_positive_rate": max(real_rates.values()),
        "ai_recall_by_source": ai_rates,
        "ai_macro_recall": macro_ai,
        "ai_worst_recall": worst_ai,
        "min_group_size": min_group_size,
    }
