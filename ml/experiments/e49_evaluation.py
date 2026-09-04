"""Fail-closed metrics for the paired E49 Module-1 comprehensive final."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Mapping, Sequence

import numpy as np

from pixelproof.benchmark_metrics import evaluate_binary_scores


BINARY_THRESHOLD = 0.07940196245908739
REAL_CUT = 0.011505939625203613
AI_CUT = BINARY_THRESHOLD
CONDITIONS = ("publisher_original", "social_q75")
BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_SEED = 49

SOURCE_COUNTS = {
    "iPhone 15 Pro": 100,
    "iPhone 15 Pro Max": 100,
    "Google Pixel 8 Pro": 100,
    "Samsung Galaxy S23 Ultra": 100,
    "iPhone 14 Pro": 100,
    "Google Pixel 7 Pro": 100,
    "iPhone 13 Pro": 100,
    "Sony ILCE-7M4": 100,
    "Canon EOS R5": 100,
    "Nikon Z 8": 100,
    "GPT Image 2": 160,
    "Z-Image Turbo": 160,
    "Seedream v5.0": 160,
    "FLUX.2 Klein 9B": 160,
    "Midjourney 7": 160,
    "StyleGAN2": 200,
}

GATES = {
    "score_coverage": 1.0,
    "binary_auc_min": 0.90,
    "binary_balanced_accuracy_min": 0.85,
    "pooled_real_false_ai_max": 0.10,
    "source_real_false_ai_worst_max": 0.20,
    "pooled_ai_recall_min": 0.80,
    "source_ai_recall_worst_min": 0.60,
    "automatic_coverage_min": 0.80,
    "covered_accuracy_min": 0.95,
    "uncertain_rate_max": 0.20,
}


def _is_ok(row: Mapping[str, Any]) -> bool:
    if row.get("status", "ok") != "ok":
        return False
    score = float(row.get("score", np.nan))
    return bool(np.isfinite(score) and 0.0 <= score <= 1.0)


def _score_or_nan(row: Mapping[str, Any]) -> float:
    if not _is_ok(row):
        return float("nan")
    return float(row["score"])


def validate_paired_final(rows: Sequence[Mapping[str, Any]]) -> None:
    """Prove the two transports are exact children of the same frozen parents."""
    if len(rows) != 4_000:
        raise ValueError(f"E49 final observation count changed: {len(rows)}/4000")
    by_condition = {condition: [] for condition in CONDITIONS}
    for row in rows:
        condition = str(row.get("condition", ""))
        if condition not in by_condition:
            raise ValueError(f"E49 unexpected condition: {condition!r}")
        by_condition[condition].append(row)

    parent_maps: dict[str, dict[str, tuple[int, str]]] = {}
    for condition, condition_rows in by_condition.items():
        if len(condition_rows) != 2_000:
            raise ValueError(f"E49 {condition} count changed: {len(condition_rows)}/2000")
        parents: dict[str, tuple[int, str]] = {}
        for row in condition_rows:
            parent_id = str(row.get("parent_id", ""))
            label, source = int(row.get("label", -1)), str(row.get("source", ""))
            if not parent_id or parent_id in parents:
                raise ValueError(f"E49 {condition} parent identity is missing or repeated")
            if label not in (0, 1) or not source:
                raise ValueError(f"E49 {condition} label/source contract changed")
            parents[parent_id] = (label, source)
        counts = Counter(source for _, source in parents.values())
        if dict(counts) != SOURCE_COUNTS:
            raise ValueError(f"E49 {condition} source quotas changed: {dict(counts)}")
        parent_maps[condition] = parents

    if parent_maps[CONDITIONS[0]] != parent_maps[CONDITIONS[1]]:
        raise ValueError("E49 paired conditions do not share exact parent label/source identities")


def source_rates(rows: Sequence[Mapping[str, Any]], threshold: float = BINARY_THRESHOLD) -> dict[str, Any]:
    """Report every source; score failures pessimistically so they cannot improve a gate."""
    real: dict[str, list[bool]] = defaultdict(list)
    ai: dict[str, list[bool]] = defaultdict(list)
    failures: Counter[str] = Counter()
    for row in rows:
        label = int(row["label"])
        source = str(row["source"])
        if _is_ok(row):
            predicted_ai = float(row["score"]) >= threshold
        else:
            failures[source] += 1
            predicted_ai = label == 0  # REAL failure -> false AI; AI failure -> missed AI.
        (ai if label == 1 else real)[source].append(predicted_ai)
    if not real or not ai:
        raise ValueError("E49 source rates require REAL and AI sources")
    real_rates = {source: float(np.mean(values)) for source, values in sorted(real.items())}
    ai_rates = {source: float(np.mean(values)) for source, values in sorted(ai.items())}
    return {
        "pooled_real_false_ai": float(np.mean([value for values in real.values() for value in values])),
        "pooled_ai_recall": float(np.mean([value for values in ai.values() for value in values])),
        "real_false_ai_by_source": real_rates,
        "ai_recall_by_source": ai_rates,
        "worst_real_false_ai": max(real_rates.values()),
        "worst_ai_recall": min(ai_rates.values()),
        "score_failures_by_source": dict(sorted(failures.items())),
    }


def selective_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    labels = np.asarray([int(row["label"]) for row in rows], dtype=np.int8)
    scores = np.asarray([_score_or_nan(row) for row in rows], dtype=np.float64)
    ok = np.asarray([_is_ok(row) for row in rows])
    decisions = np.full(len(rows), -1, dtype=np.int8)
    decisions[ok & (scores < REAL_CUT)] = 0
    decisions[ok & (scores >= AI_CUT)] = 1
    automatic = decisions >= 0
    automatic_rows = int(automatic.sum())
    return {
        "automatic_coverage": automatic_rows / len(rows),
        "covered_accuracy": (
            float((decisions[automatic] == labels[automatic]).mean()) if automatic_rows else 0.0
        ),
        "uncertain_rate": float((~automatic).mean()),
        "automatic_rows": automatic_rows,
        "uncertain_rows": int((~automatic).sum()),
    }


def bootstrap_primary(
    rows: Sequence[Mapping[str, Any]], *, samples: int = BOOTSTRAP_SAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, list[float]]:
    """Source/label-stratified parent bootstrap for one condition."""
    if not rows or not all(_is_ok(row) for row in rows):
        raise ValueError("E49 bootstrap requires complete successful parent scores")
    parent_ids = [str(row.get("parent_id", "")) for row in rows]
    if any(not value for value in parent_ids) or len(set(parent_ids)) != len(rows):
        raise ValueError("E49 bootstrap requires one unique parent per condition")
    if samples <= 0:
        raise ValueError("E49 bootstrap sample count must be positive")

    labels = np.asarray([int(row["label"]) for row in rows], dtype=np.int8)
    scores = np.asarray([float(row["score"]) for row in rows], dtype=np.float64)
    sources = np.asarray([str(row["source"]) for row in rows])
    binary = scores >= BINARY_THRESHOLD
    decisions = np.where(scores < REAL_CUT, 0, np.where(scores >= AI_CUT, 1, -1))
    order = np.argsort(scores, kind="stable")
    ordered_scores = scores[order]
    starts = np.r_[0, np.flatnonzero(np.diff(ordered_scores)) + 1]
    strata = [np.flatnonzero((sources == source) & (labels == label))
              for label, source in sorted(set(zip(labels.tolist(), sources.tolist())))]
    rng = np.random.default_rng(seed)
    output = {key: [] for key in (
        "roc_auc", "balanced_accuracy", "real_false_ai", "ai_recall",
        "automatic_coverage", "covered_accuracy", "uncertain_rate",
    )}
    for offset in range(0, samples, 100):
        chunk = min(100, samples - offset)
        counts = np.zeros((chunk, len(rows)), dtype=np.int16)
        for indices in strata:
            counts[:, indices] = rng.multinomial(
                len(indices), np.full(len(indices), 1 / len(indices)), size=chunk
            )
        real_n = counts[:, labels == 0].sum(axis=1)
        ai_n = counts[:, labels == 1].sum(axis=1)
        real_fp = (counts[:, labels == 0] * binary[labels == 0]).sum(axis=1) / real_n
        ai_recall = (counts[:, labels == 1] * binary[labels == 1]).sum(axis=1) / ai_n
        automatic = decisions != -1
        automatic_n = (counts * automatic).sum(axis=1)
        correct_n = (counts * automatic * (decisions == labels)).sum(axis=1)
        ordered = counts[:, order]
        positives = np.add.reduceat(ordered * (labels[order] == 1), starts, axis=1)
        negatives = np.add.reduceat(ordered * (labels[order] == 0), starts, axis=1)
        cumulative_negatives = np.cumsum(negatives, axis=1)
        auc = (
            positives * (cumulative_negatives - negatives + 0.5 * negatives)
        ).sum(axis=1) / (real_n * ai_n)
        output["roc_auc"].extend(auc.tolist())
        output["balanced_accuracy"].extend(((1 - real_fp + ai_recall) / 2).tolist())
        output["real_false_ai"].extend(real_fp.tolist())
        output["ai_recall"].extend(ai_recall.tolist())
        output["automatic_coverage"].extend((automatic_n / len(rows)).tolist())
        output["covered_accuracy"].extend((correct_n / automatic_n).tolist())
        output["uncertain_rate"].extend((1 - automatic_n / len(rows)).tolist())
    return {key: [float(value) for value in np.quantile(values, [0.025, 0.975])]
            for key, values in output.items()}


def evaluate_condition(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    metrics = evaluate_binary_scores(rows, threshold=BINARY_THRESHOLD)
    rates = source_rates(rows)
    selective = selective_metrics(rows)
    checks = {
        "score_coverage_eq_1": metrics["coverage"] == GATES["score_coverage"],
        "binary_auc_gte_0_90": metrics["roc_auc"] >= GATES["binary_auc_min"],
        "binary_balanced_accuracy_gte_0_85": (
            metrics["balanced_accuracy"] >= GATES["binary_balanced_accuracy_min"]
        ),
        "pooled_real_false_ai_lte_0_10": (
            rates["pooled_real_false_ai"] <= GATES["pooled_real_false_ai_max"]
        ),
        "worst_source_real_false_ai_lte_0_20": (
            rates["worst_real_false_ai"] <= GATES["source_real_false_ai_worst_max"]
        ),
        "pooled_ai_recall_gte_0_80": rates["pooled_ai_recall"] >= GATES["pooled_ai_recall_min"],
        "worst_source_ai_recall_gte_0_60": (
            rates["worst_ai_recall"] >= GATES["source_ai_recall_worst_min"]
        ),
        "automatic_coverage_gte_0_80": (
            selective["automatic_coverage"] >= GATES["automatic_coverage_min"]
        ),
        "covered_accuracy_gte_0_95": (
            selective["covered_accuracy"] >= GATES["covered_accuracy_min"]
        ),
        "uncertain_rate_lte_0_20": selective["uncertain_rate"] <= GATES["uncertain_rate_max"],
    }
    return {
        "binary_metrics": metrics,
        "binary_rates": rates,
        "selective": selective,
        "gate": {"passed": all(checks.values()), "passed_checks": sum(checks.values()),
                 "total_checks": len(checks), "checks": checks},
    }


def evaluate_final(
    rows: Sequence[Mapping[str, Any]], *, bootstrap_samples: int = BOOTSTRAP_SAMPLES,
) -> dict[str, Any]:
    validate_paired_final(rows)
    reports: dict[str, Any] = {}
    for offset, condition in enumerate(CONDITIONS):
        condition_rows = [row for row in rows if row["condition"] == condition]
        report = evaluate_condition(condition_rows)
        report["bootstrap_95pct"] = (
            bootstrap_primary(condition_rows, samples=bootstrap_samples, seed=BOOTSTRAP_SEED + offset)
            if report["binary_metrics"]["coverage"] == 1.0 else None
        )
        reports[condition] = report
    passed_checks = sum(report["gate"]["passed_checks"] for report in reports.values())
    return {
        "schema_version": 1,
        "state": "e49_comprehensive_final_passed" if passed_checks == 20 else "e49_comprehensive_final_failed",
        "candidate": {
            "binary_threshold": BINARY_THRESHOLD,
            "selective_policy": {"real_if_score_lt": REAL_CUT,
                                 "ai_if_score_gte": AI_CUT, "otherwise": "uncertain"},
        },
        "parent_count": 2_000,
        "observation_count": 4_000,
        "conditions": reports,
        "gate": {"passed": passed_checks == 20, "passed_checks": passed_checks,
                 "total_checks": 20},
        "boundary": "One-shot E49 result; thresholds, rows and sources cannot be repaired after opening.",
    }
