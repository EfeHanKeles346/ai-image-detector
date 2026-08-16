"""Pure helpers for honest image-level evaluation of tile detectors.

The model produces one score per tile.  Deployment produces one decision per
image, so the aggregation rule and the threshold calibration are part of the
model contract, not presentation details.  These helpers deliberately have no
Torch dependency; they are shared by experiment scripts and unit tests.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence

import numpy as np
from sklearn.metrics import roc_auc_score


AGGREGATION_RULES = ("top3", "top10pct", "p90", "mean", "fixed16_top3")


def aggregate_tile_scores(scores: Sequence[float], rule: str) -> float:
    """Turn a variable-length tile-score vector into one image score.

    ``fixed16_top3`` is the control for the order-statistic shortcut: large
    images are deterministically thinned to sixteen spatially ordered tiles
    before taking top-3, so they do not get hundreds more chances to produce a
    random high score than a 500 px image does.
    """
    values = np.asarray(scores, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        return float("nan")
    if rule == "top3":
        return float(np.sort(values)[-min(3, len(values)):].mean())
    if rule == "top10pct":
        count = max(1, int(np.ceil(0.10 * len(values))))
        return float(np.sort(values)[-count:].mean())
    if rule == "p90":
        return float(np.percentile(values, 90))
    if rule == "mean":
        return float(values.mean())
    if rule == "fixed16_top3":
        if len(values) > 16:
            index = np.linspace(0, len(values) - 1, 16).round().astype(int)
            values = values[np.unique(index)]
        return float(np.sort(values)[-min(3, len(values)):].mean())
    raise ValueError(f"unknown aggregation rule {rule!r}; choose from {AGGREGATION_RULES}")


def records_to_scores(records: Sequence[Mapping], rule: str) -> np.ndarray:
    """Aggregate JSON-friendly records containing a ``tile_scores`` field."""
    return np.asarray(
        [aggregate_tile_scores(record["tile_scores"], rule) for record in records],
        dtype=np.float64,
    )


def stable_calibration_split(
    records: Sequence[Mapping], fraction: float, seed: int
) -> tuple[list[Mapping], list[Mapping]]:
    """Stable, disjoint calibration/evaluation split keyed by image path.

    A hash split does not reshuffle existing images when a new file is added.
    Call this separately per source/class so every population is represented in
    both halves.  A singleton stays in evaluation; it cannot calibrate and test.
    """
    if not 0.0 < fraction < 1.0:
        raise ValueError("calibration fraction must be strictly between 0 and 1")
    ranked = sorted(
        records,
        key=lambda record: hashlib.sha256(
            f"{seed}:{record['path']}".encode("utf-8")
        ).digest(),
    )
    if len(ranked) < 2:
        return [], list(ranked)
    count = min(len(ranked) - 1, max(1, int(round(len(ranked) * fraction))))
    return list(ranked[:count]), list(ranked[count:])


def threshold_at_fpr(real_scores: Sequence[float], budget: float) -> float:
    """Highest-recall empirical threshold whose real false-positive rate fits.

    Prediction uses ``score >= threshold``.  Moving just above the first score
    outside the allowed top tail makes ties conservative and guarantees that
    the calibration false-positive rate does not silently exceed the budget.
    """
    scores = np.asarray(real_scores, dtype=np.float64)
    scores = scores[np.isfinite(scores)]
    if not len(scores):
        return float("nan")
    if not 0.0 <= budget < 1.0:
        raise ValueError("false-positive budget must be in [0, 1)")
    allowed = int(np.floor(len(scores) * budget))
    descending = np.sort(scores)[::-1]
    if allowed == 0:
        return float(np.nextafter(descending[0], np.inf))
    return float(np.nextafter(descending[allowed], np.inf))


def operating_point(
    calibration_real: Sequence[float],
    evaluation_real: Sequence[float],
    evaluation_ai: Sequence[float],
    budget: float,
) -> dict[str, float]:
    """Fit only on calibration reals, then report on untouched evaluation data."""
    cut = threshold_at_fpr(calibration_real, budget)

    def rate(values: Sequence[float]) -> float:
        array = np.asarray(values, dtype=np.float64)
        array = array[np.isfinite(array)]
        return float((array >= cut).mean()) if len(array) and np.isfinite(cut) else float("nan")

    return {
        "threshold": cut,
        "calibration_fp": rate(calibration_real),
        "evaluation_fp": rate(evaluation_real),
        "evaluation_recall": rate(evaluation_ai),
    }


def safe_auc(real_scores: Sequence[float], ai_scores: Sequence[float]) -> float:
    """AUC with the project's convention: authentic=0, synthetic=1."""
    real = np.asarray(real_scores, dtype=np.float64)
    ai = np.asarray(ai_scores, dtype=np.float64)
    real = real[np.isfinite(real)]
    ai = ai[np.isfinite(ai)]
    if not len(real) or not len(ai):
        return float("nan")
    truth = np.r_[np.zeros(len(real)), np.ones(len(ai))]
    return float(roc_auc_score(truth, np.r_[real, ai]))


def evaluate_image_score_records(
    real_records: Sequence[Mapping],
    generator_records: Mapping[str, Sequence[Mapping]],
    genimage_records: tuple[Sequence[Mapping], Sequence[Mapping]],
    forensic_records: Mapping[str, Sequence[Mapping]],
    calibration_fraction: float,
    split_seed: int,
    budget: float,
    score_key: str = "image_score",
) -> dict:
    """Evaluate a detector that emits one official score per image.

    This is the whole-image counterpart of the tile protocol.  It deliberately
    uses the same stable source-wise split and transfers the same untouched
    threshold to every forensic-real source, so an external model is compared
    with E20 on our data rather than on the headline from its paper.
    """

    def scores(records: Sequence[Mapping]) -> np.ndarray:
        values = np.asarray([record[score_key] for record in records], dtype=np.float64)
        return values[np.isfinite(values)]

    def rate(values: Sequence[float], threshold: float) -> float:
        array = np.asarray(values, dtype=np.float64)
        array = array[np.isfinite(array)]
        return float((array >= threshold).mean()) if len(array) else float("nan")

    real_cal, real_eval = stable_calibration_split(
        real_records, calibration_fraction, split_seed
    )
    generator_splits = {
        name: stable_calibration_split(records, calibration_fraction, split_seed)
        for name, records in generator_records.items()
    }
    if not real_cal or not real_eval or not generator_splits:
        raise RuntimeError("evaluation needs two Defactify reals and one generator")

    cal_real = scores(real_cal)
    eval_real = scores(real_eval)
    cal_by_generator = {
        name: scores(parts[0]) for name, parts in generator_splits.items()
    }
    eval_by_generator = {
        name: scores(parts[1]) for name, parts in generator_splits.items()
    }
    cal_ai = np.concatenate([value for value in cal_by_generator.values() if len(value)])
    eval_ai = np.concatenate([value for value in eval_by_generator.values() if len(value)])
    point = operating_point(cal_real, eval_real, eval_ai, budget)
    threshold = point["threshold"]

    per_generator = {}
    calibration_recalls = []
    for name in sorted(generator_splits):
        calibration_recall = rate(cal_by_generator[name], threshold)
        calibration_recalls.append(calibration_recall)
        per_generator[name] = {
            "calibration_recall": calibration_recall,
            "evaluation_recall": rate(eval_by_generator[name], threshold),
            "evaluation_auc": safe_auc(eval_real, eval_by_generator[name]),
            "n_calibration": len(cal_by_generator[name]),
            "n_evaluation": len(eval_by_generator[name]),
        }

    source_fp = {
        source: rate(scores(records), threshold)
        for source, records in forensic_records.items()
        if records
    }
    gen_real, gen_ai = genimage_records
    return {
        "calibration_macro_generator_recall": float(np.nanmean(calibration_recalls)),
        "threshold": threshold,
        "defactify_calibration_fp": point["calibration_fp"],
        "defactify_evaluation_fp": point["evaluation_fp"],
        "defactify_evaluation_recall": point["evaluation_recall"],
        "defactify_evaluation_auc": safe_auc(eval_real, eval_ai),
        "genimage_auc": safe_auc(scores(gen_real), scores(gen_ai)),
        "forensics_macro_fp": float(np.mean(list(source_fp.values()))) if source_fp else float("nan"),
        "forensics_worst_fp": float(max(source_fp.values())) if source_fp else float("nan"),
        "forensics_source_fp": source_fp,
        "per_generator": per_generator,
        "counts": {
            "defactify_real_calibration": len(cal_real),
            "defactify_real_evaluation": len(eval_real),
            "defactify_ai_calibration": len(cal_ai),
            "defactify_ai_evaluation": len(eval_ai),
            "forensics_real": sum(len(records) for records in forensic_records.values()),
        },
    }
