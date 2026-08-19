# =============================================================================
# e22b_bootstrap_ci.py — WHAT THIS EXPERIMENT DOES
# -----------------------------------------------------------------------------
# E22/E23a/E23b report the band's headline numbers from single splits with
# 34–100 evaluation images per source.  Before any of those numbers reaches
# the report, this script attaches honest uncertainty: a full-pipeline
# bootstrap (resample every population's calibration AND evaluation half,
# refit the worst-source threshold per replicate, recompute the metrics).
# Wide intervals are expected and must be reported as such — that is the
# point, not a defect.
#
# Three configurations, matching the decision line's evolution:
#   cf_vit       — E22's gate-passer (28% recall)
#   bfree        — E21b's best ranker (fails only NIST without the cap)
#   bfree_capped — the deployed config: B-Free + E23b's 2048px cap on NIST
#
# Output: 95% percentile intervals for worst-source FP, macro FP and macro
# recall under the deployed worst-source threshold rule (evaluation halves).
# Results: printed table + artifacts/e23/e22b_ci.json.  Usage:
#   PYTHONPATH=src .venv/bin/python experiments/e22b_bootstrap_ci.py
# =============================================================================

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from pixelproof.evaluation_protocol import (stable_calibration_split,
                                            threshold_at_fpr)

SPLIT_SEED = 2026
CAL_FRACTION = 0.5
FP_BUDGET = 0.10
REPLICATES = 2000
GENERATORS = ("dalle3", "midjourney", "sd21", "sd3", "sdxl")


def read_scores(path: str) -> dict[tuple[str, str], list[dict]]:
    populations: dict[tuple[str, str], list[dict]] = defaultdict(list)
    with open(path) as handle:
        for line in handle:
            record = json.loads(line)
            populations[(record["dataset"], record["source"])].append(record)
    return populations


def build_config(name: str) -> dict[str, dict[str, np.ndarray]]:
    """-> {population: {'cal': scores, 'ev': scores}} with E23b's cap applied
    to NIST2016 for the deployed configuration."""
    source_file = {
        "cf_vit": "artifacts/e21/cf_vit_scores.jsonl",
        "bfree": "artifacts/e21/bfree_scores.jsonl",
        "bfree_capped": "artifacts/e21/bfree_scores.jsonl",
    }[name]
    populations = read_scores(source_file)
    capped: dict[str, float] = {}
    if name == "bfree_capped":
        with open("artifacts/e23/e23b_bfree_capped_scores.jsonl") as handle:
            for line in handle:
                record = json.loads(line)
                capped[Path(record["path"]).stem] = record["image_score"]

    config: dict[str, dict[str, np.ndarray]] = {}
    for (dataset, source), records in populations.items():
        is_real = dataset == "forensics" or (dataset, source) == ("defactify", "real")
        is_generator = dataset == "defactify" and source in GENERATORS
        if not (is_real or is_generator):
            continue
        calibration, evaluation = stable_calibration_split(
            records, CAL_FRACTION, SPLIT_SEED
        )

        def value(record: dict) -> float:
            if source == "NIST2016" and capped:
                return capped.get(Path(record["path"]).stem, record["image_score"])
            return record["image_score"]

        key = f"{'real' if is_real else 'ai'}:{source if is_real else source}"
        config[key] = {
            "cal": np.asarray([value(r) for r in calibration]),
            "ev": np.asarray([value(r) for r in evaluation]),
        }
    return config


def metrics_once(config: dict, rng: np.random.Generator | None) -> dict[str, float]:
    def draw(values: np.ndarray) -> np.ndarray:
        if rng is None or not len(values):
            return values
        return values[rng.integers(0, len(values), len(values))]

    cuts = [
        threshold_at_fpr(draw(pop["cal"]), FP_BUDGET)
        for key, pop in config.items()
        if key.startswith("real:")
    ]
    t_ai = max(cut for cut in cuts if np.isfinite(cut))

    fps, recalls = [], []
    for key, pop in config.items():
        values = draw(pop["ev"])
        rate = float((values >= t_ai).mean()) if len(values) else np.nan
        (fps if key.startswith("real:") else recalls).append(rate)
    return {
        "worst_fp": float(np.nanmax(fps)),
        "macro_fp": float(np.nanmean(fps)),
        "macro_recall": float(np.nanmean(recalls)),
    }


def main() -> None:
    rng = np.random.default_rng(SPLIT_SEED)
    payload: dict[str, dict] = {}
    print(f"{'config':14s} {'metrik':13s} {'nokta':>7s} {'%95 aralık':>18s}")
    for name in ("cf_vit", "bfree", "bfree_capped"):
        config = build_config(name)
        point = metrics_once(config, rng=None)
        samples = defaultdict(list)
        for _ in range(REPLICATES):
            replicate = metrics_once(config, rng)
            for key, value in replicate.items():
                samples[key].append(value)
        payload[name] = {"point": point}
        for metric in ("worst_fp", "macro_fp", "macro_recall"):
            low, high = np.percentile(samples[metric], [2.5, 97.5])
            payload[name][metric + "_ci95"] = [float(low), float(high)]
            print(
                f"{name:14s} {metric:13s} {100 * point[metric]:6.1f}% "
                f"[{100 * low:5.1f}% … {100 * high:5.1f}%]"
            )

    output = Path("artifacts/e23/e22b_ci.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2))
    print(f"\nResults -> {output}")


if __name__ == "__main__":
    main()
