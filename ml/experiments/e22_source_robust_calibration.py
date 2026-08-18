# =============================================================================
# e22_source_robust_calibration.py — WHAT THIS EXPERIMENT DOES
# -----------------------------------------------------------------------------
# E21/E21b settled that no detector family passes the cross-source gate when
# its threshold is fitted on one real source.  This experiment asks the
# remaining question: can the DECISION RULE pass the gate, using the same
# frozen score sets?  Everything here runs on the cached per-image scores of
# three detectors (our tile ResNet-18, CF ViT-S, B-Free) — no model is loaded,
# a full run costs seconds.
#
# PRE-REGISTERED HYPOTHESES
# -----------------------------------------------------------------------------
# H1 (diagnostic) — NIST2016, the universal poison source, differs from the
#     other forensic sources in measurable metadata (resolution / bytes per
#     pixel), i.e. the failure has a pipeline explanation, not a mystery one.
# H2 (LOSO) — a threshold calibrated on MANY real pipelines transfers to an
#     unseen pipeline far better than the Defactify-only threshold, at a
#     measurable cost in AI recall.  The worst-source (max-over-pipelines)
#     rule is the most conservative and the only one with a chance of holding
#     the budget on a held-out pipeline.
# H3 (abstention) — a two-threshold band ("AI" above t_ai, "real" below
#     t_real, abstain between) turns an undeployable single threshold into a
#     deployable partial decision: bounded FP among decided images, at the
#     price of an honest abstention rate.
#
# PER BLOCK
# -----------------------------------------------------------------------------
# load_arm()        one detector's records -> {(dataset, source): scores split
#                   into calibration/evaluation halves}, identical split rules
#                   to E20-v2/E21 (seed 2026, fraction 0.5, path-hash split).
# diagnose()        H1: per forensic source, metadata medians + per-arm median
#                   score on authentic images.
# loso()            H2: for each held-out real pipeline, fit the threshold on
#                   the OTHER pipelines' calibration halves under three rules
#                   (defactify_only / pooled / worst_source), then measure FP
#                   on the held-out pipeline's evaluation half and recall on
#                   the untouched Defactify generator halves.
# abstention()      H3: t_ai from the worst-source rule over all pipelines'
#                   calibration halves, t_real from the generators'
#                   calibration halves (10% miss budget), then per-population
#                   decided/abstained rates on evaluation halves only.
#
# Results: printed tables + artifacts/e22/results.json.  Usage:
#   PYTHONPATH=src .venv/bin/python experiments/e22_source_robust_calibration.py
# =============================================================================

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from pixelproof.evaluation_protocol import (aggregate_tile_scores, safe_auc,
                                            stable_calibration_split,
                                            threshold_at_fpr)

SPLIT_SEED = 2026
CAL_FRACTION = 0.5
FP_BUDGET = 0.10
MISS_BUDGET = 0.10
GENERATORS = ("dalle3", "midjourney", "sd21", "sd3", "sdxl")

ARMS = {
    "tile_resnet18": ("artifacts/e20/raw_scores/resnet18_seed42.jsonl", "top3"),
    "cf_vit": ("artifacts/e21/cf_vit_scores.jsonl", None),
    "bfree": ("artifacts/e21/bfree_scores.jsonl", None),
}


def load_arm(path: str, rule: str | None) -> dict[tuple[str, str], dict]:
    """Read one JSONL into per-population calibration/evaluation score arrays."""
    populations: dict[tuple[str, str], list[dict]] = defaultdict(list)
    with open(path) as handle:
        for line in handle:
            record = json.loads(line)
            populations[(record["dataset"], record["source"])].append(record)

    def score_of(record: dict) -> float:
        if rule is None:
            return float(record["image_score"])
        return aggregate_tile_scores(record["tile_scores"], rule)

    split: dict[tuple[str, str], dict] = {}
    for key, records in populations.items():
        calibration, evaluation = stable_calibration_split(
            records, CAL_FRACTION, SPLIT_SEED
        )
        split[key] = {
            "cal": np.asarray([score_of(r) for r in calibration]),
            "ev": np.asarray([score_of(r) for r in evaluation]),
            "meta": records,
        }
    return split


def rate(scores: np.ndarray, threshold: float) -> float:
    scores = scores[np.isfinite(scores)]
    if not len(scores) or not np.isfinite(threshold):
        return float("nan")
    return float((scores >= threshold).mean())


def real_pipelines(arm: dict) -> dict[str, dict]:
    """The eleven real populations: Defactify real + ten forensic sources."""
    named = {"defactify_real": arm[("defactify", "real")]}
    for (dataset, source), value in arm.items():
        if dataset == "forensics":
            named[source] = value
    return named


def generator_recall(arm: dict, threshold: float) -> dict[str, float]:
    return {
        generator: rate(arm[("defactify", generator)]["ev"], threshold)
        for generator in GENERATORS
    }


def diagnose(arms: dict[str, dict]) -> list[dict]:
    """H1 — what makes NIST2016 different, in numbers every arm shares."""
    rows = []
    reference = arms["bfree"]
    for name, population in sorted(real_pipelines(reference).items()):
        records = population["meta"]
        widths = np.array([r["width"] for r in records])
        heights = np.array([r["height"] for r in records])
        bpp = np.array([r["bytes_per_pixel"] for r in records])
        row = {
            "source": name,
            "n": len(records),
            "median_megapixels": float(np.median(widths * heights) / 1e6),
            "median_bytes_per_pixel": float(np.median(bpp)),
        }
        for arm_name, arm in arms.items():
            pipeline = real_pipelines(arm)[name]
            scores = np.r_[pipeline["cal"], pipeline["ev"]]
            row[f"median_score_{arm_name}"] = float(np.median(scores))
        rows.append(row)
    return rows


def fit_threshold(rule: str, sources: dict[str, dict]) -> float:
    """Fit one threshold at FP_BUDGET from calibration halves under a rule."""
    if rule == "pooled":
        pooled = np.concatenate([s["cal"] for s in sources.values()])
        return threshold_at_fpr(pooled, FP_BUDGET)
    if rule == "worst_source":
        cuts = [threshold_at_fpr(s["cal"], FP_BUDGET) for s in sources.values()]
        return float(max(cut for cut in cuts if np.isfinite(cut)))
    raise ValueError(rule)


def loso(arm: dict) -> dict:
    """H2 — three calibration rules, each judged on held-out pipelines only."""
    pipelines = real_pipelines(arm)
    result: dict[str, dict] = {}

    defactify_cut = threshold_at_fpr(pipelines["defactify_real"]["cal"], FP_BUDGET)
    result["defactify_only"] = {
        "held_out_fp": {
            name: rate(p["ev"], defactify_cut)
            for name, p in pipelines.items()
            if name != "defactify_real"
        },
        "recall": generator_recall(arm, defactify_cut),
        "threshold": defactify_cut,
    }

    for rule in ("pooled", "worst_source"):
        held_out_fp: dict[str, float] = {}
        recalls: list[dict[str, float]] = []
        for held_out in pipelines:
            others = {n: p for n, p in pipelines.items() if n != held_out}
            cut = fit_threshold(rule, others)
            held_out_fp[held_out] = rate(pipelines[held_out]["ev"], cut)
            recalls.append(generator_recall(arm, cut))
        result[rule] = {
            "held_out_fp": held_out_fp,
            # recall barely varies across folds; report the mean per generator
            "recall": {
                generator: float(np.mean([fold[generator] for fold in recalls]))
                for generator in GENERATORS
            },
            "threshold": fit_threshold(rule, pipelines),
        }

    for rule_result in result.values():
        fps = [v for v in rule_result["held_out_fp"].values() if np.isfinite(v)]
        rule_result["macro_fp"] = float(np.mean(fps))
        rule_result["worst_fp"] = float(np.max(fps))
        rule_result["macro_recall"] = float(
            np.mean(list(rule_result["recall"].values()))
        )
    return result


def abstention(arm: dict) -> dict:
    """H3 — the two-threshold band, fitted on calibration halves only."""
    pipelines = real_pipelines(arm)
    t_ai = fit_threshold("worst_source", pipelines)
    ai_calibration = np.concatenate(
        [arm[("defactify", g)]["cal"] for g in GENERATORS]
    )
    # t_real: at most MISS_BUDGET of calibration AI scores may fall below it.
    finite = np.sort(ai_calibration[np.isfinite(ai_calibration)])
    t_real = float(finite[int(np.floor(MISS_BUDGET * len(finite)))])
    if t_real >= t_ai:  # degenerate band: fall back to a decisionless midpoint
        t_real = t_ai

    def verdicts(scores: np.ndarray) -> dict[str, float]:
        scores = scores[np.isfinite(scores)]
        if not len(scores):
            return {"ai": float("nan"), "abstain": float("nan"), "real": float("nan")}
        return {
            "ai": float((scores >= t_ai).mean()),
            "abstain": float(((scores > t_real) & (scores < t_ai)).mean()),
            "real": float((scores <= t_real).mean()),
        }

    populations: dict[str, dict] = {}
    for name, pipeline in pipelines.items():
        populations[f"real:{name}"] = verdicts(pipeline["ev"])
    for generator in GENERATORS:
        populations[f"ai:{generator}"] = verdicts(arm[("defactify", generator)]["ev"])

    real_fp = [populations[f"real:{n}"]["ai"] for n in pipelines]
    ai_rows = [populations[f"ai:{g}"] for g in GENERATORS]
    return {
        "t_ai": t_ai,
        "t_real": t_real,
        "populations": populations,
        "summary": {
            "worst_real_fp": float(np.nanmax(real_fp)),
            "macro_real_fp": float(np.nanmean(real_fp)),
            "macro_ai_recall": float(np.nanmean([row["ai"] for row in ai_rows])),
            "macro_ai_wrongly_real": float(
                np.nanmean([row["real"] for row in ai_rows])
            ),
            "macro_ai_abstain": float(np.nanmean([row["abstain"] for row in ai_rows])),
        },
    }


def main() -> None:
    arms = {
        name: load_arm(path, rule) for name, (path, rule) in ARMS.items()
    }

    print("=" * 96)
    print("E22 · H1 — the poison-source diagnostic (authentic pipelines)")
    print("=" * 96)
    diagnostic = diagnose(arms)
    header = f"{'source':22s} {'n':>4s} {'Mpx':>6s} {'B/px':>6s}" + "".join(
        f"{name:>16s}" for name in ARMS
    )
    print(header)
    for row in diagnostic:
        print(
            f"{row['source']:22s} {row['n']:4d} {row['median_megapixels']:6.2f} "
            f"{row['median_bytes_per_pixel']:6.2f}"
            + "".join(f"{row[f'median_score_{name}']:16.3f}" for name in ARMS)
        )

    loso_results: dict[str, dict] = {}
    abstention_results: dict[str, dict] = {}
    for name, arm in arms.items():
        print("\n" + "=" * 96)
        print(f"E22 · H2 — leave-one-source-out threshold transfer · {name}")
        print("=" * 96)
        loso_results[name] = loso(arm)
        print(f"{'rule':16s} {'worst FP':>9s} {'macro FP':>9s} {'recall':>8s}")
        for rule, values in loso_results[name].items():
            print(
                f"{rule:16s} {100 * values['worst_fp']:8.1f}% "
                f"{100 * values['macro_fp']:8.1f}% {100 * values['macro_recall']:7.1f}%"
            )

        abstention_results[name] = abstention(arm)
        summary = abstention_results[name]["summary"]
        print(
            f"H3 band  ·  worst real FP {100 * summary['worst_real_fp']:.1f}%  ·  "
            f"macro real FP {100 * summary['macro_real_fp']:.1f}%  ·  "
            f"AI recall {100 * summary['macro_ai_recall']:.1f}%  ·  "
            f"AI abstain {100 * summary['macro_ai_abstain']:.1f}%  ·  "
            f"AI wrongly-real {100 * summary['macro_ai_wrongly_real']:.1f}%"
        )

    output = Path("artifacts/e22/results.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "protocol": {
            "split_seed": SPLIT_SEED,
            "calibration_fraction": CAL_FRACTION,
            "fp_budget": FP_BUDGET,
            "miss_budget": MISS_BUDGET,
            "arms": {name: path for name, (path, _) in ARMS.items()},
        },
        "diagnostic": diagnostic,
        "loso": loso_results,
        "abstention": abstention_results,
    }
    output.write_text(json.dumps(payload, indent=2))
    print(f"\nResults -> {output}")


if __name__ == "__main__":
    main()
