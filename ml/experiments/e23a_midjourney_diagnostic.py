# =============================================================================
# e23a_midjourney_diagnostic.py — WHAT THIS EXPERIMENT DOES
# -----------------------------------------------------------------------------
# E22's abstention band has one honest embarrassment: 40% of Midjourney images
# receive an active "real" verdict from the B-Free arm (not abstention — a
# wrong decision).  This experiment asks why, and what the cheapest fix costs.
# Everything runs on the cached per-image scores; no model is loaded.
#
# PRE-REGISTERED HYPOTHESES
# -----------------------------------------------------------------------------
# H1 — the wrongly-real Midjourney images are not random: they differ from the
#      caught ones in a measurable way (score sits low because of resolution /
#      compression, or the whole Midjourney distribution sits low for B-Free).
# H2 — the "real" verdict is the band's weakest promise.  Tightening t_real
#      (miss budget 10% -> 5% -> 2% -> none) converts wrongly-real into
#      abstention at a measurable price: authentic sources lose "real"
#      coverage.  The frontier decides whether an asymmetric band (no "real"
#      verdict at all) is worth it.
#
# PER BLOCK
# -----------------------------------------------------------------------------
# load_arm()     same loader contract as e22 (path-hash splits, seed 2026).
# profile()      H1: wrongly-real vs caught Midjourney images — median size,
#                bytes/pixel, and score per group, for both external arms.
# frontier()     H2: sweep the miss budget; report per point — MJ wrongly-real,
#                macro AI wrongly-real, real-source "real" coverage, abstention.
#
# Results: printed tables + artifacts/e23/e23a_results.json.  Usage:
#   PYTHONPATH=src .venv/bin/python experiments/e23a_midjourney_diagnostic.py
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
GENERATORS = ("dalle3", "midjourney", "sd21", "sd3", "sdxl")
MISS_BUDGETS = (0.10, 0.05, 0.02, None)  # None = no t_real, no "real" verdict

ARMS = {
    "cf_vit": "artifacts/e21/cf_vit_scores.jsonl",
    "bfree": "artifacts/e21/bfree_scores.jsonl",
}


def load_arm(path: str) -> dict[tuple[str, str], dict]:
    populations: dict[tuple[str, str], list[dict]] = defaultdict(list)
    with open(path) as handle:
        for line in handle:
            record = json.loads(line)
            populations[(record["dataset"], record["source"])].append(record)
    split: dict[tuple[str, str], dict] = {}
    for key, records in populations.items():
        calibration, evaluation = stable_calibration_split(
            records, CAL_FRACTION, SPLIT_SEED
        )
        split[key] = {"cal": calibration, "ev": evaluation}
    return split


def scores(records: list[dict]) -> np.ndarray:
    return np.asarray([r["image_score"] for r in records], dtype=np.float64)


def real_pipelines(arm: dict) -> dict[str, dict]:
    named = {"defactify_real": arm[("defactify", "real")]}
    for (dataset, source), value in arm.items():
        if dataset == "forensics":
            named[source] = value
    return named


def worst_source_threshold(arm: dict) -> float:
    cuts = [
        threshold_at_fpr(scores(p["cal"]), FP_BUDGET)
        for p in real_pipelines(arm).values()
    ]
    return float(max(cut for cut in cuts if np.isfinite(cut)))


def t_real_for(arm: dict, miss_budget: float) -> float:
    pooled = np.concatenate(
        [scores(arm[("defactify", g)]["cal"]) for g in GENERATORS]
    )
    finite = np.sort(pooled[np.isfinite(pooled)])
    return float(finite[int(np.floor(miss_budget * len(finite)))])


def profile(arm: dict, t_real: float) -> dict:
    """H1 — who exactly gets the wrong 'real' verdict inside Midjourney?"""
    records = arm[("defactify", "midjourney")]["ev"]
    groups = {
        "wrongly_real": [r for r in records if r["image_score"] <= t_real],
        "not_real": [r for r in records if r["image_score"] > t_real],
    }
    out: dict[str, dict] = {}
    for name, group in groups.items():
        if not group:
            out[name] = {"n": 0}
            continue
        out[name] = {
            "n": len(group),
            "median_long_side": float(
                np.median([max(r["width"], r["height"]) for r in group])
            ),
            "median_bytes_per_pixel": float(
                np.median([r["bytes_per_pixel"] for r in group])
            ),
            "median_score": float(np.median([r["image_score"] for r in group])),
        }
    # distribution context: where does each generator's evaluation half sit?
    out["generator_median_scores"] = {
        g: float(np.median(scores(arm[("defactify", g)]["ev"])))
        for g in GENERATORS
    }
    out["defactify_real_median"] = float(
        np.median(scores(arm[("defactify", "real")]["ev"]))
    )
    return out


def frontier(arm: dict) -> list[dict]:
    """H2 — the price of the 'real' verdict, swept over the miss budget."""
    t_ai = worst_source_threshold(arm)
    pipelines = real_pipelines(arm)
    rows = []
    for miss_budget in MISS_BUDGETS:
        t_real = -np.inf if miss_budget is None else t_real_for(arm, miss_budget)
        if t_real >= t_ai:
            t_real = t_ai

        def rates(values: np.ndarray) -> tuple[float, float, float]:
            return (
                float((values >= t_ai).mean()),
                float(((values > t_real) & (values < t_ai)).mean()),
                float((values <= t_real).mean()),
            )

        per_generator_wrong = {}
        ai_wrong, ai_catch = [], []
        for generator in GENERATORS:
            catch, _, wrong = rates(scores(arm[("defactify", generator)]["ev"]))
            per_generator_wrong[generator] = wrong
            ai_wrong.append(wrong)
            ai_catch.append(catch)
        real_coverage = [
            rates(scores(p["ev"]))[2] for p in pipelines.values()
        ]
        rows.append(
            {
                "miss_budget": miss_budget,
                "t_real": None if np.isneginf(t_real) else t_real,
                "mj_wrongly_real": per_generator_wrong["midjourney"],
                "dalle3_wrongly_real": per_generator_wrong["dalle3"],
                "macro_ai_wrongly_real": float(np.mean(ai_wrong)),
                "macro_ai_recall": float(np.mean(ai_catch)),
                "macro_real_coverage": float(np.mean(real_coverage)),
                "min_real_coverage": float(np.min(real_coverage)),
            }
        )
    return rows


def main() -> None:
    payload: dict[str, dict] = {}
    for name, path in ARMS.items():
        arm = load_arm(path)
        t_ai = worst_source_threshold(arm)
        t_real_10 = t_real_for(arm, 0.10)

        print("=" * 96)
        print(f"E23a · {name} — H1: who gets the wrong 'real' verdict? "
              f"(t_ai {t_ai:.3f}, t_real@10% {t_real_10:.3f})")
        print("=" * 96)
        h1 = profile(arm, t_real_10)
        for group in ("wrongly_real", "not_real"):
            row = h1[group]
            if row["n"] == 0:
                print(f"{group:14s} n=0")
                continue
            print(
                f"{group:14s} n={row['n']:3d}  long side {row['median_long_side']:.0f}px"
                f"  B/px {row['median_bytes_per_pixel']:.2f}"
                f"  median score {row['median_score']:.3f}"
            )
        print("generator medians:", {
            g: round(v, 3) for g, v in h1["generator_median_scores"].items()
        })
        print(f"defactify real median: {h1['defactify_real_median']:.3f}")

        print(f"\nH2 — the miss-budget frontier ({name})")
        print(
            f"{'budget':>7s} {'MJ wrong-real':>14s} {'macro wrong-real':>17s} "
            f"{'AI recall':>10s} {'real coverage':>14s} {'min coverage':>13s}"
        )
        h2 = frontier(arm)
        for row in h2:
            budget = "none" if row["miss_budget"] is None else f"{row['miss_budget']:.0%}"
            print(
                f"{budget:>7s} {100 * row['mj_wrongly_real']:13.1f}% "
                f"{100 * row['macro_ai_wrongly_real']:16.1f}% "
                f"{100 * row['macro_ai_recall']:9.1f}% "
                f"{100 * row['macro_real_coverage']:13.1f}% "
                f"{100 * row['min_real_coverage']:12.1f}%"
            )
        payload[name] = {"h1": h1, "h2": h2, "t_ai": t_ai}
        print()

    output = Path("artifacts/e23/e23a_results.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2))
    print(f"Results -> {output}")


if __name__ == "__main__":
    main()
