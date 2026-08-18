# E16 — Step 2: a frozen backbone with a linear probe, against the 68
# hand-crafted statistics, on identical training images.
#
# E15 closed off the data explanation: three statistics models trained on 9.9k,
# 101k and a class-balanced multi-source pool all land at ~0.69 AUC on Defactify
# and ~35% AI recall at a 10% false-positive budget. More data and better-
# balanced data changed archive1 (0.706 -> 0.904) and left Defactify alone. What
# the features can express is the ceiling.
#
# This is E2's question asked with a strong representation. E2 found that every
# classical classifier landed within 0.2 points of the CNN head and concluded
# "the representation, not the classifier, is the bottleneck" — but it only ever
# tested that with a weak representation. IMAGE_FORENSICS_REFERENCE.md 4.4 names
# the strong one: CLIP-feature detectors are "currently among the best
# out-of-distribution generalizers, even trained on little data".
#
# CONTROLLED: the probe trains on the same balanced multi-source pool as v3
# (pool_balanced_v3.csv), so backbone-vs-statistics is the only variable.
#
# DINOv2 also takes 518px input against CLIP/ConvNeXt's 224, so a 1024px image
# is downscaled 2.0x instead of 4.6x. Running more than one backbone separates
# "better representation" from "less downscaling" — otherwise the two
# explanations are confounded.

import argparse
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

GENS = ["dalle3", "midjourney", "sd21", "sd3", "sdxl"]
SEEDS = [42, 1337, 2024]


def load(root: Path):
    pool = np.load(root / "pool.npz", allow_pickle=True)
    tests = {p.stem: np.load(p) for p in root.glob("*.npy")}
    return pool["x"], pool["y"], pool["sources"], tests


def fit(x, y, seed):
    return make_pipeline(StandardScaler(),
                         LogisticRegression(max_iter=3000, random_state=seed,
                                            class_weight="balanced")).fit(x, y)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", default="dinov2")
    args = parser.parse_args()

    root = Path("artifacts") / f"backbone_{args.backbone}"
    x, y, sources, tests = load(root)
    print(f"{args.backbone}: pool {x.shape}  ({int((y == 0).sum()):,} real / {int((y == 1).sum()):,} ai)")
    print(f"  real sources: {', '.join(sorted(set(sources[y == 0])))}")

    models = [fit(x, y, s) for s in SEEDS]

    def auc(xr, xa):
        yy = np.r_[np.zeros(len(xr)), np.ones(len(xa))]
        return float(np.mean([roc_auc_score(yy, m.predict_proba(np.r_[xr, xa])[:, 1])
                              for m in models]))

    def rate(xs, cut=0.5):
        return float(np.mean([(m.predict_proba(xs)[:, 1] >= cut).mean() for m in models]))

    real_def = tests["defactify_real"]
    per_gen = {g: tests[f"defactify_{g}"] for g in GENS}
    every_ai = np.concatenate([per_gen[g] for g in GENS])

    # Statistics-model numbers from E15, same evaluation, for reference.
    STATS_V3 = {"genimage test": 0.919, "archive1": 0.904, "defactify": 0.692}

    print("\n" + "=" * 72)
    print(f"AUC — {args.backbone} linear probe vs the 68 statistics (E15 v3)")
    print("=" * 72)
    print(f"{'eval set':<22}{'statistics v3':>16}{'backbone probe':>17}{'delta':>10}")
    print("-" * 65)
    pairs = {
        "genimage test": (tests["genimage_test_real"], tests["genimage_test_ai"]),
        "archive1": (tests["archive1_real"], tests["archive1_ai"]),
        "defactify": (real_def, every_ai),
    }
    for key, (xr, xa) in pairs.items():
        got = auc(xr, xa)
        print(f"{key:<22}{STATS_V3[key]:>16.3f}{got:>17.3f}{got - STATS_V3[key]:>+10.3f}")

    print("\n--- per generator on Defactify ---")
    print(f"{'generator':<14}{'src px':>8}{'stats v1':>10}{'probe AUC':>12}{'recall@0.5':>12}")
    STATS_V1 = {"dalle3": 0.808, "midjourney": 0.796, "sd21": 0.676, "sd3": 0.620, "sdxl": 0.685}
    PX = {"dalle3": 270, "midjourney": 436, "sd21": 768, "sd3": 1024, "sdxl": 1024}
    for g in GENS:
        print(f"{g:<14}{PX[g]:>8}{STATS_V1[g]:>10.3f}{auc(real_def, per_gen[g]):>12.3f}"
              f"{rate(per_gen[g]) * 100:>11.1f}%")

    print("\n--- false positives on real photographs (threshold 0.5) ---")
    for key, xs in (("genimage (trained)", tests["genimage_test_real"]),
                    ("archive1 (unseen)", tests["archive1_real"]),
                    ("defactify (unseen)", real_def)):
        print(f"  {key:<24}{rate(xs) * 100:>6.1f}%")

    print("\n--- USABLE OPERATING POINT: AI recall at a 10% false-positive budget ---")
    scores = np.mean([m.predict_proba(real_def)[:, 1] for m in models], axis=0)
    cut = float(np.percentile(scores, 90))
    print(f"  threshold {cut:.3f}   AI recall {rate(every_ai, cut) * 100:.1f}%"
          f"   (statistics v3: 33.7%)")
    for g in GENS:
        print(f"    {g:<20}{rate(per_gen[g], cut) * 100:>6.1f}%")


if __name__ == "__main__":
    main()
