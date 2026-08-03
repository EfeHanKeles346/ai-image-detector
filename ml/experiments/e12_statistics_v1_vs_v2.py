# E12 — statistics model: v1 (9,917 GenImage images) vs v2 (101k balanced pool).
#
# One variable changes: the training data. Same 68 features, same pipeline, same
# test sets. v2's pool is resolution-balanced (real 431px vs ai 400px median,
# 1.08x) because the raw merged pool carried a 3.4x gap that a native-resolution
# model reads as a shortcut — see ROADMAP 1b.
#
# Evaluated on everything we hold, and reported per generator, because a single
# pooled number hides where a detector actually fails.

import argparse
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from pixelproof.feature_experiment import stack

CACHE = Path("artifacts/features")
GENS = ["dalle3", "midjourney", "sd21", "sd3", "sdxl"]
SRC_PX = {"dalle3": 270, "midjourney": 436, "sd21": 768, "sd3": 1024, "sdxl": 1024}
SEEDS = [42, 1337, 2024]


def fit(x, y, seed):
    return make_pipeline(StandardScaler(),
                         HistGradientBoostingClassifier(random_state=seed)).fit(x, y)


def auc(model, x_real, x_ai):
    y = np.r_[np.zeros(len(x_real)), np.ones(len(x_ai))]
    s = model.predict_proba(np.r_[x_real, x_ai])[:, 1]
    return roc_auc_score(y, s), float(((s >= 0.5).astype(int) == y).mean()), \
        float((s[: len(x_real)] >= 0.5).mean())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool", type=Path, default=Path("artifacts/pool_features.npz"))
    parser.add_argument("--save", type=Path, default=Path("artifacts/feature_full_v2.joblib"))
    args = parser.parse_args()

    # ---- training data -----------------------------------------------------
    x1, y1, _ = stack(["genimage_train_real", "genimage_train_ai"], None, CACHE)
    pool = np.load(args.pool, allow_pickle=True)
    x2, y2 = pool["x"], pool["y"]
    print(f"v1 train: {len(x1):,}  ({int((y1 == 0).sum()):,} real / {int((y1 == 1).sum()):,} ai)")
    print(f"v2 train: {len(x2):,}  ({int((y2 == 0).sum()):,} real / {int((y2 == 1).sum()):,} ai)")

    # A held-out slice of the pool itself, so v2 has an in-distribution number too.
    x2_tr, x2_te, y2_tr, y2_te = train_test_split(x2, y2, test_size=0.1,
                                                  random_state=42, stratify=y2)

    # ---- evaluation sets ---------------------------------------------------
    sets = {}
    for key, names in (("genimage test", ["genimage_test_real", "genimage_test_ai"]),
                       ("archive1", ["archive1_real", "archive1_ai"])):
        x, y, _ = stack(names, None, CACHE)
        sets[key] = (x[y == 0], x[y == 1])
    dre, _, _ = stack(["defactify_real"], None, CACHE)
    per_gen = {g: stack([f"defactify_{g}"], None, CACHE)[0] for g in GENS}
    sets["defactify (all)"] = (dre, np.concatenate([per_gen[g] for g in GENS]))

    # ---- train, 3 seeds ----------------------------------------------------
    print(f"\ntraining {len(SEEDS)} seeds each…")
    v1 = [fit(x1, y1, s) for s in SEEDS]
    v2 = [fit(x2_tr, y2_tr, s) for s in SEEDS]

    def report(models, x_real, x_ai):
        rows = np.array([auc(m, x_real, x_ai) for m in models])
        return rows.mean(0), rows.std(0)

    print("\n" + "=" * 78)
    print("E12  statistics v1 (9.9k) vs v2 (101k balanced) — mean +/- std over 3 seeds")
    print("=" * 78)
    print(f"\n{'eval set':<24}{'v1 AUC':>16}{'v2 AUC':>16}{'delta':>9}")
    print("-" * 66)
    for key, (x_real, x_ai) in sets.items():
        m1, s1 = report(v1, x_real, x_ai)
        m2, s2 = report(v2, x_real, x_ai)
        print(f"{key:<24}{m1[0]:>10.3f}±{s1[0]:.3f}{m2[0]:>10.3f}±{s2[0]:.3f}{m2[0] - m1[0]:>+9.3f}")
    m2, s2 = report(v2, x2_te[y2_te == 0], x2_te[y2_te == 1])
    print(f"{'pool held-out (v2 only)':<24}{'—':>16}{m2[0]:>10.3f}±{s2[0]:.3f}")

    print(f"\n--- per generator on Defactify (AUC) ---")
    print(f"{'generator':<14}{'src px':>8}{'v1':>16}{'v2':>16}{'delta':>9}")
    for g in GENS:
        m1, s1 = report(v1, dre, per_gen[g])
        m2, s2 = report(v2, dre, per_gen[g])
        print(f"{g:<14}{SRC_PX[g]:>8}{m1[0]:>10.3f}±{s1[0]:.3f}{m2[0]:>10.3f}±{s2[0]:.3f}{m2[0] - m1[0]:>+9.3f}")

    print(f"\n--- false positives on real photographs (lower is better) ---")
    for key, (x_real, x_ai) in sets.items():
        f1 = np.mean([auc(m, x_real, x_ai)[2] for m in v1])
        f2 = np.mean([auc(m, x_real, x_ai)[2] for m in v2])
        print(f"  {key:<24}v1 {f1*100:5.1f}%   v2 {f2*100:5.1f}%   {(f2-f1)*100:+.1f}")

    joblib.dump(fit(x2, y2, 42), args.save)   # final model on the full pool
    print(f"\nsaved {args.save}")


if __name__ == "__main__":
    main()
