# E15 — Step 0: retrain with a class-balanced, multi-source real half.
#
# E14 proved real-class diversity is the dominant lever (AUC 0.55 -> 0.884) but
# left every arm at a 1:14 class imbalance, which inflated the absolute
# false-positive rates. AUC was unaffected — it is threshold-free — but the
# operating point was unreadable. This run fixes the imbalance so the numbers
# mean something in deployment.
#
# Three arms, each evaluated the same way:
#   v1              9,917 GenImage images — the shipped model
#   v2            101,027 pool images, real half still source-skewed
#   v3-balanced    class-balanced, real half drawn evenly from all five sources
#
# What matters here is not AUC. It is the FALSE-POSITIVE RATE on real
# photographs from sources the model never trained on, at a threshold that keeps
# AI recall usable — the thing E13 showed the tile model does not have.

from collections import Counter
from pathlib import Path

import csv
import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from pixelproof.feature_experiment import stack

CACHE = Path("artifacts/features")
INDEX = Path("artifacts/pool_balanced.csv")
POOL = Path("artifacts/pool_features.npz")
SEEDS = [42, 1337, 2024]
GENS = ["dalle3", "midjourney", "sd21", "sd3", "sdxl"]


def fit(x, y, seed):
    return make_pipeline(StandardScaler(),
                         HistGradientBoostingClassifier(random_state=seed)).fit(x, y)


def build_balanced(x, y, source, rng):
    """Equal real/AI, and the real half drawn evenly across every source.

    E14's arms kept all 50,940 AI against a 3,697 real budget. Here both halves
    are the same size and the real half is stratified by source, so no single
    camera pipeline dominates what 'real' means to the model.
    """
    real_sources = [s for s in dict.fromkeys(source[y == 0])
                    if ((y == 0) & (source == s)).sum() >= 1000]
    per_source = min(int(((y == 0) & (source == s)).sum()) for s in real_sources)
    real_parts = []
    for s in real_sources:
        pool = x[(y == 0) & (source == s)]
        real_parts.append(pool[rng.permutation(len(pool))[:per_source]])
    real_x = np.concatenate(real_parts)

    ai_pool = x[y == 1]
    ai_x = ai_pool[rng.permutation(len(ai_pool))[:len(real_x)]]
    return (np.concatenate([real_x, ai_x]),
            np.r_[np.zeros(len(real_x)), np.ones(len(ai_x))],
            real_sources, per_source)


def main() -> None:
    with INDEX.open() as handle:
        meta = {r["phash"]: r["source"] for r in csv.DictReader(handle)}
    pool = np.load(POOL, allow_pickle=True)
    x_pool, y_pool = pool["x"], pool["y"]
    source = np.array([meta.get(h, "?") for h in pool["hashes"]])

    rng = np.random.RandomState(42)
    x_bal, y_bal, real_sources, per_source = build_balanced(x_pool, y_pool, source, rng)
    print(f"balanced pool: {len(x_bal):,}  "
          f"({int((y_bal == 0).sum()):,} real / {int((y_bal == 1).sum()):,} ai)")
    print(f"  real half: {per_source:,} from each of {len(real_sources)} sources")
    for s in real_sources:
        print(f"    {s}")

    x1, y1, _ = stack(["genimage_train_real", "genimage_train_ai"], None, CACHE)
    arms = {
        "v1 (9.9k GenImage)": (x1, y1),
        "v2 (101k, skewed real)": (x_pool, y_pool),
        "v3 (balanced, multi-source)": (x_bal, y_bal),
    }
    models = {name: [fit(x, y, s) for s in SEEDS] for name, (x, y) in arms.items()}

    # ---- evaluation ---------------------------------------------------------
    sets = {}
    for key, names in (("genimage test", ["genimage_test_real", "genimage_test_ai"]),
                       ("archive1", ["archive1_real", "archive1_ai"])):
        x, y, _ = stack(names, None, CACHE)
        sets[key] = (x[y == 0], x[y == 1])
    real_def, _, _ = stack(["defactify_real"], None, CACHE)
    per_gen = {g: stack([f"defactify_{g}"], None, CACHE)[0] for g in GENS}
    sets["defactify"] = (real_def, np.concatenate([per_gen[g] for g in GENS]))

    def mean_auc(ms, xr, xa):
        y = np.r_[np.zeros(len(xr)), np.ones(len(xa))]
        return float(np.mean([roc_auc_score(y, m.predict_proba(np.r_[xr, xa])[:, 1]) for m in ms]))

    def fp(ms, xr, cut=0.5):
        return float(np.mean([(m.predict_proba(xr)[:, 1] >= cut).mean() for m in ms]))

    def recall(ms, xa, cut=0.5):
        return float(np.mean([(m.predict_proba(xa)[:, 1] >= cut).mean() for m in ms]))

    print("\n" + "=" * 78)
    print("AUC — ranking quality")
    print("=" * 78)
    print(f"{'eval set':<20}" + "".join(f"{k.split(' ')[0]:>18}" for k in arms))
    for key, (xr, xa) in sets.items():
        print(f"{key:<20}" + "".join(f"{mean_auc(models[k], xr, xa):>18.3f}" for k in arms))

    print("\n" + "=" * 78)
    print("FALSE POSITIVES on real photographs at threshold 0.5 — the deployment number")
    print("=" * 78)
    print(f"{'real source':<20}" + "".join(f"{k.split(' ')[0]:>18}" for k in arms))
    reals = {"genimage (trained)": sets["genimage test"][0],
             "archive1 (unseen)": sets["archive1"][0],
             "defactify (unseen)": real_def}
    for key, xr in reals.items():
        print(f"{key:<20}" + "".join(f"{fp(models[k], xr)*100:>17.1f}%" for k in arms))

    print("\n" + "=" * 78)
    print("AI recall at threshold 0.5 — what the false-positive fix costs")
    print("=" * 78)
    print(f"{'generator':<20}" + "".join(f"{k.split(' ')[0]:>18}" for k in arms))
    for g in GENS:
        print(f"{g:<20}" + "".join(f"{recall(models[k], per_gen[g])*100:>17.1f}%" for k in arms))

    print("\n" + "=" * 78)
    print("USABLE OPERATING POINT — AI recall at a 10% false-positive budget")
    print("=" * 78)
    for name in arms:
        ms = models[name]
        scores = np.mean([m.predict_proba(real_def)[:, 1] for m in ms], axis=0)
        cut = float(np.percentile(scores, 90))
        every = np.concatenate([per_gen[g] for g in GENS])
        got = np.mean([(m.predict_proba(every)[:, 1] >= cut).mean() for m in ms])
        print(f"  {name:<30}threshold {cut:.3f}   AI recall {got*100:>5.1f}%")

    joblib.dump(fit(x_bal, y_bal, 42), "artifacts/feature_full_v4.joblib")
    print("\nsaved artifacts/feature_full_v4.joblib")


if __name__ == "__main__":
    main()
