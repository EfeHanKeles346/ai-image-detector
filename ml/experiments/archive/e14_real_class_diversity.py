# E14 — is the failure caused by a NARROW REAL CLASS?
#
# E13 measured the tile model on real photographs and found the pattern that
# prompted this:
#
#     genimage reals (TRAINED ON)   median 0.461   45% called AI
#     defactify reals (unseen)      median 0.935   93% called AI
#     archive1 reals  (unseen)      median 0.939   99% called AI
#
# That is not a detector answering "does this look generated". It is a detector
# answering "does this look like ImageNet", and calling everything else AI.
#
# Every experiment so far asked whether the AI class generalises to an unseen
# GENERATOR. None asked whether the real class generalises to an unseen CAMERA
# PIPELINE. This closes that gap.
#
# HYPOTHESIS (pre-registered)
#   Training the real class on ONE source produces a model that rejects real
#   photographs from any other source. Widening the real class to several
#   sources — with the AI class held fixed — cuts the false-positive rate on
#   held-out real sources, at little or no cost to AI recall.
#
# DESIGN
#   The AI half is identical in every arm; only the real half changes. Each arm
#   is evaluated on real photographs from sources it never trained on, so the
#   number that matters is the false-positive rate on an unseen camera pipeline.
#
# Grounding: IMAGE_FORENSICS_REFERENCE.md 4.1 — a detector should read camera
# traces (PRNU, CFA, compression history), which are physics and therefore
# source-independent, not "unlike my training set", which is source identity.

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

INDEX = Path("artifacts/pool_balanced.csv")
FEATURES = Path("artifacts/pool_features.npz")
SEEDS = [42, 1337, 2024]
MIN_REAL_PER_SOURCE = 2000


def load():
    with INDEX.open() as handle:
        meta = {r["phash"]: r["source"] for r in csv.DictReader(handle)}
    data = np.load(FEATURES, allow_pickle=True)
    source = np.array([meta.get(h, "?") for h in data["hashes"]])
    return data["x"], data["y"], source


def fit(x, y, seed):
    return make_pipeline(StandardScaler(),
                         HistGradientBoostingClassifier(random_state=seed)).fit(x, y)


def main() -> None:
    x, y, source = load()
    real_sources = [s for s in dict.fromkeys(source[y == 0])
                    if (source[y == 0] == s).sum() >= MIN_REAL_PER_SOURCE]
    print("real-photograph sources in the pool:")
    for s in real_sources:
        print(f"  {s:<24}{int(((y == 0) & (source == s)).sum()):>8,}")

    ai_x = x[y == 1]                       # identical in every arm
    print(f"\nAI half (fixed across all arms): {len(ai_x):,}")

    # Each arm trains on real photographs from a subset of sources and is tested
    # on the sources it never saw. Sizes are matched so the comparison is about
    # DIVERSITY, not volume.
    per_source = min(int(((y == 0) & (source == s)).sum()) for s in real_sources)
    budget = per_source                    # total real images each arm may use
    rng = np.random.RandomState(42)
    pools = {s: x[(y == 0) & (source == s)] for s in real_sources}

    arms = {f"only {s}": [s] for s in real_sources}
    arms["all sources"] = real_sources

    print(f"\neach arm trains on {budget:,} real images (equal budget) + {len(ai_x):,} AI")
    print("=" * 92)
    print(f"{'training real sources':<28}" +
          "".join(f"{s[:11]:>13}" for s in real_sources) + f"{'held-out FP':>14}")
    print("-" * 92)

    for arm, used in arms.items():
        take = budget // len(used)
        real_x = np.concatenate([pools[s][rng.permutation(len(pools[s]))[:take]] for s in used])
        train_x = np.concatenate([real_x, ai_x])
        train_y = np.r_[np.zeros(len(real_x)), np.ones(len(ai_x))]
        models = [fit(train_x, train_y, s) for s in SEEDS]

        line, held_out = f"{arm:<28}", []
        for s in real_sources:
            # evaluate on images this arm did not train on
            pool_s = pools[s]
            rates = [float((m.predict_proba(pool_s)[:, 1] >= 0.5).mean()) for m in models]
            rate = float(np.mean(rates))
            mark = "*" if s in used else " "
            line += f"{rate * 100:>12.1f}{mark}"
            if s not in used:
                held_out.append(rate)
        line += f"{np.mean(held_out) * 100:>13.1f}%" if held_out else f"{'—':>14}"
        print(line)

    print("-" * 92)
    print("cells are FALSE-POSITIVE rates: % of real photographs called AI. * = trained on.")

    # Does widening the real class cost AI recall?
    print("\nAI recall (same threshold 0.5), to check what diversity costs:")
    for arm, used in arms.items():
        take = budget // len(used)
        real_x = np.concatenate([pools[s][rng.permutation(len(pools[s]))[:take]] for s in used])
        train_x = np.concatenate([real_x, ai_x])
        train_y = np.r_[np.zeros(len(real_x)), np.ones(len(ai_x))]
        models = [fit(train_x, train_y, s) for s in SEEDS]
        recall = np.mean([float((m.predict_proba(ai_x)[:, 1] >= 0.5).mean()) for m in models])
        auc = np.mean([roc_auc_score(np.r_[np.zeros(len(x[y == 0])), np.ones(len(ai_x))],
                                     m.predict_proba(np.concatenate([x[y == 0], ai_x]))[:, 1])
                       for m in models])
        print(f"  {arm:<28}recall {recall * 100:>5.1f}%   AUC(all pool) {auc:.3f}")


if __name__ == "__main__":
    main()
