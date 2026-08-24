# =============================================================================
# feature_experiment.py — WHAT THIS FILE DOES
# -----------------------------------------------------------------------------
# Runs the feature-based detection experiment end to end:
#   1. turn every image into a 68-number vector (features.py), in two modes
#   2. train classical models on GenImage
#   3. evaluate on data the models have never seen — above all Defactify,
#      whose five generators are all NEWER than anything in training
#
# THE QUESTION THIS ANSWERS
# -----------------------------------------------------------------------------
# The CNN pipeline downscales every image to 224x224 and E7 showed the damage:
# detection quality fell monotonically with how much an image was downscaled
# (270px -> AUC 0.896 ... 1024px -> AUC 0.670). If statistics computed over the
# whole image at native resolution really are resolution-independent, that
# ordering should DISAPPEAR. That is the decisive test.
#
# CODE BLOCKS IN THIS FILE
# -----------------------------------------------------------------------------
# DATASETS         Where each image set lives and which label it carries.
#                  Training = GenImage (5,828 real / 5,831 AI — the only
#                  perfectly balanced set we hold, 7 older generators).
#                  Testing  = Defactify (5 modern, unseen generators),
#                  archive1 (different source entirely), GenImage test.
#
# build_matrix()   Extracts features for a folder using a process pool
#                  (11 cores) and caches the result to .npz, so re-running an
#                  analysis costs seconds instead of minutes.
#
# MODES            "full" = statistics over every pixel of the whole image.
#                  "crop" = same statistics on a 128x128 NATIVE centre crop
#                  (no resampling). Crop mode gives both classes identical
#                  dimensions, so the "every AI image is a square 1024" hint
#                  present in BOTH GenImage and Defactify cannot be used.
#                  Comparing the two modes measures how big that shortcut is.
#
# shortcut_probe() Trains a model to predict image WIDTH from the features
#                  alone. If it succeeds, resolution is leaking into the
#                  vector and any accuracy gain in "full" mode is suspect.
#
# SETUPS           Three learning formulations on identical features:
#                    supervised  — real + AI, ordinary classification
#                    oneclass_real — trained on real photographs only
#                    oneclass_ai   — trained on AI images only
#                  The second is Phase 5's premise (real photographs do not
#                  change when a new generator ships). The third is the
#                  mirror hypothesis, included so the comparison is fair
#                  rather than assumed.
#
# main()           Runs every mode x setup combination and prints one table
#                  per evaluation set, plus the per-generator breakdown that
#                  the whole experiment hinges on.
# =============================================================================

import argparse
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

from pixelproof.features import FEATURE_NAMES, extract
from pixelproof.project_paths import WORK_ROOT

HOME = WORK_ROOT
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".JPEG"}

# name -> (folder, label 1=AI/0=real)
DATASETS = {
    "genimage_train_real": (HOME / "genimage_split/train/REAL", 0),
    "genimage_train_ai":   (HOME / "genimage_split/train/FAKE", 1),
    "genimage_test_real":  (HOME / "genimage_split/test/REAL", 0),
    "genimage_test_ai":    (HOME / "genimage_split/test/FAKE", 1),
    "defactify_real":      (HOME / "defactify_test/real", 0),
    "defactify_sd21":      (HOME / "defactify_test/ai/sd21", 1),
    "defactify_sdxl":      (HOME / "defactify_test/ai/sdxl", 1),
    "defactify_sd3":       (HOME / "defactify_test/ai/sd3", 1),
    "defactify_dalle3":    (HOME / "defactify_test/ai/dalle3", 1),
    "defactify_midjourney": (HOME / "defactify_test/ai/midjourney", 1),
    "archive1_real":       (HOME / "archive1/real_dataset", 0),
    "archive1_ai":         (HOME / "archive1/Ai_generated_dataset", 1),
}

DEFACTIFY_GENERATORS = ["dalle3", "midjourney", "sd21", "sd3", "sdxl"]
SOURCE_RESOLUTION = {"dalle3": 270, "midjourney": 436, "sd21": 768, "sd3": 1024, "sdxl": 1024}


def _one(args):
    path, crop = args
    try:
        from PIL import Image
        with Image.open(path) as image:
            width = image.size[0]
        return extract(path, crop), width
    except Exception:
        return None, None


def build_matrix(name: str, crop: int | None, cache_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Feature matrix for one folder. Returns (features, widths). Cached on disk."""
    tag = f"{name}_{'full' if crop is None else f'crop{crop}'}"
    cache = cache_dir / f"{tag}.npz"
    if cache.exists():
        stored = np.load(cache)
        return stored["x"], stored["w"]

    folder, _ = DATASETS[name]
    files = sorted(p for p in folder.rglob("*") if p.suffix in IMAGE_EXTENSIONS)
    start = time.time()
    with ProcessPoolExecutor() as pool:
        results = list(pool.map(_one, [(f, crop) for f in files], chunksize=16))
    rows = [(v, w) for v, w in results if v is not None]
    x = np.stack([v for v, _ in rows])
    w = np.array([w for _, w in rows], dtype=np.int32)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, x=x, w=w)
    print(f"  {tag:34s} {len(x):6d} images  {time.time() - start:5.0f}s")
    return x, w


def stack(names: list[str], crop: int | None, cache_dir: Path):
    features, labels, widths = [], [], []
    for name in names:
        x, w = build_matrix(name, crop, cache_dir)
        features.append(x)
        widths.append(w)
        labels.append(np.full(len(x), DATASETS[name][1]))
    return np.concatenate(features), np.concatenate(labels), np.concatenate(widths)


def metrics(y_true: np.ndarray, scores: np.ndarray) -> tuple[float, float, float]:
    """AUC, accuracy at 0.5, and recall on the AI class."""
    auc = roc_auc_score(y_true, scores) if len(set(y_true)) > 1 else float("nan")
    predicted = (scores >= 0.5).astype(int)
    accuracy = float((predicted == y_true).mean())
    ai = y_true == 1
    recall = float(predicted[ai].mean()) if ai.any() else float("nan")
    return auc, accuracy, recall


def shortcut_probe(x: np.ndarray, w: np.ndarray) -> float:
    """Can the features alone tell a square-1024 image from a 500px photo?

    Trained to predict 'is the image wider than 600px'. High accuracy means
    resolution is leaking into the vector and 'full' mode results are inflated.
    """
    target = (w > 600).astype(int)
    if len(set(target)) < 2:
        return float("nan")
    split = len(x) // 2
    order = np.random.RandomState(42).permutation(len(x))
    train, test = order[:split], order[split:]
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    model.fit(x[train], target[train])
    return float(model.score(x[test], target[test]))


def fit_setups(x: np.ndarray, y: np.ndarray, seed: int) -> dict:
    """Three learning formulations on identical features."""
    models = {}

    supervised = make_pipeline(StandardScaler(),
                               HistGradientBoostingClassifier(random_state=seed))
    supervised.fit(x, y)
    models["supervised_gb"] = (lambda z: supervised.predict_proba(z)[:, 1])

    logistic = make_pipeline(StandardScaler(),
                             LogisticRegression(max_iter=4000, class_weight="balanced"))
    logistic.fit(x, y)
    models["supervised_logreg"] = (lambda z: logistic.predict_proba(z)[:, 1])

    forest = make_pipeline(StandardScaler(),
                           RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1))
    forest.fit(x, y)
    models["supervised_rf"] = (lambda z: forest.predict_proba(z)[:, 1])

    # One-class: learn ONE class, score everything else as deviation.
    # Trained on real only -> "AI" means "does not look like a photograph".
    scaler_real = StandardScaler().fit(x[y == 0])
    ocsvm_real = OneClassSVM(nu=0.1, gamma="scale").fit(scaler_real.transform(x[y == 0]))
    models["oneclass_real"] = (lambda z: -ocsvm_real.decision_function(scaler_real.transform(z)))

    forest_real = IsolationForest(random_state=seed, n_estimators=300).fit(x[y == 0])
    models["oneclass_real_iforest"] = (lambda z: -forest_real.decision_function(z))

    # Trained on AI only -> "real" means "does not look like the AI we know".
    scaler_ai = StandardScaler().fit(x[y == 1])
    ocsvm_ai = OneClassSVM(nu=0.1, gamma="scale").fit(scaler_ai.transform(x[y == 1]))
    models["oneclass_ai"] = (lambda z: ocsvm_ai.decision_function(scaler_ai.transform(z)))

    return models


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, default=Path("artifacts/features"))
    parser.add_argument("--crop", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    for mode, crop in (("full", None), (f"crop{args.crop}", args.crop)):
        print(f"\n{'=' * 78}\nMODE: {mode}   ({'whole image, every pixel' if crop is None else f'{crop}x{crop} native centre crop, identical dimensions for both classes'})\n{'=' * 78}")

        print("extracting features…")
        x_train, y_train, w_train = stack(["genimage_train_real", "genimage_train_ai"], crop, args.cache)
        print(f"  train: {len(x_train)} images  ({int((y_train==0).sum())} real / {int((y_train==1).sum())} ai)")

        leak = shortcut_probe(x_train, w_train)
        print(f"\n  SHORTCUT PROBE — predicting image size from features alone: {leak*100:.1f}% accurate")
        print("  (high = resolution leaks into the vector; results below are inflated by it)")

        models = fit_setups(x_train, y_train, args.seed)

        evaluations = {
            "genimage test (in-distribution)": ["genimage_test_real", "genimage_test_ai"],
            "archive1 (different source)": ["archive1_real", "archive1_ai"],
            "DEFACTIFY (5 unseen modern generators)":
                ["defactify_real"] + [f"defactify_{g}" for g in DEFACTIFY_GENERATORS],
        }

        for title, names in evaluations.items():
            x_eval, y_eval, _ = stack(names, crop, args.cache)
            print(f"\n  --- {title} — {len(x_eval)} images ---")
            print(f"  {'model':<26}{'AUC':>8}{'acc':>8}{'AI recall':>11}")
            for model_name, score_fn in models.items():
                auc, accuracy, recall = metrics(y_eval, score_fn(x_eval))
                print(f"  {model_name:<26}{auc:8.3f}{accuracy:8.3f}{recall*100:10.1f}%")

        # The decisive table: does the resolution ordering survive?
        print(f"\n  --- PER-GENERATOR (the decisive test) ---")
        x_real, _, _ = stack(["defactify_real"], crop, args.cache)
        print(f"  {'generator':<14}{'source px':>10}" + "".join(f"{n[:14]:>16}" for n in models))
        for generator in DEFACTIFY_GENERATORS:
            x_gen, _, _ = stack([f"defactify_{generator}"], crop, args.cache)
            x_pair = np.concatenate([x_real, x_gen])
            y_pair = np.concatenate([np.zeros(len(x_real)), np.ones(len(x_gen))])
            line = f"  {generator:<14}{SOURCE_RESOLUTION[generator]:>10}"
            for score_fn in models.values():
                line += f"{roc_auc_score(y_pair, score_fn(x_pair)):16.3f}"
            print(line)
        print("\n  CNN reference on the same set (E7): dalle3 0.896 > midjourney 0.821 > "
              "sdxl 0.717 > sd21 0.696 > sd3 0.670  — ordered by resolution.")
        print("  If this method is resolution-independent, that ordering should be gone.")


if __name__ == "__main__":
    main()
