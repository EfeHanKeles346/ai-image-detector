# =============================================================================
# feature_model.py — WHAT THIS FILE DOES
# -----------------------------------------------------------------------------
# Trains the feature-based detector on GenImage and saves it to disk, so the
# web service (serve.py) can load it without re-training. Two variants are
# stored, because they behave differently and we want both visible in the demo:
#
#   full     — statistics over the WHOLE image, every pixel, no resizing.
#   crop128  — the same statistics on a 128x128 NATIVE centre crop, so that
#              every input reaches the model with identical dimensions.
#
# WHY BOTH
# -----------------------------------------------------------------------------
# Measured on Defactify (5 generators the model never saw), they fail in
# opposite places:
#   generator    src px   full    crop128
#   dalle3          270   0.808     0.377
#   midjourney      436   0.796     0.793
#   sd21            768   0.676     0.784
#   sd3            1024   0.620     0.760
#   sdxl           1024   0.685     0.867
# crop128 wins on high-resolution sources, full wins on small ones. Neither is
# strictly better, so the demo shows both rather than pretending one is right.
#
# CODE BLOCKS IN THIS FILE
# -----------------------------------------------------------------------------
# build_pipeline() StandardScaler + HistGradientBoosting. Gradient boosting is
#                  the standard strong choice on tabular data of this size, and
#                  it beat logistic regression / random forest in our runs.
#
# train()          Loads the cached GenImage feature matrices (produced by
#                  feature_experiment.py), fits one pipeline per variant and
#                  writes them to artifacts/ with joblib.
#
# load()           Used by serve.py. Returns {variant: fitted pipeline}.
#
# score_image()    PIL image -> probability of "AI", per variant. Feature
#                  extraction happens at native resolution: nothing is resized.
# =============================================================================

import argparse
from pathlib import Path

import joblib
import numpy as np
from PIL import Image
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from pixelproof.features import extract_from_image, extract_tiles, tile_positions

VARIANTS = {"full": None, "crop128": 128}
TRAIN_SETS = ["genimage_train_real", "genimage_train_ai"]
TILE_PX = 128       # must match the crop the model was trained on
TILE_GRID = 36      # 6x6 — measured optimum, see score_tiles()


def build_pipeline(seed: int = 42):
    return make_pipeline(StandardScaler(), HistGradientBoostingClassifier(random_state=seed))


def train(cache_dir: Path, output_dir: Path, seed: int = 42) -> None:
    from pixelproof.feature_experiment import stack  # imported here to keep serve.py light

    output_dir.mkdir(parents=True, exist_ok=True)
    for variant, crop in VARIANTS.items():
        x, y, _ = stack(TRAIN_SETS, crop, cache_dir)
        pipeline = build_pipeline(seed).fit(x, y)
        target = output_dir / f"feature_{variant}.joblib"
        joblib.dump(pipeline, target)
        print(f"{variant:8s} trained on {len(x)} images "
              f"({int((y == 0).sum())} real / {int((y == 1).sum())} ai) -> {target.name}")


def load(artifacts_dir: Path) -> dict:
    return {variant: joblib.load(artifacts_dir / f"feature_{variant}.joblib")
            for variant in VARIANTS
            if (artifacts_dir / f"feature_{variant}.joblib").exists()}


def score_image(models: dict, image: Image.Image) -> dict[str, float]:
    scores = {}
    for variant, pipeline in models.items():
        vector = extract_from_image(image, VARIANTS[variant]).reshape(1, -1)
        scores[variant] = float(pipeline.predict_proba(vector)[0, 1])
    return scores


def score_tiles(models: dict, image: Image.Image, grid: int = TILE_GRID) -> dict | None:
    """Cut the image into a grid of 128px NATIVE tiles and score every one.

    No retraining involved: the crop128 model was fitted on 128x128 native
    crops and each tile IS one, so this is the same input distribution
    evaluated several times instead of once.

    Aggregation is the mean of the TOP 3 tiles, not the mean of all of them.
    Measured on Defactify's high-resolution generators, a 6x6 grid with top-3
    beats a plain mean (0.821 vs 0.781) because flat tiles -- sky, a wall,
    plain clothing -- score around 0.5 and drag an ordinary average toward
    "no idea", drowning the tiles that carry real evidence.

    The per-tile scores returned here are also the raw material for a
    localisation heat-map ("where does it look manipulated"), i.e. Module 2.
    """
    pipeline = models.get("crop128")
    if pipeline is None:
        return None
    features, texture = extract_tiles(image, tile=TILE_PX, max_tiles=grid)
    probabilities = pipeline.predict_proba(features)[:, 1]
    width, height = image.size
    positions = (tile_positions(width, height, TILE_PX, grid)
                 if width >= TILE_PX and height >= TILE_PX else [(0, 0)])
    top = float(np.sort(probabilities)[-3:].mean())
    return {
        "p_ai": top,
        "tiles": [
            {"x": x, "y": y, "p_ai": round(float(p), 4), "texture": round(float(t), 4)}
            for (x, y), p, t in zip(positions, probabilities, texture)
        ],
        "tile_px": TILE_PX,
        "image_w": width,
        "image_h": height,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, default=Path("artifacts/features"))
    parser.add_argument("--output", type=Path, default=Path("artifacts"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train(args.cache, args.output, args.seed)


if __name__ == "__main__":
    main()
