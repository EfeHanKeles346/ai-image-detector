# How many tiles, and which aggregation rule? Optimised on the HIGH-RESOLUTION
# generators only — low-resolution inputs are the CNNs' job (SmallCNN / ResNet).
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import joblib
import numpy as np
from PIL import Image
from sklearn.metrics import roc_auc_score

from pixelproof.features import extract_tiles

H = Path.home() / "Desktop"
HIRES = {"sd21": 768, "sd3": 1024, "sdxl": 1024, "midjourney": 436}
GRIDS = [4, 9, 16, 25, 36]
LIMIT = 800


def _tiles(path, max_tiles):
    with Image.open(path) as image:
        features, _ = extract_tiles(image, tile=128, max_tiles=max_tiles)
    return features


def scores(folder, model, max_tiles):
    files = sorted(p for p in folder.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"})[:LIMIT]
    with ProcessPoolExecutor() as pool:
        packs = list(pool.map(partial(_tiles, max_tiles=max_tiles), files, chunksize=8))
    return [model.predict_proba(f)[:, 1] for f in packs]


RULES = {
    "mean": lambda p: p.mean(),
    "top3": lambda p: np.sort(p)[-3:].mean(),
    "top half": lambda p: np.sort(p)[len(p) // 2:].mean(),
    "median": lambda p: np.median(p),
    "max": lambda p: p.max(),
}


def main():
    model = joblib.load("artifacts/feature_crop128.joblib")
    print(f"{'grid':>6}{'tiles/img':>11}" + "".join(f"{r:>11}" for r in RULES))
    print("-" * 72)
    best = None
    for grid in GRIDS:
        real = scores(H / "defactify_test/real", model, grid)
        gens = {g: scores(H / "defactify_test/ai" / g, model, grid) for g in HIRES}
        actual = np.mean([len(p) for p in real])
        line = f"{int(np.sqrt(grid))}x{int(np.sqrt(grid)):<4}{actual:11.1f}"
        for name, rule in RULES.items():
            aucs = []
            for g in HIRES:
                sr = np.array([rule(p) for p in real])
                sg = np.array([rule(p) for p in gens[g]])
                y = np.r_[np.zeros(len(sr)), np.ones(len(sg))]
                aucs.append(roc_auc_score(y, np.r_[sr, sg]))
            mean_auc = float(np.mean(aucs))
            line += f"{mean_auc:11.3f}"
            if best is None or mean_auc > best[0]:
                best = (mean_auc, grid, name, dict(zip(HIRES, aucs)))
        print(line)

    auc, grid, rule, detail = best
    print(f"\nBEST: {int(np.sqrt(grid))}x{int(np.sqrt(grid))} grid + '{rule}'  ->  mean AUC {auc:.3f}")
    print(f"{'generator':<13}{'src px':>8}{'tiled':>9}{'CNN (E7)':>11}{'gain':>9}")
    cnn = {"sd21": 0.696, "sd3": 0.670, "sdxl": 0.717, "midjourney": 0.821}
    for g, value in detail.items():
        print(f"{g:<13}{HIRES[g]:>8}{value:9.3f}{cnn[g]:11.3f}{value - cnn[g]:+9.3f}")


if __name__ == "__main__":
    main()
