# Tile inference with the EXISTING crop128 model — no retraining.
#
# Each tile is a 128x128 native crop, exactly the input the model was fitted on.
# We score every tile of an image and compare aggregation rules, because the
# obvious one (mean) is diluted by flat tiles that carry no measurable trace.

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import joblib
import numpy as np
from PIL import Image
from sklearn.metrics import roc_auc_score

from pixelproof.features import extract_tiles

H = Path.home() / "Desktop"
GENS = ["dalle3", "midjourney", "sd21", "sd3", "sdxl"]
SRC_PX = {"dalle3": 270, "midjourney": 436, "sd21": 768, "sd3": 1024, "sdxl": 1024}
CACHE = Path("/private/tmp/claude-501/-Users-efehankeles-Desktop-ai-image-detector/"
             "238e1a7f-1cab-4e73-92da-2333ce2ae064/scratchpad/tilecache")
TEXTURE_FLOOR = 0.04          # below this a tile is effectively featureless
LIMIT = 1200                  # images per class — enough for a clear signal


def _tiles(path):
    with Image.open(path) as image:
        return extract_tiles(image, tile=128, max_tiles=9)


def tile_scores(folder: Path, model, tag: str) -> list[tuple[np.ndarray, np.ndarray]]:
    """[(per-tile p_ai, per-tile texture)] for each image in the folder."""
    CACHE.mkdir(parents=True, exist_ok=True)
    cached = CACHE / f"{tag}.npz"
    if cached.exists():
        stored = np.load(cached, allow_pickle=True)
        return list(zip(stored["p"], stored["t"]))

    files = sorted(p for p in folder.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"})[:LIMIT]
    with ProcessPoolExecutor() as pool:
        packs = list(pool.map(_tiles, files, chunksize=8))
    out = [(model.predict_proba(features)[:, 1], texture) for features, texture in packs]
    np.savez_compressed(cached,
                        p=np.array([p for p, _ in out], dtype=object),
                        t=np.array([t for _, t in out], dtype=object))
    print(f"  {tag:26s} {len(out):5d} images, {sum(len(p) for p, _ in out)} tiles")
    return out


RULES = {
    "mean":        lambda p, t: float(p.mean()),
    "max":         lambda p, t: float(p.max()),
    "top3 mean":   lambda p, t: float(np.sort(p)[-3:].mean()),
    "median":      lambda p, t: float(np.median(p)),
    "textured only": lambda p, t: float(p[t >= TEXTURE_FLOOR].mean()) if (t >= TEXTURE_FLOOR).any() else float(p.mean()),
    "texture-weighted": lambda p, t: float(np.average(p, weights=np.maximum(t, 1e-3))),
}


def main():
    model = joblib.load("artifacts/feature_crop128.joblib")
    print("scoring tiles…")
    sets = {"real": tile_scores(H / "defactify_test/real", model, "defactify_real")}
    for g in GENS:
        sets[g] = tile_scores(H / "defactify_test/ai" / g, model, f"defactify_{g}")

    # Reference: the single centre crop the demo uses today.
    centre = {k: np.array([p[len(p) // 2] for p, _ in v]) for k, v in sets.items()}

    print("\n" + "=" * 88)
    print("TILE AGGREGATION — same crop128 model, 9 tiles per image, no retraining")
    print("=" * 88)
    header = ["single centre"] + list(RULES)
    print(f"\n{'generator':<13}{'px':>6}" + "".join(f"{h[:15]:>17}" for h in header))
    print("-" * 88)

    per_rule = {h: [] for h in header}
    for g in GENS:
        line = f"{g:<13}{SRC_PX[g]:>6}"
        y = np.r_[np.zeros(len(sets['real'])), np.ones(len(sets[g]))]
        auc = roc_auc_score(y, np.r_[centre["real"], centre[g]])
        per_rule["single centre"].append(auc)
        line += f"{auc:17.3f}"
        for name, rule in RULES.items():
            sr = np.array([rule(p, t) for p, t in sets["real"]])
            sg = np.array([rule(p, t) for p, t in sets[g]])
            auc = roc_auc_score(y, np.r_[sr, sg])
            per_rule[name].append(auc)
            line += f"{auc:17.3f}"
        print(line)

    print("-" * 88)
    print(f"{'MEAN':<13}{'':>6}" + "".join(f"{np.mean(per_rule[h]):17.3f}" for h in header))
    print("\nCNN reference (E7): dalle3 0.896 · midjourney 0.821 · sd21 0.696 · sd3 0.670 · sdxl 0.717   (mean 0.760)")

    # How flat are the tiles in practice? This is what dilutes a plain mean.
    flat = np.concatenate([t for _, t in sets["real"]] + [t for g in GENS for _, t in sets[g]])
    print(f"\ntiles below the texture floor ({TEXTURE_FLOOR}): "
          f"{float((flat < TEXTURE_FLOOR).mean())*100:.1f}% of all tiles")


if __name__ == "__main__":
    main()
