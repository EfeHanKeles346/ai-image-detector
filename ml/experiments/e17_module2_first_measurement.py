# E17 — Module 2's first measurement: does the tile map find the tampered region?
#
# The tile scorer was built for Module 1 (E11) and produces a per-region
# probability as a side effect. ROADMAP 9c called localisation "a well-founded
# hypothesis that is still unvalidated" because there was no ground truth. There
# is now: 13 forensic datasets with pixel-level masks.
#
# NO TRAINING. This asks a narrower question than "can we localise": it asks
# whether a model trained to answer "does this tile look GENERATED" already
# answers "was this tile EDITED". Those coincide for a diffusion-inpainted
# region and diverge for a Photoshop splice, so the result is reported PER
# SUB-DATASET. A single pooled number would hide exactly the distinction that
# matters.
#
# PREDICTION (pre-registered)
#   CocoGlide — diffusion inpainting — should work: the pasted region really is
#   generated texture, which is what the model was trained to see.
#   CASIA 2.0 and Columbia — classic splices from other photographs — should
#   not: the pasted region is camera output, just from a different camera.
#
# METRICS
#   image level: can the tile scores tell a tampered image from an authentic one
#   pixel level: do the high-scoring tiles land on the mask (IoU, and how much
#               better than chance the tile ranking is)

import warnings
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
from PIL import Image
from sklearn.metrics import roc_auc_score

from pixelproof.features import extract_tiles, tile_positions

warnings.filterwarnings("ignore")
Image.MAX_IMAGE_PIXELS = None

ROOT = Path("/tmp/m2")
TILE = 128
GRID = 36
LIMIT = 120


def tile_scores(model, path: Path):
    """Per-tile p(AI) plus each tile's box, so scores can be matched to a mask."""
    with Image.open(path) as image:
        image = image.convert("RGB")
        width, height = image.size
        if width < TILE or height < TILE:
            return None
        features, _ = extract_tiles(image, tile=TILE, max_tiles=GRID)
        boxes = tile_positions(width, height, TILE, GRID)
    return model.predict_proba(features)[:, 1], boxes, (width, height)


def mask_coverage(mask_path: Path, boxes, size):
    """Fraction of each tile that falls inside the ground-truth mask."""
    with Image.open(mask_path) as mask:
        mask = mask.convert("L").resize(size, Image.NEAREST)
        array = np.asarray(mask) > 127
    return np.array([array[y:y + TILE, x:x + TILE].mean() for x, y in boxes]), array.mean()


def main() -> None:
    model = joblib.load("artifacts/feature_crop128.joblib")

    datasets = sorted(d.name for d in ROOT.iterdir() if d.is_dir())
    print(f"sub-datasets found: {', '.join(datasets)}\n")

    for name in datasets:
        manip_dir, auth_dir = ROOT / name / "manip", ROOT / name / "auth"
        if not manip_dir.exists():
            continue

        manip = sorted(p for p in manip_dir.glob("*.png") if ".mask" not in p.name)[:LIMIT]
        auth = sorted(p for p in auth_dir.glob("*.png") if ".mask" not in p.name)[:LIMIT] \
            if auth_dir.exists() else []

        # --- pixel level: do high-scoring tiles land on the mask? -----------
        inside, outside, ious, mask_fracs = [], [], [], []
        image_scores_manip = []
        for path in manip:
            mask_path = path.with_suffix(".mask.png")
            if not mask_path.exists():
                continue
            got = tile_scores(model, path)
            if got is None:
                continue
            scores, boxes, size = got
            coverage, mask_frac = mask_coverage(mask_path, boxes, size)
            if coverage.max() < 0.5 or coverage.min() > 0.5:
                continue                      # no tile clearly in or clearly out
            inside += list(scores[coverage >= 0.5])
            outside += list(scores[coverage < 0.5])
            mask_fracs.append(mask_frac)
            image_scores_manip.append(float(np.sort(scores)[-3:].mean()))

            # IoU between "tiles the model flags" and "tiles the mask covers"
            flagged = scores >= np.percentile(scores, 100 * (1 - mask_frac)) if mask_frac > 0 else scores >= 1
            truth = coverage >= 0.5
            union = (flagged | truth).sum()
            if union:
                ious.append(float((flagged & truth).sum() / union))

        image_scores_auth = []
        for path in auth:
            got = tile_scores(model, path)
            if got is not None:
                image_scores_auth.append(float(np.sort(got[0])[-3:].mean()))

        if not inside:
            print(f"{name}: no usable image/mask pairs")
            continue

        inside_a, outside_a = np.array(inside), np.array(outside)
        y = np.r_[np.zeros(len(outside_a)), np.ones(len(inside_a))]
        tile_auc = roc_auc_score(y, np.r_[outside_a, inside_a])

        print(f"--- {name} ---")
        print(f"  images {len(image_scores_manip)} manip / {len(image_scores_auth)} auth"
              f"   mask covers {np.mean(mask_fracs)*100:.0f}% of the image on average")
        print(f"  PIXEL LEVEL   tampered tiles {inside_a.mean():.3f} vs clean tiles "
              f"{outside_a.mean():.3f}   tile AUC {tile_auc:.3f}   IoU {np.mean(ious):.3f}")
        if image_scores_auth:
            yi = np.r_[np.zeros(len(image_scores_auth)), np.ones(len(image_scores_manip))]
            img_auc = roc_auc_score(yi, np.r_[image_scores_auth, image_scores_manip])
            print(f"  IMAGE LEVEL   manip {np.mean(image_scores_manip):.3f} vs auth "
                  f"{np.mean(image_scores_auth):.3f}   AUC {img_auc:.3f}")
        print()

    print("tile AUC 0.5 = the model cannot tell a tampered tile from a clean one in the")
    print("same image. Above 0.5 = the tile map carries localisation signal.")


if __name__ == "__main__":
    main()
