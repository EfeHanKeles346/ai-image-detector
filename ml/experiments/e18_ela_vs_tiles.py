# E18 — Phase 7a: does ELA cover the case the tile model cannot?
#
# E17 split Module 2 cleanly in two:
#   CocoGlide (diffusion inpainting)  tile AUC 0.648  — the pasted region really
#                                                       is generated texture
#   CASIA 2.0 (Photoshop splice)      tile AUC 0.606, image AUC 0.481 — the
#                                                       pasted region is camera
#                                                       output, just from another
#                                                       camera
#
# ELA is built for exactly the second case. IMAGE_FORENSICS_REFERENCE.md 4.3:
# it works on "JPEG splices where donor and host had different qualities or
# histories", and fails BY DESIGN on fully generated images and uniformly
# re-encoded ones. So the prediction is the mirror image of E17.
#
# PREDICTION (pre-registered)
#   ELA beats the tile model on CASIA, and loses to it on CocoGlide.
#   If that holds, Module 2 is two detectors with a clean division of labour
#   rather than one detector that must handle both.
#
# The doc also warns ELA false-positives on sharp edges, so a POSITIVE RESULT
# here has to be checked against authentic images too — a map that lights up on
# every edge would score well on tampered images and be useless.
#
# Same protocol as E17: per-tile scores, matched to ground-truth masks, reported
# per sub-dataset.

import warnings
from pathlib import Path

import joblib
import numpy as np
from PIL import Image
from sklearn.metrics import roc_auc_score

from pixelproof.ela import ela_map
from pixelproof.features import extract_tiles, tile_positions

warnings.filterwarnings("ignore")
Image.MAX_IMAGE_PIXELS = None

ROOT = Path("/tmp/m2")
TILE = 128
GRID = 36
LIMIT = 120


def ela_tile_scores(image: Image.Image, boxes):
    """Mean ELA response per tile, normalised within the image.

    Normalising inside the image is the point: a splice is betrayed by being
    INCONSISTENT with its surroundings, not by an absolute error level, and
    absolute levels vary with the image's own compression history.
    """
    heat = np.asarray(ela_map(image).convert("L"), dtype=np.float32)
    scores = np.array([heat[y:y + TILE, x:x + TILE].mean() for x, y in boxes])
    return (scores - scores.mean()) / (scores.std() + 1e-6)


def cnn_tile_scores(model, image: Image.Image):
    features, _ = extract_tiles(image, tile=TILE, max_tiles=GRID)
    return model.predict_proba(features)[:, 1]


def mask_coverage(mask_path: Path, boxes, size):
    with Image.open(mask_path) as mask:
        array = np.asarray(mask.convert("L").resize(size, Image.NEAREST)) > 127
    return np.array([array[y:y + TILE, x:x + TILE].mean() for x, y in boxes])


def main() -> None:
    model = joblib.load("artifacts/feature_crop128.joblib")

    for name in sorted(d.name for d in ROOT.iterdir() if d.is_dir()):
        manip_dir, auth_dir = ROOT / name / "manip", ROOT / name / "auth"
        if not manip_dir.exists():
            continue
        files = sorted(p for p in manip_dir.glob("*.png") if ".mask" not in p.name)[:LIMIT]

        ela_in, ela_out, tile_in, tile_out = [], [], [], []
        ela_img, tile_img = [], []
        for path in files:
            mask_path = path.with_suffix(".mask.png")
            if not mask_path.exists():
                continue
            with Image.open(path) as image:
                image = image.convert("RGB")
                if min(image.size) < TILE:
                    continue
                boxes = tile_positions(image.width, image.height, TILE, GRID)
                coverage = mask_coverage(mask_path, boxes, image.size)
                if coverage.max() < 0.5 or coverage.min() > 0.5:
                    continue
                ela_s = ela_tile_scores(image, boxes)
                tile_s = cnn_tile_scores(model, image)
            ela_in += list(ela_s[coverage >= 0.5]); ela_out += list(ela_s[coverage < 0.5])
            tile_in += list(tile_s[coverage >= 0.5]); tile_out += list(tile_s[coverage < 0.5])
            ela_img.append(float(ela_s.max()))
            tile_img.append(float(np.sort(tile_s)[-3:].mean()))

        # authentic images — the positive control the reference doc demands
        ela_auth, tile_auth = [], []
        for path in sorted(p for p in auth_dir.glob("*.png") if ".mask" not in p.name)[:LIMIT] \
                if auth_dir.exists() else []:
            with Image.open(path) as image:
                image = image.convert("RGB")
                if min(image.size) < TILE:
                    continue
                boxes = tile_positions(image.width, image.height, TILE, GRID)
                ela_auth.append(float(ela_tile_scores(image, boxes).max()))
                tile_auth.append(float(np.sort(cnn_tile_scores(model, image))[-3:].mean()))

        if not ela_in:
            print(f"{name}: no usable pairs\n")
            continue

        def auc(out, inn):
            return roc_auc_score(np.r_[np.zeros(len(out)), np.ones(len(inn))], np.r_[out, inn])

        print(f"--- {name} ---  ({len(ela_img)} manip / {len(ela_auth)} auth)")
        print(f"  {'':22}{'ELA':>10}{'tile model':>13}")
        print(f"  {'PIXEL tile AUC':<22}{auc(ela_out, ela_in):>10.3f}{auc(tile_out, tile_in):>13.3f}")
        if ela_auth:
            print(f"  {'IMAGE AUC':<22}{auc(ela_auth, ela_img):>10.3f}{auc(tile_auth, tile_img):>13.3f}")
        print()

    print("A pixel AUC of 0.5 means the method cannot tell a tampered tile from a clean one")
    print("in the SAME image. The image AUC uses authentic images as the control, which")
    print("matters for ELA: a map that lights up on every sharp edge would score well on")
    print("tampered images and be useless in practice.")


if __name__ == "__main__":
    main()
