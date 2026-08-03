# E13 — how often does the tile model call a REAL photograph "AI"?
#
# E11 measured the tile method's ranking quality (SDXL 0.948) and stopped there.
# AUC is threshold-free: a model can rank almost perfectly and still put its 0.5
# line in the wrong place. Manual testing surfaced exactly that — a real photo
# scored 99% AI — so this measures the false-positive rate directly, on three
# sets of real photographs with different processing histories.

import warnings
from pathlib import Path

import joblib
import numpy as np
from PIL import Image

from pixelproof.features import extract_tiles

warnings.filterwarnings("ignore")
Image.MAX_IMAGE_PIXELS = None

HOME = Path.home() / "Desktop"
LIMIT = 300
EXT = {".jpg", ".jpeg", ".png", ".webp"}

SETS = {
    "defactify real (MS-COCO)": HOME / "defactify_test/real",
    "archive1 real (Instagram)": HOME / "archive1/real_dataset",
    "genimage real (ImageNet)": HOME / "genimage_split/test/REAL",
}


def files_in(folder: Path):
    return sorted(p for p in folder.rglob("*")
                  if p.suffix.lower() in EXT and not p.name.startswith("._"))[:LIMIT]


def main() -> None:
    model = joblib.load("artifacts/feature_crop128.joblib")

    def score(path: Path) -> float | None:
        try:
            with Image.open(path) as image:
                features, _ = extract_tiles(image, tile=128, max_tiles=36)
            probabilities = model.predict_proba(features)[:, 1]
            return float(np.sort(probabilities)[-3:].mean())   # top-3, as in serve.py
        except Exception:
            return None

    print(f"{'real-photograph set':<30}{'n':>5}{'called AI':>11}{'median p':>11}{'> 90%':>8}{'> 99%':>8}")
    print("-" * 73)
    results = {}
    for name, folder in SETS.items():
        scores = np.array([s for s in (score(p) for p in files_in(folder)) if s is not None])
        results[name] = scores
        print(f"{name:<30}{len(scores):>5}{100 * (scores >= .5).mean():>10.1f}%"
              f"{np.median(scores):>11.3f}{100 * (scores >= .9).mean():>7.1f}%"
              f"{100 * (scores >= .99).mean():>7.1f}%")

    every = np.concatenate(list(results.values()))
    print("-" * 73)
    print(f"{'ALL REAL PHOTOGRAPHS':<30}{len(every):>5}{100 * (every >= .5).mean():>10.1f}%"
          f"{np.median(every):>11.3f}{100 * (every >= .9).mean():>7.1f}%"
          f"{100 * (every >= .99).mean():>7.1f}%")

    print("\nwhat threshold would a 5% false-positive rate need?")
    for name, scores in results.items():
        print(f"  {name:<30}{np.percentile(scores, 95):>8.3f}")
    print(f"  {'ALL':<30}{np.percentile(every, 95):>8.3f}")


if __name__ == "__main__":
    main()


# --- appended: the AI side, so the operating point can be judged -------------
def operating_point() -> None:
    """A detector is only usable where BOTH error rates are acceptable."""
    model = joblib.load("artifacts/feature_crop128.joblib")

    def score(path: Path):
        try:
            with Image.open(path) as image:
                features, _ = extract_tiles(image, tile=128, max_tiles=36)
            return float(np.sort(model.predict_proba(features)[:, 1])[-3:].mean())
        except Exception:
            return None

    real = np.array([s for s in (score(p) for p in files_in(HOME / "defactify_test/real"))
                     if s is not None])
    gens = {g: np.array([s for s in (score(p) for p in files_in(HOME / "defactify_test/ai" / g))
                         if s is not None])
            for g in ["dalle3", "midjourney", "sd21", "sd3", "sdxl"]}

    print("\n\nOPERATING POINT — real photos vs generators, same scale")
    print(f"{'set':<22}{'median p':>10}{'>= 0.5':>9}{'>= 0.9':>9}{'>= 0.99':>9}")
    print("-" * 59)
    print(f"{'REAL (defactify)':<22}{np.median(real):>10.3f}"
          f"{100*(real >= .5).mean():>8.1f}%{100*(real >= .9).mean():>8.1f}%{100*(real >= .99).mean():>8.1f}%")
    for g, s in gens.items():
        print(f"{g:<22}{np.median(s):>10.3f}"
              f"{100*(s >= .5).mean():>8.1f}%{100*(s >= .9).mean():>8.1f}%{100*(s >= .99).mean():>8.1f}%")

    print("\nAI recall at the threshold that gives 5% false positives on real photos:")
    cut = float(np.percentile(real, 95))
    print(f"  threshold = {cut:.3f}")
    for g, s in gens.items():
        print(f"    {g:<20}{100*(s >= cut).mean():>7.1f}%")
    every = np.concatenate(list(gens.values()))
    print(f"    {'ALL AI':<20}{100*(every >= cut).mean():>7.1f}%")


if __name__ == "__main__":
    operating_point()
