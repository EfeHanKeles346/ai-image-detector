# archive1 aspect-ratio control.
#
# archive1's AI half is 100% square (512x512); its real half is 98% rectangular
# (all 1080px wide). Both models resize every input to a SQUARE (224 or 32), so
# real images get geometrically squashed while AI images pass through undistorted.
# That distortion is itself a potential cue.
#
# Conditions (each adds one control, nothing else changes):
#   A  as downloaded                          real=JPEG rect   ai=PNG square
#   C  format equalised                       real=JPEG rect   ai=JPEG square
#   D  format AND geometry equalised          both JPEG, both centre-cropped square
#
# Plus a METADATA-ONLY baseline: a model that sees nothing but width, height and
# aspect ratio. Its score is the ceiling of what a pure shortcut could achieve.

import shutil
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from pixelproof.evaluate import FolderPairDataset, collect_predictions, eval_transform
from pixelproof.models import create_model

SRC = Path.home() / "Desktop/archive1"
WORK = Path("/private/tmp/claude-501/-Users-efehankeles-Desktop-ai-image-detector/"
            "238e1a7f-1cab-4e73-92da-2333ce2ae064/scratchpad/archive1_fmt")
QUALITY = 90


def source_files(sub: str) -> list[Path]:
    return sorted(p for p in (SRC / sub).rglob("*") if p.is_file() and not p.name.startswith("."))


def build(condition: str) -> tuple[Path, Path]:
    real_out, ai_out = WORK / condition / "real", WORK / condition / "ai"
    if real_out.exists():
        return real_out, ai_out
    real_out.mkdir(parents=True), ai_out.mkdir(parents=True)
    square = condition.startswith("D")
    for sub, out, recode in (("real_dataset", real_out, condition != "A_original"),
                             ("Ai_generated_dataset", ai_out, condition != "A_original")):
        for i, path in enumerate(source_files(sub)):
            if not recode:
                shutil.copy(path, out / f"{i:05d}{path.suffix}")
                continue
            with Image.open(path) as image:
                image = image.convert("RGB")
                if square:  # centre crop to a square at NATIVE resolution — no resampling
                    side = min(image.size)
                    left, top = (image.width - side) // 2, (image.height - side) // 2
                    image = image.crop((left, top, left + side, top + side))
                image.save(out / f"{i:05d}.jpg", "JPEG", quality=QUALITY)
    return real_out, ai_out


def metadata_ceiling() -> float:
    """How well can width/height/aspect ALONE separate the two classes?"""
    rows, labels = [], []
    for sub, label in (("real_dataset", 0), ("Ai_generated_dataset", 1)):
        for path in source_files(sub):
            with Image.open(path) as image:
                width, height = image.size
            rows.append([width, height, width / height, width * height])
            labels.append(label)
    x, y = np.array(rows, dtype=float), np.array(labels)
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    scores = cross_val_predict(model, x, y, cv=5, method="predict_proba")[:, 1]
    return roc_auc_score(y, scores)


def main() -> None:
    print(f"\nMETADATA-ONLY CEILING (width/height/aspect, no pixels): "
          f"AUC {metadata_ceiling():.3f}")
    print("  ^ what a model could score by reading image dimensions alone.\n")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    conditions = {
        "A_original":     "A  as downloaded         real=JPEG rect   ai=PNG square",
        "C_both_jpeg":    "C  format equalised      real=JPEG rect   ai=JPEG square",
        "D_square_jpeg":  "D  format + geometry     both JPEG, both cropped square",
    }

    print("=" * 80)
    print("ARCHIVE1 ASPECT-RATIO CONTROL")
    print("=" * 80)
    for checkpoint, title in (("best_genimage.pt", "ResNet-18 / GenImage   (E6 headline: AUC 0.888)"),
                              ("best.pt", "SmallCNN / CIFAKE      (E1: 77.1% acc)")):
        loaded = torch.load(f"artifacts/{checkpoint}", map_location=device, weights_only=False)
        config = loaded["config"]
        model = create_model(config["model"]["name"], dropout=config["model"]["dropout"]).to(device)
        model.load_state_dict(loaded["model"])
        model.eval()
        transform = eval_transform(config["data"]["image_size"],
                                   config["data"].get("normalization", "default"))

        print(f"\n--- {title} ---")
        print(f"{'condition':<62}{'AUC':>8}{'acc':>8}")
        baseline = None
        for condition, text in conditions.items():
            real_dir, ai_dir = build(condition)
            dataset = FolderPairDataset(ai_dir, real_dir, transform)
            labels, probabilities = collect_predictions(
                model, DataLoader(dataset, batch_size=128, num_workers=0), device)
            labels, probabilities = np.array(labels), np.array(probabilities)
            auc = roc_auc_score(labels, probabilities)
            accuracy = float(((probabilities >= 0.5).astype(int) == labels).mean())
            baseline = auc if baseline is None else baseline
            print(f"{text:<62}{auc:8.3f}{accuracy:8.3f}")
        print(f"{'net change A -> D (both confounds removed):':<62}{auc - baseline:+8.3f}")


if __name__ == "__main__":
    main()
