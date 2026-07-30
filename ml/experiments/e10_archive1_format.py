# archive1 format-control experiment.
#
# archive1's real half is 100% JPEG and its AI half is 100% PNG. A model can
# separate them by compression history alone, without ever detecting generation.
# This script changes ONE variable — file format — and re-measures.
#
#   condition A (original)  : real = JPEG(as-is),   ai = PNG
#   condition B (ai->jpeg)  : real = JPEG(as-is),   ai = JPEG q90
#   condition C (both jpeg) : real = JPEG q90 re-encoded, ai = JPEG q90
#
# C is the fair one: both classes pass through the identical final encoder.
# If AUC survives C, the model is reading generation traces. If it collapses,
# E6's headline 0.888 was a codec artefact.

import shutil
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from pixelproof.evaluate import FolderPairDataset, collect_predictions, eval_transform
from pixelproof.models import create_model

SRC = Path.home() / "Desktop/archive1"
WORK = Path("/private/tmp/claude-501/-Users-efehankeles-Desktop-ai-image-detector/"
            "238e1a7f-1cab-4e73-92da-2333ce2ae064/scratchpad/archive1_fmt")
QUALITY = 90


def build(condition: str) -> tuple[Path, Path]:
    real_out, ai_out = WORK / condition / "real", WORK / condition / "ai"
    if real_out.exists():
        return real_out, ai_out
    real_out.mkdir(parents=True), ai_out.mkdir(parents=True)
    for sub, out, recode in (("real_dataset", real_out, condition == "C_both_jpeg"),
                             ("Ai_generated_dataset", ai_out, condition != "A_original")):
        for i, path in enumerate(sorted(p for p in (SRC / sub).rglob("*")
                                        if p.is_file() and not p.name.startswith("."))):
            if recode:
                with Image.open(path) as image:
                    image.convert("RGB").save(out / f"{i:05d}.jpg", "JPEG", quality=QUALITY)
            else:
                shutil.copy(path, out / f"{i:05d}{path.suffix}")
    return real_out, ai_out


def main() -> None:
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    results = {}
    for checkpoint in ("best_genimage.pt", "best.pt"):
        loaded = torch.load(f"artifacts/{checkpoint}", map_location=device, weights_only=False)
        config = loaded["config"]
        model = create_model(config["model"]["name"], dropout=config["model"]["dropout"]).to(device)
        model.load_state_dict(loaded["model"])
        model.eval()
        transform = eval_transform(config["data"]["image_size"],
                                   config["data"].get("normalization", "default"))

        for condition in ("A_original", "B_ai_to_jpeg", "C_both_jpeg"):
            real_dir, ai_dir = build(condition)
            dataset = FolderPairDataset(ai_dir, real_dir, transform)
            labels, probabilities = collect_predictions(
                model, DataLoader(dataset, batch_size=128, num_workers=0), device)
            labels = np.array(labels)
            probabilities = np.array(probabilities)
            auc = roc_auc_score(labels, probabilities)
            accuracy = float(((probabilities >= 0.5).astype(int) == labels).mean())
            recall = float((probabilities[labels == 1] >= 0.5).mean())
            false_positive = float((probabilities[labels == 0] >= 0.5).mean())
            results[(checkpoint, condition)] = (auc, accuracy, recall, false_positive)

    print("\n" + "=" * 78)
    print("ARCHIVE1 FORMAT-CONTROL — one variable changes: the file format")
    print("=" * 78)
    labels_map = {
        "A_original":   "A  real=JPEG  ai=PNG      (as downloaded)",
        "B_ai_to_jpeg": "B  real=JPEG  ai=JPEG90   (ai re-encoded)",
        "C_both_jpeg":  "C  real=JPEG90 ai=JPEG90  (both re-encoded — the fair test)",
    }
    for checkpoint, title in (("best_genimage.pt", "ResNet-18 / GenImage  (E6 headline: AUC 0.888)"),
                              ("best.pt", "SmallCNN / CIFAKE     (E1: 77.1% acc)")):
        print(f"\n--- {title} ---")
        print(f"{'condition':<52}{'AUC':>8}{'acc':>8}{'real->AI':>10}")
        for condition, text in labels_map.items():
            auc, accuracy, _, false_positive = results[(checkpoint, condition)]
            print(f"{text:<52}{auc:8.3f}{accuracy:8.3f}{false_positive*100:9.1f}%")
        drop = results[(checkpoint, "A_original")][0] - results[(checkpoint, "C_both_jpeg")][0]
        print(f"{'AUC change A -> C:':<52}{-drop:+8.3f}")


if __name__ == "__main__":
    main()
