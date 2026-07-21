"""Phase 2c: evaluate the learning-curve checkpoints on the shared test set and plot the curve."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
from torchvision import datasets

from pixelproof.data import invert_label
from pixelproof.evaluate import collect_predictions, eval_transform
from pixelproof.models import create_model

RUNS = [(10_000, "best_10k.pt"), (20_000, "best_20k.pt"), (50_000, "best_50k.pt"), (90_000, "best.pt")]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/figures/learning_curve.png"))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    test_loader = None
    sizes, val_accuracies, test_accuracies = [], [], []
    for size, filename in RUNS:
        checkpoint = torch.load(args.artifacts / filename, map_location=device, weights_only=False)
        config = checkpoint["config"]
        if test_loader is None:
            transform = eval_transform(config["data"]["image_size"])
            test_set = datasets.ImageFolder(Path(config["data"]["root"]) / "test", transform=transform, target_transform=invert_label)
            test_loader = DataLoader(test_set, batch_size=256, num_workers=4)
        model = create_model(config["model"]["name"], dropout=config["model"]["dropout"]).to(device)
        model.load_state_dict(checkpoint["model"])
        labels, probabilities = collect_predictions(model, test_loader, device)
        test_accuracy = accuracy_score(labels, [int(p >= 0.5) for p in probabilities])
        sizes.append(size)
        val_accuracies.append(checkpoint["val_accuracy"])
        test_accuracies.append(test_accuracy)
        print(f"train_size={size}: best_val_acc={checkpoint['val_accuracy']:.4f} test_acc={test_accuracy:.4f}")

    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(sizes, [a * 100 for a in test_accuracies], "o-", label="test accuracy")
    axis.plot(sizes, [a * 100 for a in val_accuracies], "s--", alpha=0.6, label="best validation accuracy")
    axis.set_xscale("log")
    axis.set_xticks(sizes, [f"{s // 1000}k" for s in sizes])
    axis.set_xlabel("training-set size (log scale)")
    axis.set_ylabel("accuracy (%)")
    axis.set_title("Learning curve — SmallCNN on CIFAKE")
    axis.grid(alpha=0.3)
    axis.legend()
    figure.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=150)
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
