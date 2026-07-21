"""Extract penultimate-layer embeddings from a trained checkpoint and cache them as .npz.

The 128-dim vector after global average pooling is the CNN's learned representation
of each image; Phase 2 experiments (classical classifiers, clustering) build on it.
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets

from pixelproof.data import invert_label
from pixelproof.evaluate import FolderPairDataset, eval_transform
from pixelproof.models import create_model


@torch.no_grad()
def extract(model, loader, device):
    model.eval()
    embeddings, probabilities, labels = [], [], []
    for images, batch_labels in loader:
        features = model.features(images.to(device)).flatten(1)
        logits = model.classifier(features)
        embeddings.append(features.cpu().numpy())
        probabilities.append(torch.sigmoid(logits).squeeze(1).cpu().numpy())
        labels.append(np.asarray(batch_labels))
    return np.concatenate(embeddings), np.concatenate(probabilities), np.concatenate(labels)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/best.pt"))
    parser.add_argument("--external-ai", type=Path, default=None)
    parser.add_argument("--external-real", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("artifacts/embeddings"))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = checkpoint["config"]
    model = create_model(config["model"]["name"], dropout=config["model"]["dropout"]).to(device)
    model.load_state_dict(checkpoint["model"])

    transform = eval_transform(config["data"]["image_size"], config["data"].get("normalization", "default"))
    root = Path(config["data"]["root"])
    splits = {
        "train": datasets.ImageFolder(root / "train", transform=transform, target_transform=invert_label),
        "test": datasets.ImageFolder(root / "test", transform=transform, target_transform=invert_label),
    }
    if args.external_ai and args.external_real:
        splits["external"] = FolderPairDataset(args.external_ai, args.external_real, transform)

    args.output.mkdir(parents=True, exist_ok=True)
    for name, dataset in splits.items():
        loader = DataLoader(dataset, batch_size=512, num_workers=4)
        embeddings, probabilities, labels = extract(model, loader, device)
        np.savez(args.output / f"{name}.npz", embeddings=embeddings, probabilities=probabilities, labels=labels)
        print(f"{name}: {embeddings.shape[0]} images -> {embeddings.shape[1]}-dim embeddings")


if __name__ == "__main__":
    main()
