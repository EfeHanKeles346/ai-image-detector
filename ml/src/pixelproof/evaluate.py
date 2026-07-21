import argparse
from pathlib import Path

import torch
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, roc_auc_score
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from PIL import Image

from pixelproof.data import invert_label
from pixelproof.models import create_model

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def eval_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])


class FolderPairDataset(Dataset):
    """Recursively pairs images from an AI folder (label 1) and a real folder (label 0)."""

    def __init__(self, ai_root: Path, real_root: Path, transform: transforms.Compose) -> None:
        self.samples = [
            (path, label)
            for root, label in ((ai_root, 1), (real_root, 0))
            for path in sorted(root.rglob("*"))
            if path.suffix.lower() in IMAGE_EXTENSIONS
        ]
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, label = self.samples[index]
        return self.transform(Image.open(path).convert("RGB")), label


@torch.no_grad()
def collect_predictions(model, loader, device):
    model.eval()
    all_labels, all_probabilities = [], []
    for images, labels in loader:
        probabilities = torch.sigmoid(model(images.to(device)))
        all_labels.extend(labels.tolist())
        all_probabilities.extend(probabilities.cpu().tolist())
    return all_labels, all_probabilities


def report(name: str, labels: list[int], probabilities: list[float]) -> None:
    predictions = [int(p >= 0.5) for p in probabilities]
    accuracy = sum(p == l for p, l in zip(predictions, labels)) / len(labels)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average="binary", zero_division=0)
    auc = roc_auc_score(labels, probabilities)
    matrix = confusion_matrix(labels, predictions)
    print(f"\n=== {name} ({len(labels)} images) ===")
    print(f"accuracy={accuracy:.4f} precision={precision:.4f} recall={recall:.4f} f1={f1:.4f} roc_auc={auc:.4f}")
    print(f"confusion matrix (rows=truth real/ai, cols=pred real/ai):\n{matrix}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/best.pt"))
    parser.add_argument("--external-ai", type=Path, help="Folder of AI images for out-of-distribution eval")
    parser.add_argument("--external-real", type=Path, help="Folder of real images for out-of-distribution eval")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = checkpoint["config"]
    model = create_model(config["model"]["name"], dropout=config["model"]["dropout"]).to(device)
    model.load_state_dict(checkpoint["model"])

    transform = eval_transform(config["data"]["image_size"])
    test_set = datasets.ImageFolder(Path(config["data"]["root"]) / "test", transform=transform, target_transform=invert_label)
    labels, probabilities = collect_predictions(model, DataLoader(test_set, batch_size=256, num_workers=4), device)
    report("held-out test set", labels, probabilities)

    if args.external_ai and args.external_real:
        external = FolderPairDataset(args.external_ai, args.external_real, transform)
        labels, probabilities = collect_predictions(model, DataLoader(external, batch_size=256, num_workers=4), device)
        report("external dataset", labels, probabilities)


if __name__ == "__main__":
    main()
