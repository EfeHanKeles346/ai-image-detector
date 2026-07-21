import argparse
from pathlib import Path

import torch
from PIL import Image

from pixelproof.evaluate import eval_transform
from pixelproof.models import create_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("images", type=Path, nargs="+")
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/best.pt"))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = checkpoint["config"]
    model = create_model(config["model"]["name"], dropout=config["model"]["dropout"]).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    transform = eval_transform(config["data"]["image_size"], config["data"].get("normalization", "default"))
    with torch.no_grad():
        for path in args.images:
            image = transform(Image.open(path).convert("RGB")).unsqueeze(0).to(device)
            probability = torch.sigmoid(model(image)).item()
            verdict = "AI-generated" if probability >= 0.5 else "real"
            print(f"{path}: {verdict} (p_ai={probability:.3f})")


if __name__ == "__main__":
    main()
