import argparse
from pathlib import Path

import torch
import yaml
from torch import nn

from pixelproof.data import build_loaders
from pixelproof.models import create_model


def run_epoch(model, loader, loss_fn, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss = total_correct = total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.float().to(device)
        with torch.set_grad_enabled(training):
            logits = model(images)
            loss = loss_fn(logits, labels)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        total_loss += loss.item() * labels.size(0)
        total_correct += ((logits >= 0) == labels.bool()).sum().item()
        total += labels.size(0)
    return total_loss / total, total_correct / total


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/baseline.yaml"))
    parser.add_argument("--train-size", type=int, default=None, help="Use only the first N training images (learning-curve runs)")
    parser.add_argument("--tag", default=None, help="Suffix for the checkpoint filename, e.g. best_10k.pt")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    torch.manual_seed(config["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    train_loader, val_loader = build_loaders(Path(config["data"]["root"]), config["data"]["image_size"], config["training"]["batch_size"], config["data"]["validation_ratio"], config["seed"], args.train_size)
    print(f"run tag={args.tag or 'full'} train_images={len(train_loader.dataset)}")
    model = create_model(config["model"]["name"], dropout=config["model"]["dropout"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["training"]["learning_rate"], weight_decay=config["training"]["weight_decay"])
    loss_fn = nn.BCEWithLogitsLoss()
    output = Path("artifacts"); output.mkdir(exist_ok=True)
    best_accuracy = 0.0
    for epoch in range(1, config["training"]["epochs"] + 1):
        train_loss, train_accuracy = run_epoch(model, train_loader, loss_fn, device, optimizer)
        val_loss, val_accuracy = run_epoch(model, val_loader, loss_fn, device)
        print(f"epoch={epoch:02d} train_loss={train_loss:.4f} train_acc={train_accuracy:.3f} val_loss={val_loss:.4f} val_acc={val_accuracy:.3f}")
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            checkpoint_name = f"best_{args.tag}.pt" if args.tag else "best.pt"
            torch.save({"model": model.state_dict(), "config": config, "val_accuracy": val_accuracy}, output / checkpoint_name)


if __name__ == "__main__":
    main()
