# =============================================================================
# train.py — WHAT THIS FILE DOES
# -----------------------------------------------------------------------------
# The training loop. Reads a YAML config (all hyperparameters live there, not
# in code), trains the selected model on the selected data, and saves the
# best-validation-accuracy checkpoint to artifacts/.
#
# CODE BLOCKS IN THIS FILE
# -----------------------------------------------------------------------------
# imports          argparse = CLI flags; yaml = parse the config file;
#                  torch/nn = training engine + loss; build_loaders and
#                  create_model = the data and model factories from this
#                  package (train.py never hard-codes either).
#
# run_epoch()      ONE pass over a dataset, used for BOTH training and
#                  validation: pass an optimizer -> gradients flow and weights
#                  update; pass None -> pure evaluation (no gradients,
#                  dropout/batchnorm switch to eval behavior via model.train).
#                  Per batch: forward -> loss -> (backward -> step) and
#                  accumulate running loss + accuracy (logit >= 0 means "AI",
#                  same as probability >= 0.5, without needing sigmoid).
#
# main()           Orchestration, in order:
#                  1. parse --config / --train-size / --tag flags
#                  2. seed torch (reproducible weight init + shuffles)
#                  3. pick device: cuda -> mps (Apple GPU) -> cpu
#                  4. build loaders + model from the config
#                  5. AdamW optimizer + BCEWithLogitsLoss (binary task)
#                  6. epoch loop: train pass, then validation pass; if this
#                     epoch's val accuracy is the best so far, save weights +
#                     the full config + val accuracy into one .pt checkpoint
#                     (the checkpoint describes itself — evaluate.py/serve.py
#                     rebuild the exact model from it with no extra info).
# =============================================================================

import argparse
from pathlib import Path

import torch
import yaml
from torch import nn

from pixelproof.data import build_loaders
from pixelproof.models import create_model


def run_epoch(model, loader, loss_fn, device, optimizer=None): # device = cpu or gpu or mps
    training = optimizer is not None
    model.train(training) # traning is a flag:
    total_loss = total_correct = total = 0 #initialize counters for loss, correct predictions, and total samples
    for images, labels in loader: # take batch of images and labels from the data loader
        images, labels = images.to(device), labels.float().to(device) # move data to the same device as the model
        with torch.set_grad_enabled(training): # if training is True, enable gradient computation; if False, disable it (saves memory and computation)
            logits = model(images)
            loss = loss_fn(logits, labels) #loss function: BCEWithLogitsLoss() = binary cross entropy with logits
            if training:
                optimizer.zero_grad(set_to_none=True) # clear previous batch's gradients to avoid accumulation (set_to_none=True is a memory optimization)
                loss.backward() #backpropagation: compute gradients of the loss w.r.t. model parameters
                optimizer.step() # update model parameters using the computed gradients
        # count correct predictions and accumulate loss for reporting        
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
    train_loader, val_loader = build_loaders(Path(config["data"]["root"]), config["data"]["image_size"], config["training"]["batch_size"], config["data"]["validation_ratio"], config["seed"], args.train_size, config["data"].get("normalization", "default"), config["data"].get("crop_augmentation", False))
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
