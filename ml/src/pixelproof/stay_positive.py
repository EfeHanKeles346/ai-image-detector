"""Independent Stay-Positive head training for the project-owned E20 backbone.

This module implements the small algorithm described in Sundar et al. (ICML
2025) without importing upstream code or weights: freeze a feature extractor,
zero-initialize a linear head, optimize binary cross entropy, and project the
feature weights onto the non-negative orthant after every update.  The bias is
not constrained.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch import nn

from pixelproof.artifact_registry import load_manifest, sha256_file
from pixelproof.data import NORMALIZATION
from pixelproof.models import create_model
from pixelproof.project_paths import ML_ROOT
from pixelproof.project_model import ARTIFACT_ID


METHOD = "stay_positive_independent_v1"


def source_stratified_holdout(
    labels: np.ndarray,
    sources: np.ndarray,
    *,
    fraction: float = 0.10,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return deterministic train/validation indices stratified by source and label."""
    labels = np.asarray(labels)
    sources = np.asarray(sources)
    if labels.ndim != 1 or sources.ndim != 1 or len(labels) != len(sources):
        raise ValueError("labels and sources must be equally sized one-dimensional arrays")
    if not 0.0 < fraction < 1.0:
        raise ValueError("fraction must be strictly between zero and one")

    rng = np.random.RandomState(seed)
    train: list[int] = []
    validation: list[int] = []
    for source in sorted(set(str(value) for value in sources.tolist())):
        for label in sorted(set(int(value) for value in labels.tolist())):
            group = np.flatnonzero((sources.astype(str) == source) & (labels == label))
            rng.shuffle(group)
            cut = max(1, int(len(group) * fraction)) if len(group) > 1 else 0
            validation.extend(group[:cut].tolist())
            train.extend(group[cut:].tolist())
    if not train or len(set(labels[validation].tolist())) != 2:
        raise ValueError("source-stratified validation must contain both labels")
    return np.asarray(sorted(train), dtype=np.int64), np.asarray(
        sorted(validation), dtype=np.int64
    )


def stratified_sample_indices(
    labels: np.ndarray, sources: np.ndarray, *, limit: int, seed: int
) -> np.ndarray:
    """Balanced deterministic subset for smoke tests; full experiments do not subsample."""
    labels = np.asarray(labels)
    sources = np.asarray(sources)
    if limit <= 0:
        raise ValueError("limit must be positive")
    if limit >= len(labels):
        return np.arange(len(labels), dtype=np.int64)

    rng = np.random.RandomState(seed)
    groups: list[list[int]] = []
    keys = sorted({(str(source), int(label)) for source, label in zip(sources, labels)})
    for source, label in keys:
        group = np.flatnonzero((sources.astype(str) == source) & (labels == label))
        rng.shuffle(group)
        groups.append(group.tolist())

    selected: list[int] = []
    while len(selected) < limit and any(groups):
        for group in groups:
            if group and len(selected) < limit:
                selected.append(group.pop())
    return np.asarray(sorted(selected), dtype=np.int64)


def create_zero_positive_head(in_features: int) -> nn.Linear:
    if in_features <= 0:
        raise ValueError("in_features must be positive")
    head = nn.Linear(in_features, 1)
    nn.init.zeros_(head.weight)
    nn.init.zeros_(head.bias)
    return head


def train_positive_head(
    features: np.ndarray,
    labels: np.ndarray,
    sources: np.ndarray,
    *,
    seed: int,
    epochs: int = 15,
    batch_size: int = 1024,
    learning_rate: float = 1e-3,
    validation_fraction: float = 0.10,
) -> tuple[nn.Linear, dict[str, Any]]:
    """Fit a non-negative linear head and retain the validation-AUC-best epoch."""
    features = np.asarray(features, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64)
    sources = np.asarray(sources)
    if features.ndim != 2 or len(features) != len(labels):
        raise ValueError("features must be [n, d] and match labels")
    if len(labels) != len(sources):
        raise ValueError("sources must match labels")
    if set(labels.tolist()) != {0, 1}:
        raise ValueError("labels must contain both 0 and 1")
    if not np.isfinite(features).all():
        raise ValueError("features must be finite")
    if float(features.min()) < -1e-7:
        raise ValueError("Stay-Positive requires non-negative features")
    if epochs <= 0 or batch_size <= 0 or learning_rate <= 0:
        raise ValueError("epochs, batch_size and learning_rate must be positive")

    torch.manual_seed(seed)
    rng = np.random.RandomState(seed)
    train_index, validation_index = source_stratified_holdout(
        labels, sources, fraction=validation_fraction, seed=seed
    )
    feature_tensor = torch.from_numpy(features)
    label_tensor = torch.from_numpy(labels.astype(np.float32))
    head = create_zero_positive_head(features.shape[1])
    optimizer = torch.optim.AdamW(head.parameters(), lr=learning_rate, weight_decay=0.0)
    loss_fn = nn.BCEWithLogitsLoss()

    best_auc = -1.0
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    epoch_records: list[dict[str, float | int]] = []
    for epoch in range(1, epochs + 1):
        head.train()
        order = rng.permutation(train_index)
        running_loss = 0.0
        for start in range(0, len(order), batch_size):
            index = order[start : start + batch_size]
            logits = head(feature_tensor[index]).squeeze(1)
            loss = loss_fn(logits, label_tensor[index])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                head.weight.clamp_(min=0.0)
            running_loss += float(loss.item()) * len(index)

        head.eval()
        with torch.no_grad():
            validation_scores = torch.sigmoid(
                head(feature_tensor[validation_index]).squeeze(1)
            ).numpy()
        validation_auc = float(roc_auc_score(labels[validation_index], validation_scores))
        epoch_records.append(
            {
                "epoch": epoch,
                "training_loss": running_loss / len(train_index),
                "validation_auc": validation_auc,
            }
        )
        if validation_auc > best_auc:
            best_auc = validation_auc
            best_epoch = epoch
            best_state = {
                key: value.detach().clone() for key, value in head.state_dict().items()
            }

    if best_state is None:  # defensive; epochs > 0 guarantees a state
        raise RuntimeError("training produced no checkpoint")
    head.load_state_dict(best_state)
    head.eval()
    minimum_weight = float(head.weight.detach().min().item())
    if minimum_weight < 0.0:
        raise RuntimeError("non-negative projection invariant was violated")
    return head, {
        "method": METHOD,
        "best_epoch": best_epoch,
        "validation_auc": best_auc,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "validation_fraction": validation_fraction,
        "train_count": int(len(train_index)),
        "validation_count": int(len(validation_index)),
        "minimum_feature_weight": minimum_weight,
        "negative_feature_weights": int((head.weight.detach() < 0).sum().item()),
        "epoch_records": epoch_records,
    }


@torch.no_grad()
def extract_nonnegative_features(
    model: nn.Module,
    tiles: np.ndarray,
    *,
    device: torch.device,
    batch_size: int = 256,
) -> np.ndarray:
    """Extract E20 ResNet18 embeddings under its exact ImageNet normalization."""
    if tiles.ndim != 4 or tiles.shape[-1] != 3 or tiles.dtype != np.uint8:
        raise ValueError("tiles must be uint8 [n, height, width, 3]")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    model = model.to(device).eval()
    for parameter in model.features.parameters():
        parameter.requires_grad_(False)
    mean, std = NORMALIZATION["imagenet"]
    mean_tensor = torch.tensor(mean, device=device).view(1, 3, 1, 1)
    std_tensor = torch.tensor(std, device=device).view(1, 3, 1, 1)
    extracted: list[np.ndarray] = []
    for start in range(0, len(tiles), batch_size):
        batch = (
            torch.from_numpy(tiles[start : start + batch_size])
            .to(device)
            .permute(0, 3, 1, 2)
            .contiguous()
            .float()
            / 255.0
        )
        features = model.features((batch - mean_tensor) / std_tensor).flatten(1)
        extracted.append(torch.relu(features).cpu().numpy().astype(np.float32, copy=False))
    return np.concatenate(extracted) if extracted else np.empty((0, 512), dtype=np.float32)


def install_head(model: nn.Module, head: nn.Linear) -> None:
    target = model.classifier[-1]
    if not isinstance(target, nn.Linear) or target.in_features != head.in_features:
        raise ValueError("model classifier is incompatible with the trained head")
    with torch.no_grad():
        target.weight.copy_(head.weight)
        target.bias.copy_(head.bias)


def _device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train an isolated Stay-Positive E20 candidate")
    parser.add_argument(
        "--base-checkpoint",
        type=Path,
        default=ML_ROOT / "artifacts/tile_resnet18_seed2024.pt",
    )
    parser.add_argument("--tiles", type=Path, default=ML_ROOT / "artifacts/tiles_v1.npz")
    parser.add_argument(
        "--output",
        type=Path,
        default=ML_ROOT / "artifacts/e28/stay_positive_seed2024.pt",
    )
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--feature-batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--train-limit", type=int, help="balanced smoke-only tile subset")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    args = parser.parse_args()

    base_sha256 = sha256_file(args.base_checkpoint)
    matches = [
        entry
        for entry in load_manifest(ML_ROOT)["artifacts"]
        if entry.get("id") == ARTIFACT_ID
    ]
    if len(matches) != 1 or base_sha256 != matches[0].get("sha256"):
        parser.error(
            "base checkpoint must match the hash-verified canonical E20 seed-2024 artifact"
        )
    payload = torch.load(args.base_checkpoint, map_location="cpu", weights_only=False)
    if payload.get("arm") != "resnet18" or "model" not in payload:
        parser.error("base checkpoint must be an E20 ResNet18 checkpoint")
    model = create_model("resnet18", dropout=0.0, pretrained=False)
    model.load_state_dict(payload["model"])

    with np.load(args.tiles, allow_pickle=True) as data:
        tiles = data["x"]
        labels = data["y"]
        sources = data["sources"]
        if args.train_limit and args.train_limit < len(labels):
            chosen = stratified_sample_indices(
                labels, sources, limit=args.train_limit, seed=args.seed
            )
            tiles, labels, sources = tiles[chosen], labels[chosen], sources[chosen]

    device = _device(args.device)
    print(f"extracting frozen E20 features: n={len(labels):,} device={device}")
    features = extract_nonnegative_features(
        model, tiles, device=device, batch_size=args.feature_batch_size
    )
    print(f"features={features.shape} min={float(features.min()):.6f}")
    head, training = train_positive_head(
        features,
        labels,
        sources,
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )
    install_head(model, head)

    candidate = {
        "model": model.state_dict(),
        "arm": "resnet18",
        "seed": args.seed,
        "training": {
            **training,
            "base_checkpoint": str(args.base_checkpoint),
            "base_checkpoint_sha256": base_sha256,
            "tile_dataset": str(args.tiles),
            "tile_dataset_bytes": args.tiles.stat().st_size,
            "n_tiles": int(len(labels)),
            "labels": dict(Counter(int(value) for value in labels)),
            "sources": dict(Counter(str(value) for value in sources)),
            "feature_extractor_frozen": True,
            "feature_relu": True,
            "head_zero_initialized": True,
            "bias_constrained": False,
        },
        "inference": {
            "tile_px": 128,
            "texture_floor": 0.04,
            "normalization": "imagenet",
            "aggregation_candidates": ["top3", "top10pct_mean", "p90"],
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(candidate, args.output)
    print(
        f"saved {args.output} best_epoch={training['best_epoch']} "
        f"val_auc={training['validation_auc']:.4f} min_weight={training['minimum_feature_weight']:.6f}"
    )


if __name__ == "__main__":
    main()
