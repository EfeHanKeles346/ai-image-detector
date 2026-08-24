"""Verified loader and tile scorer for PixelProof's canonical project-owned model."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from pixelproof.artifact_registry import DEFAULT_REPO_ROOT, load_manifest, verify_registry
from pixelproof.data import NORMALIZATION
from pixelproof.evaluation_protocol import AGGREGATION_RULES, aggregate_tile_scores
from pixelproof.models import create_model


ARTIFACT_ID = "e20-tile-resnet18-seed2024"
ARTIFACT_GROUP = "project_model"
SUPPORTED_ARM = "resnet18"
SUPPORTED_TILE_PX = 128
SUPPORTED_NORMALIZATION = "imagenet"


class ProjectModelContractError(RuntimeError):
    """The canonical checkpoint is absent, unverified or incompatible with this runtime."""


@dataclass(frozen=True)
class ProjectModelMetadata:
    artifact_id: str
    sha256: str
    revision: str
    arm: str
    seed: int
    tile_px: int
    texture_floor: float
    normalization: str
    aggregation: str
    threshold: float
    calibration_fraction: float
    split_seed: int
    validation_auc: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProjectModelContractError(f"checkpoint field {field!r} must be numeric")
    result = float(value)
    if not np.isfinite(result):
        raise ProjectModelContractError(f"checkpoint field {field!r} must be finite")
    return result


def _integer(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ProjectModelContractError(f"checkpoint field {field!r} must be an integer")
    return value


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ProjectModelContractError(f"checkpoint field {field!r} must be a mapping")
    return value


def _artifact_entry(repo_root: Path) -> dict[str, Any]:
    matches = [
        entry for entry in load_manifest(repo_root)["artifacts"]
        if entry.get("id") == ARTIFACT_ID
    ]
    if len(matches) != 1:
        raise ProjectModelContractError(
            f"artifact manifest must contain exactly one {ARTIFACT_ID!r} entry"
        )
    entry = matches[0]
    if entry.get("kind") != "file" or entry.get("group") != ARTIFACT_GROUP:
        raise ProjectModelContractError(
            f"artifact {ARTIFACT_ID!r} must be a file in group {ARTIFACT_GROUP!r}"
        )
    return entry


def _metadata(checkpoint: Mapping[str, Any], entry: Mapping[str, Any]) -> ProjectModelMetadata:
    arm = checkpoint.get("arm")
    if arm != SUPPORTED_ARM:
        raise ProjectModelContractError(
            f"checkpoint arm must be {SUPPORTED_ARM!r}, got {arm!r}"
        )
    seed = _integer(checkpoint.get("seed"), "seed")
    _mapping(checkpoint.get("model"), "model")
    training = _mapping(checkpoint.get("training"), "training")
    inference = _mapping(checkpoint.get("inference"), "inference")

    tile_px = _integer(inference.get("tile_px"), "inference.tile_px")
    if tile_px != SUPPORTED_TILE_PX:
        raise ProjectModelContractError(
            f"checkpoint tile size must be {SUPPORTED_TILE_PX}, got {tile_px}"
        )
    normalization = inference.get("normalization")
    if normalization != SUPPORTED_NORMALIZATION or normalization not in NORMALIZATION:
        raise ProjectModelContractError(
            f"checkpoint normalization must be {SUPPORTED_NORMALIZATION!r}, got {normalization!r}"
        )
    aggregation = inference.get("selected_aggregation")
    if aggregation not in AGGREGATION_RULES:
        raise ProjectModelContractError(
            f"unsupported checkpoint aggregation {aggregation!r}"
        )

    texture_floor = _number(inference.get("texture_floor"), "inference.texture_floor")
    threshold = _number(inference.get("threshold"), "inference.threshold")
    calibration_fraction = _number(
        inference.get("calibration_fraction"), "inference.calibration_fraction"
    )
    validation_auc = _number(training.get("validation_auc"), "training.validation_auc")
    if not 0.0 <= texture_floor <= 1.0:
        raise ProjectModelContractError("checkpoint texture floor must be in [0, 1]")
    if not 0.0 <= threshold <= 1.0:
        raise ProjectModelContractError("checkpoint threshold must be in [0, 1]")
    if not 0.0 < calibration_fraction < 1.0:
        raise ProjectModelContractError("checkpoint calibration fraction must be in (0, 1)")
    if not 0.0 <= validation_auc <= 1.0:
        raise ProjectModelContractError("checkpoint validation AUC must be in [0, 1]")

    return ProjectModelMetadata(
        artifact_id=ARTIFACT_ID,
        sha256=str(entry["sha256"]),
        revision=str(entry.get("revision", "unknown")),
        arm=arm,
        seed=seed,
        tile_px=tile_px,
        texture_floor=texture_floor,
        normalization=normalization,
        aggregation=str(aggregation),
        threshold=threshold,
        calibration_fraction=calibration_fraction,
        split_seed=_integer(inference.get("split_seed"), "inference.split_seed"),
        validation_auc=validation_auc,
    )


class ProjectTileModel:
    """Loaded E20 model with its checkpoint-owned preprocessing and aggregation contract."""

    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        metadata: ProjectModelMetadata,
    ) -> None:
        self.model = model.to(device).eval()
        self.device = device
        self.metadata = metadata
        mean, std = NORMALIZATION[metadata.normalization]
        self._mean = torch.tensor(mean, device=device).view(1, 3, 1, 1)
        self._std = torch.tensor(std, device=device).view(1, 3, 1, 1)

    @torch.inference_mode()
    def score_tiles(self, tiles: Sequence[Image.Image], batch_size: int = 256) -> np.ndarray:
        if not tiles:
            raise ValueError("at least one tile is required")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        arrays = []
        for tile in tiles:
            rgb = tile.convert("RGB")
            if rgb.size != (self.metadata.tile_px, self.metadata.tile_px):
                raise ValueError(
                    f"tile must be {self.metadata.tile_px}x{self.metadata.tile_px}, got {rgb.size}"
                )
            arrays.append(np.asarray(rgb, dtype=np.uint8))

        output = []
        for start in range(0, len(arrays), batch_size):
            chunk = np.stack(arrays[start:start + batch_size])
            tensor = (
                torch.from_numpy(chunk)
                .to(self.device)
                .permute(0, 3, 1, 2)
                .contiguous()
                .float()
                / 255.0
            )
            tensor = (tensor - self._mean) / self._std
            output.append(torch.sigmoid(self.model(tensor)).cpu().numpy())
        return np.concatenate(output).astype(np.float64, copy=False)

    def aggregate(self, tile_scores: Sequence[float]) -> float:
        return aggregate_tile_scores(tile_scores, self.metadata.aggregation)

    def triggered(self, image_score: float) -> bool:
        return bool(image_score >= self.metadata.threshold)


def load_project_model(
    repo_root: Path = DEFAULT_REPO_ROOT,
    device: torch.device | None = None,
) -> ProjectTileModel:
    """Verify the artifact before deserializing it, then enforce the E20 schema."""
    root = repo_root.resolve()
    entry = _artifact_entry(root)
    report = verify_registry(root, groups={ARTIFACT_GROUP})
    if not report["ok"]:
        raise ProjectModelContractError("; ".join(report["issues"]))

    checkpoint_path = root / entry["path"]
    active_device = device or torch.device("cpu")
    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location=active_device,
            weights_only=False,
        )
    except Exception as error:
        raise ProjectModelContractError(
            f"checkpoint could not be deserialized: {type(error).__name__}: {error}"
        ) from error
    checkpoint = _mapping(checkpoint, "root")
    metadata = _metadata(checkpoint, entry)

    model = create_model(SUPPORTED_ARM, dropout=0.0, pretrained=False)
    try:
        model.load_state_dict(checkpoint["model"], strict=True)
    except Exception as error:
        raise ProjectModelContractError(
            f"checkpoint model state is incompatible: {type(error).__name__}: {error}"
        ) from error
    return ProjectTileModel(model, active_device, metadata)
