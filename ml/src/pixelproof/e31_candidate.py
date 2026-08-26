"""Hash-verified single-DINOv2 E31 candidate scorer."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn.functional as functional
from huggingface_hub import snapshot_download
from PIL import Image

from pixelproof.build_tile_dataset import pick_tile

CANDIDATE_SHA256 = "99901219ec47e49a36fca7edd35a1c1737eb1cd9088f6465893054023914d860"
DINO_REPO_ID = "timm/vit_small_patch14_dinov2.lvd142m"
DINO_MODEL_ID = "vit_small_patch14_dinov2.lvd142m"
DINO_WEIGHT_SHA256 = "04d27f3400d059fc0cfd7d17dd1909a75bf3ea8fb3eeb48b97cb99e57ee20081"
DEFAULT_CANDIDATE = Path(__file__).resolve().parents[2] / "artifacts/e31/b4_candidate.joblib"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tile_for_content(image: Image.Image, content_key: str) -> np.ndarray:
    """One uniform seeded native tile; derivatives share the parent content key."""
    seed_hex = hashlib.sha256(f"e31-b5-tile-v1:{content_key}".encode()).hexdigest()[:8]
    tile = pick_tile(image.convert("RGB"), np.random.RandomState(int(seed_hex, 16)))
    if tile is None:
        raise ValueError("image cannot produce a 128 px texture-qualified E31 tile")
    return tile


@dataclass(frozen=True)
class E31Score:
    score: float
    threshold: float
    predicted_ai: bool


class E31Candidate:
    def __init__(self, candidate_path: Path = DEFAULT_CANDIDATE, device: torch.device | None = None):
        if sha256_file(candidate_path) != CANDIDATE_SHA256:
            raise RuntimeError("E31 candidate artifact SHA-256 changed")
        artifact = joblib.load(candidate_path)
        if artifact.get("rule") != "single_dinov2" or artifact.get("components") != (
            "r1_dinov2",
        ):
            raise RuntimeError("E31 candidate rule contract changed")
        if artifact.get("dinov2", {}).get("weight_sha256") != DINO_WEIGHT_SHA256:
            raise RuntimeError("E31 candidate DINO weight contract changed")
        snapshot = Path(snapshot_download(DINO_REPO_ID, local_files_only=True))
        if sha256_file(snapshot / "model.safetensors") != DINO_WEIGHT_SHA256:
            raise RuntimeError("cached DINOv2 weights changed")

        import timm

        active = device or torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        model = timm.create_model(
            DINO_MODEL_ID, pretrained=True, num_classes=0, img_size=224
        ).to(active).eval()
        config = timm.data.resolve_data_config({}, model=model)
        self.model = model
        self.device = active
        self.mean = torch.tensor(config["mean"], device=active).view(1, 3, 1, 1)
        self.std = torch.tensor(config["std"], device=active).view(1, 3, 1, 1)
        self.head = artifact["component_models"]["r1_dinov2"]
        self.threshold = float(artifact["threshold"])

    @torch.no_grad()
    def score_tile(self, tile: np.ndarray) -> float:
        if tile.shape != (128, 128, 3) or tile.dtype != np.uint8:
            raise ValueError("E31 tile must be uint8 [128,128,3]")
        batch = torch.from_numpy(tile.copy()[None]).to(self.device)
        batch = batch.permute(0, 3, 1, 2).float().div_(255.0)
        batch = functional.interpolate(
            batch,
            size=(224, 224),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        )
        embedding = self.model((batch - self.mean) / self.std).float().cpu().numpy()
        score = float(self.head.predict_proba(embedding)[0, 1])
        if not np.isfinite(score):
            raise RuntimeError("E31 candidate produced a non-finite score")
        return score

    def score_image(self, image: Image.Image, content_key: str) -> E31Score:
        score = self.score_tile(tile_for_content(image, content_key))
        return E31Score(score, self.threshold, score >= self.threshold)

    def contract(self) -> dict[str, object]:
        return {
            "detector_id": "e31_single_dinov2",
            "candidate_sha256": CANDIDATE_SHA256,
            "dino_repo_id": DINO_REPO_ID,
            "dino_weight_sha256": DINO_WEIGHT_SHA256,
            "tile": "one seeded native 128px texture-qualified tile keyed by content_id",
            "encoder_input_px": 224,
            "threshold": self.threshold,
            "decision": "AI detected at/above threshold; otherwise insufficient evidence",
        }
