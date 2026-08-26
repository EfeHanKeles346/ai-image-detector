"""Hash-verified inference for the successful E32 R0 DINOv2-S candidate."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import joblib
import numpy as np
import torch
from huggingface_hub import snapshot_download
from PIL import Image, ImageOps

from pixelproof.project_paths import DATA_ROOT


ARTIFACT_SHA256 = "7f170340ab050543f60ceda129224a67b5adaf22a628e52049d401bc96e8a85e"
DINO_REPO_ID = "timm/vit_small_patch14_dinov2.lvd142m"
DINO_MODEL_ID = "vit_small_patch14_dinov2.lvd142m"
DINO_WEIGHT_SHA256 = "04d27f3400d059fc0cfd7d17dd1909a75bf3ea8fb3eeb48b97cb99e57ee20081"
DEFAULT_ARTIFACT = DATA_ROOT / "e32" / "models" / "e32_r0_dinov2s.joblib"
INPUT_SIZE = 224
RESIZE_SHORT_SIDE = 256
SUPPORTED_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def standardized_array(image: Image.Image) -> np.ndarray:
    oriented = ImageOps.exif_transpose(image).convert("RGB")
    scale = RESIZE_SHORT_SIDE / min(oriented.size)
    resized = oriented.resize(
        (round(oriented.width * scale), round(oriented.height * scale)),
        Image.Resampling.LANCZOS,
    )
    left = (resized.width - INPUT_SIZE) // 2
    top = (resized.height - INPUT_SIZE) // 2
    cropped = resized.crop((left, top, left + INPUT_SIZE, top + INPUT_SIZE))
    encoded = io.BytesIO()
    cropped.save(
        encoded,
        format="JPEG",
        quality=90,
        subsampling=0,
        optimize=False,
        progressive=False,
    )
    with Image.open(io.BytesIO(encoded.getvalue())) as decoded:
        return np.asarray(decoded.convert("RGB"), dtype=np.uint8).copy()


@dataclass(frozen=True)
class E32Score:
    path: str
    score: float
    threshold: float
    predicted_ai: bool
    verdict: str


class E32Candidate:
    def __init__(self, artifact_path: Path = DEFAULT_ARTIFACT, device: torch.device | None = None):
        if sha256_file(artifact_path) != ARTIFACT_SHA256:
            raise RuntimeError("E32 candidate artifact SHA-256 changed")
        artifact = joblib.load(artifact_path)
        if artifact.get("model_id") != DINO_MODEL_ID or artifact.get("input_size") != INPUT_SIZE:
            raise RuntimeError("E32 candidate model/input contract changed")
        preprocessing = artifact.get("preprocessing", {})
        if preprocessing.get("crop") != "center-224" or "quality=90" not in preprocessing.get("encoding", ""):
            raise RuntimeError("E32 candidate preprocessing contract changed")
        snapshot = Path(snapshot_download(DINO_REPO_ID, local_files_only=True))
        if sha256_file(snapshot / "model.safetensors") != DINO_WEIGHT_SHA256:
            raise RuntimeError("cached DINOv2 weights changed")
        import timm

        active = device or torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        model = timm.create_model(
            DINO_MODEL_ID, pretrained=True, num_classes=0, img_size=INPUT_SIZE
        ).to(active).eval()
        config = timm.data.resolve_data_config({}, model=model)
        self.model = model
        self.device = active
        self.mean = torch.tensor(config["mean"], device=active).view(1, 3, 1, 1)
        self.std = torch.tensor(config["std"], device=active).view(1, 3, 1, 1)
        self.head = artifact["head"]
        self.threshold = float(artifact["threshold"])

    @torch.inference_mode()
    def score_paths(self, paths: Sequence[Path], batch_size: int = 48) -> list[E32Score]:
        results: list[E32Score] = []
        for start in range(0, len(paths), batch_size):
            batch_paths = paths[start : start + batch_size]
            arrays = []
            for path in batch_paths:
                with Image.open(path) as image:
                    arrays.append(standardized_array(image))
            tensor = torch.from_numpy(np.stack(arrays)).to(self.device)
            tensor = tensor.permute(0, 3, 1, 2).float().div_(255.0)
            embeddings = self.model((tensor - self.mean) / self.std).float().cpu().numpy()
            scores = self.head.predict_proba(embeddings)[:, 1]
            for path, score_value in zip(batch_paths, scores, strict=True):
                score = float(score_value)
                if not np.isfinite(score):
                    raise RuntimeError(f"non-finite score for {path}")
                predicted_ai = score >= self.threshold
                results.append(
                    E32Score(
                        path=str(path),
                        score=score,
                        threshold=self.threshold,
                        predicted_ai=predicted_ai,
                        verdict="ai_detected" if predicted_ai else "real_below_threshold",
                    )
                )
        return results

    def contract(self) -> dict[str, object]:
        return {
            "detector_id": "e32_r0_dinov2s_global",
            "artifact_sha256": ARTIFACT_SHA256,
            "dino_repo_id": DINO_REPO_ID,
            "dino_weight_sha256": DINO_WEIGHT_SHA256,
            "input": "EXIF transpose, RGB, short-side 256, center 224, JPEG q90 4:4:4",
            "threshold": self.threshold,
            "boundary": "group-held-out runnable candidate; not final authenticity proof",
        }


def image_paths(values: Sequence[str]) -> list[Path]:
    paths: list[Path] = []
    for value in values:
        candidate = Path(value).expanduser().resolve()
        if candidate.is_dir():
            paths.extend(
                sorted(
                    path for path in candidate.rglob("*")
                    if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
                )
            )
        elif candidate.is_file() and candidate.suffix.lower() in SUPPORTED_SUFFIXES:
            paths.append(candidate)
        else:
            raise ValueError(f"unsupported or missing image path: {candidate}")
    if not paths:
        raise ValueError("no supported image found")
    return paths


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+")
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--batch-size", type=int, default=48)
    args = parser.parse_args(argv)
    candidate = E32Candidate(args.artifact)
    print(json.dumps({"contract": candidate.contract()}, sort_keys=True))
    for result in candidate.score_paths(image_paths(args.paths), args.batch_size):
        print(json.dumps(asdict(result), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
