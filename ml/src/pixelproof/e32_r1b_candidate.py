"""Hash-verified inference for the selected E32 R1b CF-ViT candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import joblib
import numpy as np
import torch
from huggingface_hub import snapshot_download
from PIL import Image

from pixelproof.e32_candidate import E32Score, image_paths, sha256_file, standardized_array
from pixelproof.e32_cfvit_candidate import MODEL_REPO, MODEL_REVISION, MODEL_WEIGHT_SHA256
from pixelproof.project_paths import DATA_ROOT, ML_ROOT


ARTIFACT_SHA256 = "68a54aa2a5a7f85713669302ba5151983e3a9217cb2f49cc894e2a802802701c"
DEFAULT_ARTIFACT = DATA_ROOT / "e32" / "models" / "e32_r1b_cf.joblib"
SELECTION = ML_ROOT.parent / "evidence" / "e32_r1b_selection.json"


class E32R1bCandidate:
    def __init__(
        self,
        artifact_path: Path = DEFAULT_ARTIFACT,
        device: torch.device | None = None,
        *,
        model: Any | None = None,
        processor: Any | None = None,
    ):
        selection = json.loads(SELECTION.read_text())
        if selection.get("selected_arm") != "cf" or selection.get("selected_artifact_sha256") != ARTIFACT_SHA256:
            raise RuntimeError("R1b arm selection changed")
        if sha256_file(artifact_path) != ARTIFACT_SHA256:
            raise RuntimeError("R1b artifact SHA-256 changed")
        artifact = joblib.load(artifact_path)
        expected = (MODEL_REPO, MODEL_REVISION, MODEL_WEIGHT_SHA256)
        actual = (artifact.get("model_repo"), artifact.get("model_revision"), artifact.get("model_weight_sha256"))
        if actual != expected:
            raise RuntimeError("R1b CF model contract changed")
        active = device or torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        if (model is None) != (processor is None):
            raise ValueError("model and processor must be shared together")
        local = Path(snapshot_download(MODEL_REPO, revision=MODEL_REVISION, local_files_only=True))
        if sha256_file(local / "model.safetensors") != MODEL_WEIGHT_SHA256:
            raise RuntimeError("cached CF-ViT weights changed")
        if model is None:
            from transformers import ViTForImageClassification, ViTImageProcessor

            model = ViTForImageClassification.from_pretrained(local, local_files_only=True).to(active).eval()
            processor = ViTImageProcessor.from_pretrained(local, local_files_only=True)
        self.model = model
        self.processor = processor
        self.device = active
        self.head = artifact["head"]
        self.threshold = float(artifact["threshold"])

    @torch.inference_mode()
    def score_image(self, picture: Image.Image) -> float:
        standardized = Image.fromarray(standardized_array(picture), mode="RGB")
        pixels = self.processor(images=[standardized], return_tensors="pt")["pixel_values"].to(self.device)
        embedding = self.model.vit(pixel_values=pixels).last_hidden_state[:, 0].cpu().numpy()
        score = float(self.head.predict_proba(embedding)[0, 1])
        if not np.isfinite(score):
            raise RuntimeError("non-finite R1b score")
        return score

    @torch.inference_mode()
    def score_paths(self, paths: Sequence[Path], batch_size: int = 24) -> list[E32Score]:
        results = []
        for start in range(0, len(paths), batch_size):
            batch_paths = paths[start : start + batch_size]
            standardized = []
            for path in batch_paths:
                with Image.open(path) as image:
                    standardized.append(Image.fromarray(standardized_array(image), mode="RGB"))
            pixels = self.processor(images=standardized, return_tensors="pt")["pixel_values"].to(self.device)
            embeddings = self.model.vit(pixel_values=pixels).last_hidden_state[:, 0].cpu().numpy()
            scores = self.head.predict_proba(embeddings)[:, 1]
            for path, value in zip(batch_paths, scores, strict=True):
                score = float(value)
                if not np.isfinite(score):
                    raise RuntimeError(f"non-finite R1b score for {path}")
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
            "detector_id": "e32_r1b_cfvit_iphone_correction",
            "artifact_sha256": ARTIFACT_SHA256,
            "model_repo": MODEL_REPO,
            "model_revision": MODEL_REVISION,
            "model_weight_sha256": MODEL_WEIGHT_SHA256,
            "threshold": self.threshold,
            "input": "standardized JPEG round-trip then official CF-ViT processor",
            "boundary": "research candidate; external DEVELOPMENT cannot refit it",
        }


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+")
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--batch-size", type=int, default=24)
    args = parser.parse_args(argv)
    candidate = E32R1bCandidate(args.artifact)
    print(json.dumps({"contract": candidate.contract()}, sort_keys=True))
    for result in candidate.score_paths(image_paths(args.paths), args.batch_size):
        print(json.dumps(asdict(result), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
