"""Hash-verified inference for the E32 R1a Community-Forensics CLS candidate."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Iterable, Sequence

import joblib
import numpy as np
import torch
from huggingface_hub import snapshot_download
from PIL import Image

from pixelproof.e32_candidate import E32Score, image_paths, sha256_file, standardized_array
from pixelproof.project_paths import DATA_ROOT


ARTIFACT_SHA256 = "6288acba5e50f11588b48907351cbd0fd1b741d3dab079376491bae5938ed670"
MODEL_REPO = "buildborderless/CommunityForensics-DeepfakeDet-ViT"
MODEL_REVISION = "ac6ee457bea904a373065754107451793b56db00"
MODEL_WEIGHT_SHA256 = "275ba982236ddd6afddf7131f8133e89f537574b964cf8fa5825b4956d741692"
DEFAULT_ARTIFACT = DATA_ROOT / "e32" / "models" / "e32_r1a_cfvit.joblib"


class E32CFViTCandidate:
    def __init__(self, artifact_path: Path = DEFAULT_ARTIFACT, device: torch.device | None = None):
        if sha256_file(artifact_path) != ARTIFACT_SHA256:
            raise RuntimeError("E32 R1a artifact SHA-256 changed")
        artifact = joblib.load(artifact_path)
        expected = (MODEL_REPO, MODEL_REVISION, MODEL_WEIGHT_SHA256)
        actual = (
            artifact.get("model_repo"),
            artifact.get("model_revision"),
            artifact.get("model_weight_sha256"),
        )
        if actual != expected:
            raise RuntimeError("E32 R1a model contract changed")
        local = Path(snapshot_download(MODEL_REPO, revision=MODEL_REVISION, local_files_only=True))
        if sha256_file(local / "model.safetensors") != MODEL_WEIGHT_SHA256:
            raise RuntimeError("cached CF-ViT weights changed")
        from transformers import ViTForImageClassification, ViTImageProcessor

        active = device or torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.model = ViTForImageClassification.from_pretrained(
            local, local_files_only=True
        ).to(active).eval()
        self.processor = ViTImageProcessor.from_pretrained(local, local_files_only=True)
        self.device = active
        self.head = artifact["head"]
        self.threshold = float(artifact["threshold"])

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
            for path, score_value in zip(batch_paths, scores, strict=True):
                score = float(score_value)
                if not np.isfinite(score):
                    raise RuntimeError(f"non-finite R1a score for {path}")
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
            "detector_id": "e32_r1a_cfvit_cls",
            "artifact_sha256": ARTIFACT_SHA256,
            "model_repo": MODEL_REPO,
            "model_revision": MODEL_REVISION,
            "model_weight_sha256": MODEL_WEIGHT_SHA256,
            "input": "R0 standardized JPEG round-trip then official CF-ViT processor",
            "threshold": self.threshold,
            "boundary": "research candidate; owner gallery cannot refit it",
        }


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+")
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--batch-size", type=int, default=24)
    args = parser.parse_args(argv)
    candidate = E32CFViTCandidate(args.artifact)
    print(json.dumps({"contract": candidate.contract()}, sort_keys=True))
    for result in candidate.score_paths(image_paths(args.paths), args.batch_size):
        print(json.dumps(asdict(result), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
