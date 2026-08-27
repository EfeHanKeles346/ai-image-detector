"""Hash-pinned inference adapter for the official NeurIPS 2025 DDA detector.

The LoRA layer names and published preprocessing are adapted from the Apache-2.0
``roy-ch/Dual-Data-Alignment`` inference release. The base DINOv2-L architecture is
constructed without downloading weights; the verified official checkpoint supplies every tensor.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
from PIL import Image
from torch import nn
from torchvision import transforms

from pixelproof.e32_candidate import image_paths, sha256_file
from pixelproof.project_paths import DATA_ROOT


MODEL_REPO_ID = "Junwei-Xi/Dual-Data-Alignment"
MODEL_REVISION = "4390d9023899196b437480bb6a441915ef5d816c"
CHECKPOINT_SHA256 = "b27a31d39374803ddeff02bfabb2be76e190b04300490cddfafb24f683f37e3e"
DEFAULT_CHECKPOINT = DATA_ROOT / "e35_dda_model" / "DDA_ckpt.pth"
THRESHOLD = 0.5
INPUT_SIZE = 336
TARGET_MODULES = ("attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2")


class LoRALayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, rank: int = 8, alpha: float = 1.0):
        super().__init__()
        self.alpha = alpha
        self.rank = rank
        self.lora_A = nn.Parameter(torch.zeros((rank, in_dim)))
        self.lora_B = nn.Parameter(torch.zeros((out_dim, rank)))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        reduced = torch.einsum("...d,rd->...r", values, self.lora_A)
        expanded = torch.einsum("...r,or->...o", reduced, self.lora_B)
        return expanded * (self.alpha / self.rank)


class LoRALinear(nn.Module):
    def __init__(self, original_layer: nn.Linear, rank: int = 8, alpha: float = 1.0):
        super().__init__()
        self.original_layer = original_layer
        self.lora = LoRALayer(original_layer.in_features, original_layer.out_features, rank, alpha)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.original_layer(values) + self.lora(values)


def _submodule(root: nn.Module, name: str) -> nn.Module:
    current = root
    for part in name.split(".") if name else ():
        current = current[int(part)] if part.isdigit() else getattr(current, part)
    return current


def apply_lora_to_linear_layers(model: nn.Module) -> nn.Module:
    replacements = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and any(target in name for target in TARGET_MODULES):
            parent_name, _, child_name = name.rpartition(".")
            replacements.append((_submodule(model, parent_name), child_name, module))
    for parent, child_name, module in replacements:
        setattr(parent, child_name, LoRALinear(module))
    return model


class _DINOHead(nn.Module):
    def __init__(self):
        super().__init__()
        import timm

        self.model = timm.create_model(
            "vit_large_patch14_dinov2.lvd142m",
            pretrained=False,
            num_classes=0,
            img_size=518,
        )
        # torch.hub DINOv2 exposes this training-time token; timm omits it from inference.
        # Retain it so the published full checkpoint loads strictly without changing forward.
        self.model.mask_token = nn.Parameter(torch.zeros(1, 1024))
        self.model = apply_lora_to_linear_layers(self.model)
        self.fc = nn.Linear(1024, 1)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        features = self.model.forward_features(values)
        if isinstance(features, dict):
            cls = features["x_norm_clstoken"]
        else:
            cls = features[:, 0]
        return self.fc(cls)


class _OfficialDDA(nn.Module):
    def __init__(self):
        super().__init__()
        self.base_model = _DINOHead()

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.base_model(values)


@dataclass(frozen=True)
class DDAScore:
    path: str
    score: float
    threshold: float
    predicted_ai: bool
    verdict: str


def official_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.CenterCrop(INPUT_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711],
        ),
    ])


class OfficialDDACandidate:
    def __init__(
        self,
        checkpoint: Path = DEFAULT_CHECKPOINT,
        device: torch.device | None = None,
    ):
        if sha256_file(checkpoint) != CHECKPOINT_SHA256:
            raise RuntimeError("official DDA checkpoint SHA-256 changed")
        active = device or torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if not isinstance(payload, dict) or not isinstance(payload.get("model"), dict):
            raise RuntimeError("official DDA checkpoint payload changed")
        model = _OfficialDDA()
        model.load_state_dict(payload["model"], strict=True)
        self.model = model.to(active).eval()
        self.device = active
        self.transform = official_transform()

    @torch.inference_mode()
    def score_paths(self, paths: Sequence[Path], batch_size: int = 2) -> list[DDAScore]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        output = []
        for start in range(0, len(paths), batch_size):
            batch_paths = paths[start : start + batch_size]
            tensors = []
            for path in batch_paths:
                with Image.open(path) as picture:
                    tensors.append(self.transform(picture.convert("RGB")))
            logits = self.model(torch.stack(tensors).to(self.device)).sigmoid().flatten().cpu().numpy()
            for path, value in zip(batch_paths, logits, strict=True):
                score = float(value)
                if not np.isfinite(score) or not 0.0 <= score <= 1.0:
                    raise RuntimeError(f"official DDA returned invalid score for {path}")
                predicted_ai = score >= THRESHOLD
                output.append(DDAScore(
                    path=str(path),
                    score=score,
                    threshold=THRESHOLD,
                    predicted_ai=predicted_ai,
                    verdict="ai_detected" if predicted_ai else "real_below_threshold",
                ))
        return output

    def contract(self) -> dict[str, object]:
        return {
            "detector_id": "official_dda_dinov2l_lora",
            "model_repo_id": MODEL_REPO_ID,
            "model_revision": MODEL_REVISION,
            "checkpoint_sha256": CHECKPOINT_SHA256,
            "input": "official RGB center-crop 336 and published CLIP normalization",
            "threshold": THRESHOLD,
            "score_direction": "higher means AI-generated",
            "boundary": "external research candidate; DEVELOPMENT cannot tune it",
        }


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args(argv)
    candidate = OfficialDDACandidate(args.checkpoint)
    print(json.dumps({"contract": candidate.contract()}, sort_keys=True))
    for result in candidate.score_paths(image_paths(args.paths), args.batch_size):
        print(json.dumps(asdict(result), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
