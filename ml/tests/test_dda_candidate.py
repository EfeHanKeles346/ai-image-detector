from __future__ import annotations

import torch
from PIL import Image
from torch import nn

from pixelproof.dda_candidate import (
    _DINOHead,
    LoRALinear,
    apply_lora_to_linear_layers,
    official_transform,
)


class _Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv = nn.Linear(4, 12)
        self.proj = nn.Linear(4, 4)


class _Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = _Attention()
        self.other = nn.Linear(4, 4)


def test_official_lora_targets_only_published_linear_layers() -> None:
    model = nn.Sequential(_Block())
    apply_lora_to_linear_layers(model)
    assert isinstance(model[0].attn.qkv, LoRALinear)
    assert isinstance(model[0].attn.proj, LoRALinear)
    assert isinstance(model[0].other, nn.Linear)


def test_official_preprocessing_is_fixed_336_rgb_tensor() -> None:
    tensor = official_transform()(Image.new("RGB", (500, 400), "white"))
    assert tensor.shape == (3, 336, 336)
    assert tensor.dtype == torch.float32


def test_timm_dinov2_adapter_accepts_official_336_crop() -> None:
    model = _DINOHead().eval()
    with torch.inference_mode():
        result = model(torch.zeros(1, 3, 336, 336))
    assert result.shape == (1, 1)
