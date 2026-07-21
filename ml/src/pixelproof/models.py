from collections.abc import Callable

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18


class SmallCNN(nn.Module):
    """Parameter-efficient 32x32 RGB binary classifier returning one logit."""

    def __init__(self, dropout: float = 0.25) -> None:
        super().__init__()
        self.features = nn.Sequential(
            self._block(3, 32),
            nn.MaxPool2d(2),
            self._block(32, 64),
            nn.MaxPool2d(2),
            self._block(64, 128),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(128, 1))

    @staticmethod
    def _block(in_channels: int, out_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(images)).squeeze(1)


class ResNet18Transfer(nn.Module):
    """ImageNet-pretrained ResNet-18 backbone with a one-logit binary head.

    Mirrors SmallCNN's features/classifier split so embeddings.py and the
    training loop work unchanged (embeddings are 512-dim here).
    """

    def __init__(self, dropout: float = 0.0, pretrained: bool = True) -> None:
        super().__init__()
        backbone = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
        self.features = nn.Sequential(*list(backbone.children())[:-1])
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(512, 1))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(images)).squeeze(1)


MODEL_REGISTRY: dict[str, Callable[..., nn.Module]] = {"small_cnn": SmallCNN, "resnet18": ResNet18Transfer}


def create_model(name: str, **kwargs: object) -> nn.Module:
    try:
        return MODEL_REGISTRY[name](**kwargs)
    except KeyError as error:
        raise ValueError(f"Unknown model: {name}. Available: {sorted(MODEL_REGISTRY)}") from error
