from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


def invert_label(label: int) -> int:
    """ImageFolder sorts folders as FAKE=0, REAL=1; invert so our API contract is AI=1."""
    return 1 - label


def build_loaders(root: Path, image_size: int, batch_size: int, validation_ratio: float, seed: int):
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    validation_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    augmented = datasets.ImageFolder(root / "train", transform=train_transform, target_transform=invert_label)
    deterministic = datasets.ImageFolder(root / "train", transform=validation_transform, target_transform=invert_label)
    indices = torch.randperm(len(augmented), generator=torch.Generator().manual_seed(seed)).tolist()
    val_size = int(len(augmented) * validation_ratio)
    train = Subset(augmented, indices[val_size:])
    validation = Subset(deterministic, indices[:val_size])
    return (
        DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True),
        DataLoader(validation, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True),
    )
