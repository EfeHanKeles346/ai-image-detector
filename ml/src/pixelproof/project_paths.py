"""Portable defaults for datasets and disposable experiment work directories."""

from __future__ import annotations

import os
from pathlib import Path


ML_ROOT = Path(__file__).resolve().parents[2]


def configured_path(environment_name: str, default: Path) -> Path:
    value = os.environ.get(environment_name)
    return Path(value).expanduser().resolve() if value else default.resolve()


DATA_ROOT = configured_path("PIXELPROOF_DATA_ROOT", ML_ROOT / "data")
WORK_ROOT = configured_path("PIXELPROOF_WORK_ROOT", ML_ROOT / "work")
LEGACY_DATA_ROOT = configured_path(
    "PIXELPROOF_LEGACY_DATA_ROOT",
    Path.home() / "Desktop" / "PixelProof Workspace" / "Legacy Datasets",
)
