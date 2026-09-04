from __future__ import annotations

from pathlib import Path

from PIL import Image
import pytest

from experiments.e51_transfer import ieee_destination, inspect_ieee_file


def _row(size: int = 1) -> dict:
    return {
        "identity": "id",
        "label": 0,
        "role": "DEVELOPMENT",
        "source": "IEEE-SP-Cup-2018-test",
        "transport_cell": "unaltered",
        "remote_path": "test/test/img_0000000_unalt.tif",
        "expected_bytes": size,
    }


def test_ieee_destination_rejects_unsafe_or_wrong_split():
    assert ieee_destination(_row()).name == "img_0000000_unalt.tif"
    for remote in ("../bad.tif", "train/train/a.tif", "/test/test/a.tif", "test/test/a.jpg"):
        row = _row()
        row["remote_path"] = remote
        with pytest.raises(ValueError, match="unsafe IEEE"):
            ieee_destination(row)


def test_ieee_inspection_verifies_tiff_and_bytes(tmp_path: Path):
    path = tmp_path / "image.tif"
    Image.new("RGB", (512, 512), "red").save(path, format="PNG")
    row = _row(path.stat().st_size)
    found = inspect_ieee_file(path, row)
    assert found["format"] == "PNG"
    assert (found["width"], found["height"]) == (512, 512)
    assert len(found["sha256"]) == 64


def test_ieee_inspection_rejects_byte_drift(tmp_path: Path):
    path = tmp_path / "image.tif"
    Image.new("RGB", (512, 512)).save(path, format="PNG")
    with pytest.raises(ValueError, match="byte length"):
        inspect_ieee_file(path, _row(path.stat().st_size + 1))
