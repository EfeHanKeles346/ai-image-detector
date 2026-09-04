from __future__ import annotations

from PIL import Image
import pytest

from experiments.e49_openfake_download import inspect_file


def test_download_inspection_requires_exact_jpeg_geometry_and_bytes(tmp_path):
    path = tmp_path / "sample.jpg"
    Image.new("RGB", (32, 24), (20, 30, 40)).save(path, format="JPEG")
    expected = {
        "record_id": "openfake:core:test:1",
        "row_index": 1,
        "model": "gpt-image-2",
        "content_length": path.stat().st_size,
        "width": 32,
        "height": 24,
    }
    found = inspect_file(path, expected)
    assert found["format"] == "JPEG"
    assert found["bytes"] == path.stat().st_size
    assert len(found["sha256"]) == 64

    with pytest.raises(ValueError, match="dimensions changed"):
        inspect_file(path, {**expected, "width": 31})
