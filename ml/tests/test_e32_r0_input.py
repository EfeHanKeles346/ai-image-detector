from __future__ import annotations

import io

import pytest
from PIL import Image

from experiments import e32_r0_input as r0


def _png(size: tuple[int, int] = (400, 300), mode: str = "RGBA") -> bytes:
    image = Image.new(mode, size, (20, 80, 140, 120) if mode == "RGBA" else (20, 80, 140))
    output = io.BytesIO()
    image.save(output, "PNG")
    return output.getvalue()


def test_standardized_jpeg_has_fixed_contract() -> None:
    raw = r0.standardized_jpeg(_png())
    with Image.open(io.BytesIO(raw)) as image:
        assert image.format == "JPEG"
        assert image.mode == "RGB"
        assert image.size == (224, 224)


def test_standardized_jpeg_is_deterministic() -> None:
    raw = _png((300, 500), "RGB")
    assert r0.standardized_jpeg(raw) == r0.standardized_jpeg(raw)


def test_raw_image_accepts_arrow_style_mapping() -> None:
    assert r0.raw_image({"bytes": b"abc", "path": None}) == b"abc"


def test_raw_image_rejects_missing_payload() -> None:
    with pytest.raises(ValueError, match="no byte payload"):
        r0.raw_image({"path": "image.png"})


def test_file_source_resolution(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.setattr(r0, "DATA_ROOT", tmp_path)
    monkeypatch.setattr(r0, "E32_ROOT", tmp_path / "e32")
    expected = tmp_path / "e32" / "real" / "vision" / "D01/a.jpg"
    assert r0.source_path("vision-base-native", "D01/a.jpg") == expected
