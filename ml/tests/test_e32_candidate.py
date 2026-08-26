from __future__ import annotations

from PIL import Image

from pixelproof.e32_candidate import image_paths, standardized_array


def test_standardized_array_matches_candidate_shape() -> None:
    image = Image.new("RGBA", (500, 300), (10, 20, 30, 100))
    array = standardized_array(image)
    assert array.shape == (224, 224, 3)
    assert str(array.dtype) == "uint8"


def test_image_paths_filters_nonimages(tmp_path) -> None:
    Image.new("RGB", (16, 16)).save(tmp_path / "a.png")
    (tmp_path / "video.mov").write_bytes(b"not a still")
    assert image_paths([str(tmp_path)]) == [tmp_path / "a.png"]
