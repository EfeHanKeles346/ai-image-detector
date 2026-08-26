from __future__ import annotations

import numpy as np
from PIL import Image

from pixelproof.e31_candidate import tile_for_content


def test_content_tile_is_reproducible() -> None:
    values = np.random.default_rng(42).integers(0, 256, size=(256, 320, 3), dtype=np.uint8)
    image = Image.fromarray(values)
    first = tile_for_content(image, "same-parent")
    second = tile_for_content(image, "same-parent")
    assert first.shape == (128, 128, 3)
    assert np.array_equal(first, second)


def test_content_key_changes_uniform_tile_choice() -> None:
    y, x = np.mgrid[:256, :320]
    values = np.stack((x % 256, y % 256, (x + y) % 256), axis=-1).astype(np.uint8)
    image = Image.fromarray(values)
    assert not np.array_equal(
        tile_for_content(image, "parent-a"), tile_for_content(image, "parent-b")
    )
