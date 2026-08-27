from __future__ import annotations

import numpy as np
from PIL import Image

from experiments.e42_features import (
    TRANSPORTS,
    aggregate_tokens,
    assigned_transport,
    texture_crops,
    transport_image,
    view_plan,
)


def test_view_plan_is_symmetric_and_has_frozen_cardinality() -> None:
    rows = [
        {"parent_id": "train-a", "path": "a", "sha256": "a", "label": 0, "source": "r", "role": "train"},
        {"parent_id": "train-b", "path": "b", "sha256": "b", "label": 1, "source": "a", "role": "train"},
        {"parent_id": "dev", "path": "c", "sha256": "c", "label": 0, "source": "d", "role": "development"},
    ]
    planned = view_plan(rows)
    assert len(planned) == 2 + 2 + 5
    assert {row["condition"] for row in planned if row["parent_id"] == "dev"} == {"clean", *TRANSPORTS}
    assert assigned_transport("train-a") in TRANSPORTS
    assert assigned_transport("train-a") == assigned_transport("train-a")


def test_texture_crops_are_three_deterministic_224_rgb_arrays() -> None:
    y, x = np.mgrid[:420, :620]
    array = np.stack([(x % 256), (y % 256), ((x + y) % 256)], axis=2).astype(np.uint8)
    image = Image.fromarray(array)
    first = texture_crops(image)
    second = texture_crops(image)
    assert len(first) == 3
    assert all(item.shape == (224, 224, 3) and item.dtype == np.uint8 for item in first)
    assert all(np.array_equal(a, b) for a, b in zip(first, second, strict=True))


def test_every_transport_decodes_to_rgb() -> None:
    image = Image.new("RGB", (3000, 2200), (10, 20, 30))
    for condition in ("clean", *TRANSPORTS):
        result = transport_image(image, condition)
        assert result.mode == "RGB"
        assert result.width > 0 and result.height > 0
        assert max(result.size) <= 2048


def test_intermediate_crop_aggregation_shape_and_values() -> None:
    tokens = np.arange(2 * 3 * 4 * 5, dtype=np.float32).reshape(6, 4, 5)
    result = aggregate_tokens(tokens, views=2)
    assert result.shape == (2, 40)
    grouped = tokens.reshape(2, 3, 4, 5)
    assert np.allclose(result[:, :20], grouped.mean(axis=1).reshape(2, -1))
    assert np.allclose(result[:, 20:], grouped.std(axis=1).reshape(2, -1))
