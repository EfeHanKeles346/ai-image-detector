import numpy as np
import pytest
from experiments.e69_features import shuffle_crops, PERMUTATION


def test_shuffle_moves_whole_patches_without_changing_local_rgb_values():
    crops = np.zeros((3, 3, 224, 224, 3), dtype=np.uint8)
    for cond in range(3):
        for crop in range(3):
            for patch in range(64):
                y, x = divmod(patch, 8)
                crops[cond, crop, y*28:(y+1)*28, x*28:(x+1)*28] = [patch, cond, crop]
    out = shuffle_crops(crops)
    for dest, source in enumerate(PERMUTATION):
        y, x = divmod(dest, 8)
        assert np.all(out[:, :, y*28:(y+1)*28, x*28:(x+1)*28, 0] == source)
    np.testing.assert_array_equal(out[..., 1:], crops[..., 1:])
    assert out.dtype == crops.dtype and out.shape == crops.shape
    assert not np.array_equal(out, crops)


def test_shuffle_is_deterministic_and_invertible_with_no_pixel_loss():
    crops = np.random.default_rng(169).integers(0, 256, (3, 3, 224, 224, 3), dtype=np.uint8)
    saved = crops.copy(); out = shuffle_crops(crops)
    np.testing.assert_array_equal(out, shuffle_crops(crops))
    recovered = np.empty_like(crops)
    for dest, source in enumerate(PERMUTATION):
        dy, dx = divmod(dest, 8); sy, sx = divmod(int(source), 8)
        recovered[:, :, sy*28:(sy+1)*28, sx*28:(sx+1)*28] = out[:, :, dy*28:(dy+1)*28, dx*28:(dx+1)*28]
    np.testing.assert_array_equal(recovered, saved)
    np.testing.assert_array_equal(crops, saved)


def test_shuffle_rejects_wrong_shape_or_dtype():
    with pytest.raises(ValueError): shuffle_crops(np.zeros((3, 224, 224, 3), dtype=np.uint8))
    with pytest.raises(ValueError): shuffle_crops(np.zeros((3, 3, 224, 224, 3), dtype=np.float32))
