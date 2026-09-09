import numpy as np
import pytest

from experiments.e51_features import (
    FEATURE_NAMES, aggregate_crops, candidate_matrix, crop_statistics,
)


def test_flat_crop_is_finite_without_spurious_frequency_energy():
    stats = crop_statistics(np.full((224,224,3), 127, dtype=np.uint8))
    assert stats.shape == (16,)
    assert np.isfinite(stats).all()
    assert np.allclose(stats, 0, atol=1e-8)


def test_pattern_has_normalized_frequency_energy():
    pattern = np.tile((np.indices((224,224)).sum(axis=0) % 2 * 255)[..., None], (1,1,3)).astype(np.uint8)
    stats = crop_statistics(pattern)
    assert np.isclose(stats[:8].sum(), 1)
    assert stats[-1] > 0


def test_shared_crop_aggregation_has_fixed_order_and_size():
    crop = np.random.default_rng(51).integers(0, 256, (224,224,3), dtype=np.uint8)
    result = aggregate_crops([crop, crop, crop])
    assert result.shape == (len(FEATURE_NAMES),) == (32,)
    assert np.allclose(result[16:], 0, atol=1e-7)
    assert np.array_equal(result, aggregate_crops([crop, crop, crop]))


def test_unknown_candidate_or_wrong_feature_shape_is_rejected():
    dino, residual = np.zeros((2,3072)), np.zeros((2,32))
    assert candidate_matrix(dino, residual, 'A').shape == (2,3072)
    assert candidate_matrix(dino, residual, 'B').shape == (2,3104)
    with pytest.raises(ValueError):
        candidate_matrix(dino, residual, 'C')
    with pytest.raises(ValueError):
        crop_statistics(np.zeros((100,100,3), dtype=np.uint8))
