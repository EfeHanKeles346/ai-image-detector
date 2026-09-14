import numpy as np
import pytest
from pixelproof.patch_drift import haar_attenuate, half_cosine_drift, median3, expand_grid


def test_haar_preserves_constant_and_block_means_and_attenuates_detail():
    a = np.zeros((8, 8, 3), np.uint8)
    a[:4, :4] = 100
    a[4:, 4:] = np.tile(np.array([0, 255, 0, 255], np.uint8), (4, 3, 1)).transpose(0, 2, 1)
    out = haar_attenuate(a)
    assert out.dtype == np.float32
    np.testing.assert_allclose(out[:4, :4], 100/255.)
    assert out[4:, 4:].mean() == pytest.approx(.5)
    np.testing.assert_allclose(out[4:, 4:].max()-out[4:, 4:].min(), .8, atol=1e-7)


def test_cosine_geometry_identity_opposition_and_scaling():
    a = np.ones((2, 3, 4), np.float32)
    np.testing.assert_allclose(half_cosine_drift(a, a), 0)
    np.testing.assert_allclose(half_cosine_drift(a, -a), 1)
    np.testing.assert_allclose(half_cosine_drift(a, a*7), 0, atol=1e-7)
    right = a.copy(); right[:, :, 2:] = -1
    np.testing.assert_allclose(half_cosine_drift(a, right), .5)


def test_median_removes_isolated_spike_without_normalizing_map():
    x = np.full((5, 5), .002, np.float32); x[2, 2] = 1
    np.testing.assert_allclose(median3(x), .002)


def test_expansion_is_nearest_grid_cell_not_interpolation():
    x = np.array([[0., 1.], [.2, .3]])
    out = expand_grid(x, (4, 6))
    assert out.shape == (4, 6)
    np.testing.assert_array_equal(out[:2, 3:], np.ones((2, 3)))
    np.testing.assert_array_equal(out[2:, :3], np.full((2, 3), .2))


@pytest.mark.parametrize('value', [np.zeros((2, 3, 4)), np.full((2, 3, 4), np.nan)])
def test_undefined_cosine_rejected(value):
    with pytest.raises(ValueError): half_cosine_drift(value, value)


@pytest.mark.parametrize('value', [np.zeros((7, 8, 3), np.uint8), np.zeros((8, 8, 3)), np.zeros((8, 8), np.uint8)])
def test_haar_does_not_silently_change_invalid_geometry_or_quantization(value):
    with pytest.raises(ValueError): haar_attenuate(value)


def test_bad_expansion_and_scores_rejected():
    with pytest.raises(ValueError): expand_grid(np.zeros((3, 3)), (4, 4))
    with pytest.raises(ValueError): median3(np.full((3, 3), float('nan')))
    with pytest.raises(ValueError): expand_grid(np.full((3, 3), 1.1), (6, 6))


def test_pixel_response_control_is_exact_rms_without_ai_label():
    from pixelproof.patch_drift import pixel_response
    image=np.zeros((28,28,3),np.uint8);perturbed=np.zeros((28,28,3),np.float32)
    perturbed[:14,:14]=.2
    np.testing.assert_allclose(pixel_response(image,perturbed),[[.2,0],[0,0]],atol=1e-7)
    with pytest.raises(ValueError):pixel_response(image,perturbed[:27])
