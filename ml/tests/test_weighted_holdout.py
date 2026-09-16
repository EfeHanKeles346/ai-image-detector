import numpy as np
import pytest
from pixelproof.holdout_linear import fit_map as unweighted_map, project
from pixelproof.weighted_holdout import fit_map


def test_uniform_weights_recover_unweighted_geometry_and_whitening():
    x = np.random.default_rng(147).normal(size=(50, 8))
    before = unweighted_map(x, 3)
    after = fit_map(x, 3, np.full(len(x), 1 / len(x)))
    np.testing.assert_allclose(after['center'], before['center'], atol=1e-14)
    np.testing.assert_allclose(after['scale'], before['scale'], atol=1e-14)
    np.testing.assert_allclose(after['components'], before['components'], atol=1e-12)
    np.testing.assert_allclose(after['latent_scale'], before['latent_scale'], atol=1e-12)
    np.testing.assert_allclose(project(x, after), project(x, before), atol=1e-12)


def test_weighted_map_matches_direct_covariance_eigendecomposition():
    x = np.random.default_rng(1).normal(size=(70, 7))
    x[:50] += 4
    w = np.r_[np.full(50, .5 / 50), np.full(20, .5 / 20)]
    a = fit_map(x, 3, w)
    center = w @ x
    scale = np.sqrt(w @ ((x - center) ** 2))
    z = (x - center) / scale
    cov = z.T @ (w[:, None] * z) / (1 - w @ w)
    values, vectors = np.linalg.eigh(cov)
    np.testing.assert_allclose(a['center'], center, atol=1e-13)
    np.testing.assert_allclose(a['scale'], scale, atol=1e-13)
    np.testing.assert_allclose(a['latent_scale'] ** 2, values[-3:][::-1], atol=1e-12)
    np.testing.assert_allclose(a['components'].T @ a['components'], vectors[:, -3:] @ vectors[:, -3:].T, atol=1e-12)
    transformed = project(x, a)
    np.testing.assert_allclose(w @ transformed, 0, atol=1e-12)
    np.testing.assert_allclose(transformed.T @ (w[:, None] * transformed) / (1 - w @ w), np.eye(3), atol=1e-12)


def test_rejects_bad_weights_and_does_not_mutate_input():
    x = np.random.default_rng(2).normal(size=(20, 5)).astype(np.float32)
    original = x.copy()
    fit_map(x, 2, np.full(20, .05))
    np.testing.assert_array_equal(x, original)
    for w in [np.ones(20), np.full(19, 1 / 19), np.r_[0., np.full(19, 1 / 19)], np.full(20, np.nan)]:
        with pytest.raises(ValueError):
            fit_map(x, 2, w)
