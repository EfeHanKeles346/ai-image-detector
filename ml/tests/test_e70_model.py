import numpy as np
import pytest
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from experiments import e70_model as m


def test_fft_sketch_matches_direct_hashed_outer_products():
    rng = np.random.default_rng(170); left = rng.normal(size=(3, 5)); right = rng.normal(size=(3, 5))
    p = m.sketch_parameters(5); hashes = p['sketch_hash']; signs = p['sketch_sign']
    expected = np.zeros((3, m.SKETCH_DIM))
    for i in range(5):
        for j in range(5):
            expected[:, (hashes[0, i] + hashes[1, j]) % m.SKETCH_DIM] += signs[0, i] * signs[1, j] * left[:, i] * right[:, j]
    np.testing.assert_allclose(m.bilinear_sketch(left, right, hashes, signs), expected, atol=1e-13)


def test_sketch_is_bilinear_and_preserves_inputs():
    rng = np.random.default_rng(171); a, b, c = rng.normal(size=(3, 4, 6)); saved = a.copy()
    p = m.sketch_parameters(6)
    def f(x, y): return m.bilinear_sketch(x, y, p['sketch_hash'], p['sketch_sign'])
    np.testing.assert_allclose(f(a+b, c), f(a, c)+f(b, c), atol=1e-13)
    np.testing.assert_allclose(f(a, 2*c), 2*f(a, c), atol=1e-13)
    assert not np.any(f(a, np.zeros_like(c)))
    np.testing.assert_array_equal(a, saved)


def fixture():
    rng = np.random.default_rng(270); original = rng.normal(size=(96, 80)).astype(np.float32)
    blurred = (original*.8+rng.normal(scale=.2, size=original.shape)).astype(np.float32)
    head = make_pipeline(StandardScaler(), LogisticRegression(C=.1)).fit(original, original[:, 0] > 0)
    basis, _ = m.linear.fit_basis(head, original, blurred)
    basis['weights'][:] = .3
    arrays = m.fit_map(head, original, blurred, basis)
    assert np.all(basis['weights'] == .3)
    return original, blurred, head, arrays


def test_zero_init_is_exact_reference_and_replaces_prior_weights():
    original, blurred, head, arrays = fixture()
    assert arrays['weights'].shape == (257,) and not arrays['weights'].any()
    np.testing.assert_array_equal(m.predict(head, original, blurred, arrays), head.predict_proba(original)[:, 1])
    assert m.project(head, original, blurred, arrays).shape == (96, 257)


def test_serialized_map_replays_and_batch_partition_is_stable(tmp_path):
    original, blurred, head, arrays = fixture(); arrays['weights'][129] = .05
    expected = m.predict(head, original, blurred, arrays)
    path = tmp_path / 'candidate.npz'; np.savez_compressed(path, **arrays)
    with np.load(path, allow_pickle=False) as saved:
        np.testing.assert_array_equal(expected, m.predict(head, original, blurred, saved))
        batched = np.concatenate([m.predict(head, original[i:i+8], blurred[i:i+8], saved) for i in range(0, 96, 8)])
        np.testing.assert_allclose(expected, batched, rtol=0, atol=1e-12)


def test_invalid_hashes_and_nonfinite_features_rejected():
    p = m.sketch_parameters(3); x = np.ones((2, 3))
    with pytest.raises(ValueError): m.bilinear_sketch(x, x, p['sketch_hash'].astype(float), p['sketch_sign'])
    with pytest.raises(ValueError): m.bilinear_sketch(x*np.nan, x, p['sketch_hash'], p['sketch_sign'])
    with pytest.raises(ValueError): m.bilinear_sketch(x, x[:, :2], p['sketch_hash'], p['sketch_sign'])
