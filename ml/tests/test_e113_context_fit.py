import numpy as np
import pytest
from experiments.e113_context_fit import representation, coordinates


def test_saved_representation_is_deterministic_and_batch_independent(tmp_path):
    rng = np.random.default_rng(113)
    train = rng.normal(size=(96, 12)); heldout = rng.normal(size=(17, 12))
    a = representation(train, width=4)
    again = representation(train, width=4)
    assert all(np.array_equal(a[k], again[k]) for k in a)
    path = tmp_path/'representation.npz'; np.savez(path, **a)
    with np.load(path, allow_pickle=False) as saved:
        full = coordinates(heldout, saved)
        chunks = np.concatenate([coordinates(heldout[i:i+3], saved) for i in range(0, len(heldout), 3)])
    assert np.allclose(full, chunks, rtol=0, atol=1e-12)
    assert all(np.array_equal(a[k], again[k]) for k in a)
    with pytest.raises(ValueError): coordinates(np.full((3, 12), np.nan), a)


def test_flat_train_dimensions_remain_finite_without_using_evaluation_statistics():
    train = np.random.default_rng(1).normal(size=(96, 12)); train[:, 0] = 3
    a = representation(train, width=4)
    assert a['feature_scale'][0] == 1e-6
    assert np.isfinite(coordinates(train[:2], a)).all()
    before = a['feature_center'].copy()
    coordinates(np.full((2, 12), 50.), a)
    assert np.array_equal(before, a['feature_center'])
