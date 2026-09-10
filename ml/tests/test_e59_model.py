import numpy as np
import pytest

from experiments import e59_model as m


def test_representation_order_and_dimensions():
    d = np.ones((2, 3072))
    c = np.full((2, 1536), 2.)
    assert m.representation('dino3072', d, c) is d
    assert m.representation('clip1536', d, c) is c
    joined = m.representation('dino_clip4608', d, c)
    np.testing.assert_array_equal(joined[:, :3072], d)
    np.testing.assert_array_equal(joined[:, 3072:], c)
    with pytest.raises(ValueError):
        m.representation('other', d, c)
    with pytest.raises(ValueError):
        m.representation('dino3072', d, c[:1])


def test_weights_keep_class_mass_and_determinism():
    y = np.array([0, 0, 0, 1, 1, 1])
    s = np.array(['real']*3+['ai']*3)
    p = np.array(['r']*3+['a']*3)
    w = m.loss_weights(y, s, p, 12)
    assert w.sum() == 12
    assert w[y == 1].sum() == w[y == 0].sum() == 6
    assert len({m.hashlib_array(m.loss_weights(y, s, p, 12)) for _ in m.ARMS}) == 1


def test_control_parity_rejects_flip_and_truth_changes():
    row = dict(parent_id='p', source='s', label=1, condition='clean', score=.6, predicted_ai=True)
    base = dict(threshold=.5, observations=[row])
    assert m.control_parity(base, base)['passed']
    near = dict(threshold=.5, observations=[dict(row, score=.600001)])
    assert m.control_parity(near, base)['passed']
    flipped = dict(threshold=.7, observations=[dict(row, predicted_ai=False)])
    assert not m.control_parity(flipped, base)['passed']
    with pytest.raises(ValueError):
        m.control_parity(dict(threshold=.5, observations=[dict(row, label=0)]), base)


def test_training_refuses_incomplete_feature_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(m, 'FEATURES', tmp_path / 'missing.npz')
    with pytest.raises(RuntimeError, match='both completion receipts'):
        m.require_features()
