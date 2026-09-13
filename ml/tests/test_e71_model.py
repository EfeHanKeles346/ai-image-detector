import numpy as np
import pytest
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from experiments import e71_model as m


def fixture(monkeypatch):
    monkeypatch.setattr(m, 'RANK', 4)
    rng = np.random.default_rng(71)
    x = rng.normal(size=(80, 12)).astype(np.float32)
    clip = rng.normal(size=(80, 9)).astype(np.float32)
    head = make_pipeline(StandardScaler(), LogisticRegression(C=.1)).fit(x, (x[:, 0] > 0).astype(int))
    pca = PCA(n_components=4).fit(head[0].transform(x).astype(np.float64))
    old = dict(base_mean=pca.mean_, base_components=pca.components_,
               base_scales=np.sqrt(pca.explained_variance_), weights=np.ones(9),
               response_center=np.ones(12))
    a, _ = m.fit_basis(x, clip, old)
    return x, clip, head, old, a


def test_exact_zero_init_and_no_old_correction_leakage(monkeypatch):
    x, clip, head, old, a = fixture(monkeypatch)
    assert np.array_equal(m.predict(head, x, clip, a), head.predict_proba(x)[:, 1])
    assert 'response_center' not in a
    assert not np.any(a['weights']) and np.all(old['weights'] == 1)
    for key in m.BASE_KEYS:
        assert np.array_equal(a[key], old[key])
        assert not np.shares_memory(a[key], old[key])


def test_original_coordinates_unchanged_and_independent_clip_branch(monkeypatch):
    x, clip, head, old, a = fixture(monkeypatch)
    p = m.project(head, x, clip, a)
    expected = (head[0].transform(x).astype(np.float64)-old['base_mean']) @ old['base_components'].T / old['base_scales']
    assert np.array_equal(p[:, :4], expected)
    q = m.project(head, x, clip+.25, a)
    assert np.array_equal(p[:, :4], q[:, :4]) and not np.allclose(p[:, 4:8], q[:, 4:8])
    q = m.project(head, x+.25, clip, a)
    assert np.array_equal(p[:, 4:8], q[:, 4:8]) and not np.allclose(p[:, :4], q[:, :4])


def test_serialization_and_batch_invariant_predictions(monkeypatch, tmp_path):
    x, clip, head, _, a = fixture(monkeypatch)
    a['weights'][4] = .2
    path = tmp_path/'correction.npz'
    np.savez_compressed(path, **a)
    expected = m.predict(head, x, clip, a)
    with np.load(path, allow_pickle=False) as saved:
        assert np.array_equal(m.predict(head, x, clip, saved), expected)
        actual = np.concatenate([m.predict(head, x[i:i+7], clip[i:i+7], saved) for i in range(0, len(x), 7)])
        # The unchanged float32 reference has ~6e-8 batch rounding itself.
        np.testing.assert_allclose(actual, expected, atol=1e-7, rtol=0)
        assert np.array_equal(actual >= m.AI_CUT, expected >= m.AI_CUT)
        coordinates = np.concatenate([m.project(head, x[i:i+7], clip[i:i+7], saved) for i in range(0, len(x), 7)])
        np.testing.assert_allclose(coordinates, m.project(head, x, clip, saved), atol=1e-14, rtol=1e-14)


def test_invalid_alignment_and_bases_fail_closed(monkeypatch):
    x, clip, head, old, a = fixture(monkeypatch)
    with pytest.raises(ValueError, match='aligned'): m.predict(head, x, clip[:-1], a)
    broken = clip.copy(); broken[0, 0] = np.nan
    with pytest.raises(ValueError, match='aligned'): m.fit_basis(x, broken, old)
    old['base_components'] = old['base_components'][:3]
    with pytest.raises(ValueError, match='dimensions'): m.fit_basis(x, clip, old)
