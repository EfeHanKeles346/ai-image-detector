import numpy as np
import pytest
from experiments.e147_weighted_representation import fit_project, check_replay
from pixelproof.source_holdout import balanced_weights


def test_held_out_features_cannot_change_fitted_map():
    x = np.random.default_rng(147).normal(size=(80, 9))
    mask = np.arange(len(x)) % 3 != 0
    weights = np.full(mask.sum(), 1 / mask.sum())
    a, projected = fit_project(x, mask, 4, weights)
    modified = x.copy()
    modified[~mask] = modified[~mask] * 1000 + 10000
    b, changed = fit_project(modified, mask, 4, weights)
    for key in a:
        np.testing.assert_array_equal(a[key], b[key])
    np.testing.assert_array_equal(projected[mask], changed[mask])
    assert not np.allclose(projected[~mask], changed[~mask])
    with pytest.raises(ValueError):
        fit_project(x, np.ones(len(x), dtype=bool), 4, np.full(len(x), 1 / len(x)))


def test_real_source_weights_reach_map_without_held_out_mass():
    rows = [{'parent_id': str(i), 'label': int(i >= 6)} for i in range(10)]
    groups = {str(i): 'a' if i < 5 else 'b' if i == 5 else 'c' for i in range(10)}
    weights = balanced_weights(rows, groups, views=4)
    x = np.random.default_rng(2).normal(size=(48, 7))
    mask = np.arange(48) < 40
    mapping, _ = fit_project(x, mask, 3, weights)
    np.testing.assert_allclose(mapping['center'], weights @ x[mask], atol=1e-14)
    assert weights[:24].sum() == pytest.approx(.5)
    assert weights[:20].sum() == pytest.approx(.25)
    assert weights[20:24].sum() == pytest.approx(.25)


def test_replay_rejects_decision_flip_even_below_numeric_tolerance():
    before = np.array([.5 - 1e-12, .8])
    assert check_replay(before, before.copy()) == 0
    with pytest.raises(ValueError):
        check_replay(before, np.array([.5 + 1e-12, .8]))
    with pytest.raises(ValueError):
        check_replay(before, before + .01)
    with pytest.raises(ValueError):
        check_replay(np.array([np.nan]), np.array([.1]))
