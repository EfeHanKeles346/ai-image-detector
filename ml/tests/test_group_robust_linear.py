import numpy as np
import pytest
from pixelproof import group_robust_linear as g


def data():
    y = np.repeat([0, 0, 1, 1, 0, 1], 4)
    groups = np.repeat(['a', 'b', 'a', 'b', 'a', 'b'], 4)
    rng = np.random.default_rng(138)
    return rng.normal(size=(24, 3)) + (y * 2 - 1)[:, None], y, groups


def test_gradient_matches_finite_differences_including_intercept():
    x, y, groups = data(); structure = g.grouping(y, groups); p = np.array([.3, -.2, .1, .4])
    loss, grad = g.objective(p, x, y, structure)
    for i in range(len(p)):
        step = np.zeros_like(p); step[i] = 1e-6
        actual = (g.objective(p + step, x, y, structure)[0] - g.objective(p - step, x, y, structure)[0]) / 2e-6
        assert actual == pytest.approx(grad[i], abs=1e-7)
    assert np.isfinite(loss)


def test_group_weights_follow_fit_difficulty_with_half_mass_per_class():
    _, y, groups = data(); structure = g.grouping(y, groups)
    logits = np.where(groups == 'b', y * 6 - 3, 3 - y * 6)
    risk, weights, losses, mass = g.risk(logits, y, structure)
    assert weights.sum() == pytest.approx(1.)
    for label in (0, 1):
        take = structure[2] == label
        assert weights[y == label].sum() == pytest.approx(.5)
        assert mass[take][np.argmax(losses[take])] > .49
    mean = .5 * sum(losses[structure[2] == label].mean() for label in (0, 1))
    worst = .5 * sum(losses[structure[2] == label].max() for label in (0, 1))
    assert mean <= risk <= worst


def test_single_group_per_class_matches_equal_class_bce():
    x, y, _ = data(); groups = np.array(['single'] * len(y)); p = np.array([.1, .2, -.1, .3])
    value, grad = g.objective(p, x, y, g.grouping(y, groups))
    from pixelproof.holdout_linear import objective
    weights = np.array([.5 / (y == label).sum() for label in y])
    expected, expected_grad = objective(p, x, y, weights)
    assert value == pytest.approx(expected)
    np.testing.assert_allclose(grad, expected_grad)


def test_real_solver_converges_and_invalid_parent_groups_fail():
    x, y, groups = data(); zero = g.objective(np.zeros(4), x, y, g.grouping(y, groups))[0]
    p, report = g.fit_head(x, y, groups)
    assert report['objective'] < zero and report['gradient_max_abs'] <= 1e-5
    assert np.isfinite(p).all()
    with pytest.raises(ValueError): g.grouping(y[:-1], groups[:-1])
    bad = groups.copy(); bad[1] = 'x'
    with pytest.raises(ValueError): g.grouping(y, bad)
    bad_y = y.copy(); bad_y[1] = 1
    with pytest.raises(ValueError): g.grouping(bad_y, groups)
