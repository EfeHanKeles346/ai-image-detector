import numpy as np
import pytest
from scipy.optimize._numdiff import approx_derivative
from scipy.special import expit, logit
from experiments.e68_minimax import AI_CUT, GroupRisks, objective, fit


def population():
    rng = np.random.default_rng(68)
    x = rng.normal(size=(12, 4)); logits = rng.normal(size=12) + logit(AI_CUT)
    y = np.repeat([0, 1], 6); sources = np.tile(['a', 'a', 'a', 'b', 'b', 'b'], 2)
    parents = np.array([f'p{i}' for i in range(12)]); conditions = np.tile(['clean', 'clean', 'q75'], 4)
    return x, logits, y, sources, parents, conditions


def test_risk_and_epigraph_gradients_match_finite_differences():
    risk = GroupRisks(*population()); v = np.array([.1, -.2, .3, -.4, 1., 2.])
    values, gradient = risk.risks(v[:-2])
    assert np.isfinite(values).all()
    np.testing.assert_allclose(gradient, approx_derivative(lambda w: risk.risks(w)[0], v[:-2]), atol=1e-8)
    np.testing.assert_allclose(risk.epigraph(v)[1], approx_derivative(lambda z: risk.epigraph(z)[0], v), atol=1e-8)
    np.testing.assert_allclose(objective(v)[1], approx_derivative(lambda z: objective(z)[0], v).ravel(), atol=1e-8)


def test_each_class_bound_must_cover_its_worst_group():
    risk = GroupRisks(*population()); w = np.zeros(4); values, _ = risk.risks(w)
    bounds = np.array([values[risk.classes == y].max() for y in (0, 1)])
    assert np.min(risk.epigraph(np.r_[w, bounds])[0]) >= 0
    bounds[0] -= .01
    assert np.min(risk.epigraph(np.r_[w, bounds])[0][risk.classes == 0]) < 0
    assert np.min(risk.epigraph(np.r_[w, bounds])[0][risk.classes == 1]) >= 0


def test_duplicate_views_do_not_increase_a_parents_group_mass():
    x = np.zeros((4, 1)); y = np.array([0, 0, 0, 1])
    offsets = np.array([0., 0., 2., 0.])
    risk = GroupRisks(x, offsets + logit(AI_CUT), y, ['real']*3+['ai'],
                      ['a', 'a', 'b', 'c'], ['clean']*4)
    values, _ = risk.risks(np.zeros(1))
    assert values[0] == pytest.approx(.5 * (np.logaddexp(0, 0) + np.logaddexp(0, 2)))


def test_conflicting_groups_cannot_trade_away_protected_decisions():
    x = np.ones((4, 1)); margins = np.array([.1, -1., 1., -.1])
    y = np.array([1, 1, 0, 0]); logits = margins + logit(AI_CUT)
    w, result = fit(x, logits, y, np.array(['ai','ai','real','real']),
                    np.array(['a','b','c','d']), np.array(['clean']*4))
    assert result['success'] and result['max_constraint_violation'] <= 1e-8
    scores = expit(logits + x @ w)
    assert scores[0] >= AI_CUT and scores[3] < AI_CUT
    assert abs(w[0]) <= .1


def test_parent_cannot_cross_source_or_label():
    args = list(population()); args[4][7] = args[4][0]
    with pytest.raises(ValueError, match='parent crosses'):
        GroupRisks(*args)
