import numpy as np
from scipy.optimize import check_grad
from scipy.special import expit, logit

from experiments import e64_constrained as model


def test_objective_gradient():
    rng = np.random.default_rng(62)
    x = rng.normal(size=(12, 4)); w = rng.normal(size=4)
    baseline = rng.normal(size=12); labels = np.arange(12) % 2; weights = np.ones(12)/12
    args = (x, baseline, labels, weights)
    error = check_grad(lambda w: model.objective(w, *args)[0],
                       lambda w: model.objective(w, *args)[1], w)
    assert error < 1e-6


def test_feasible_correction_rescues_real_without_losing_ai():
    # Same initial AI-positive score for the hard REAL and AI; a useful feature separates them.
    x = np.array([[-1., 1.], [-2., 1.], [1., 1.], [2., 1.]])
    baseline = logit(model.AI_CUT) + np.array([1., -1., .2, 1.])
    labels = np.array([0, 0, 1, 1])
    w, result = model.fit(x, baseline, labels, np.ones(4)/4)
    scores = expit(baseline + x @ w)
    assert result['success']
    assert result['max_constraint_violation'] < 1e-8
    assert np.all(scores[:2] < model.AI_CUT)
    assert np.all(scores[2:] >= model.AI_CUT)


def test_exact_ai_boundary_and_conflicting_features_cannot_be_traded_for_real_gain():
    x = np.array([[1., 1.], [-1., 1.], [1., 1.]])
    baseline = logit(model.AI_CUT) + np.array([.3, -1., 0.])
    labels = np.array([0, 0, 1])
    w, result = model.fit(x, baseline, labels, np.array([.45, .05, .5]))
    assert result['success']
    # Impossible to rescue the hard REAL with the shared correction without losing boundary AI.
    assert (baseline + x @ w)[0] >= logit(model.AI_CUT)
    assert (x @ w)[2] >= -1e-10


def test_correct_real_can_move_within_margin_without_becoming_false_positive():
    x = np.array([[-1., 1.], [1., 1.], [1., 1.]])
    baseline = logit(model.AI_CUT) + np.array([1., -10., .2])
    labels = np.array([0, 0, 1])
    w, result = model.fit(x, baseline, labels, np.array([.25, .25, .5]))
    logits = baseline + x @ w
    assert result['success']
    assert 0 < (x @ w)[1] < 10  # Confidence may move; the REAL decision remains protected.
    assert np.all(logits[:2] < logit(model.AI_CUT))
    assert logits[2] >= logit(model.AI_CUT)
