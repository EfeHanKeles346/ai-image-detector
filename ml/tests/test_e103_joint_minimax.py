import numpy as np
import pytest
from scipy.optimize._numdiff import approx_derivative
from scipy.special import logit
from experiments.e64_constrained import AI_CUT
from experiments.e68_minimax import GroupRisks, objective
from experiments.e103_joint_minimax import fit


def population():
    # The second AI is currently missed; both REAL cases are already correct.
    x = np.array([[1., 1.], [2., 1.], [-1., 1.], [-2., 1.]])
    logits = logit(AI_CUT) + np.array([1., -1., -1., -2.])
    return x, logits, np.array([1, 1, 0, 0]), np.array(['ai', 'ai', 'real', 'real']), \
        np.array(['a', 'b', 'c', 'd']), np.array(['clean'] * 4)


def test_joint_derivatives_and_missed_ai_incentive():
    risk = GroupRisks(*population())
    v = np.array([.1, -.2, 1., 1.])
    np.testing.assert_allclose(risk.epigraph(v)[1], approx_derivative(lambda z: risk.epigraph(z)[0], v), atol=1e-8)
    np.testing.assert_allclose(objective(v)[1], approx_derivative(lambda z: objective(z)[0], v).ravel(), atol=1e-8)
    args = list(population()); args[1] = args[1].copy(); args[1][1] -= 2
    other, _ = GroupRisks(*args).risks(np.zeros(2))
    original, _ = risk.risks(np.zeros(2))
    assert other[risk.classes == 1][0] > original[risk.classes == 1][0]
    np.testing.assert_array_equal(other[risk.classes == 0], original[risk.classes == 0])


def test_joint_fit_can_rescue_ai_without_losing_previous_decisions():
    args = population(); w, report = fit(*args); shift = args[0] @ w
    assert report['success'] and report['max_constraint_violation'] <= 1e-8
    assert shift[:2].min() >= -1e-8
    assert np.all(args[1][:2] + shift[:2] >= logit(AI_CUT))
    assert np.all(args[1][2:] + shift[2:] < logit(AI_CUT))
    assert all(g['final_mean_cut_bce'] < g['initial_mean_cut_bce'] for g in report['groups'])


def test_conflicting_features_do_not_waive_correct_real_or_ai_bounds():
    args = list(population())
    args[0] = np.ones((4, 1))
    args[1] = logit(AI_CUT) + np.array([1., -1., -1e-5, -1e-5])
    w, report = fit(*args); shift = args[0] @ w
    assert report['success'] and report['max_constraint_violation'] <= 1e-8
    assert shift[:2].min() >= -1e-8
    assert np.all(args[1][2:] + shift[2:] < logit(AI_CUT))
    assert args[1][1] + shift[1] < logit(AI_CUT)  # Inseparability is kept visible.


def test_invalid_or_cross_label_parents_are_rejected():
    args = list(population()); args[4] = np.array(['same'] * 4)
    with pytest.raises(ValueError, match='parent crosses'):
        fit(*args)
    args = list(population()); args[0][0, 0] = np.nan
    with pytest.raises(ValueError, match='finite'):
        fit(*args)
