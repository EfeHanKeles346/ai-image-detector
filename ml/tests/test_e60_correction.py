import numpy as np
import pytest
from scipy.special import expit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from experiments.e60_correction import objective, delta, predict, BOUND


def test_analytic_retention_gradient_matches_finite_difference():
    rng = np.random.default_rng(60)
    x = rng.normal(size=(8, 4)); w = rng.normal(size=4) * .1
    base = rng.normal(size=8); labels = np.arange(8) % 2
    supervised = np.ones(8) / 8; retention = labels / labels.sum()
    _, gradient = objective(w, x, base, labels, supervised, retention)
    numerical = np.zeros(4)
    for j in range(4):
        e = np.eye(4)[j] * 1e-6
        numerical[j] = (objective(w+e, x, base, labels, supervised, retention)[0] -
                        objective(w-e, x, base, labels, supervised, retention)[0]) / 2e-6
    np.testing.assert_allclose(gradient, numerical, atol=1e-8)


def test_zero_correction_exactly_preserves_saturated_and_boundary_scores():
    scaler = StandardScaler()
    scaler.mean_ = np.zeros(3072); scaler.scale_ = np.ones(3072); scaler.n_features_in_ = 3072
    classifier = LogisticRegression()
    classifier.classes_ = np.array([0, 1]); classifier.coef_ = np.zeros((1, 3072))
    classifier.coef_[0, 0] = 1.; classifier.intercept_ = np.zeros(1); classifier.n_features_in_ = 3072
    head = make_pipeline(scaler, classifier)
    x = np.zeros((5, 3072), dtype=np.float32)
    x[:, 0] = [-1000, -2.45, 0, 3, 1000]
    np.testing.assert_array_equal(predict(head, x, np.zeros(3072)), head.predict_proba(x)[:, 1])
    with pytest.raises(ValueError):
        predict(head, x, np.zeros(2))


def test_correction_is_bounded_on_extreme_unseen_features():
    result = delta(np.array([[1e12, 1e12], [-1e12, -1e12]]), np.ones(2))
    assert np.max(np.abs(result)) <= BOUND
    np.testing.assert_array_equal(result, [BOUND, -BOUND])
