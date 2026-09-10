"""Bounded additive correction to an immutable E43 classifier; no threshold fitting."""
import numpy as np
from scipy.special import expit

BOUND = 2.0
RETENTION = 10.0
L2 = 0.01
STEPS = 200
LR = 0.02
AI_CUT = 0.07940196245908739
REAL_CUT = 0.011505939625203613


def correction_features(head, features):
    features = np.asarray(features)
    if features.ndim != 2 or features.shape[1] != 3072 or not np.isfinite(features).all():
        raise ValueError('finite E43 3072-D features required')
    # Preserve the original scaler and its float32 transform before using CPU float64.
    return head[0].transform(features).astype(np.float64) / np.sqrt(features.shape[1])


def delta(x, weights):
    if x.ndim != 2 or weights.shape != (x.shape[1],) or not np.isfinite(weights).all():
        raise ValueError('invalid correction weights')
    return BOUND * np.tanh((x @ weights) / BOUND)


def predict(head, features, weights):
    if np.asarray(weights).shape != (3072,) or not np.isfinite(weights).all():
        raise ValueError('invalid saved correction')
    if not np.any(weights):
        return head.predict_proba(features)[:, 1]  # Exact original path, including saturation.
    return expit(head.decision_function(features) + delta(correction_features(head, features), weights))


def objective(weights, x, baseline_logit, labels, supervised_weights, retention_weights):
    t = np.tanh((x @ weights) / BOUND)
    shift = BOUND * t
    logit = baseline_logit + shift - np.log(AI_CUT / (1 - AI_CUT))
    loss = np.dot(supervised_weights, np.logaddexp(0.0, logit) - labels * logit)
    derivative = supervised_weights * (expit(logit) - labels)
    # Preserve the reference's correct-AI margin, not a pseudo-label target.
    down = np.minimum(shift, 0.0)
    loss += RETENTION * 0.5 * np.dot(retention_weights, down**2)
    derivative += RETENTION * retention_weights * down
    loss += L2 * 0.5 * np.dot(weights, weights)
    gradient = x.T @ (derivative * (1 - t**2)) + L2 * weights
    return float(loss), gradient


def fit_fixed(x, baseline_logit, labels, supervised_weights, retention_weights, check=lambda: None):
    weights = np.zeros(x.shape[1], dtype=np.float64)
    first = np.zeros_like(weights)
    second = np.zeros_like(weights)
    trace = []
    for step in range(1, STEPS + 1):
        check()
        loss, gradient = objective(weights, x, baseline_logit, labels, supervised_weights, retention_weights)
        if not np.isfinite(loss) or not np.isfinite(gradient).all():
            raise ValueError('nonfinite correction optimization')
        first = .9 * first + .1 * gradient
        second = .999 * second + .001 * gradient**2
        weights -= LR * (first / (1 - .9**step)) / (np.sqrt(second / (1 - .999**step)) + 1e-8)
        if step == 1 or step % 25 == 0:
            trace.append({'step': step, 'training_objective_before_step': loss})
    return weights, trace
