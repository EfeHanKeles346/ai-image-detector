"""Convex TRAIN correction preserving original correct decisions in both classes."""
import numpy as np
from scipy.optimize import LinearConstraint, minimize
from scipy.special import expit, logit

AI_CUT = 0.07940196245908739
REAL_CUT = 0.011505939625203613
RANK = 64
L2 = 0.01
MAX_ITER = 200
FTOL = 1e-9
GUARD_MARGIN = 1e-7


def objective(w, x, baseline_logits, labels, sample_weights):
    z = baseline_logits + x @ w - logit(AI_CUT)
    loss = np.dot(sample_weights, np.logaddexp(0, z) - labels * z) + .5 * L2 * np.dot(w, w)
    grad = x.T @ (sample_weights * (expit(z) - labels)) + L2 * w
    return float(loss), grad


def fit(x, baseline_logits, labels, weights, check=lambda: None):
    old_ai = baseline_logits >= logit(AI_CUT)
    protected_ai = (labels == 1) & old_ai
    protected_real = (labels == 0) & ~old_ai
    # Zero is feasible even for a reference exactly on the AI threshold.
    slack = np.maximum(baseline_logits[protected_ai] - logit(AI_CUT) - GUARD_MARGIN, 0)
    real_slack = np.maximum(logit(AI_CUT) - baseline_logits[protected_real] - GUARD_MARGIN, 0)
    constraints = [LinearConstraint(x[protected_ai], -slack, np.inf),
                   LinearConstraint(x[protected_real], -np.inf, real_slack)]
    trace = []
    def callback(w):
        check()
        trace.append(float(objective(w, x, baseline_logits, labels, weights)[0]))
    result = minimize(objective, np.zeros(x.shape[1]),
                      args=(x, baseline_logits, labels, weights), method='SLSQP', jac=True,
                      constraints=constraints, callback=callback,
                      options={'maxiter': MAX_ITER, 'ftol': FTOL, 'disp': False})
    shift = x @ result.x
    violation = max(0., float(np.max(-slack - shift[protected_ai])),
                    float(np.max(shift[protected_real] - real_slack)))
    return result.x, {'success': bool(result.success), 'message': str(result.message),
                      'iterations': int(result.nit), 'objective': float(result.fun),
                      'max_constraint_violation': violation, 'trace': trace}


def project(head, features, mean, components, scales):
    scaled = head[0].transform(features).astype(np.float64)
    z = ((scaled - mean) @ components.T) / scales
    return np.column_stack((z, np.ones(len(z))))


def predict(head, features, arrays):
    if not np.any(arrays['weights']):
        return head.predict_proba(features)[:, 1]
    x = project(head, features, arrays['mean'], arrays['components'], arrays['scales'])
    return expit(head.decision_function(features) + x @ arrays['weights'])
