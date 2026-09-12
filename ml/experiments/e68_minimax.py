"""Convex class-balanced worst-source/condition loss with frozen decision guards."""
from collections import Counter
import numpy as np
from scipy.optimize import LinearConstraint, minimize
from scipy.special import expit, logit
from experiments.e64_constrained import AI_CUT, L2, MAX_ITER, FTOL, GUARD_MARGIN


class GroupRisks:
    def __init__(self, x, logits, labels, sources, parents, conditions):
        self.x = np.asarray(x, dtype=np.float64)
        self.logits = np.asarray(logits, dtype=np.float64) - logit(AI_CUT)
        self.labels = np.asarray(labels)
        n = len(self.x)
        if self.x.ndim != 2 or any(np.asarray(a).shape != (n,) for a in
                [logits, labels, sources, parents, conditions]):
            raise ValueError('aligned group training arrays required')
        if set(self.labels.tolist()) != {0, 1} or not np.isfinite(self.x).all() or not np.isfinite(self.logits).all():
            raise ValueError('finite two-class training arrays required')
        keys = list(zip(self.labels.tolist(), sources, conditions))
        self.keys = sorted(set(keys))
        self.classes = np.array([key[0] for key in self.keys])
        self.groups = []
        identities = {}
        for y, source, parent in zip(labels, sources, parents):
            if not source or not parent: raise ValueError('empty source/parent')
            if parent in identities and identities[parent] != (y, source):
                raise ValueError('parent crosses label/source boundary')
            identities[parent] = (y, source)
        for key in self.keys:
            idx = np.array([i for i, k in enumerate(keys) if k == key])
            counts = Counter(parents[i] for i in idx)
            weights = np.array([1 / (len(counts) * counts[parents[i]]) for i in idx])
            self.groups.append((idx, weights))

    def risks(self, w):
        z = self.logits + self.x @ w
        loss = np.logaddexp(0, z) - self.labels * z
        derivative = expit(z) - self.labels
        values = np.empty(len(self.groups))
        jac = np.empty((len(self.groups), self.x.shape[1]))
        for i, (idx, weight) in enumerate(self.groups):
            values[i] = weight @ loss[idx]
            jac[i] = (weight * derivative[idx]) @ self.x[idx]
        return values, jac

    def epigraph(self, v):
        values, jac = self.risks(v[:-2])
        gradient = np.zeros((len(values), len(v)))
        gradient[:, :-2] = -jac
        gradient[np.arange(len(values)), self.x.shape[1] + self.classes] = 1
        return v[-2:][self.classes] - values, gradient


def objective(v):
    return (.5 * float(v[-2:].sum()) + .5 * L2 * float(v[:-2] @ v[:-2]),
            np.r_[L2 * v[:-2], .5, .5])


def fit(x, baseline_logits, labels, sources, parents, conditions, check=lambda: None):
    risk = GroupRisks(x, baseline_logits, labels, sources, parents, conditions)
    old_ai = baseline_logits >= logit(AI_CUT)
    ai = (labels == 1) & old_ai
    real = (labels == 0) & ~old_ai
    if not ai.any() or not real.any(): raise ValueError('both protected classes required')
    ai_slack = np.maximum(baseline_logits[ai] - logit(AI_CUT) - GUARD_MARGIN, 0)
    real_slack = np.maximum(logit(AI_CUT) - baseline_logits[real] - GUARD_MARGIN, 0)
    extended = np.column_stack([x, np.zeros((len(x), 2))])
    zero = np.zeros(x.shape[1]); values, _ = risk.risks(zero)
    initial = np.r_[zero, [values[risk.classes == y].max() + 1e-10 for y in (0, 1)]]
    trace = []
    def callback(v):
        check(); trace.append(objective(v)[0])
    result = minimize(objective, initial, jac=True, method='SLSQP', constraints=[
        LinearConstraint(extended[ai], -ai_slack, np.inf),
        LinearConstraint(extended[real], -np.inf, real_slack),
        {'type': 'ineq', 'fun': lambda v: risk.epigraph(v)[0], 'jac': lambda v: risk.epigraph(v)[1]},
    ], callback=callback, options={'maxiter': MAX_ITER, 'ftol': FTOL, 'disp': False})
    w = result.x[:-2]; shift = x @ w
    linear_violation = max(0., float(np.max(-ai_slack - shift[ai])),
                           float(np.max(shift[real] - real_slack)))
    epigraph_violation = max(0., float(-np.min(risk.epigraph(result.x)[0])))
    final_values, _ = risk.risks(w)
    return w, {'success': bool(result.success), 'message': str(result.message),
        'iterations': int(result.nit), 'objective': float(result.fun), 'trace': trace,
        'max_constraint_violation': max(linear_violation, epigraph_violation),
        'linear_constraint_violation': linear_violation, 'epigraph_violation': epigraph_violation,
        'class_bounds': result.x[-2:].tolist(), 'groups': [
            {'label': int(k[0]), 'source': str(k[1]), 'condition': str(k[2]),
             'initial_mean_cut_bce': float(values[i]), 'final_mean_cut_bce': float(final_values[i])}
            for i, k in enumerate(risk.keys)]}
