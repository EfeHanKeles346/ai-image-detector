"""Two-class worst-group correction under all-AI confidence preservation.

E68 supplies parent-balanced class/source/condition risks; E73 supplies the
stronger all-AI bounds. Neither historical implementation is changed.
"""
import numpy as np
from scipy.optimize import minimize
from experiments.e68_minimax import GroupRisks, objective
from experiments.e64_constrained import MAX_ITER, FTOL
from experiments.e73_fit import constraints_for


def fit(x, baseline_logits, labels, sources, parents, conditions, check=lambda: None):
    risk = GroupRisks(x, baseline_logits, labels, sources, parents, conditions)
    extended = np.column_stack([x, np.zeros((len(x), 2))])
    bounds, ai, real, slack = constraints_for(extended, baseline_logits, labels)
    values, _ = risk.risks(np.zeros(x.shape[1]))
    initial = np.r_[np.zeros(x.shape[1]),
                    [values[risk.classes == y].max() + 1e-10 for y in (0, 1)]]
    trace = []

    def callback(v):
        check()
        trace.append(float(objective(v)[0]))

    result = minimize(objective, initial, jac=True, method='SLSQP',
        constraints=[*bounds, {'type': 'ineq', 'fun': lambda v: risk.epigraph(v)[0],
                              'jac': lambda v: risk.epigraph(v)[1]}],
        callback=callback, options={'maxiter': MAX_ITER, 'ftol': FTOL, 'disp': False})
    w = result.x[:-2]
    shift = x @ w
    linear = max(0., float(np.max(-shift[ai])),
                 float(np.max(shift[real] - slack)) if real.any() else 0.)
    nonlinear = max(0., float(-risk.epigraph(result.x)[0].min()))
    final_values, _ = risk.risks(w)
    return w, {
        'success': bool(result.success), 'message': str(result.message),
        'iterations': int(result.nit), 'objective': float(result.fun), 'trace': trace,
        'max_constraint_violation': max(linear, nonlinear),
        'linear_constraint_violation': linear, 'epigraph_violation': nonlinear,
        'all_ai_views_protected': int(ai.sum()),
        'minimum_ai_logit_shift': float(shift[ai].min()),
        'class_bounds': result.x[-2:].tolist(),
        'initial_class_bounds': initial[-2:].tolist(),
        'groups': [{'label': int(k[0]), 'source': str(k[1]), 'condition': str(k[2]),
                    'initial_mean_cut_bce': float(values[i]),
                    'final_mean_cut_bce': float(final_values[i])}
                   for i, k in enumerate(risk.keys)]}
