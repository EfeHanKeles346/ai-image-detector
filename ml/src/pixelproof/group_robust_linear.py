"""Fixed class-balanced entropic risk over FIT source-component mean BCE losses."""
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logsumexp

TEMPERATURE = .1
PENALTY = .01


def grouping(y, groups):
    y = np.asarray(y); groups = np.asarray(groups)
    if y.ndim != 1 or not len(y) or len(y) % 4 or groups.shape != y.shape or \
            y.dtype.kind not in 'iu' or set(y.tolist()) != {0, 1} or groups.dtype.kind not in 'US' or \
            any(not str(g) for g in groups) or \
            not np.all(y.reshape(-1, 4) == y.reshape(-1, 4)[:, :1]) or \
            not np.all(groups.reshape(-1, 4) == groups.reshape(-1, 4)[:, :1]):
        raise ValueError('Complete four-view FIT parents, fixed labels and source groups required')
    keys = sorted(set(zip(y.tolist(), groups.tolist(), strict=True)))
    mapping = {key: i for i, key in enumerate(keys)}
    index = np.array([mapping[key] for key in zip(y.tolist(), groups.tolist(), strict=True)])
    return index, np.bincount(index), np.array([key[0] for key in keys])


def risk(logits, y, structure):
    index, counts, labels = structure
    losses = np.bincount(index, weights=np.logaddexp(0, logits) - y * logits) / counts
    mass = np.zeros(len(counts)); value = 0.
    for label in (0, 1):
        take = labels == label; scaled = losses[take] / TEMPERATURE
        normalizer = logsumexp(scaled)
        value += .5 * TEMPERATURE * (normalizer - np.log(take.sum()))
        mass[take] = .5 * np.exp(scaled - normalizer)
    return float(value), mass[index] / counts[index], losses, mass


def objective(parameters, x, y, structure):
    logits = x @ parameters[:-1] + parameters[-1]
    value, weights, _, _ = risk(logits, y, structure)
    residual = weights * (expit(logits) - y)
    return value + .5 * PENALTY * float(parameters[:-1] @ parameters[:-1]), \
        np.r_[x.T @ residual + PENALTY * parameters[:-1], residual.sum()]


def diagnostics(parameters, x, y, groups):
    structure = grouping(y, groups)
    value, _, losses, mass = risk(x @ parameters[:-1] + parameters[-1], y, structure)
    return dict(entropic_risk=value, group_mean_losses=losses.tolist(),
                group_weight_mass=mass.tolist(), group_classes=structure[2].tolist(),
                class_worst_loss={str(label): float(losses[structure[2] == label].max()) for label in (0, 1)})


def fit_head(x, y, groups, check=lambda: None):
    x = np.asarray(x, dtype=np.float64); y = np.asarray(y)
    structure = grouping(y, groups)
    if x.ndim != 2 or len(x) != len(y) or not x.shape[1] or not np.isfinite(x).all():
        raise ValueError('Finite aligned FIT features required')
    check(); trace = []
    def callback(parameters):
        check(); trace.append(float(objective(parameters, x, y, structure)[0]))
    result = minimize(objective, np.zeros(x.shape[1] + 1), args=(x, y, structure), jac=True,
        method='L-BFGS-B', callback=callback, options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-7})
    loss, gradient = objective(result.x, x, y, structure)
    if not result.success or not np.isfinite(result.x).all() or not np.isfinite(loss) or \
            np.abs(gradient).max() > 1e-5:
        raise RuntimeError('Registered group-robust head did not converge')
    return result.x, dict(success=bool(result.success), iterations=int(result.nit), objective=float(loss),
        gradient_max_abs=float(np.abs(gradient).max()), trace=trace,
        FIT_group_diagnostics=diagnostics(result.x, x, y, groups))
