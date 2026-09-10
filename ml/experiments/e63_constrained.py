"""Nonlinear Gaussian basis, convex correction coefficients and frozen TRAIN constraints."""
import numpy as np
from scipy.spatial.distance import cdist
from scipy.special import expit
from experiments.e62_constrained import (AI_CUT, REAL_CUT, RANK, L2, MAX_ITER, FTOL,
                                        GUARD_MARGIN, fit, objective)
from experiments.e62_constrained import project as pca_project


def kernels(coordinates, centers, sigma):
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError('positive finite TRAIN-derived bandwidth required')
    return np.exp(-cdist(coordinates, centers, metric='sqeuclidean') / (2 * sigma**2))


def project(head, features, arrays):
    z = pca_project(head, features, arrays['mean'], arrays['components'], arrays['scales'])[:, :-1]
    k = kernels(z, arrays['centers'], float(arrays['sigma']))
    k = (k - arrays['kernel_mean']) / arrays['kernel_scales']
    return np.column_stack((k, np.ones(len(k))))


def predict(head, features, arrays):
    if not np.any(arrays['weights']):
        return head.predict_proba(features)[:, 1]
    return expit(head.decision_function(features) + project(head, features, arrays) @ arrays['weights'])
