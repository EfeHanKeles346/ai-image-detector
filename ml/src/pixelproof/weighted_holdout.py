"""FIT-only weighted normalization and covariance PCA, compatible with E131 maps."""
import numpy as np
from sklearn.utils.extmath import randomized_svd, svd_flip


def fit_map(train, width, weights):
    train = np.asarray(train)
    weights = np.asarray(weights, dtype=np.float64)
    if train.ndim != 2 or type(width) is not int or width < 1 or min(train.shape) <= width or \
            not np.isfinite(train).all() or weights.shape != (len(train),) or \
            not np.isfinite(weights).all() or np.any(weights <= 0) or \
            not np.isclose(weights.sum(), 1., rtol=0, atol=1e-10):
        raise ValueError('Finite FIT rows, fixed rank and positive unit-mass weights required')
    weights = weights / weights.sum()
    correction = 1. - float(weights @ weights)
    if correction <= 1e-12:
        raise ValueError('Weighted covariance has insufficient effective support')
    normalized = np.array(train, dtype=np.float64, copy=True)
    center = weights @ normalized
    normalized -= center
    scale = np.maximum(np.sqrt(np.maximum(np.einsum('i,ij,ij->j', weights, normalized, normalized), 0.)), 1e-6)
    normalized /= scale
    mean = weights @ normalized
    normalized -= mean
    normalized *= np.sqrt(weights)[:, None]
    u, singular, components = randomized_svd(normalized, n_components=width,
        n_iter=3, random_state=131, flip_sign=False)
    _, components = svd_flip(u, components, u_based_decision=False)
    # Uniform weights reproduce E131's sample covariance denominator n-1.
    # This numeric convention does not claim independent views or confidence bounds.
    latent_scale = np.sqrt(np.maximum(singular ** 2 / correction, 1e-12))
    result = dict(center=center, scale=scale, mean=mean, components=components, latent_scale=latent_scale)
    if not all(np.isfinite(v).all() for v in result.values()):
        raise ValueError('Nonfinite weighted transform')
    return result
