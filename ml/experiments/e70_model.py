"""Frozen original/blur coordinates plus a bilinear TensorSketch correction map."""
import numpy as np
from scipy.special import expit
from sklearn.preprocessing import StandardScaler
from experiments import e67_model as linear
from experiments.e64_constrained import AI_CUT, REAL_CUT, fit, objective

SKETCH_DIM = 128
SEED = 70


def sketch_parameters(width=64):
    rng = np.random.default_rng(SEED)
    return {'sketch_hash': rng.integers(SKETCH_DIM, size=(2, width)),
            'sketch_sign': rng.choice(np.array([-1., 1.]), size=(2, width))}


def bilinear_sketch(left, right, hashes, signs):
    left = np.asarray(left, dtype=np.float64); right = np.asarray(right, dtype=np.float64)
    hashes = np.asarray(hashes); signs = np.asarray(signs)
    if left.ndim != 2 or right.shape != left.shape or hashes.shape != (2, left.shape[1]) or signs.shape != hashes.shape:
        raise ValueError('aligned bilinear inputs and hash/sign arrays required')
    if not np.isfinite(left).all() or not np.isfinite(right).all() or \
            not np.issubdtype(hashes.dtype, np.integer) or np.any((hashes < 0) | (hashes >= SKETCH_DIM)) or \
            not np.isin(signs, [-1., 1.]).all():
        raise ValueError('invalid bilinear inputs or sketch parameters')
    a = np.zeros((len(left), SKETCH_DIM)); b = np.zeros_like(a)
    for j in range(left.shape[1]):
        a[:, hashes[0, j]] += signs[0, j] * left[:, j]
        b[:, hashes[1, j]] += signs[1, j] * right[:, j]
    result = np.fft.ifft(np.fft.fft(a, axis=1) * np.fft.fft(b, axis=1), axis=1).real
    if not np.isfinite(result).all(): raise ValueError('nonfinite bilinear map')
    return result


def fit_map(head, original, blurred, basis):
    a = {k: np.array(v, copy=True) for k, v in basis.items() if k not in ['weights', 'contract_sha256', 'reference_sha256']}
    coordinates = linear.project(head, original, blurred, a)[:, :-1]
    if coordinates.shape[1] != 128: raise ValueError('expected frozen original64/response64 basis')
    a.update(sketch_parameters())
    raw = bilinear_sketch(coordinates[:, :64], coordinates[:, 64:], a['sketch_hash'], a['sketch_sign'])
    scaler = StandardScaler().fit(raw)
    a.update(interaction_mean=scaler.mean_, interaction_scale=scaler.scale_, weights=np.zeros(257))
    return a


def project(head, original, blurred, a):
    coordinates = linear.project(head, original, blurred, a)[:, :-1]
    raw = bilinear_sketch(coordinates[:, :64], coordinates[:, 64:], a['sketch_hash'], a['sketch_sign'])
    interactions = (raw - a['interaction_mean']) / a['interaction_scale']
    result = np.column_stack([coordinates, interactions, np.ones(len(coordinates))])
    if result.shape[1] != 257 or not np.isfinite(result).all(): raise ValueError('invalid interaction projection')
    return result


def predict(head, original, blurred, a):
    if not np.any(a['weights']): return head.predict_proba(original)[:, 1]
    return expit(head.decision_function(original) + project(head, original, blurred, a) @ a['weights'])
