"""E43 residual with its frozen TRAIN original basis and a TRAIN CLIP basis."""
import numpy as np
from scipy.special import expit
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from experiments.e64_constrained import AI_CUT, REAL_CUT, fit, objective

RANK = 64
BASE_KEYS = ('base_mean', 'base_components', 'base_scales')


def aligned(original, clip):
    if original.ndim != 2 or clip.ndim != 2 or len(original) != len(clip) or \
            not np.isfinite(original).all() or not np.isfinite(clip).all():
        raise ValueError('finite aligned original and CLIP matrices required')


def fit_basis(original, clip, previous):
    aligned(original, clip)
    # Copy only original coordinates: neither E67 response nor learned weights survive.
    a = {key: np.array(previous[key], copy=True) for key in BASE_KEYS}
    if a['base_components'].shape != (RANK, original.shape[1]) or \
            a['base_mean'].shape != (original.shape[1],) or a['base_scales'].shape != (RANK,):
        raise ValueError('frozen original basis dimensions differ')
    scaler = StandardScaler().fit(clip.astype(np.float64))
    standardized = scaler.transform(clip.astype(np.float64))
    pca = PCA(n_components=RANK, svd_solver='randomized', random_state=71,
              iterated_power=3).fit(standardized)
    a.update(clip_center=scaler.mean_, clip_scale=scaler.scale_, clip_mean=pca.mean_,
             clip_components=pca.components_, clip_scales=np.sqrt(pca.explained_variance_),
             weights=np.zeros(2*RANK+1))
    if any(not np.isfinite(v).all() for v in a.values()) or \
            any(np.any(a[k] <= 0) for k in ['base_scales', 'clip_scale', 'clip_scales']):
        raise ValueError('invalid finite positive basis scales')
    return a, {'clip_variance_explained': float(pca.explained_variance_ratio_.sum()),
               'original_basis_reused_exactly': True}


def project(head, original, clip, a):
    aligned(original, clip)
    base = (head[0].transform(original).astype(np.float64)-a['base_mean']) @ a['base_components'].T / a['base_scales']
    standardized = (clip.astype(np.float64)-a['clip_center'])/a['clip_scale']
    semantic = (standardized-a['clip_mean']) @ a['clip_components'].T / a['clip_scales']
    result = np.column_stack([base, semantic, np.ones(len(original))])
    if result.shape[1] != len(a['weights']) or not np.isfinite(result).all():
        raise ValueError('invalid correction projection')
    return result


def predict(head, original, clip, a):
    aligned(original, clip)
    if not np.any(a['weights']): return head.predict_proba(original)[:, 1]
    return expit(head.decision_function(original)+project(head, original, clip, a) @ a['weights'])
