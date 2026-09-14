"""Matched AI-localization diagnostics; masks are evaluation input, never model input.

Threshold and boundary width must be fixed by the calling experiment before scores.
These descriptive per-parent metrics do not establish independent generalization.
"""
from numbers import Integral, Real

import numpy as np
from scipy.ndimage import maximum_filter, minimum_filter
from sklearn.metrics import average_precision_score, roc_auc_score


def regions(intended_mask, boundary_width):
    """Partition using in-image Chebyshev distance to the opposite mask class.

    Replicated image borders do not invent an edit boundary outside the image.
    Thin edits may have no interior; the caller must retain that missing endpoint.
    """
    mask = np.asarray(intended_mask)
    if mask.ndim != 2 or mask.dtype != bool or not mask.any() or mask.all():
        raise ValueError('A nondegenerate two-dimensional boolean intended mask is required')
    if isinstance(boundary_width, bool) or not isinstance(boundary_width, Integral) or boundary_width < 1:
        raise ValueError('A positive integer boundary width is required')
    size = 2 * int(boundary_width) + 1
    inner = minimum_filter(mask, size=size, mode='nearest')
    outer = maximum_filter(mask, size=size, mode='nearest')
    return {'interior': inner, 'inner_boundary': mask & ~inner,
            'outer_boundary': outer & ~mask, 'background': ~outer}


def _score(value, shape):
    score = np.asarray(value)
    if score.shape != shape or score.dtype.kind != 'f' or not np.isfinite(score).all() or \
            np.any((score < 0) | (score > 1)):
        raise ValueError('Aligned finite floating-point scores in [0,1] are required')
    return score


def _ranking(target, score):
    if target.size == 0 or not target.any() or target.all():
        return {'auc': None, 'ap': None, 'prevalence': None,
                'reason': 'Both target classes are required'}
    return {'auc': float(roc_auc_score(target, score)),
            'ap': float(average_precision_score(target, score)),
            'prevalence': float(target.mean()), 'reason': None}


def evaluate_triplet(authentic, classical_edit, ai_composite, intended_mask, *, threshold, boundary_width):
    """Compare three maps on the same parent, geometry and intended edit region.

    Authentic and classical edit have all-zero AI truth. The intended edit mask
    only stratifies their false alarms. Raw full-frame diffusion output is not
    accepted as a locally labelled composite by this interface's data contract.
    """
    if isinstance(threshold, bool) or not isinstance(threshold, Real) or \
            not np.isfinite(threshold) or not 0 < threshold < 1:
        raise ValueError('A prespecified finite threshold strictly inside (0,1) is required')
    partition = regions(intended_mask, boundary_width)
    mask = np.asarray(intended_mask)
    score_maps = {name: _score(value, mask.shape) for name, value in
                  [('authentic', authentic), ('classical_edit', classical_edit)]}
    # A rejected generation has no usable AI map. Retain both matched negatives
    # without inventing an all-zero positive prediction or dropping its parent.
    if ai_composite is not None:
        score_maps['ai_composite'] = _score(ai_composite, mask.shape)
    selected = partition['interior'] | partition['background']
    result = {'threshold': float(threshold), 'boundary_width_pixels': int(boundary_width),
              'distance': 'Chebyshev, in-image boundary only',
              'region_pixels': {name: int(region.sum()) for name, region in partition.items()},
              'interior_and_background_available': bool(partition['interior'].any() and partition['background'].any()),
              'composite_available': ai_composite is not None, 'maps': {}}
    for name, score in score_maps.items():
        predicted = score >= threshold
        truth = mask if name == 'ai_composite' else np.zeros(mask.shape, dtype=bool)
        region_metrics = {}
        for region_name, region in partition.items():
            count = int(region.sum())
            region_metrics[region_name] = {'pixels': count,
                'mean_score': float(score[region].mean()) if count else None,
                'flagged_fraction': float(predicted[region].mean()) if count else None}
        tp, fp, fn = (int(value.sum()) for value in
                      (predicted & truth, predicted & ~truth, ~predicted & truth))
        result['maps'][name] = {'regions': region_metrics,
            'flagged_area_fraction': float(predicted.mean()),
            'confusion_pixels': {'tp': tp, 'fp': fp, 'fn': fn, 'tn': int((~predicted & ~truth).sum())},
            'iou': float(tp / (tp + fp + fn)) if tp + fp + fn else None,
            'all_pixel_ranking': _ranking(truth.ravel(), score.ravel()),
            'interior_background_ranking': _ranking(truth[selected], score[selected])}
    result['limits'] = ('Per-parent diagnostic only. Classical edits are AI-negative. Empty regions remain null; '
                        'no threshold search, pixel-independence claim, training admission or promotion decision.')
    return result
