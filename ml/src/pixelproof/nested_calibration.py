"""Declared-component splits and CAL-only fixed-budget threshold selection."""
from collections import Counter
from itertools import permutations
import numpy as np


def split_roster(rows, components, folds):
    ids = [r['parent_id'] for r in rows]
    if not rows or len(set(ids)) != len(ids) or set(ids) != set(components) or set(ids) != set(folds):
        raise ValueError('Complete unique parent/component/fold mappings required')
    if any(str(r['role']).upper() != 'TRAIN' or type(r['label']) is not int or r['label'] not in (0, 1) for r in rows):
        raise ValueError('Explicit TRAIN binary labels required')
    if set(folds.values()) != {0, 1, 2}:
        raise ValueError('Exactly three inherited folds required')
    for group in set(components.values()):
        if len({folds[p] for p in ids if components[p] == group}) != 1:
            raise ValueError('Declared component crosses folds')
    support = {}
    for fold in range(3):
        selected = [r for r in rows if folds[r['parent_id']] == fold]
        counts = Counter(r['label'] for r in selected)
        if set(counts) != {0, 1}:
            raise ValueError('Each FIT/CAL/EVAL fold needs both classes')
        support[str(fold)] = dict(parents=len(selected), class_counts={str(k): v for k, v in counts.items()},
            class_components={str(y): len({components[r['parent_id']] for r in selected if r['label'] == y}) for y in (0, 1)},
            sources=sorted({r['source'] for r in selected}))
    return dict(fold_support=support, assignments=[dict(FIT=f, CAL=c, EVAL=e) for f, c, e in permutations(range(3))])


def validate_scores(labels, scores, groups):
    labels = np.asarray(labels)
    scores = np.asarray(scores, dtype=np.float64)
    groups = np.asarray(groups)
    if scores.ndim != 2 or not len(scores) or scores.shape[1] != 4 or labels.shape != (len(scores),) or \
            groups.shape != labels.shape or set(labels.tolist()) != {0, 1} or \
            not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
        raise ValueError('Both classes and complete finite four-view CAL/EVAL scores required')
    return labels, scores, groups


def outcomes(labels, scores, groups, cut):
    labels, scores, groups = validate_scores(labels, scores, groups)
    if not np.isfinite(cut) or not 0 <= cut <= np.nextafter(1., np.inf):
        raise ValueError('Finite score cutoff required')
    result = []
    for j in range(4):
        pred = scores[:, j] >= cut
        base = scores[:, j] >= .5
        real, ai = labels == 0, labels == 1
        fprs = [float(pred[real & (groups == g)].mean()) for g in sorted(set(groups[real]))]
        result.append(dict(REAL_parents=int(real.sum()), AI_parents=int(ai.sum()),
            REAL_FP=int(pred[real].sum()), AI_TP=int(pred[ai].sum()),
            REAL_FPR=float(pred[real].mean()), AI_recall=float(pred[ai].mean()),
            worst_REAL_component_FPR=max(fprs),
            new_AI_misses=int((base & ~pred & ai).sum()), rescued_AI=int((~base & pred & ai).sum()),
            new_REAL_alerts=int((~base & pred & real).sum()), rescued_REAL=int((base & ~pred & real).sum())))
    return result


def select_cut(labels, scores, groups):
    """Smallest CAL cut satisfying <=10% pooled/<=20% component REAL FP in all views.

    Reject if it loses any .5-caught CAL AI or yields <95% CAL recall in any view.
    No EVAL inputs, fitted probabilities, grid search or fallback tuned on EVAL.
    """
    labels, scores, groups = validate_scores(labels, scores, groups)
    real = labels == 0
    constraints = [(real, .10)] + [(real & (groups == g), .20) for g in sorted(set(groups[real]))]
    cut = 0.
    for mask, budget in constraints:
        allowed = int(np.floor(budget * int(mask.sum()) + 1e-12))
        for j in range(4):
            ranked = np.sort(scores[mask, j])[::-1]
            # >=cut is positive: move strictly above the first disallowed score.
            cut = max(cut, float(np.nextafter(ranked[allowed], np.inf)))
    measured = outcomes(labels, scores, groups, cut)
    reasons = []
    if any(r['new_AI_misses'] for r in measured):
        reasons.append('CAL_AI_retention_failed')
    if any(r['AI_recall'] < .95 for r in measured):
        reasons.append('CAL_AI_recall_below_95_percent')
    if any(r['REAL_FPR'] > .10 + 1e-12 or r['worst_REAL_component_FPR'] > .20 + 1e-12 for r in measured):
        raise AssertionError('Order-statistic budget calculation failed')
    return dict(accepted=not reasons, cut=cut, rejection_reasons=reasons, conditions=measured)
