"""Strict, paired TRAIN replay check. A pass never authorizes model promotion."""
from __future__ import annotations

import numpy as np


def check_train_retention(parents, reference_scores, candidate_scores, *, conditions, ai_cut):
    """Require every previously caught AI view to remain caught at the frozen cut.

    Scores must be N parents x K conditions in the supplied manifest order. The caller
    must verify artifact/manifest bindings before calling. This is an empirical replay
    gate, not a guarantee on unseen generators or an independent evaluation.
    """
    if not np.isscalar(ai_cut) or not np.isfinite(ai_cut) or not 0 < ai_cut < 1:
        raise ValueError('finite probability cut strictly inside (0, 1) required')
    if not conditions or any(not isinstance(c, str) or not c for c in conditions):
        raise ValueError('named conditions required')
    if len(set(conditions)) != len(conditions):
        raise ValueError('duplicate condition')
    if not parents:
        raise ValueError('complete nonempty TRAIN manifest required')
    ids = []
    for row in parents:
        if str(row.get('role', '')).upper() != 'TRAIN':
            raise ValueError('only admitted TRAIN parents permitted')
        if (not isinstance(row.get('label'), (int, np.integer))
                or isinstance(row.get('label'), bool) or row['label'] not in (0, 1)):
            raise ValueError('binary integer labels required')
        for key in ('parent_id', 'source'):
            if not isinstance(row.get(key), str) or not row[key]:
                raise ValueError('parent identity and source required')
        ids.append(row['parent_id'])
    if len(set(ids)) != len(ids):
        raise ValueError('duplicate parent; derivatives cannot inflate sample size')
    labels = np.asarray([r['label'] for r in parents])
    if set(labels) != {0, 1}:
        raise ValueError('both REAL and AI classes required')
    old, new = (np.asarray(a, dtype=np.float64) for a in (reference_scores, candidate_scores))
    for scores in (old, new):
        if scores.shape != (len(parents), len(conditions)):
            raise ValueError('complete parent-by-condition scores required')
        if not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
            raise ValueError('finite probabilities required')
    old_ai, new_ai = old >= ai_cut, new >= ai_cut
    ai = labels == 1
    if not old_ai[ai].any():
        raise ValueError('reference catches no AI; retention would be vacuous')
    lost = ai[:, None] & old_ai & ~new_ai
    sources = np.asarray([r['source'] for r in parents])

    def compare(mask, j):
        before, after = old_ai[mask, j], new_ai[mask, j]
        return {'parents': int(mask.sum()), 'old_ai_count': int(before.sum()),
                'new_ai_count': int(after.sum()), 'old_ai_rate': float(before.mean()),
                'new_ai_rate': float(after.mean()),
                'ai_to_non_ai': int((before & ~after).sum()),
                'non_ai_to_ai': int((~before & after).sum())}

    comparisons = {}
    for j, condition in enumerate(conditions):
        comparisons[condition] = {
            'real': compare(~ai, j), 'ai': compare(ai, j),
            'by_source': {f'{y}:{source}': compare((labels == y) & (sources == source), j)
                          for y in (0, 1) for source in sorted(set(sources[labels == y]))}}
    return {'state': 'pass_train_retention' if not lost.any() else 'reject_ai_replay_regression',
            'passes_train_retention': not bool(lost.any()), 'parents': len(parents),
            'ai_parents': int(ai.sum()), 'views': int(old.size),
            'new_ai_miss_views': int(lost.sum()),
            'new_ai_miss_parents': int(lost.any(axis=1).sum()),
            'misses_by_condition': {c: int(lost[:, j].sum()) for j, c in enumerate(conditions)},
            'comparisons': comparisons,
            'promotion_allowed': False, 'independent_quality_claim': False,
            'scope': 'Necessary TRAIN replay condition only; no unseen-data guarantee. '
                     'New AI rescues cannot cancel previously caught AI losses.'}
