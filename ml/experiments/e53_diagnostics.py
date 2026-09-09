"""Descriptive audit of frozen E53 predictions; never tunes or refits a model.

ROC operating points are computed separately for each fitted fold head. They are
optimistic, evaluation-label-derived diagnostics, NOT transferable product cuts.
"""
from collections import defaultdict
import json

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

from experiments.e53_offline import EVIDENCE, ROOT, contract, digest, fixed_write
from experiments.e53_coverage import load as coverage_contract


def describe(rows, threshold):
    """Conditional class/source rates, without mixing differently scaled heads."""
    labels = np.asarray([r['label'] for r in rows])
    scores = np.asarray([r['score'] for r in rows], dtype=float)
    if set(labels) != {0, 1} or not np.isfinite(scores).all():
        raise ValueError('finite scores and both labels required')
    predicted = scores >= threshold
    if any(bool(r['predicted_ai']) != bool(p) for r, p in zip(rows, predicted, strict=True)):
        raise ValueError('frozen decision disagrees with cut')
    sources = np.asarray([r['source'] for r in rows])
    per_class = {}
    for label, name in ((0, 'real'), (1, 'ai')):
        values = {}
        for source in sorted(set(sources[labels == label])):
            mask = (labels == label) & (sources == source)
            values[str(source)] = {'parents': int(mask.sum()), 'predicted_ai': int(predicted[mask].sum()),
                                   'predicted_ai_rate': float(predicted[mask].mean())}
        rates = [r['predicted_ai_rate'] for r in values.values()]
        per_class[name] = {'sources': values, 'macro_predicted_ai_rate': float(np.mean(rates)),
                           'worst_predicted_ai_rate': float(max(rates) if label == 0 else min(rates))}
    fpr, tpr, _ = roc_curve(labels, scores, drop_intermediate=False)
    return {'parents': len(rows), 'auc': float(roc_auc_score(labels, scores)),
            'diagnostic_tpr_at_fpr_le_10pct': float(tpr[fpr <= .1].max()),
            'fixed_cut_ai_recall': float(predicted[labels == 1].mean()),
            'fixed_cut_real_fpr': float(predicted[labels == 0].mean()),
            'by_class': per_class}


def transitions(rows, baseline):
    """Count both rescued and newly broken decisions; gains cannot hide losses."""
    if set(rows) != set(baseline):
        raise ValueError('paired populations differ')
    result = defaultdict(lambda: {'parents': 0, 'rescued': 0, 'new_errors': 0})
    for key, candidate in rows.items():
        old = baseline[key]
        if (candidate['label'], candidate['source']) != (old['label'], old['source']):
            raise ValueError('paired identity/label mismatch')
        name = f"{candidate['label']}:{candidate['source']}"
        item = result[name]
        before = int(old['predicted_ai']) == old['label']
        after = int(candidate['predicted_ai']) == candidate['label']
        item['parents'] += 1
        item['rescued'] += int(after and not before)
        item['new_errors'] += int(before and not after)
    return dict(sorted(result.items()))


def report():
    base, binding = contract()
    _, _, v2 = coverage_contract()
    known = {r['parent_id']: r for r in base['rows']}
    output = {'state': 'frozen_E53_descriptive_audit_complete', 'code_sha256': digest(__file__),
              'protocols': {}, 'serving_changed': False, 'new_model_scores': 0,
              'limitations': [
                  'No pooled raw-score ROC across independently calibrated fold heads.',
                  'TPR@FPR10 uses held-out labels, is optimistic, and must not choose a product threshold.',
                  'Source macro/worst rates are not unseen-device or independent-prompt guarantees.',
                  'Both protocols reuse the same consumed TRAIN validation; neither is E52 final.',
                  'Transitions describe all arms, not a post-hoc deployment or winner selection.']}
    for protocol, receipt, folds, expected_contract in (
        ('v1', EVIDENCE/'e53_source_held_out_expanded_controls_result.json', base['folds'], binding),
        ('v2', EVIDENCE/'e53_coverage_result.json', v2['folds'], digest(ROOT/'coverage_v2/contract.json')),
    ):
        summary = json.loads(receipt.read_text())
        grouped = defaultdict(list)
        for filename, expected in summary['result_bindings'].items():
            if digest(filename) != expected:
                raise ValueError('bound predictions changed')
            with open(filename) as stream:
                result = json.load(stream)
            if result['contract_sha256'] != expected_contract:
                raise ValueError('mixed protocol')
            grouped[result['arm']].append(result)
        arms, paired = {}, {}
        for arm, results in sorted(grouped.items()):
            maps = {'clean': {}, 'q75': {}}
            diagnostics = []
            if sorted(r['fold'] for r in results) != [0, 1, 2]:
                raise ValueError('missing/duplicate fold')
            for result in sorted(results, key=lambda r: r['fold']):
                fold = folds[result['fold']]
                for row in result['observations']:
                    parent, condition = row['parent_id'], row['condition']
                    if condition not in maps or parent in maps[condition]:
                        raise ValueError('unexpected/duplicate observation')
                    original = known[parent]
                    if (row['label'], row['source']) != (original['label'], original['source']):
                        raise ValueError('identity metadata changed')
                    if fold['roles'][parent] != 'VALIDATION':
                        raise ValueError('non-validation prediction')
                    maps[condition][parent] = row
                diagnostics.append({'fold': result['fold'], 'cal_selected_cut': result['threshold'],
                    'conditions': {c: describe([r for r in result['observations'] if r['condition'] == c],
                                               result['threshold']) for c in maps}})
            if any(set(rows) != set(known) for rows in maps.values()):
                raise ValueError('incomplete population')
            # Decisions may be pooled; score magnitudes may not. Use boolean scores/cut .5
            # only for class/source rates, deliberately discarding resulting synthetic ROC.
            pooled = {}
            for condition, rows in maps.items():
                binary = [dict(r, score=float(r['predicted_ai'])) for r in rows.values()]
                stats = describe(binary, .5)
                pooled[condition] = {k: stats[k] for k in ('parents', 'fixed_cut_ai_recall',
                                                          'fixed_cut_real_fpr', 'by_class')}
            paired[arm] = maps
            arms[arm] = {'folds': diagnostics, 'pooled_decisions': pooled}
        for arm in arms:
            arms[arm]['paired_transitions'] = {
                ref: {c: transitions(paired[arm][c], paired[ref][c]) for c in ('clean', 'q75')}
                for ref in ('full_old2', 'full_e51_3')}
        output['protocols'][protocol] = {'summary_sha256': digest(receipt), 'arms': arms}
    fixed_write(EVIDENCE/'e53_diagnostics.json', output)
    return {'state': output['state'], 'protocols': list(output['protocols']), 'serving_changed': False}


if __name__ == '__main__':
    print(json.dumps(report(), indent=2))
