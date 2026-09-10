"""Score-only, optimistic ROC diagnosis. Never exports an operating threshold."""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from experiments.e53_offline import EVIDENCE, digest, fixed_write

CONTRACT = EVIDENCE / 'e58_diagnostic_contract.json'
RESULT = EVIDENCE / 'e58_score_diagnosis.json'
ARMS = ('native64', 'native64_fivek')


def envelope(labels, scores):
    y = np.asarray(labels)
    s = np.asarray(scores, dtype=float)
    if y.ndim != 1 or s.shape != y.shape or set(y) != {0, 1} or not np.isfinite(s).all():
        raise ValueError('finite binary two-class scores required')
    order = np.argsort(-s, kind='stable')
    y, s = y[order], s[order]
    # A deterministic threshold cannot split equal-score photographs.
    ends = np.r_[np.flatnonzero(s[:-1] != s[1:]), len(s)-1]
    tpr = np.r_[0., np.cumsum(y == 1)[ends] / np.sum(y == 1)]
    fpr = np.r_[0., np.cumsum(y == 0)[ends] / np.sum(y == 0)]
    return {
        'real_parents': int(np.sum(y == 0)), 'ai_parents': int(np.sum(y == 1)),
        'max_ai_recall_at_real_fpr_10pct': float(tpr[fpr <= .1].max()),
        'min_real_fpr_at_ai_recall_80pct': float(fpr[tpr >= .8].min()),
        'min_real_fpr_at_ai_recall_95pct': float(fpr[tpr >= .95].min()),
        'oracle_80recall_10fpr_feasible': bool(np.any((tpr >= .8) & (fpr <= .1))),
        'oracle_95recall_10fpr_feasible': bool(np.any((tpr >= .95) & (fpr <= .1))),
    }


def transitions(baseline, candidate):
    def index(rows):
        result = {}
        for r in rows:
            key = (r['parent_id'], r['condition'])
            if key in result:
                raise ValueError('duplicate observation')
            result[key] = r
        return result
    b, c = index(baseline), index(candidate)
    if b.keys() != c.keys():
        raise ValueError('unpaired observations')
    groups = defaultdict(lambda: {'parents': 0, 'baseline_errors': 0, 'candidate_errors': 0,
                                 'rescued_errors': 0, 'new_errors': 0})
    for key, before in b.items():
        after = c[key]
        if any(before[k] != after[k] for k in ('label', 'source')):
            raise ValueError('truth/source changed')
        if before['label'] not in (0, 1):
            raise ValueError('unknown label')
        old_error = before['predicted_ai'] != bool(before['label'])
        new_error = after['predicted_ai'] != bool(after['label'])
        out = groups[(before['source'], before['label'], before['condition'])]
        out['parents'] += 1
        out['baseline_errors'] += int(old_error)
        out['candidate_errors'] += int(new_error)
        out['rescued_errors'] += int(old_error and not new_error)
        out['new_errors'] += int(new_error and not old_error)
    return [dict(source=s, label=y, condition=c, **v) for (s, y, c), v in sorted(groups.items())]


def freeze():
    summary = json.loads((EVIDENCE / 'e57_result.json').read_text())
    if summary['eligible'] or summary['serving_changed']:
        raise ValueError('diagnosis assumes rejected, unserved E57')
    inputs = {str(EVIDENCE / 'e57_result.json'): digest(EVIDENCE / 'e57_result.json')}
    paths = {}
    for path, sha in summary['result_bindings'].items():
        p = Path(path)
        if digest(p) != sha:
            raise ValueError('bound E57 result changed')
        for arm in ARMS:
            for fold in range(3):
                if p.name != f'{arm}_fold{fold}.json':
                    continue
                inputs[path] = sha
                value = json.loads(p.read_text())
                artifact = p.with_suffix('.joblib')
                if digest(artifact) != value['artifact_sha256'] or not value['artifact_replay']['passed']:
                    raise ValueError('artifact/replay invalid')
                inputs[str(artifact)] = digest(artifact)
                paths[f'{arm}:{fold}'] = path
    if len(paths) != 6:
        raise ValueError('six paired folds required')
    contract = {'state': 'E58_score_diagnosis_frozen', 'code_sha256': digest(__file__),
                'inputs': inputs, 'paths': paths, 'policy': 'Existing scores only; label-oracle diagnostic, not deployable cuts.',
                'recall_targets': [.8, .95], 'real_fpr_budget': .1}
    fixed_write(CONTRACT, contract)
    return contract


def run():
    contract = json.loads(CONTRACT.read_text())
    if digest(__file__) != contract['code_sha256']:
        raise ValueError('frozen code changed')
    for path, sha in contract['inputs'].items():
        if digest(path) != sha:
            raise ValueError('frozen input changed: ' + path)
    folds = []
    for fold in range(3):
        values = {a: json.loads(Path(contract['paths'][f'{a}:{fold}']).read_text()) for a in ARMS}
        for a, v in values.items():
            if v['arm'] != a or v['fold'] != fold:
                raise ValueError('arm/fold mismatch')
            for r in v['observations']:
                if r['predicted_ai'] != (r['score'] >= v['threshold']):
                    raise ValueError('saved decision mismatch')
        cells = {}
        for condition in ('clean', 'q75'):
            cells[condition] = {}
            for arm, value in values.items():
                rows = [r for r in value['observations'] if r['condition'] == condition]
                cells[condition][arm] = envelope([r['label'] for r in rows], [r['score'] for r in rows])
        folds.append({'fold': fold, 'oracle_envelopes': cells,
                      'source_error_transitions': transitions(values[ARMS[0]]['observations'], values[ARMS[1]]['observations'])})
    result = {'state': 'E58_existing_score_diagnosis_complete', 'contract_sha256': digest(CONTRACT),
              'folds': folds, 'new_inference': 0, 'fits': 0, 'candidate_saved': False,
              'serving_changed': False, 'independent_reserves_opened': False,
              'limitations': ['Optimistic label-oracle envelopes on consumed research folds, not measured deployable thresholds.',
                              'Failure rules out a scalar monotone threshold on these frozen scores, not every future model.',
                              'No saved oracle cuts, interpolation across tied scores, new final scores or CAL adjustment.',
                              'Per-source error changes are paired descriptions, not causal evidence or independent final proof.']}
    fixed_write(RESULT, result)
    return {'state': result['state'], 'folds': folds}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=('freeze', 'run'))
    args = parser.parse_args()
    print(json.dumps(globals()[args.phase](), indent=2))
