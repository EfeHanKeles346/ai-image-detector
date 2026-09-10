"""Fixed DINO/CLIP representation comparison; no acquisition or serving mutation."""
import argparse
import json
import os
from pathlib import Path
import time
import warnings

import joblib
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from experiments.e53_offline import EVIDENCE, digest, fixed_write, contract as old_contract
from experiments.e53_coverage import OUT as BASELINE
from experiments.e53_report import paired_interval, preservation_guard
from experiments.e51_pipeline import metrics, real_safe_threshold
from experiments.e54_data import CONTRACT as DATA_CONTRACT, TEACHER
from experiments.e54_replay import compare_scores
from experiments.e59_features import ROOT, CONTRACT as FEATURE_CONTRACT, FEATURES, RECEIPT, load as feature_load
from experiments.e57_data import safe
from pixelproof.training_weights import balanced_parent_weights

CONTRACT = ROOT / 'model_contract.json'
RESULT = ROOT / 'result.json'
ARMS = ('dino3072', 'clip1536', 'dino_clip4608')
REFERENCES = ('full_old2', 'full_e51_3', 'full_expanded3', 'e57_native64')
DIMENSIONS = dict(zip(ARMS, (3072, 1536, 4608), strict=True))


def reference_path(arm, fold):
    if arm == 'e57_native64':
        return ROOT.parent / 'e57/v2/training' / f'native64_fold{fold}.json'
    if arm not in REFERENCES:
        raise ValueError('unknown reference')
    return BASELINE / f'{arm}_fold{fold}.json'


def require_features():
    compact = EVIDENCE / 'e59_features.json'
    if not all(p.exists() for p in (FEATURES, RECEIPT, compact)):
        raise RuntimeError('full CLIP archive and both completion receipts required before training')
    receipt = json.loads(RECEIPT.read_text())
    if digest(RECEIPT) != json.loads(compact.read_text())['receipt_sha256']:
        raise ValueError('feature completion receipt changed')
    if receipt['contract_sha256'] != digest(FEATURE_CONTRACT) or digest(FEATURES) != receipt['features_sha256']:
        raise ValueError('feature archive binding changed')
    if receipt['parents'] != 11630 or receipt['views'] != 34890:
        raise ValueError('incomplete feature population')
    return receipt


def loss_weights(labels, sources, parents, mass):
    if mass <= 0:
        raise ValueError('positive original FIT mass required')
    w = balanced_parent_weights(labels, sources, parents)
    return w * (mass / w.sum())


def representation(arm, dino, clip):
    if dino.shape[0] != clip.shape[0] or dino.ndim != 2 or clip.ndim != 2:
        raise ValueError('aligned feature rows required')
    if dino.shape[1] != 3072 or clip.shape[1] != 1536:
        raise ValueError('feature dimension mismatch')
    if arm == 'dino3072':
        return dino
    if arm == 'clip1536':
        return clip
    if arm == 'dino_clip4608':
        return np.concatenate([dino, clip], axis=1)
    raise ValueError('unknown fixed arm')


def observation_map(value):
    result = {}
    for row in value['observations']:
        key = (row['parent_id'], row['condition'])
        if key in result or row['condition'] not in {'clean', 'q75'}:
            raise ValueError('duplicate/unknown observation')
        if row['label'] not in (0, 1) or not np.isfinite(row['score']):
            raise ValueError('invalid class/score')
        if row['predicted_ai'] != (row['score'] >= value['threshold']):
            raise ValueError('decision/cut mismatch')
        result[key] = row
    return result


def control_parity(current, reference):
    a, b = observation_map(current), observation_map(reference)
    if not a or a.keys() != b.keys():
        raise ValueError('unpaired control population')
    if any(any(a[k][field] != b[k][field] for field in ('source', 'label')) for k in a):
        raise ValueError('control truth changed')
    error = max(abs(a[k]['score'] - b[k]['score']) for k in a)
    flips = sum(a[k]['predicted_ai'] != b[k]['predicted_ai'] for k in a)
    cut_error = abs(current['threshold'] - reference['threshold'])
    return {'max_score_error': float(error), 'threshold_error': float(cut_error),
            'decision_changes': int(flips), 'passed': error <= 5e-5 and cut_error <= 5e-5 and flips == 0}


def freeze():
    safe(time.monotonic() + 3600)
    require_features()
    data, feature_config = feature_load()
    registered = feature_config['training']
    if registered['arms'] != list(ARMS) or len(data['folds']) != 3:
        raise ValueError('registered representation study changed')
    paths = [FEATURE_CONTRACT, FEATURES, RECEIPT, EVIDENCE / 'e59_features.json', DATA_CONTRACT, TEACHER,
             EVIDENCE / 'e57_result.json', EVIDENCE / 'e57_model_contract.json',
             Path(__file__).with_name('e59_features.py'), Path(__file__).with_name('e53_report.py'),
             Path(__file__).with_name('e51_pipeline.py'), Path(__file__).with_name('e54_replay.py'),
             EVIDENCE.parent / 'ml/src/pixelproof/training_weights.py']
    historical_bindings = json.loads((EVIDENCE / 'e57_result.json').read_text())['result_bindings']
    for arm in REFERENCES:
        for fold in range(3):
            path = reference_path(arm, fold)
            if digest(path) != historical_bindings[str(path)]:
                raise ValueError('historically bound reference changed')
            value = json.loads(path.read_text())
            if digest(path.with_suffix('.joblib')) != value['artifact_sha256']:
                raise ValueError('historical artifact changed')
            paths.extend([path, path.with_suffix('.joblib')])
    value = {'state': 'E59_nine_fit_contract_frozen', 'code_sha256': digest(__file__),
             'inputs': {str(p): digest(p) for p in paths}, **registered,
             'dimensions': DIMENSIONS, 'control_parity_tolerance': 5e-5,
             'references': list(REFERENCES), 'policy': 'Existing FIT only, unchanged CAL and OOF; no FiveK or new final.',
             'feature_contract_sha256': digest(FEATURE_CONTRACT), 'serving_changed': False}
    fixed_write(CONTRACT, value)
    fixed_write(EVIDENCE / 'e59_model_contract.json', value | {'contract_sha256': digest(CONTRACT)})
    return {'state': value['state'], 'contract_sha256': digest(CONTRACT)}


def load():
    require_features()
    data, _ = feature_load()
    value = json.loads(CONTRACT.read_text())
    if digest(__file__) != value['code_sha256'] or digest(CONTRACT) != json.loads((EVIDENCE / 'e59_model_contract.json').read_text())['contract_sha256']:
        raise ValueError('frozen model contract changed')
    for p, sha in value['inputs'].items():
        if digest(p) != sha:
            raise ValueError('bound model input changed: ' + p)
    return data, value


def fit():
    data, config = load()
    deadline = time.monotonic() + 3600
    binding = digest(CONTRACT)
    with np.load(TEACHER, allow_pickle=False) as a:
        if str(a['binding']) != digest(DATA_CONTRACT) or a['features'].shape != (11630, 3, 3072):
            raise ValueError('DINO features/order contract mismatch')
        dino = a['features'].reshape(-1, 3072).astype(np.float64)
    rows = data['rows']
    with np.load(FEATURES, allow_pickle=False) as a:
        if str(a['binding']) != digest(FEATURE_CONTRACT) or list(a['parents']) != [r['parent_id'] for r in rows]:
            raise ValueError('CLIP parent order/binding mismatch')
        if a['features'].shape != (11630, 3, 1536):
            raise ValueError('CLIP feature shape mismatch')
        clip = a['features'].reshape(-1, 1536).astype(np.float64)
    if not np.isfinite(dino).all() or not np.isfinite(clip).all():
        raise ValueError('nonfinite training features')
    parents = np.repeat([r['parent_id'] for r in rows], 3)
    labels = np.repeat([r['label'] for r in rows], 3)
    sources = np.repeat([r['source'] for r in rows], 3)
    native = np.repeat([r['native'] for r in rows], 3)
    conditions = np.tile(['clean', 'assigned_transport', 'q75'], len(rows))
    for arm in ARMS:
        features = representation(arm, dino, clip)
        for fold in data['folds']:
            safe(deadline)
            path = ROOT / 'training' / f"{arm}_fold{fold['fold']}.json"
            artifact = path.with_suffix('.joblib')
            if path.exists():
                previous = json.loads(path.read_text())
                if previous['contract_sha256'] != binding or digest(artifact) != previous['artifact_sha256']:
                    raise ValueError('completed fit binding changed')
                continue
            if artifact.exists():
                raise RuntimeError('orphaned saved head requires explicit audit; no overwrite')
            started = time.monotonic()
            roles = np.repeat(fold['roles'], 3)
            fit_mask = roles == 'FIT'
            cal = (roles == 'CAL') & (conditions != 'assigned_transport')
            val = (roles == 'VALIDATION') & (conditions != 'assigned_transport')
            mass = int((fit_mask & ~native).sum())
            y, s, p = labels[fit_mask], sources[fit_mask], parents[fit_mask]
            weights = loss_weights(y, s, p, mass)
            expected_ai = json.loads(reference_path('e57_native64', fold['fold']).read_text())['AI_views_preserved']
            if int((y == 1).sum()) != expected_ai:
                raise ValueError('AI replay population changed')
            head = make_pipeline(StandardScaler(), LogisticRegression(
                C=config['C'], tol=config['tol'], max_iter=config['max_iter'],
                random_state=config['seed'], solver='lbfgs'))
            with warnings.catch_warnings(), threadpool_limits(limits=2):
                warnings.simplefilter('error', ConvergenceWarning)
                head.fit(features[fit_mask], y, standardscaler__sample_weight=weights, logisticregression__sample_weight=weights)
                if list(head.classes_) != [0, 1]:
                    raise ValueError('class orientation changed')
                cal_scores = head.predict_proba(features[cal])[:, 1]
                cut = real_safe_threshold(labels[cal], cal_scores, sources[cal], conditions[cal])
                score = head.predict_proba(features[val])[:, 1]
            safe(deadline)
            artifact.parent.mkdir(parents=True, exist_ok=True)
            partial = artifact.with_suffix('.joblib.part')
            with partial.open('wb') as stream:
                joblib.dump({'head': head, 'threshold': cut, 'binding': binding, 'arm': arm,
                             'fold': fold['fold'], 'restriction': 'Research FIT fold only; never serve'}, stream)
                stream.flush()
                os.fsync(stream.fileno())
            partial.replace(artifact)
            with threadpool_limits(limits=2):
                again = joblib.load(artifact)['head'].predict_proba(features[val])[:, 1]
            error, flips = compare_scores(again, score, cut)
            observations = [dict(parent_id=str(p), source=str(s), label=int(y), condition=str(c),
                                 score=float(v), predicted_ai=bool(v >= cut))
                            for p, s, y, c, v in zip(parents[val], sources[val], labels[val], conditions[val], score, strict=True)]
            value = {'state': 'E59_fold_complete', 'contract_sha256': binding, 'arm': arm, 'fold': fold['fold'],
                     'threshold': cut, 'observations': observations, 'artifact_sha256': digest(artifact),
                     'AI_views_preserved': expected_ai, 'fit_parents': int(fit_mask.sum()) // 3,
                     'original_total_weight_mass': mass, 'actual_weight_mass': float(weights.sum()),
                     'AI_weight_sha256': hashlib_array(weights[y == 1]),
                     'artifact_replay': {'max_score_error': error, 'decision_changes': flips, 'passed': error == 0 and flips == 0},
                     'rates': {c: metrics(labels[val][conditions[val] == c], score[conditions[val] == c], sources[val][conditions[val] == c], cut, cut)
                               for c in ('clean', 'q75')}, 'seconds': time.monotonic() - started, 'serving_changed': False}
            if arm == 'dino3072':
                value['e57_control_parity'] = control_parity(value, json.loads(reference_path('e57_native64', fold['fold']).read_text()))
            fixed_write(path, value)
            print(json.dumps({k: v for k, v in value.items() if k not in {'observations', 'rates'}}), flush=True)
        del features
    return {'state': 'E59_nine_fits_complete'}


def hashlib_array(a):
    import hashlib
    return hashlib.sha256(np.ascontiguousarray(a, dtype=np.float64).tobytes()).hexdigest()


def report():
    data, _ = load()
    binding = digest(CONTRACT)
    known = {r['parent_id']: r for r in data['rows'] if not r['native']}
    parents = sorted(known)
    labels = np.asarray([known[p]['label'] for p in parents])
    sources = np.asarray([known[p]['source'] for p in parents])
    old, _ = old_contract()
    groups = np.asarray([old['components'][p] for p in parents])
    arms, predictions, bindings = {}, {}, {}
    for arm in (*ARMS, *REFERENCES):
        joined, folds = {}, []
        for fold in data['folds']:
            path = (ROOT / 'training' / f"{arm}_fold{fold['fold']}.json") if arm in ARMS else reference_path(arm, fold['fold'])
            value = json.loads(path.read_text())
            bindings[str(path)] = digest(path)
            if arm in ARMS and (value['contract_sha256'] != binding or value['arm'] != arm or value['fold'] != fold['fold']):
                raise ValueError('mixed trained fold/contract')
            if digest(path.with_suffix('.joblib')) != value['artifact_sha256']:
                raise ValueError('saved head changed')
            roles = {r['parent_id']: role for r, role in zip(data['rows'], fold['roles'], strict=True)}
            for key, row in observation_map(value).items():
                if key in joined or roles[row['parent_id']] != 'VALIDATION':
                    raise ValueError('OOF role overlap')
                truth = known[row['parent_id']]
                if any(row[k] != truth[k] for k in ('label', 'source')):
                    raise ValueError('OOF truth mismatch')
                joined[key] = row['predicted_ai']
            folds.append({k: v for k, v in value.items() if k != 'observations'})
        if joined.keys() != {(p, c) for p in parents for c in ('clean', 'q75')}:
            raise ValueError('incomplete OOF population')
        pred = np.asarray([[joined[(p, c)] for c in ('clean', 'q75')] for p in parents])
        predictions[arm] = pred
        arms[arm] = {'folds': folds, 'conditions': {
            c: {'ai_recall': float(pred[labels == 1, j].mean()), 'real_false_ai': float(pred[labels == 0, j].mean()),
                'balanced_accuracy': float((1-pred[labels == 0, j].mean()+pred[labels == 1, j].mean())/2),
                'mean_fold_auc': float(np.mean([f['rates'][c]['auc'] for f in folds]))}
            for j, c in enumerate(('clean', 'q75'))}}
    parity = all(f['e57_control_parity']['passed'] for f in arms['dino3072']['folds'])
    equal_ai_weights = all(len({arms[a]['folds'][i]['AI_weight_sha256'] for a in ARMS}) == 1 for i in range(3))
    for arm in ARMS:
        refs = REFERENCES if arm == 'dino3072' else (*REFERENCES, 'dino3072')
        comparisons = {}
        for ref in refs:
            ci = paired_interval(labels, groups, predictions[arm], predictions[ref])
            comparisons[ref] = {'intervals': ci, **preservation_guard(labels, sources, predictions[arm], predictions[ref], ci)}
        checks = {'all_relative_guards': all(v['passed'] for v in comparisons.values()),
                  'all_absolute_fold_gates': all(f['rates'][c]['passed'] for f in arms[arm]['folds'] for c in ('clean', 'q75')),
                  'exact_artifact_replay': all(f['artifact_replay']['passed'] for f in arms[arm]['folds']),
                  'stable_control_parity': parity, 'identical_AI_loss_weights_across_arms': equal_ai_weights}
        arms[arm].update(comparisons=comparisons, checks=checks, eligible_for_next_research_stage=all(checks.values()))
    value = {'state': 'E59_representation_comparison_complete', 'contract_sha256': binding,
             'arms': arms, 'result_bindings': bindings,
             'eligible': [a for a in ARMS[1:] if arms[a]['eligible_for_next_research_stage']],
             'serving_changed': False, 'independent_final_passed': False,
             'limitations': ['Reused TRAIN source-held-out exploration, not independent final or three seeds.',
                             'Per-pair publisher intervals do not certify a winner selected among representations.',
                             'No raw-score AUC pooled across differently fitted/calibrated folds.',
                             'Any survivor requires separately frozen follow-up; no automatic promotion.']}
    fixed_write(RESULT, value)
    fixed_write(EVIDENCE / 'e59_result.json', value)
    return {k: v for k, v in value.items() if k not in {'arms', 'result_bindings'}}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=('freeze', 'fit', 'report'))
    args = parser.parse_args()
    print(json.dumps(globals()[args.phase](), indent=2))
