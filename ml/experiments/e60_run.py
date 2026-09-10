"""One offline champion-preserving engineering fit, followed by consumed E49 diagnosis."""
from __future__ import annotations

import argparse
import fcntl
import json
from importlib.metadata import version
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
import time


def disable_network():
    def denied(*args, **kwargs):
        raise RuntimeError('E60 network connections are disabled')
    socket.socket.connect = denied
    socket.socket.connect_ex = denied
    socket.create_connection = denied
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'


if __name__ == '__main__':
    disable_network()  # Before third-party imports or model loading.

import joblib
import numpy as np
from scipy.special import expit
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits

from experiments import e60_correction as correction
from experiments.e60_audit import (ROOT, EVIDENCE, REFERENCE, REFERENCE_SHA, RESULT as AUDIT,
    DATA_CONTRACT, TEACHER, INDEX, read, write, digest, verified, verify_contract)
from pixelproof.project_paths import DATA_ROOT
from pixelproof.training_weights import balanced_parent_weights

CONTRACT = ROOT / 'correction_contract.json'
CANDIDATE = ROOT / 'correction.npz'
FIT_RESULT = ROOT / 'fit.json'
REGRESSION_CONTRACT = ROOT / 'regression_contract.json'
REGRESSION_SCORES = ROOT / 'e49_scores.json'
REGRESSION_RESULT = ROOT / 'e49_report.json'


def resource_check(deadline):
    if time.monotonic() > deadline:
        raise TimeoutError('E60 bounded execution deadline reached')
    if not Path('/Volumes/LaCie').is_mount() or not ROOT.is_dir():
        raise RuntimeError('verified external data volume unavailable')
    if shutil.disk_usage(ROOT).free < 10 * 1024**3:
        raise RuntimeError('less than 10 GiB available on external volume')
    power = subprocess.check_output(['pmset', '-g', 'batt'], text=True)
    percent = re.search(r'(\d+)%;', power)
    if 'Battery Power' in power and (not percent or int(percent.group(1)) < 30):
        raise RuntimeError('battery below bounded-work floor')


def save_arrays(path, **arrays):
    if path.exists():
        raise FileExistsError(path)
    temporary = path.with_suffix('.npz.part')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def freeze():
    verify_contract()
    audit = read(AUDIT)
    verified(AUDIT, read(EVIDENCE / 'e60_audit.json')['report_sha256'])
    if (audit['admitted_ai_replay_parents'], audit['admitted_real_parents']) != (4595, 7035):
        raise ValueError('admitted replay budget changed')
    inputs = [AUDIT, DATA_CONTRACT, TEACHER, INDEX, REFERENCE,
              Path(correction.__file__), Path(__file__).with_name('e60_audit.py'),
              EVIDENCE.parent / 'ml/src/pixelproof/training_weights.py',
              Path(__file__).with_name('e42_features.py'),
              Path(__file__).with_name('e49_evaluation.py'),
              EVIDENCE.parent / 'ml/src/pixelproof/benchmark_metrics.py',
              EVIDENCE.parent / 'ml/src/pixelproof/e32_candidate.py']
    value = {'state': 'E60_single_correction_frozen_before_fit', 'code_sha256': digest(__file__),
        'inputs': {str(p): digest(p) for p in inputs}, 'reference_sha256': REFERENCE_SHA,
        'rows': 11630, 'views': 34890, 'features': 3072,
        'representation': 'Original E43 scaler transform, then float64 / sqrt(3072); scaler/head/backbone frozen.',
        'correction': '2*tanh(X*w/2), no intercept, zero initialization, added to original E43 logit.',
        'supervised_loss': 'Class/source/parent-balanced BCE at fixed AI operating cut; '
                           'teacher false-positive REAL views get 2x weight, normalized within REAL to class mass .5.',
        'retention_loss': '10/2 times balanced mean squared negative correction on teacher-correct TRAIN AI.',
        'l2': correction.L2, 'steps': correction.STEPS, 'learning_rate': correction.LR,
        'optimizer': 'CPU float64 full-batch Adam beta1=.9 beta2=.999 eps=1e-8; final step only.',
        'bound': correction.BOUND, 'retention_coefficient': correction.RETENTION,
        'ai_cut': correction.AI_CUT, 'real_cut': correction.REAL_CUT,
        'cpu_threads': 2, 'max_training_seconds': 1200, 'network_connections': 'denied',
        'runtime': {name: version(name) for name in ('numpy', 'scipy', 'scikit-learn', 'torch', 'timm', 'pillow')},
        'reference_parity': 'Zero initialization: exact score array and zero binary/selective decision changes.',
        'comparison': 'One fixed consumed E49 regression only after final candidate hash freeze. '
                      'No threshold, hyperparameter or winner selection from that report.',
        'quality_guards': 'REAL FPR lower on both conditions with familywise paired intervals; '
                          'AI pooled and supported-source recall no loss with paired intervals; '
                          'existing 20 E49 absolute/selective checks unchanged.',
        'serving_promotion_allowed': False, 'fresh_independent_dev_available': False,
        'source_downloads': 0, 'seed': None, 'validation_early_stopping': False}
    write(CONTRACT, value)
    write(EVIDENCE / 'e60_correction_contract.json', value | {'contract_sha256': digest(CONTRACT)})
    return {'state': value['state'], 'steps': value['steps']}


def validate():
    config = read(CONTRACT)
    verified(CONTRACT, read(EVIDENCE / 'e60_correction_contract.json')['contract_sha256'])
    verified(__file__, config['code_sha256'])
    for path, sha in config['inputs'].items():
        verified(path, sha)
    verified(REFERENCE, REFERENCE_SHA)
    return config


def pair_rates(labels, sources, old, new):
    labels = np.asarray(labels); sources = np.asarray(sources)
    old_ai = old >= correction.AI_CUT; new_ai = new >= correction.AI_CUT
    groups = {}
    for y in (0, 1):
        for source in sorted(set(sources[labels == y])):
            mask = (labels == y) & (sources == source)
            groups[f'{y}:{source}'] = {'parents_or_views': int(mask.sum()),
                'old_ai_rate': float(old_ai[mask].mean()), 'new_ai_rate': float(new_ai[mask].mean()),
                'new_ai_minus_old': float(new_ai[mask].mean() - old_ai[mask].mean())}
    def rates(scores, predicted):
        real, ai = labels == 0, labels == 1
        automatic = (scores < correction.REAL_CUT) | predicted
        return {'real_false_ai': float(predicted[real].mean()), 'ai_recall': float(predicted[ai].mean()),
                'balanced_accuracy': float((1-predicted[real].mean()+predicted[ai].mean())/2),
                'auc': float(roc_auc_score(labels, scores)), 'uncertain_rate': float((~automatic).mean()),
                'covered_accuracy': float((predicted[automatic] == labels[automatic]).mean()) if automatic.any() else 0.}
    return {'old': rates(old, old_ai), 'new': rates(new, new_ai), 'by_source': groups,
        'real_rescued': int(((labels == 0) & old_ai & ~new_ai).sum()),
        'real_new_errors': int(((labels == 0) & ~old_ai & new_ai).sum()),
        'ai_rescued': int(((labels == 1) & ~old_ai & new_ai).sum()),
        'ai_new_errors': int(((labels == 1) & old_ai & ~new_ai).sum())}


def fit():
    config = validate()
    if CANDIDATE.exists() or FIT_RESULT.exists():
        raise FileExistsError('E60 candidate/fit exists; do not refit or replace')
    started = time.monotonic(); deadline = started + config['max_training_seconds']
    resource_check(deadline)
    rows = read(DATA_CONTRACT)['rows']
    if len(rows) != config['rows'] or any(str(r['role']).upper() != 'TRAIN' for r in rows):
        raise ValueError('non-TRAIN or missing parent')
    with np.load(TEACHER, allow_pickle=False) as a:
        if str(a['binding']) != digest(DATA_CONTRACT) or a['features'].shape != (11630, 3, 3072):
            raise ValueError('teacher feature alignment changed')
        features = a['features'].reshape(-1, 3072)
    head = joblib.load(REFERENCE)['head']
    if not np.array_equal(head[-1].classes_, [0, 1]):
        raise ValueError('AI positive class changed')
    labels = np.repeat([r['label'] for r in rows], 3)
    sources = np.repeat([r['source'] for r in rows], 3)
    parents = np.repeat([r['parent_id'] for r in rows], 3)
    with threadpool_limits(limits=2):
        baseline = head.predict_proba(features)[:, 1]
        zero = correction.predict(head, features, np.zeros(3072))
        if not np.array_equal(baseline, zero):
            raise ValueError('zero correction does not exactly reproduce E43')
        logits = head.decision_function(features)
        x = correction.correction_features(head, features)
        weights = balanced_parent_weights(labels, sources, parents)
        weights /= weights.sum()
        supervised = weights.copy()
        supervised[(labels == 0) & (baseline >= correction.AI_CUT)] *= 2
        for y in (0, 1):
            supervised[labels == y] *= .5 / supervised[labels == y].sum()
        retention = weights * ((labels == 1) & (baseline >= correction.AI_CUT))
        retention /= retention.sum()
        last_check = [0.]
        def bounded():
            if time.monotonic() - last_check[0] > 10:
                resource_check(deadline); last_check[0] = time.monotonic()
        fitted, trace = correction.fit_fixed(x, logits, labels, supervised, retention, bounded)
        scores = correction.predict(head, features, fitted)
        resource_check(deadline)
        verified(REFERENCE, REFERENCE_SHA)
        save_arrays(CANDIDATE, weights=fitted, contract_sha256=np.asarray(digest(CONTRACT)),
                    reference_sha256=np.asarray(REFERENCE_SHA))
        with np.load(CANDIDATE, allow_pickle=False) as a:
            replay = correction.predict(head, features, a['weights'])
        if not np.array_equal(scores, replay):
            raise ValueError('saved correction replay differs')
    comparisons = {c: pair_rates(labels[m], sources[m], baseline[m], scores[m])
                  for j, c in enumerate(('clean', 'assigned_transport', 'q75'))
                  for m in [np.arange(len(labels)) % 3 == j]}
    result = {'state': 'E60_bounded_TRAIN_engineering_fit_complete', 'contract_sha256': digest(CONTRACT),
        'candidate_sha256': digest(CANDIDATE), 'reference_sha256_after_fit': digest(REFERENCE),
        'parents': len(rows), 'views': len(labels), 'all_admitted_ai_parents_used': 4595,
        'zero_initialization_max_score_error': float(np.max(np.abs(baseline-zero))),
        'saved_replay_max_score_error': float(np.max(np.abs(scores-replay))),
        'max_absolute_correction': float(np.max(np.abs(correction.delta(x, fitted)))),
        'training_trace': trace, 'training_comparisons': comparisons, 'seconds': time.monotonic()-started,
        'independent_quality_claim': False, 'serving_changed': False, 'source_downloads': 0}
    write(FIT_RESULT, result); write(EVIDENCE / 'e60_fit.json', result)
    return {k: v for k, v in result.items() if k not in {'training_trace', 'training_comparisons'}}


def regression_freeze():
    validate()
    result = read(FIT_RESULT)
    verified(CANDIDATE, result['candidate_sha256'])
    from experiments.e49_evaluation import validate_paired_final
    folder = DATA_ROOT / 'e49/final'
    manifest = folder / 'manifest_unscored.json'; old_scores = folder / 'generalist_scores.jsonl'
    verified(manifest, '9744a9d2385ef2f105b7a132bfee76a7099d280ac9d075d81220f572429c5909')
    verified(old_scores, '249f005c83acbc65ac987c9fdeea91a9a3c7dfc4a388fdd4e20a436bc53610a8')
    validate_paired_final(read(manifest)['rows'])
    value = {'state': 'E60_consumed_E49_regression_frozen_before_scoring',
        'candidate_sha256': digest(CANDIDATE), 'fit_result_sha256': digest(FIT_RESULT),
        'correction_contract_sha256': digest(CONTRACT), 'code_sha256': digest(__file__),
        'manifest': str(manifest), 'manifest_sha256': digest(manifest),
        'old_scores': str(old_scores), 'old_scores_sha256': digest(old_scores),
        'source_downloads': 0, 'observations': 4000, 'parents': 2000, 'batch_views': 16,
        'max_seconds': 900, 'reference_max_score_error': 5e-5,
        'reference_binary_and_selective_decision_changes': 0,
        'report_intervals': '20,000 paired source-cluster resamples stratified by label; '
                            'Bonferroni 4 primary deltas, quantiles .00625/.99375, seed60. '
                            'Conditional on these 16 sources; not new-publisher certification.',
        'purpose': 'Single consumed regression diagnosis; no selection, refit, threshold change, final or serving promotion.'}
    write(REGRESSION_CONTRACT, value)
    write(EVIDENCE / 'e60_regression_contract.json', value | {'contract_sha256': digest(REGRESSION_CONTRACT)})
    return {'state': value['state']}


def regression_inputs():
    validate()
    c = read(REGRESSION_CONTRACT)
    verified(REGRESSION_CONTRACT, read(EVIDENCE / 'e60_regression_contract.json')['contract_sha256'])
    verified(CANDIDATE, c['candidate_sha256']); verified(FIT_RESULT, c['fit_result_sha256'])
    verified(c['manifest'], c['manifest_sha256']); verified(c['old_scores'], c['old_scores_sha256'])
    verified(__file__, c['code_sha256'])
    rows = read(c['manifest'])['rows']
    baseline = {r['record_id']: r for r in [json.loads(line) for line in Path(c['old_scores']).read_text().splitlines()]}
    if len(baseline) != len(rows):
        raise ValueError('baseline coverage changed')
    for r in rows:
        if any(r[k] != baseline[r['record_id']][k] for k in ('parent_id', 'label', 'source', 'condition')):
            raise ValueError('baseline label/source/parent alignment changed')
    return c, rows, baseline


def regression_score():
    c, rows, baseline = regression_inputs()
    if REGRESSION_SCORES.exists():
        raise FileExistsError('complete regression scores already exist')
    from concurrent.futures import ThreadPoolExecutor
    from PIL import Image
    import torch
    from experiments import e42_features as dino
    torch.set_num_threads(2)
    started = time.monotonic(); deadline = started+c['max_seconds']; resource_check(deadline)
    head = joblib.load(REFERENCE)['head']
    with np.load(CANDIDATE, allow_pickle=False) as a:
        if str(a['contract_sha256']) != digest(CONTRACT) or str(a['reference_sha256']) != REFERENCE_SHA:
            raise ValueError('candidate binding changed')
        weights = a['weights']
    model, means, stds, _ = dino._load_small()  # Cached pinned local weights; network denied.
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    model = model.to(device).eval()
    mean = torch.tensor(means, device=device).view(1, 3, 1, 1)
    std = torch.tensor(stds, device=device).view(1, 3, 1, 1)
    partial = ROOT / 'e49_scores.partial.jsonl'
    done = [json.loads(line) for line in partial.read_text().splitlines()] if partial.exists() else []
    if len(done) > len(rows):
        raise ValueError('invalid score prefix')
    for result, row in zip(done, rows, strict=False):
        if any(result[k] != row[k] for k in ('record_id', 'parent_id', 'label', 'source', 'condition')):
            raise ValueError('resumed score identities differ')
        if result['binding'] != digest(REGRESSION_CONTRACT) or not all(
                np.isfinite(result[k]) and 0 <= result[k] <= 1 for k in ('old_reproduced', 'score')):
            raise ValueError('resumed score contract differs')
        verified(row['path'], row['sha256'])
    binding = digest(REGRESSION_CONTRACT)
    def prepare(row):
        verified(row['path'], row['sha256'])
        with Image.open(row['path']) as im:
            return dino.texture_crops(dino.transport_image(im, 'clean'))
    with torch.inference_mode(), ThreadPoolExecutor(max_workers=4) as pool, threadpool_limits(limits=2):
        for start in range(len(done), len(rows), c['batch_views']):
            resource_check(deadline)
            batch = rows[start:start+c['batch_views']]
            arrays = np.stack([crop for prepared in pool.map(prepare, batch) for crop in prepared])
            tensor = torch.from_numpy(arrays).to(device).permute(0, 3, 1, 2).float().div_(255)
            blocks = model.forward_intermediates((tensor-mean)/std, indices=list(dino.BLOCKS['small']),
                return_prefix_tokens=True, norm=True, intermediates_only=True)
            tokens = torch.stack([b[1][:, 0, :] for b in blocks], dim=1).float().cpu().numpy()
            features = dino.aggregate_tokens(tokens, len(batch))
            old = head.predict_proba(features)[:, 1]
            new = correction.predict(head, features, weights)
            results = [dict({k: r[k] for k in ('record_id', 'parent_id', 'label', 'source', 'condition')},
                           old_reproduced=float(old[i]), score=float(new[i]), status='ok', binding=binding)
                       for i, r in enumerate(batch)]
            with partial.open('a') as stream:
                for r in results:
                    stream.write(json.dumps(r, sort_keys=True)+'\n')
                stream.flush(); os.fsync(stream.fileno())
            done.extend(results)
            if len(done) % 160 == 0:
                print(json.dumps({'phase': 'E60_frozen_E49_scores', 'rows': len(done), 'total': len(rows),
                                  'seconds': round(time.monotonic()-started)}), flush=True)
    differences = [abs(r['old_reproduced']-baseline[r['record_id']]['score']) for r in done]
    changes = {str(cut): sum((r['old_reproduced'] >= cut) != (baseline[r['record_id']]['score'] >= cut)
               for r in done) for cut in (correction.AI_CUT, correction.REAL_CUT)}
    if max(differences) > c['reference_max_score_error'] or any(changes.values()):
        raise ValueError('E43 replay drift prevents regression interpretation')
    write(REGRESSION_SCORES, {'state': 'E60_scores_locked_before_metrics', 'rows': done,
        'contract_sha256': binding, 'reference_max_score_error': max(differences),
        'reference_decision_changes': changes, 'seconds': time.monotonic()-started})
    write(EVIDENCE / 'e60_regression_scores.json', {'rows': len(done), 'sha256': digest(REGRESSION_SCORES),
        'reference_max_score_error': max(differences), 'reference_decision_changes': changes,
        'metrics_opened': False, 'source_downloads': 0, 'contract_sha256': binding})
    return {'rows': len(done), 'reference_max_score_error': max(differences)}


def regression_report():
    c, manifest, baseline = regression_inputs()
    verified(REGRESSION_SCORES, read(EVIDENCE / 'e60_regression_scores.json')['sha256'])
    scored = read(REGRESSION_SCORES)['rows']
    from experiments.e49_evaluation import evaluate_final, validate_paired_final
    validate_paired_final(scored)
    full = evaluate_final(scored)
    parents = sorted({r['parent_id'] for r in manifest})
    by_key = {(r['parent_id'], r['condition']): r for r in scored}
    conditions = ('publisher_original', 'social_q75')
    labels = np.array([by_key[p, conditions[0]]['label'] for p in parents])
    sources = np.array([by_key[p, conditions[0]]['source'] for p in parents])
    new = np.array([[by_key[p, cond]['score'] for cond in conditions] for p in parents])
    old = np.array([[baseline[by_key[p, cond]['record_id']]['score'] for cond in conditions] for p in parents])
    comparisons = {cond: pair_rates(labels, sources, old[:, j], new[:, j]) for j, cond in enumerate(conditions)}
    delta = (new >= correction.AI_CUT).astype(float) - (old >= correction.AI_CUT)
    rng = np.random.default_rng(60); samples = []
    for y in (0, 1):
        groups = sorted(set(sources[labels == y]))
        counts = np.array([np.sum((labels == y) & (sources == s)) for s in groups])
        changes = np.array([delta[(labels == y) & (sources == s)].sum(axis=0) for s in groups])
        draws = rng.integers(len(groups), size=(20000, len(groups)))
        samples.append(changes[draws].sum(axis=1) / counts[draws].sum(axis=1)[:, None])
    intervals = {name: np.quantile(np.concatenate(samples, axis=1)[:, j], [.00625, .99375]).tolist()
                 for j, name in enumerate(('real_clean', 'real_q75', 'ai_clean', 'ai_q75'))}
    checks = {
        'real_fpr_lower_both': all(v['new']['real_false_ai'] < v['old']['real_false_ai'] for v in comparisons.values()),
        'ai_recall_no_point_loss_both': all(v['new']['ai_recall'] >= v['old']['ai_recall'] for v in comparisons.values()),
        'supported_ai_sources_no_point_loss': all(g['new_ai_minus_old'] >= 0 for v in comparisons.values()
            for key, g in v['by_source'].items() if key.startswith('1:') and g['parents_or_views'] >= 50),
        'real_improvement_intervals_both': all(intervals[f'real_{cond}'][1] < 0 for cond in ('clean', 'q75')),
        'ai_preservation_intervals_both': all(intervals[f'ai_{cond}'][0] >= 0 for cond in ('clean', 'q75')),
        'absolute_selective_20_checks': full['gate']['passed']}
    result = {'state': 'E60_consumed_E49_regression_complete_not_fresh_final', 'checks': checks,
        'preservation_and_absolute_checks_passed': all(checks.values()), 'comparisons': comparisons,
        'source_cluster_paired_intervals': intervals, 'candidate_e49_diagnostics': full,
        'candidate_sha256': c['candidate_sha256'], 'contract_sha256': digest(REGRESSION_CONTRACT),
        'raw_scores_sha256': digest(REGRESSION_SCORES), 'reference_sha256': digest(REFERENCE),
        'serving_changed': False, 'independent_final_passed': False, 'winner_selection_allowed': False,
        'limitations': ['Consumed E49 diagnostic, never a new validation/final or tuning set.',
                       'Intervals condition on observed camera/generator sources; unknown prompt dependence remains.',
                       'Candidate remains research-only even if these diagnostic checks pass.']}
    write(REGRESSION_RESULT, result); write(EVIDENCE / 'e60_regression.json', result)
    return {'checks': checks, 'comparisons': {k: {a: b for a, b in v.items() if a != 'by_source'}
                                             for k, v in comparisons.items()}, 'intervals': intervals}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=['freeze', 'fit', 'regression-freeze', 'score', 'report'])
    args = parser.parse_args()
    ROOT.mkdir(parents=True, exist_ok=True)
    with (ROOT / 'execution.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        function = {'freeze': freeze, 'fit': fit, 'regression-freeze': regression_freeze,
                    'score': regression_score, 'report': regression_report}[args.phase]
        print(json.dumps(function(), indent=2), flush=True)
