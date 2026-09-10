"""One offline constrained correction. TRAIN feasibility precedes any benchmark comparison."""
from __future__ import annotations
import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
import time

if __name__ == '__main__':
    def denied(*args, **kwargs):
        raise RuntimeError('E64 network connections disabled')
    socket.socket.connect = denied
    socket.socket.connect_ex = denied
    socket.create_connection = denied
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_HUB_DISABLE_TELEMETRY='1')

import joblib
import numpy as np
from sklearn.decomposition import PCA
from threadpoolctl import threadpool_limits
from experiments import e64_constrained as model
from experiments.e61_replay_gate import digest, read, write_once, REFERENCE_SHA
from pixelproof.retention_gate import check_train_retention
from pixelproof.training_weights import balanced_parent_weights
from pixelproof.project_paths import DATA_ROOT, ML_ROOT

ROOT = DATA_ROOT / 'e64'
EVIDENCE = ML_ROOT.parent / 'evidence'
CONTRACT = ROOT / 'contract.json'
CANDIDATE = ROOT / 'correction.npz'
RESULT = ROOT / 'fit.json'


def resource_check(deadline):
    if time.monotonic() > deadline:
        raise TimeoutError('E64 runtime budget exceeded')
    if not Path('/Volumes/LaCie').is_mount() or shutil.disk_usage(ROOT).free < 10 * 1024**3:
        raise RuntimeError('external volume unavailable or less than 10 GiB free')
    power = subprocess.check_output(['pmset', '-g', 'batt'], text=True)
    percent = re.search(r'(\d+)%;', power)
    if 'Battery Power' in power and (not percent or int(percent.group(1)) < 30):
        raise RuntimeError('battery below 30 percent')


def freeze():
    previous = read(EVIDENCE / 'e61_replay_contract.json')
    inputs = {k: previous['inputs'][k] for k in ('manifest', 'features', 'reference', 'gate_code')}
    for key, path in {'model_code': Path(model.__file__), 'runner_code': Path(__file__),
                      'weights_code': ML_ROOT / 'src/pixelproof/training_weights.py',
                      'helper_code': Path(__file__).with_name('e61_replay_gate.py')}.items():
        inputs[key] = {'path': str(path), 'sha256': digest(path)}
    for item in inputs.values():
        if digest(item['path']) != item['sha256']:
            raise ValueError('admitted input changed')
    config = {'state': 'E64_constrained_TRAIN_fit_registered', 'inputs': inputs,
              'reference_sha256': REFERENCE_SHA, 'parents': 11630, 'ai_parents': 4595,
              'conditions': ['clean', 'assigned_transport', 'q75'],
              'representation': 'Frozen E43 scaler; TRAIN-only PCA64 randomized seed62 iterated_power3. '
                                'Project explicitly after fit; divide by sqrt(explained_variance), append intercept.',
              'correction': 'Unbounded linear additive logit in the frozen TRAIN-fitted basis. Zero initial weights.',
              'objective': 'Class/source/parent-balanced operating-cut BCE; reference false-positive REAL '
                           'views weighted2x then REAL/AI each renormalized to mass.5; L2.01 including intercept.',
              'constraints': 'All reference-caught TRAIN AI stay at/above the fixed AI logit cut with '
                             'min(existing_margin,1e-7) retained. Correct TRAIN REAL correction <=max(logit(cut)-reference_logit-1e-7,0).',
              'optimizer': 'CPU float64 SLSQP, maxiter200 ftol1e-9, final solution only; no sweep.',
              'ai_cut': model.AI_CUT, 'real_cut': model.REAL_CUT,
              'cpu_threads': 2, 'max_seconds': 1200,
              'fit_guard': 'Solver success, violation<=1e-8, exact saved replay; E61 zero new AI misses; '
                           'zero new REAL false alarms; REAL FPR <=10 percent in all3 TRAIN conditions.',
              'comparison': 'If TRAIN guard passes, permit one separately preregistered consumed E49 '
                            'benchmark comparison of this immutable candidate. No test-based parameter selection. '
                            'E49 is already consumed, cannot supply independent final/generalization evidence.',
              'promotion_allowed': False, 'fresh_independent_dev_available': False,
              'source_downloads': 0, 'new_role_admission': False}
    write_once(CONTRACT, config)
    write_once(EVIDENCE / 'e64_contract.json', config | {'contract_sha256': digest(CONTRACT)})
    return {'state': config['state'], 'contract_sha256': digest(CONTRACT)}


def validate():
    config = read(CONTRACT)
    if digest(CONTRACT) != read(EVIDENCE / 'e64_contract.json')['contract_sha256']:
        raise ValueError('contract changed')
    for item in config['inputs'].values():
        if digest(item['path']) != item['sha256']:
            raise ValueError(f"bound input changed: {item['path']}")
    return config


def fit():
    config = validate()
    if RESULT.exists() or CANDIDATE.exists():
        raise FileExistsError('E64 already executed; no refit or replacement')
    started = time.monotonic(); deadline = started + config['max_seconds']
    resource_check(deadline)
    paths = {k: Path(v['path']) for k, v in config['inputs'].items()}
    rows = read(paths['manifest'])['rows']
    if len(rows) != config['parents'] or any(str(r['role']).upper() != 'TRAIN' for r in rows):
        raise ValueError('invalid TRAIN population')
    labels = np.repeat([r['label'] for r in rows], 3)
    sources = np.repeat([r['source'] for r in rows], 3)
    parents = np.repeat([r['parent_id'] for r in rows], 3)
    if int((labels == 1).sum()) != config['ai_parents'] * 3:
        raise ValueError('incomplete AI replay')
    with np.load(paths['features'], allow_pickle=False) as a:
        if str(a['binding']) != digest(paths['manifest']) or a['features'].shape != (11630, 3, 3072):
            raise ValueError('feature/manifest binding changed')
        features = a['features'].reshape(-1, 3072)
    head = joblib.load(paths['reference'])['head']
    if not np.array_equal(head[-1].classes_, [0, 1]):
        raise ValueError('AI class orientation changed')
    with threadpool_limits(limits=2):
        baseline = head.predict_proba(features)[:, 1]
        logits = head.decision_function(features)
        scaled = head[0].transform(features).astype(np.float64)
        print(json.dumps({'stage': 'TRAIN_PCA64', 'views': len(labels)}), flush=True)
        pca = PCA(n_components=model.RANK, svd_solver='randomized', random_state=62, iterated_power=3)
        pca.fit(scaled)
        del scaled
        arrays = {'mean': pca.mean_, 'components': pca.components_,
                  'scales': np.sqrt(pca.explained_variance_), 'weights': np.zeros(model.RANK+1)}
        if not np.isfinite(arrays['scales']).all() or np.any(arrays['scales'] <= 0):
            raise ValueError('invalid PCA scales')
        x = model.project(head, features, arrays['mean'], arrays['components'], arrays['scales'])
        if not np.array_equal(baseline, model.predict(head, features, arrays)):
            raise ValueError('zero initialization not exact')
        sample_weights = balanced_parent_weights(labels, sources, parents)
        sample_weights[(labels == 0) & (baseline >= model.AI_CUT)] *= 2
        for y in (0, 1):
            sample_weights[labels == y] *= .5 / sample_weights[labels == y].sum()
        count = [0]
        def check():
            resource_check(deadline); count[0] += 1
            if count[0] % 10 == 0:
                print(json.dumps({'stage': 'constrained_fit', 'iteration': count[0],
                                  'seconds': round(time.monotonic()-started)}), flush=True)
        arrays['weights'], solver = model.fit(x, logits, labels, sample_weights, check)
        candidate = model.predict(head, features, arrays)
        with CANDIDATE.open('xb') as f:
            np.savez_compressed(f, **arrays, contract_sha256=np.asarray(digest(CONTRACT)),
                                reference_sha256=np.asarray(REFERENCE_SHA))
        with np.load(CANDIDATE, allow_pickle=False) as a:
            replay = model.predict(head, features, a)
        if not np.array_equal(replay, candidate):
            raise ValueError('saved candidate replay changed')
    gate = check_train_retention(rows, baseline.reshape(-1, 3), candidate.reshape(-1, 3),
                                conditions=config['conditions'], ai_cut=model.AI_CUT)
    real = [gate['comparisons'][c]['real'] for c in config['conditions']]
    accepted = (solver['success'] and solver['max_constraint_violation'] <= 1e-8
                and gate['passes_train_retention'] and all(r['non_ai_to_ai'] == 0 for r in real)
                and all(r['new_ai_rate'] <= .10 for r in real))
    resource_check(deadline)
    if digest(paths['reference']) != REFERENCE_SHA:
        raise ValueError('reference changed')
    result = {'state': 'E64_TRAIN_feasible' if accepted else 'E64_TRAIN_guard_failed',
              'consumed_regression_permitted': bool(accepted), 'solver': solver, 'train_gate': gate,
              'contract_sha256': digest(CONTRACT), 'candidate_sha256': digest(CANDIDATE),
              'reference_sha256': REFERENCE_SHA, 'seconds': time.monotonic()-started,
              'pca_explained_variance_ratio_sum': float(pca.explained_variance_ratio_.sum()),
              'source_downloads': 0, 'evaluation_rows_read': 0, 'promotion_allowed': False}
    write_once(RESULT, result); write_once(EVIDENCE / 'e64_fit.json', result)
    return {'state': result['state'], 'seconds': result['seconds'],
            'ai_new_miss_views': gate['new_ai_miss_views'], 'solver_success': solver['success'],
            'real_rates': {c: gate['comparisons'][c]['real'] for c in config['conditions']}}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('freeze', 'fit'))
    args = parser.parse_args()
    ROOT.mkdir(parents=True, exist_ok=True)
    with (ROOT / 'execution.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        print(json.dumps({'freeze': freeze, 'fit': fit}[args.stage](), indent=2))
