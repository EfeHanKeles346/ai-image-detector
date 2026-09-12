"""Single TRAIN-only E70 original/blur interaction fit, with unchanged admission gates."""
from __future__ import annotations
import argparse
import fcntl
import json
import os
from pathlib import Path
import socket
import time

if __name__ == '__main__':
    def denied(*args, **kwargs): raise RuntimeError('E70 fit network disabled')
    socket.socket.connect = denied; socket.socket.connect_ex = denied; socket.create_connection = denied
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_HUB_DISABLE_TELEMETRY='1')

import joblib
import numpy as np
from threadpoolctl import threadpool_limits
from experiments import e70_model as model, e67_fit as prior
from experiments.e65_acquisition import digest, read, write_once
from experiments.e67_features import resource_check
from pixelproof.project_paths import DATA_ROOT, ML_ROOT
from pixelproof.retention_gate import check_train_retention
from pixelproof.training_weights import balanced_parent_weights

ROOT = DATA_ROOT / 'e70'
EVIDENCE = ML_ROOT.parent / 'evidence'
CONTRACT = ROOT / 'fit_contract.json'
CANDIDATE = ROOT / 'correction.npz'
REPORT = ROOT / 'fit.json'


def freeze():
    old = prior.validate(); old_result = read(prior.REPORT)
    if digest(prior.REPORT) != digest(EVIDENCE / 'e67_fit.json') or old_result['state'] != 'E67_TRAIN_guard_failed':
        raise ValueError('locked failed E67 prerequisite missing')
    if digest(prior.CANDIDATE) != old_result['candidate_sha256']:
        raise ValueError('basis artifact changed')
    if (prior.ROOT / 'dev_scores.json').exists(): raise ValueError('E67 unexpectedly scored DEV')
    population = read(old['dev_manifest'])
    if population['model_scores_created'] != 0 or population['class_counts'] != {'0': 160, '1': 160}:
        raise ValueError('complete unscored DEV required')
    files = {'runner_code': Path(__file__), 'model_code': Path(model.__file__),
             'prior_fit_contract': prior.CONTRACT, 'prior_fit_report': prior.REPORT,
             'basis_artifact': prior.CANDIDATE, 'train_diagnostic': prior.ROOT / 'training_diagnostic.json'}
    later = DATA_ROOT / 'e69/fit.json'
    if digest(later) != digest(EVIDENCE / 'e69_fit.json') or read(later)['state'] != 'E69_TRAIN_guard_failed':
        raise ValueError('locked failed E69 prerequisite missing')
    if (DATA_ROOT / 'e69/dev_scores.json').exists(): raise ValueError('E69 unexpectedly scored DEV')
    files['later_failed_report'] = later
    inputs = dict(old['inputs'])
    # Preserve all previous bindings, with a separate name for this version's runner.
    inputs['prior_runner_code'] = inputs.pop('runner_code')
    inputs['prior_model_code'] = inputs.pop('model_code')
    inputs.update({k: {'path': str(p), 'sha256': digest(p)} for k, p in files.items()})
    if digest(files['train_diagnostic']) != digest(EVIDENCE / 'e67_training_diagnostic.json'):
        raise ValueError('TRAIN diagnostic differs')
    c = {k: old[k] for k in ['dev_manifest', 'dev_manifest_sha256', 'parents', 'ai_parents',
        'conditions', 'ai_cut', 'real_cut', 'max_seconds', 'cpu_threads', 'train_guard', 'evaluation']}
    c.update(state='E70_TRAIN_bilinear_original_response_registered', inputs=inputs,
        representation='Exact E67 original64 + blur-response64 coordinates, plus128 bilinear TensorSketch '
                       'coordinates standardized on TRAIN only, plus intercept:257 coefficients. '
                       'Two independent64-to128 CountSketch hash/sign arrays, NumPy default_rng(seed70), '
                       'FFT circular convolution. Discard old learned weights, initialize all257 to zero.',
        objective='Exactly E67 class/source/parent-balanced operating-cut BCE, hard REAL2x, '
                  'class mass0.5 each and L2.01 on all coefficients, including intercept.',
        constraints='Unchanged E64 per-image caught-AI and correct-REAL decision margins; violation<=1e-8.',
        optimizer='CPU float64 SLSQP200,ftol1e-9; final iterate only; no dimension/hash/seed/L2/threshold sweep.',
        dev_labels_used_in_fitting=False, e49_read_allowed=False, promotion_allowed=False,
        independent_final_allowed=False, downloads=0)
    write_once(CONTRACT, c)
    write_once(EVIDENCE / 'e70_fit_contract.json', c | {'contract_sha256': digest(CONTRACT)})
    return {'state': c['state'], 'dev_manifest_sha256': c['dev_manifest_sha256']}


def validate():
    c = read(CONTRACT)
    if digest(CONTRACT) != read(EVIDENCE / 'e70_fit_contract.json')['contract_sha256']:
        raise ValueError('E70 contract changed')
    for v in c['inputs'].values():
        if digest(v['path']) != v['sha256']: raise ValueError('E70 bound input changed')
    if digest(c['dev_manifest']) != c['dev_manifest_sha256']: raise ValueError('DEV identity changed')
    return c


def fit():
    c = validate()
    if CANDIDATE.exists() or REPORT.exists(): raise FileExistsError('E70 already fitted')
    started = time.monotonic(); deadline = started + c['max_seconds']; resource_check(deadline)
    p = {k: Path(v['path']) for k, v in c['inputs'].items()}; rows = read(p['manifest'])['rows']
    if len(rows) != c['parents'] or any(r['role'].upper() != 'TRAIN' for r in rows):
        raise ValueError('TRAIN population differs')
    with np.load(p['features'], allow_pickle=False) as a:
        if str(a['binding']) != digest(p['manifest']) or a['features'].shape != (11630, 3, 3072):
            raise ValueError('original features differ')
        original = a['features'].reshape(-1, 3072)
    with np.load(p['blur'], allow_pickle=False) as a:
        if str(a['binding']) != digest(p['feature_contract']) or a['features'].shape != (11630, 3, 3072) or \
                list(a['parent_ids']) != [r['parent_id'] for r in rows] or set(a['roles']) != {'TRAIN'}:
            raise ValueError('blur feature alignment differs')
        blurred = a['features'].reshape(-1, 3072)
    with np.load(p['basis_artifact'], allow_pickle=False) as a:
        if str(a['contract_sha256']) != digest(p['prior_fit_contract']) or \
                str(a['reference_sha256']) != c['inputs']['reference']['sha256']:
            raise ValueError('prior basis binding differs')
        arrays = {k: a[k] for k in a.files if k not in ['contract_sha256', 'reference_sha256']}
    labels = np.repeat([r['label'] for r in rows], 3); sources = np.repeat([r['source'] for r in rows], 3)
    parents = np.repeat([r['parent_id'] for r in rows], 3); conditions = np.tile(c['conditions'], len(rows))
    if int(np.sum(labels == 1)) != c['ai_parents'] * 3: raise ValueError('AI replay incomplete')
    head = joblib.load(p['reference'])['head']
    with threadpool_limits(limits=c['cpu_threads']):
        baseline = head.predict_proba(original)[:, 1]; logits = head.decision_function(original)
        arrays = model.fit_map(head, original, blurred, arrays)
        if not np.array_equal(baseline, model.predict(head, original, blurred, arrays)):
            raise ValueError('zero-init parity failed')
        x = model.project(head, original, blurred, arrays); count = [0]
        def check():
            resource_check(deadline); count[0] += 1
            if count[0] % 10 == 0:
                print(json.dumps({'stage': 'E70_interaction_fit', 'iteration': count[0],
                                  'seconds': round(time.monotonic() - started)}), flush=True)
        weights = balanced_parent_weights(labels, sources, parents)
        weights[(labels == 0) & (baseline >= model.AI_CUT)] *= 2
        for y in (0, 1): weights[labels == y] *= .5 / weights[labels == y].sum()
        arrays['weights'], solver = model.fit(x, logits, labels, weights, check)
        candidate = model.predict(head, original, blurred, arrays)
        with CANDIDATE.open('xb') as out:
            np.savez_compressed(out, **arrays, contract_sha256=np.array(digest(CONTRACT)),
                                reference_sha256=np.array(c['inputs']['reference']['sha256']))
        with np.load(CANDIDATE, allow_pickle=False) as a:
            if not np.array_equal(candidate, model.predict(head, original, blurred, a)):
                raise ValueError('serialized replay differs')
    gate = check_train_retention(rows, baseline.reshape(-1, 3), candidate.reshape(-1, 3),
                                conditions=c['conditions'], ai_cut=model.AI_CUT)
    passed = solver['success'] and solver['max_constraint_violation'] <= 1e-8 and gate['passes_train_retention'] and \
        all(gate['comparisons'][k]['real']['non_ai_to_ai'] == 0 and
            gate['comparisons'][k]['real']['new_ai_rate'] <= .10 for k in c['conditions'])
    result = {'state': 'E70_TRAIN_guard_passed' if passed else 'E70_TRAIN_guard_failed',
        'dev_scoring_permitted': bool(passed), 'solver': solver, 'train_gate': gate,
        'seconds': time.monotonic() - started, 'candidate_sha256': digest(CANDIDATE),
        'contract_sha256': digest(CONTRACT), 'reference_sha256': c['inputs']['reference']['sha256'],
        'dev_features_or_scores_read': 0, 'e49_rows_read': 0, 'promotion_allowed': False,
        'independent_final_passed': False}
    write_once(REPORT, result); write_once(EVIDENCE / 'e70_fit.json', result)
    return {k: result[k] for k in ['state', 'seconds']} | {'new_ai_miss_views': gate['new_ai_miss_views'],
        'real_rates': {k: gate['comparisons'][k]['real'] for k in c['conditions']}}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('stage', choices=('freeze', 'fit'))
    ROOT.mkdir(parents=True, exist_ok=True)
    with (ROOT / 'fit_execution.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        print(json.dumps({'freeze': freeze, 'fit': fit}[parser.parse_args().stage](), indent=2))
