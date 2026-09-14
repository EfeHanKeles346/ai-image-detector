"""Joint worst-class/source correction from the frozen E102 candidate on identical TRAIN."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import socket
import time

if __name__ == '__main__':
    def denied(*args, **kwargs):
        raise RuntimeError('E103 fit is offline')
    socket.socket.connect = denied; socket.socket.connect_ex = denied; socket.create_connection = denied
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')

import numpy as np
from scipy.special import expit
import torch
from threadpoolctl import threadpool_limits
from experiments import e92_model as model
from experiments.e65_acquisition import digest, read, write_once
from experiments.e72_acquisition import resource_check
from experiments.e103_joint_minimax import fit as fit_delta
from experiments.e91_data import load as load_old, CONDITIONS
from experiments.e102_fit import validate as validate_base
from experiments.e92_gates import full_training_gates as old_numeric, population_gates as old_population
from experiments.e49_evaluation import evaluate_condition
from experiments.e80_fit import runtime_checks
from experiments.e101_features import validate as validate_features, OUTPUT as CACHE, MANIFEST
from pixelproof.project_paths import DATA_ROOT, ML_ROOT
from pixelproof.retention_gate import check_train_retention

ROOT = DATA_ROOT/'e103'; EVIDENCE = ML_ROOT.parent/'evidence'
CONTRACT = ROOT/'fit_contract.json'; CANDIDATE = ROOT/'correction.npz'; REPORT = ROOT/'fit.json'
BASE = DATA_ROOT/'e102/correction.npz'


def new_sensor_gates(rows, scores):
    if scores.shape != (len(rows), 4) or not len(rows) or not np.isfinite(scores).all() or \
            np.any((scores < 0) | (scores > 1)) or any(r['label'] != 0 for r in rows):
        raise ValueError('Complete bounded new REAL four-view cohort required')
    sources = np.array([r['source'] for r in rows]); detected = scores >= model.AI_CUT
    result = {}
    for j, condition in enumerate(CONDITIONS):
        by_source = {s: float(detected[sources == s, j].mean()) for s in sorted(set(sources))}
        pooled = float(detected[:, j].mean())
        result[condition] = {'pooled_fpr': pooled, 'by_sensor_fpr': by_source,
                             'passed': pooled <= .1 and max(by_source.values()) <= .2}
    return result


def freeze():
    validate_base(); validate_features()
    feature_report = read(DATA_ROOT/'e101/features.json')
    old_report = read(DATA_ROOT/'e102/fit.json')
    if digest(DATA_ROOT/'e101/features.json') != read(EVIDENCE/'e101_features.json')['report_sha256'] or \
            feature_report['state'] != 'E101_MIDD_extension_four_condition_TRAIN_features_complete' or \
            feature_report['parents'] != 256 or feature_report['views'] != 1024 or \
            not feature_report['parity']['passed'] or feature_report['new_image_classifier_scores'] or \
            digest(CACHE) != feature_report['feature_sha256'] or \
            digest(DATA_ROOT/'e102/fit.json') != digest(EVIDENCE/'e102_fit.json') or \
            digest(BASE) != old_report['candidate_sha256'] or not old_report['dev_scoring_permitted']:
        raise ValueError('Complete E101 cache and frozen E102 required')
    paths = [Path(__file__), Path(__file__).with_name('e101_features.py'), Path(__file__).with_name('e91_data.py'),
        Path(__file__).with_name('e92_gates.py'), Path(__file__).with_name('e83_model.py'),
        Path(__file__).with_name('e103_joint_minimax.py'), Path(__file__).with_name('e68_minimax.py'), Path(__file__).with_name('e73_fit.py'),
        Path(__file__).with_name('e64_constrained.py'), Path(__file__).with_name('e80_fit.py'),
        ML_ROOT/'src/pixelproof/retention_gate.py', BASE, CACHE, MANIFEST,
        DATA_ROOT/'e101/features_contract.json', DATA_ROOT/'e101/features.json', DATA_ROOT/'e102/fit.json']
    c = {'state': 'E103_joint_worst_group_correction_registered', 'inputs': {str(p): digest(p) for p in paths},
        'old_parents': 12269, 'new_parents': 256, 'parents': 12525, 'views': 50100, 'ai_views': 18380,
        'conditions': CONDITIONS, 'ai_cut': model.AI_CUT, 'real_cut': model.REAL_CUT, 'max_seconds': 3600,
        'objective': 'Frozen E102/E92 map,450 zero-initialized delta weights; .5 worst REAL group BCE '
            '+ .5 worst AI group BCE + .5*.01 squared delta norm. Parent-balanced within '
            'class/source/condition. All E102 AI logits nondecreasing, correct REAL binary '
            'decisions protected; no sweep. E68 risks plus stronger E73 all-AI bounds.',
        'optimizer': 'SLSQP,maxiter200,ftol1e-9,L2=.01; CPU2 threads',
        'acceptance': 'All old E92 population/numeric gates, expanded numeric gates, new pooled<=10% '
            'and worst sensor<=20%, combined REAL<=10%, solver<=1e-8, no lost AI/new REAL errors, '
            'full batch8 runtime<=1e-6 and zero two-cut changes.',
        'dev_final_rows_read': 0, 'promotion_allowed': False, 'downloads': 0,
        'limits': 'TRAIN-only objective experiment on the same E102 cohort; no new features or data. '
            'No learning from consumed E66/gallery or score-based row selection. A passing fit only '
            'permits separate consumed regression; the E43 DEV retention deficit cannot be waived.'}
    write_once(CONTRACT, c)
    write_once(EVIDENCE/'e103_fit_contract.json', c | {'contract_sha256': digest(CONTRACT)})
    return {'state': c['state'], 'views': c['views']}


def validate():
    validate_base(); validate_features(); c = read(CONTRACT)
    if digest(CONTRACT) != read(EVIDENCE/'e103_fit_contract.json')['contract_sha256']:
        raise ValueError('Fit contract changed')
    for path, sha in c['inputs'].items():
        if digest(path) != sha:
            raise ValueError('Bound fit input changed')
    return c


def fit():
    c = validate(); torch.set_num_threads(2)
    if CANDIDATE.exists() or REPORT.exists():
        raise FileExistsError('Fit already recorded')
    start = time.monotonic(); deadline = start+c['max_seconds']; resource_check(deadline)
    old, features, head = load_old(); new = read(MANIFEST)['rows']; rows = old+new
    if len(old) != c['old_parents'] or len(new) != c['new_parents'] or \
            len({r['parent_id'] for r in rows}) != len(rows) or len({r['sha256'] for r in rows}) != len(rows) or \
            any(r['role'].upper() != 'TRAIN' for r in rows):
        raise ValueError('Complete disjoint old+new TRAIN required')
    with np.load(CACHE, allow_pickle=False) as a:
        if str(a['binding']) != digest(DATA_ROOT/'e101/features_contract.json') or \
                list(a['parents']) != [r['parent_id'] for r in new] or list(a['conditions']) != CONDITIONS or \
                list(a['roles']) != ['TRAIN']*len(new):
            raise ValueError('New TRAIN feature pairing changed')
        for key in features:
            features[key] = np.concatenate([features[key], a[key]]).reshape(len(rows)*4, -1)
    with np.load(BASE, allow_pickle=False) as a:
        arrays = {k: np.array(a[k], copy=True) for k in a.files if k not in ['contract_sha256', 'reference_sha256', 'parent_artifact_sha256']}
        reference_sha = str(a['reference_sha256'])
    labels = np.repeat([r['label'] for r in rows], 4)
    if int((labels == 1).sum()) != c['ai_views']:
        raise ValueError('AI cohort changed')
    with threadpool_limits(limits=2):
        x = model.project(head, features['dino'], features['clip'], features['dear'], arrays)
        raw_logits = head.decision_function(features['dino'])
        baseline_logits = raw_logits+x@arrays['weights']; baseline = expit(baseline_logits)
        if not np.array_equal(baseline, model.predict(head, features['dino'], features['clip'], features['dear'], arrays)):
            raise ValueError('Zero-delta E102 replay differs')
        if old_numeric(old, baseline[:len(old)*4].reshape(-1, 4), 11630, 12141) != read(DATA_ROOT/'e102/fit.json')['previous_numeric_gates']:
            raise ValueError('Frozen E102 TRAIN metrics differ')
        baseline_scores = baseline.reshape(-1, 4)
        for j, condition in enumerate(CONDITIONS):
            replay = evaluate_condition([{'parent_id': r['parent_id'], 'source': r['source'],
                'role': 'TRAIN', 'label': r['label'], 'score': float(baseline_scores[i, j])}
                for i, r in enumerate(rows)])
            if replay != read(DATA_ROOT/'e102/fit.json')['expanded_numeric_gates'][condition]:
                raise ValueError('Complete E102 expanded TRAIN replay differs')
        initial_new = new_sensor_gates(new, baseline[len(old)*4:].reshape(-1, 4))
        calls = [0]
        def check():
            resource_check(deadline); calls[0] += 1
            if calls[0] % 10 == 0:
                print(json.dumps({'E103_iteration': calls[0], 'seconds': round(time.monotonic()-start)}), flush=True)
        delta, solver = fit_delta(x, baseline_logits, labels, np.repeat([r['source'] for r in rows], 4),
                                  np.repeat([r['parent_id'] for r in rows], 4), np.tile(CONDITIONS, len(rows)), check)
        arrays['weights'] += delta
        candidate = expit(raw_logits+x@arrays['weights'])
        with CANDIDATE.open('xb') as stream:
            np.savez_compressed(stream, **arrays, contract_sha256=np.array(digest(CONTRACT)),
                reference_sha256=np.array(reference_sha), parent_artifact_sha256=np.array(digest(BASE)))
        with np.load(CANDIDATE, allow_pickle=False) as saved:
            if not np.array_equal(candidate, model.predict(head, features['dino'], features['clip'], features['dear'], saved)):
                raise ValueError('Saved candidate replay differs')
    scores = candidate.reshape(-1, 4)
    retention = check_train_retention(rows, baseline.reshape(-1, 4), scores, conditions=CONDITIONS, ai_cut=model.AI_CUT)
    old_gates = old_numeric(old, scores[:len(old)], 11630, 12141)
    old_slices = old_population(old, scores[:len(old)], 11630, 12141)
    new_gates = new_sensor_gates(new, scores[len(old):])
    expanded = {}
    for j, condition in enumerate(CONDITIONS):
        expanded[condition] = evaluate_condition([{'parent_id': r['parent_id'], 'source': r['source'],
            'role': 'TRAIN', 'label': r['label'], 'score': float(scores[i, j])} for i, r in enumerate(rows)])
    passed = solver['success'] and solver['max_constraint_violation'] <= 1e-8 and solver['minimum_ai_logit_shift'] >= -1e-8 and \
        retention['passes_train_retention'] and all(retention['comparisons'][k]['real']['non_ai_to_ai'] == 0 for k in CONDITIONS) and \
        all(v['gate']['passed'] for group in old_gates.values() for v in group.values()) and \
        all(old_slices[k]['passed'] and new_gates[k]['passed'] and expanded[k]['gate']['passed'] for k in CONDITIONS)
    runtime = {'executed': False, 'reason': 'Provisional TRAIN guards failed'}
    if passed:
        actual, shifts = [], []
        with threadpool_limits(limits=2):
            for offset in range(0, len(labels), 8):
                resource_check(deadline); sl = slice(offset, offset+8)
                chunk_x = model.project(head, features['dino'][sl], features['clip'][sl], features['dear'][sl], arrays)
                actual.append(expit(head.decision_function(features['dino'][sl])+chunk_x@arrays['weights']))
                shifts.append(chunk_x@delta)
        runtime = {'executed': True, 'batch_size': 8} | runtime_checks(candidate, np.concatenate(actual), np.concatenate(shifts), labels)
        passed = runtime['passed']
    result = {'state': 'E103_TRAIN_guard_passed' if passed else 'E103_TRAIN_guard_failed',
        'dev_scoring_permitted': bool(passed), 'candidate_sha256': digest(CANDIDATE), 'contract_sha256': digest(CONTRACT),
        'solver': solver, 'train_gate': retention, 'previous_numeric_gates': old_gates, 'previous_population_gates': old_slices,
        'expanded_numeric_gates': expanded, 'predecessor': 'E102', 'baseline_new_sensor_gates': initial_new, 'new_sensor_gates': new_gates,
        'runtime_batch_replay': runtime, 'seconds': time.monotonic()-start, 'dev_final_rows_read': 0, 'promotion_allowed': False}
    write_once(REPORT, result); write_once(EVIDENCE/'e103_fit.json', result)
    return {k: result[k] for k in ['state', 'seconds', 'dev_scoring_permitted', 'baseline_new_sensor_gates', 'new_sensor_gates']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('stage', choices=['freeze', 'fit'])
    ROOT.mkdir(parents=True, exist_ok=True)
    with (ROOT/'execution.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        print(json.dumps({'freeze': freeze, 'fit': fit}[parser.parse_args().stage](), indent=2))
