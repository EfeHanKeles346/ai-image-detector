"""Paired full-TRAIN context versus pooled extensions under unchanged E103 retention."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import socket
import time
if __name__ == '__main__':
    def denied(*a, **kw): raise RuntimeError('E113 fitting is offline')
    socket.socket.connect = denied; socket.socket.connect_ex = denied; socket.create_connection = denied
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
import numpy as np
from scipy.special import expit
from sklearn.decomposition import PCA
from threadpoolctl import threadpool_limits
from experiments import e92_model as base_model
from experiments.e65_acquisition import digest, read, write_once
from experiments.e71_features import save_npz
from experiments.e72_acquisition import resource_check
from experiments.e91_data import load as load_old, CONDITIONS
from experiments.e103_fit import validate as validate_base, new_sensor_gates
from experiments.e103_joint_minimax import fit as fit_delta
from experiments.e92_gates import full_training_gates, population_gates
from experiments.e49_evaluation import evaluate_condition
from experiments.e80_fit import runtime_checks
from experiments.e112_context_features import validate as validate_features, pooled
from pixelproof.retention_gate import check_train_retention
from pixelproof.project_paths import DATA_ROOT, ML_ROOT

ROOT = DATA_ROOT/'e113'; EVIDENCE = ML_ROOT.parent/'evidence'; CONTRACT = ROOT/'contract.json'
BASE = DATA_ROOT/'e103/correction.npz'
BRANCHES = ('pooled_control', 'ordered_context')


def representation(values, width=128):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or not np.isfinite(values).all() or min(values.shape) <= width:
        raise ValueError('Complete finite TRAIN feature matrix required')
    center = values.mean(axis=0); scale = np.maximum(values.std(axis=0), 1e-6)
    normalized = (values-center)/scale
    pca = PCA(n_components=width, svd_solver='randomized', iterated_power=3, random_state=113, copy=False)
    pca.fit(normalized)
    return {'feature_center': center, 'feature_scale': scale, 'pca_mean': pca.mean_,
            'components': pca.components_, 'latent_scale': np.sqrt(np.maximum(pca.explained_variance_, 1e-12))}


def coordinates(values, a):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != len(a['feature_center']) or not np.isfinite(values).all():
        raise ValueError('Finite aligned representation inputs required')
    result = (((values-a['feature_center'])/a['feature_scale']-a['pca_mean'])@a['components'].T)/a['latent_scale']
    if not np.isfinite(result).all(): raise ValueError('Invalid latent features')
    return result


def predict(head, features, values, base, a):
    old = base_model.project(head, features['dino'], features['clip'], features['dear'], base)
    x = np.column_stack([old, coordinates(values, a)])
    return expit(head.decision_function(features['dino'])+old@base['weights']+x@a['delta'])


def freeze():
    validate_base(); validate_features()
    receipt = DATA_ROOT/'e112/report.json'; report = read(receipt)
    if digest(receipt) != digest(EVIDENCE/'e112_context_features.json') or \
            report['state'] != 'E112_complete_context_features_ready' or report['parents'] != 12525 or \
            report['views'] != 50100 or report['max_aggregate_error'] > 1e-5 or report['model_scores'] or \
            digest(DATA_ROOT/'e112/features.npz') != report['feature_sha256']:
        raise ValueError('Complete bound E112 features required before registering fit')
    if digest(BASE) != read(DATA_ROOT/'e103/fit.json')['candidate_sha256'] or \
            digest(DATA_ROOT/'e103/fit.json') != digest(EVIDENCE/'e103_fit.json'):
        raise ValueError('Frozen E103 baseline required')
    previous = read(DATA_ROOT/'e103/fit_contract.json')
    files = [Path(__file__), BASE, DATA_ROOT/'e103/fit.json', DATA_ROOT/'e103/fit_contract.json',
        DATA_ROOT/'e112/contract.json', receipt, DATA_ROOT/'e112/features.npz',
        Path(fit_delta.__code__.co_filename), Path(check_train_retention.__code__.co_filename),
        ML_ROOT/'experiments/e112_context_features.py', ML_ROOT/'experiments/e68_minimax.py',
        ML_ROOT/'experiments/e73_fit.py', ML_ROOT/'experiments/e64_constrained.py',
        ML_ROOT/'experiments/e91_data.py', ML_ROOT/'experiments/e83_model.py',
        ML_ROOT/'experiments/e92_gates.py', ML_ROOT/'experiments/e80_fit.py']
    c = {'state': 'E113_paired_complete_TRAIN_context_fit_registered',
        'inputs': previous['inputs'] | {str(p): digest(p) for p in files}, 'branches': list(BRANCHES),
        'parents': 12525, 'views': 50100, 'conditions': CONDITIONS, 'max_seconds_per_branch': 3600,
        'representation': 'All-TRAIN per-coordinate center/std (floor1e-6), randomized PCA128 seed113 power3, whiten by TRAIN explained variance (floor1e-12). Same transform capacity for pooled mean/std1536 and ordered center plus center-minus-two-local-mean1536. Frozen E103450 coordinates appended with128 new coordinates; zero578 delta weights.',
        'objective': 'Identical E103 joint worst-class/source/condition BCE, L2.01, SLSQP200 ftol1e-9, CPU2. All E103 AI logits nondecreasing; every correct REAL binary decision protected. Two fixed branches, no sweep or DEV selection.',
        'acceptance': 'Solver success and violation<=1e-8; all old population/numeric gates plus new sensor/expanded gates, zero lost AI and zero new REAL errors. Saved artifact replay exact; complete batch8 runtime score error<=1e-6, no changes at either cut. Only a TRAIN pass permits a separately registered consumed DEV screen.',
        'ai_cut': base_model.AI_CUT, 'real_cut': base_model.REAL_CUT,
        'promotion_allowed': False, 'dev_final_reads': 0, 'downloads': 0,
        'limits': 'TRAIN-only paired ablation; both PCA and head fitted only on complete TRAIN. No calibration, user-gallery learning or independent evidence. Retain rejected branch artifacts and report both, including failure.'}
    ROOT.mkdir(exist_ok=True); write_once(CONTRACT, c)
    write_once(EVIDENCE/'e113_context_fit_contract.json', c | {'contract_sha256': digest(CONTRACT)})
    return {'branches': list(BRANCHES), 'views_each': 50100}


def validate():
    c = read(CONTRACT)
    if digest(CONTRACT) != read(EVIDENCE/'e113_context_fit_contract.json')['contract_sha256']:
        raise ValueError('Fit contract differs')
    for p, sha in c['inputs'].items():
        if digest(p) != sha: raise ValueError('Fit input changed')
    return c


def fit():
    c = validate(); binding = digest(CONTRACT)
    if (ROOT/'report.json').exists(): raise FileExistsError('Paired fit already complete')
    old, features, head = load_old(); new = read(DATA_ROOT/'e100/training_manifest.json')['rows']; rows = old+new
    with np.load(DATA_ROOT/'e101/midd_features.npz', allow_pickle=False) as cache:
        if list(cache['parents']) != [r['parent_id'] for r in new] or \
                str(cache['binding']) != digest(DATA_ROOT/'e101/features_contract.json'):
            raise ValueError('New feature identity differs')
        for key in features:
            features[key] = np.concatenate([features[key], cache[key]]).reshape(len(rows)*4, -1)
    with np.load(BASE, allow_pickle=False) as saved: base = {k: saved[k].copy() for k in saved.files}
    with np.load(DATA_ROOT/'e112/features.npz', allow_pickle=False) as cache:
        if str(cache['binding']) != digest(DATA_ROOT/'e112/contract.json') or \
                list(cache['parents']) != [r['parent_id'] for r in rows] or \
                list(cache['roles']) != ['TRAIN']*len(rows) or list(cache['conditions']) != CONDITIONS:
            raise ValueError('Complete ordered TRAIN context pairing required')
        raw = cache['raw']; context = cache['context'].reshape(-1, 1536)
    control = pooled(raw.reshape(-1, 3, 768)); del raw
    if len(rows) != 12525 or context.shape != (50100, 1536) or \
            max(float(np.abs(control-features['clip']).max()), 0.) > 1e-5:
        raise ValueError('Complete matched context/control population required')
    labels = np.repeat([r['label'] for r in rows], 4)
    sources = np.repeat([r['source'] for r in rows], 4)
    parents = np.repeat([r['parent_id'] for r in rows], 4); conditions = np.tile(CONDITIONS, len(rows))
    reports = {}
    with threadpool_limits(limits=2):
        old_x = base_model.project(head, features['dino'], features['clip'], features['dear'], base)
        baseline_logits = head.decision_function(features['dino'])+old_x@base['weights']; baseline = expit(baseline_logits)
        previous = read(DATA_ROOT/'e103/fit.json')
        for j, condition in enumerate(CONDITIONS):
            replay = evaluate_condition([{'parent_id': r['parent_id'], 'source': r['source'], 'label': r['label'],
                'role': 'TRAIN', 'score': float(baseline.reshape(-1, 4)[i, j])} for i, r in enumerate(rows)])
            if replay != previous['expanded_numeric_gates'][condition]: raise ValueError('E103 TRAIN replay changed')
        for branch in BRANCHES:
            start = time.monotonic(); deadline = start+c['max_seconds_per_branch']; resource_check(deadline)
            report_path = ROOT/(branch+'_report.json')
            if report_path.exists():
                prior = read(report_path)
                if prior['contract_sha256'] != binding or digest(ROOT/(branch+'.npz')) != prior['candidate_sha256']:
                    raise ValueError('Completed branch identity differs')
                reports[branch] = prior
                continue
            values = control if branch == 'pooled_control' else context
            a = representation(values); x = np.column_stack([old_x, coordinates(values, a)])
            calls = [0]
            def check():
                resource_check(deadline); calls[0] += 1
                if calls[0] % 10 == 0:
                    print(json.dumps({'E113_branch': branch, 'iteration': calls[0], 'seconds': round(time.monotonic()-start)}), flush=True)
            delta, solver = fit_delta(x, baseline_logits, labels, sources, parents, conditions, check)
            a['delta'] = delta; candidate = expit(baseline_logits+x@delta)
            artifact = ROOT/(branch+'.npz')
            save_npz(artifact, **a, contract_sha256=binding, base_sha256=digest(BASE), branch=branch)
            with np.load(artifact, allow_pickle=False) as saved:
                if not np.array_equal(candidate, predict(head, features, values, base, saved)):
                    raise ValueError('Saved candidate replay differs')
            scores = candidate.reshape(-1, 4)
            retention = check_train_retention(rows, baseline.reshape(-1, 4), scores, conditions=CONDITIONS, ai_cut=base_model.AI_CUT)
            numeric = full_training_gates(old, scores[:len(old)], 11630, 12141)
            slices = population_gates(old, scores[:len(old)], 11630, 12141)
            sensors = new_sensor_gates(new, scores[len(old):]); expanded = {}
            for j, condition in enumerate(CONDITIONS):
                expanded[condition] = evaluate_condition([{'parent_id': r['parent_id'], 'source': r['source'],
                    'label': r['label'], 'role': 'TRAIN', 'score': float(scores[i, j])} for i, r in enumerate(rows)])
            passed = solver['success'] and solver['max_constraint_violation'] <= 1e-8 and solver['minimum_ai_logit_shift'] >= -1e-8 and \
                retention['passes_train_retention'] and all(retention['comparisons'][k]['real']['non_ai_to_ai'] == 0 for k in CONDITIONS) and \
                all(v['gate']['passed'] for g in numeric.values() for v in g.values()) and \
                all(slices[k]['passed'] and sensors[k]['passed'] and expanded[k]['gate']['passed'] for k in CONDITIONS)
            runtime = {'executed': False, 'reason': 'Provisional TRAIN gate failed'}
            if passed:
                actual, shifts = [], []
                for i in range(0, len(labels), 8):
                    resource_check(deadline); sl = slice(i, i+8)
                    fs = {k: v[sl] for k, v in features.items()}
                    actual.append(predict(head, fs, values[sl], base, a))
                    chunk_x = np.column_stack([base_model.project(head, fs['dino'], fs['clip'], fs['dear'], base), coordinates(values[sl], a)])
                    shifts.append(chunk_x@delta)
                runtime = {'executed': True, 'batch_size': 8} | runtime_checks(candidate, np.concatenate(actual), np.concatenate(shifts), labels)
                passed = runtime['passed']
            result = {'state': 'E113_TRAIN_guard_passed' if passed else 'E113_TRAIN_guard_failed',
                'branch': branch, 'contract_sha256': binding, 'candidate_sha256': digest(artifact),
                'dev_scoring_permitted': bool(passed), 'solver': solver, 'retention': retention,
                'previous_numeric_gates': numeric, 'previous_population_gates': slices,
                'new_sensor_gates': sensors, 'expanded_numeric_gates': expanded,
                'runtime': runtime, 'seconds': time.monotonic()-start, 'promotion_allowed': False}
            write_once(report_path, result); reports[branch] = result
            print(json.dumps({'branch': branch, 'state': result['state'], 'seconds': result['seconds']}), flush=True)
            del x, a
    summary = {name: {'dev_scoring_permitted': r['dev_scoring_permitted'],
        'solver_success': r['solver']['success'],
        'false_REAL_alerts_by_condition': {k: v['binary_metrics']['confusion']['fp'] for k, v in r['expanded_numeric_gates'].items()},
        'AI_caught_by_condition': {k: v['binary_metrics']['confusion']['tp'] for k, v in r['expanded_numeric_gates'].items()}}
        for name, r in reports.items()}
    result = {'state': 'E113_paired_TRAIN_ablation_complete', 'contract_sha256': binding,
        'summary': summary, 'reports': reports, 'dev_final_reads': 0, 'promotion_allowed': False,
        'next': 'Separately register consumed DEV only for passing branches; no independent or universal claim.'}
    write_once(ROOT/'report.json', result); write_once(EVIDENCE/'e113_context_fit.json', result)
    with (ML_ROOT.parent/'PLAN.md').open('a') as plan:
        fcntl.flock(plan, fcntl.LOCK_EX)
        plan.write('\n### E113 automatic execution outcome\n\nBoth fixed TRAIN branches completed. '+
            json.dumps(summary, sort_keys=True)+
            '\n\nNext: separately register consumed DEV only for branches whose TRAIN guards passed. '
            'If neither passed, diagnose TRAIN failure before another preregistered experiment. '
            'No model promotion or independent evidence has been obtained. '
            'Detailed results: evidence/e113_context_fit.json.\n')
    return {'state': result['state'], 'summary': summary}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('stage', choices=['freeze', 'fit'])
    ROOT.mkdir(exist_ok=True)
    with (ROOT/'execution.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        print(json.dumps({'freeze': freeze, 'fit': fit}[p.parse_args().stage](), indent=2))
