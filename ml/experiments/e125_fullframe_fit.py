"""Paired full-frame versus center-only TRAIN extension under unchanged E103 retention."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import socket
import time
if __name__ == '__main__':
    def denied(*a, **kw): raise RuntimeError('E125 fitting is offline')
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
from experiments.e124_fullframe_features import validate as validate_features
from experiments.e113_context_fit import representation, coordinates, predict, BASE
from pixelproof.retention_gate import check_train_retention
from pixelproof.project_paths import DATA_ROOT, ML_ROOT

ROOT = DATA_ROOT/'e125'; EVIDENCE = ML_ROOT.parent/'evidence'; CONTRACT = ROOT/'contract.json'
BRANCHES = ('center_control', 'full_frame')


def freeze():
    validate_base(); validate_features()
    receipt = DATA_ROOT/'e124/report.json'; report = read(receipt)
    if digest(receipt) != digest(EVIDENCE/'e124_fullframe_features.json') or \
            report['state'] != 'E124_fullframe_TRAIN_cache_complete' or report['parents'] != 12525 or \
            report['views'] != 50100 or report['pilot_views_replayed'] != 160 or report['max_pilot_error'] > 1e-5 or \
            digest(DATA_ROOT/'e124/features.npz') != report['feature_sha256']:
        raise ValueError('Complete bound full-frame cache required before registering fit')
    if digest(BASE) != read(DATA_ROOT/'e103/fit.json')['candidate_sha256']:
        raise ValueError('Frozen E103 baseline differs')
    previous = read(DATA_ROOT/'e113/contract.json')
    if digest(DATA_ROOT/'e113/contract.json') != read(EVIDENCE/'e113_context_fit_contract.json')['contract_sha256']:
        raise ValueError('Previous paired fit contract differs')
    for p, sha in previous['inputs'].items():
        if digest(p) != sha: raise ValueError('Bound previous fit inputs differ')
    files = [Path(__file__), DATA_ROOT/'e124/contract.json', receipt, DATA_ROOT/'e124/features.npz',
        ML_ROOT/'experiments/e113_context_fit.py', DATA_ROOT/'e113/contract.json',
        ML_ROOT/'experiments/e124_fullframe_features.py']
    c = {'state': 'E125_complete_TRAIN_fullframe_fit_registered',
        'inputs': previous['inputs'] | read(DATA_ROOT/'e124/contract.json')['inputs'] | {str(p): digest(p) for p in files},
        'branches': list(BRANCHES), 'parents': 12525, 'views': 50100, 'conditions': CONDITIONS,
        'max_seconds_per_branch': 3600,
        'representation': 'Equal768-dimensional raw CLIP input: old center only versus uncropped available-frame warp. Each uses128 whitened PCA coordinates fitted only on all50100 TRAIN views with unchanged E113 scaling/seed113/power3. Append to frozen E103450 coordinates;578 zero delta coefficients.',
        'objective': 'Unchanged E103 joint worst-class/source/condition BCE,L2.01,SLSQP200,ftol1e-9,CPU2. Preserve all AI logits and correct REAL binary decisions. Both branches required, no DEV-driven selection.',
        'acceptance': 'Every E113 old/expanded/source/sensor/retention guard, solver success/violation<=1e-8 and minimum AI shift>=-1e-8, exact saved replay and complete batch8 runtime parity<=1e-6 with no cut changes. Only complete passing TRAIN permits a new consumed DEV registration.',
        'ai_cut': base_model.AI_CUT, 'real_cut': base_model.REAL_CUT,
        'downloads': 0, 'dev_final_reads': 0, 'promotion_allowed': False,
        'limits': 'TRAIN capacity-controlled comparison, not independent proof. Full-frame warping can introduce geometric/source bias. Repeated development comparisons increase selection bias; one additional fixed hypothesis, no margin/cut sweep or exception for the known missed AI.'}
    ROOT.mkdir(exist_ok=True); write_once(CONTRACT, c)
    write_once(EVIDENCE/'e125_context_fit_contract.json', c | {'contract_sha256': digest(CONTRACT)})
    return {'branches': list(BRANCHES), 'views_each': 50100}


def validate():
    c = read(CONTRACT)
    if digest(CONTRACT) != read(EVIDENCE/'e125_context_fit_contract.json')['contract_sha256']:
        raise ValueError('Full-frame fit contract differs')
    for p, sha in c['inputs'].items():
        if digest(p) != sha: raise ValueError('Frozen full-frame fit input differs')
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
            raise ValueError('Complete center-only TRAIN pairing required')
        control = cache['raw'][:, :, 0, :].reshape(-1, 768).copy()
    with np.load(DATA_ROOT/'e124/features.npz', allow_pickle=False) as cache:
        if str(cache['binding']) != digest(DATA_ROOT/'e124/contract.json') or \
                list(cache['parents']) != [r['parent_id'] for r in rows] or \
                list(cache['roles']) != ['TRAIN']*len(rows) or list(cache['conditions']) != CONDITIONS:
            raise ValueError('Complete full-frame TRAIN pairing required')
        context = cache['full'].reshape(-1, 768).copy()
    if len(rows) != 12525 or context.shape != (50100, 768) or control.shape != context.shape or \
            not np.isfinite(context).all() or not np.isfinite(control).all():
        raise ValueError('Complete matched full-frame/center-only population required')
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
            values = control if branch == 'center_control' else context
            a = representation(values); x = np.column_stack([old_x, coordinates(values, a)])
            calls = [0]
            def check():
                resource_check(deadline); calls[0] += 1
                if calls[0] % 10 == 0:
                    print(json.dumps({'E125_branch': branch, 'iteration': calls[0], 'seconds': round(time.monotonic()-start)}), flush=True)
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
            result = {'state': 'E125_TRAIN_guard_passed' if passed else 'E125_TRAIN_guard_failed',
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
    result = {'state': 'E125_paired_TRAIN_ablation_complete', 'contract_sha256': binding,
        'summary': summary, 'reports': reports, 'dev_final_reads': 0, 'promotion_allowed': False,
        'next': 'Separately register consumed DEV only for passing branches; no independent or universal claim.'}
    write_once(ROOT/'report.json', result); write_once(EVIDENCE/'e125_context_fit.json', result)
    with (ML_ROOT.parent/'PLAN.md').open('a') as plan:
        fcntl.flock(plan, fcntl.LOCK_EX)
        plan.write('\n### E125 automatic execution outcome\n\nBoth fixed TRAIN branches completed. '+
            json.dumps(summary, sort_keys=True)+
            '\n\nNext: separately register consumed DEV only for branches whose TRAIN guards passed. '
            'If neither passed, diagnose TRAIN failure before another preregistered experiment. '
            'No model promotion or independent evidence has been obtained. '
            'Detailed results: evidence/e125_context_fit.json.\n')
    return {'state': result['state'], 'summary': summary}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('stage', choices=['freeze', 'fit'])
    ROOT.mkdir(exist_ok=True)
    with (ROOT/'execution.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        print(json.dumps({'freeze': freeze, 'fit': fit}[p.parse_args().stage](), indent=2))
