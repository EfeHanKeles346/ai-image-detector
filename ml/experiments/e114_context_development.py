"""One gated, paired consumed-DEV comparison of the E113 representation ablation."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import socket
import time
if __name__ == '__main__':
    def denied(*a, **kw): raise RuntimeError('E114 DEV is offline')
    socket.socket.connect = denied; socket.socket.connect_ex = denied; socket.create_connection = denied
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
import joblib
import numpy as np
from PIL import Image
from threadpoolctl import threadpool_limits
from experiments import e42_features as crops, e92_model as base_model
from experiments.e65_acquisition import digest, read, write_once
from experiments.e71_features import load_encoder, save_npz, BACKBONE, BACKBONE_SHA
from experiments.e72_acquisition import resource_check
from experiments.e71_development import validate_pairs, check_reference, transitions
from experiments.e83_development import validate as validate_old_cache, FEATURES, CONTRACT as OLD_CONTRACT
from experiments.e83_diagnostic import check_cache
from experiments.e104_context_audit import context_coordinates
from experiments.e112_context_features import pooled
from experiments.e113_context_fit import validate as validate_fit, predict, BRANCHES, BASE
from experiments.e49_evaluation import evaluate_condition, CONDITIONS
from pixelproof.project_paths import DATA_ROOT, ML_ROOT

ROOT = DATA_ROOT/'e114'; EVIDENCE = ML_ROOT.parent/'evidence'; CONTRACT = ROOT/'contract.json'


def eligible(report):
    if report.get('state') != 'E113_paired_TRAIN_ablation_complete' or set(report.get('reports', {})) != set(BRANCHES):
        raise ValueError('Both complete preregistered TRAIN branch reports required')
    passing = []
    for name in BRANCHES:
        r = report['reports'][name]
        if r.get('dev_scoring_permitted') is True:
            if r['state'] != 'E113_TRAIN_guard_passed' or not r['solver']['success'] or \
                    not np.isfinite(r['solver']['max_constraint_violation']) or \
                    r['solver']['max_constraint_violation'] > 1e-8 or \
                    not r['runtime'].get('passed') or not r['retention']['passes_train_retention']:
                raise ValueError('Inconsistent TRAIN permission')
            passing.append(name)
    if not passing: raise ValueError('Neither TRAIN branch passed; DEV access denied')
    return passing


def pair_identity(a, b):
    fields = ('parent_id', 'condition', 'sha256', 'role', 'source', 'label', 'record_id')
    if len(a) != len(b) or any(any(x[k] != y[k] for k in fields) for x, y in zip(a, b, strict=True)):
        raise ValueError('Paired predecessor identity/order differs')


def comparison(rows):
    result = {}
    for condition in CONDITIONS:
        selected = [r for r in rows if r['condition'] == condition]
        metrics = evaluate_condition(selected)
        changes = {name: transitions([r | {'reference_score': r[field]} for r in selected])
                   for name, field in [('E43', 'reference_score'), ('E92', 'E92_score'), ('E103', 'E103_score')]}
        checks = {'twenty_numeric_gates_condition_passed': metrics['gate']['passed']}
        for name, values in changes.items():
            checks[f'zero_lost_{name}_AI'] = all(v['new_ai_misses'] == 0 for v in values.values())
            if name != 'E43':
                checks[f'zero_new_{name}_REAL_errors'] = all(v['new_real_false_ai'] == 0 for v in values.values())
        result[condition] = {'metrics': metrics, 'transitions': changes, 'checks': checks,
                             'passed': all(checks.values())}
    return result


def freeze():
    validate_fit()
    fit_path = DATA_ROOT/'e113/report.json'
    if digest(fit_path) != digest(EVIDENCE/'e113_context_fit.json'):
        raise ValueError('Paired TRAIN report differs')
    fit = read(fit_path); branches = eligible(fit)  # Must precede any DEV metadata or cache read.
    for name in branches:
        if digest(DATA_ROOT/'e113'/f'{name}.npz') != fit['reports'][name]['candidate_sha256']:
            raise ValueError('Candidate differs')
    old = validate_old_cache()
    if digest(BACKBONE) != BACKBONE_SHA: raise ValueError('Pinned CLIP encoder differs')
    files = [Path(__file__), fit_path, DATA_ROOT/'e113/contract.json', BASE, BACKBONE,
        ML_ROOT/'experiments/e113_context_fit.py', ML_ROOT/'experiments/e112_context_features.py',
        ML_ROOT/'experiments/e104_context_audit.py', ML_ROOT/'experiments/e71_features.py',
        ML_ROOT/'experiments/e71_development.py', ML_ROOT/'experiments/e83_development.py',
        ML_ROOT/'experiments/e83_diagnostic.py', Path(crops.__file__),
        ML_ROOT/'src/pixelproof/e32_candidate.py', FEATURES, OLD_CONTRACT,
        Path(old['manifest']), Path(old['reference']['path'])]
    for experiment in ('e83', 'e92', 'e103'):
        report = DATA_ROOT/experiment/'dev_report.json'; scores = DATA_ROOT/experiment/'dev_scores.json'
        if digest(report) != digest(EVIDENCE/(experiment+'_development.json')) or digest(scores) != read(report)['scores_sha256']:
            raise ValueError('Predecessor DEV evidence differs')
        files += [report, scores]
    if digest(FEATURES) != read(DATA_ROOT/'e83/dev_scores.json')['features_sha256']:
        raise ValueError('Frozen DEV feature body differs')
    rows = read(DATA_ROOT/'e103/dev_scores.json')['rows']
    validate_pairs(rows, read(old['manifest'])['rows'])
    pair_identity(rows, read(DATA_ROOT/'e92/dev_scores.json')['rows'])
    pair_identity(rows, read(DATA_ROOT/'e83/dev_scores.json')['rows'])
    files += [DATA_ROOT/'e113'/f'{name}.npz' for name in branches]
    c = {'state': 'E114_paired_consumed_DEV_registered',
        'inputs': read(DATA_ROOT/'e113/contract.json')['inputs'] | {str(p): digest(p) for p in files},
        'branches': branches, 'manifest': old['manifest'], 'reference': old['reference'],
        'views': 640, 'conditions': list(CONDITIONS), 'max_seconds': 2400,
        'parity_max_abs': 1e-5, 'reference_score_max_abs': 1e-6, 'mps_limit_bytes': 6*1024**3,
        'protocol': 'Complete original/social640-view E83 cache. Re-encode only CLIP raw ordered crops with identical E83 native crop pipeline/batch3; every pooled feature must reproduce old CLIP<=1e-5 before using context. Reuse DINO/DEAR and frozen E113 PCA/head; no fitting or threshold change.',
        'selection': 'Score every E113 branch that passed its prespecified complete TRAIN/runtime guards; compare both if both passed. No branch selected by DEV results. Lock all candidate scores before metrics.',
        'acceptance': 'All20 numerical gates plus zero lost E43/E92/E103 AI and zero new E92/E103 REAL errors per condition/source. No tradeoff of rescues against new errors. No gallery or E49 access.',
        'limits': old['limits']+' This comparison is consumed development, never independent final. 160 REAL photographs share10 SIDD scenes; AI families/prompt linkage are limited.',
        'training_allowed': False, 'promotion_allowed': False, 'downloads': 0}
    ROOT.mkdir(exist_ok=True); write_once(CONTRACT, c)
    write_once(EVIDENCE/'e114_context_dev_contract.json', c | {'contract_sha256': digest(CONTRACT)})
    return {'branches': branches, 'views_per_branch': 640}


def validate():
    c = read(CONTRACT)
    if digest(CONTRACT) != read(EVIDENCE/'e114_context_dev_contract.json')['contract_sha256']:
        raise ValueError('DEV contract differs')
    for p, sha in c['inputs'].items():
        if digest(p) != sha: raise ValueError('Bound DEV input differs')
    if eligible(read(DATA_ROOT/'e113/report.json')) != c['branches']:
        raise ValueError('TRAIN permissions differ')
    return c


def score():
    import torch
    c = validate(); binding = digest(CONTRACT)
    if (ROOT/'scores.json').exists(): raise FileExistsError('DEV scores already locked')
    start = time.monotonic(); deadline = start+c['max_seconds']; resource_check(deadline)
    torch.set_num_threads(2)
    rows = read(DATA_ROOT/'e103/dev_scores.json')['rows']; e92 = read(DATA_ROOT/'e92/dev_scores.json')['rows']
    validate_pairs(rows, read(c['manifest'])['rows']); pair_identity(rows, e92)
    head = joblib.load(c['reference']['path'])['head']
    with np.load(BASE, allow_pickle=False) as a: base = {k: a[k].copy() for k in a.files}
    with np.load(FEATURES, allow_pickle=False) as a:
        check_cache(a, rows, digest(OLD_CONTRACT)); features = {k: a[k].copy() for k in ('dino', 'clip', 'dear')}
    model, preprocess, device = load_encoder(); raw_parts = []; max_error = 0.; peak = 0
    with torch.inference_mode():
        for i, row in enumerate(rows):
            resource_check(deadline)
            if digest(row['path']) != row['sha256']: raise ValueError('DEV source/transport body differs')
            with Image.open(row['path']) as image:
                pack = crops.texture_crops(crops.transport_image(image, 'clean'))
            raw = model.encode_image(torch.stack([preprocess(Image.fromarray(crop)) for crop in pack]).to(device)).float().cpu().numpy()
            error = float(np.abs(pooled(raw[None])[0]-features['clip'][i]).max())
            if error > c['parity_max_abs']: raise ValueError('DEV raw aggregate parity failed')
            max_error = max(max_error, error); raw_parts.append(raw)
            peak = max(peak, torch.mps.driver_allocated_memory() if device.type == 'mps' else 0)
            if peak > c['mps_limit_bytes']: raise RuntimeError('DEV memory ceiling reached')
            if (i+1) % 64 == 0:
                print(json.dumps({'E114_encoded_views': i+1, 'max_aggregate_error': max_error}), flush=True)
    raw = np.stack(raw_parts).astype(np.float32); ordered = context_coordinates(raw); control = pooled(raw)
    save_npz(ROOT/'raw.npz', raw=raw, binding=binding, roles=np.repeat('DEVELOPMENT', len(rows)),
        record_ids=np.array([r['record_id'] for r in rows]), image_sha256=np.array([r['sha256'] for r in rows]))
    locked = {}; ref_error = 0.; predecessor_error = 0.
    with threadpool_limits(limits=2):
        for name in c['branches']:
            with np.load(DATA_ROOT/'e113'/f'{name}.npz', allow_pickle=False) as a:
                if str(a['contract_sha256']) != digest(DATA_ROOT/'e113/contract.json') or str(a['base_sha256']) != digest(BASE):
                    raise ValueError('Candidate bindings differ')
                done = []; values = control if name == 'pooled_control' else ordered
                for offset in range(0, len(rows), 8):
                    resource_check(deadline); sl = slice(offset, offset+8)
                    fs = {k: v[sl] for k, v in features.items()}
                    old = head.predict_proba(fs['dino'])[:, 1]
                    prior = base_model.predict(head, fs['dino'], fs['clip'], fs['dear'], base)
                    scores = predict(head, fs, values[sl], base, a)
                    if not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
                        raise ValueError('Invalid candidate DEV score')
                    for j, row in enumerate(rows[sl]):
                        ref_error = max(ref_error, check_reference(row, old[j], row, c['reference_score_max_abs']))
                        error = abs(float(prior[j])-row['score']); predecessor_error = max(predecessor_error, error)
                        if error > 1e-6 or any((prior[j] >= cut) != (row['score'] >= cut) for cut in (base_model.AI_CUT, base_model.REAL_CUT)):
                            raise ValueError('E103 predecessor replay differs')
                        done.append(row | {'E103_score': row['score'], 'E92_score': e92[offset+j]['score'],
                            'score': float(scores[j]), 'status': 'ok'})
                validate_pairs(done, read(c['manifest'])['rows']); locked[name] = done
    result = {'state': 'E114_all_passing_branch_scores_locked', 'branches': locked,
        'contract_sha256': binding, 'raw_sha256': digest(ROOT/'raw.npz'),
        'max_aggregate_error': max_error, 'E43_replay_max_error': ref_error, 'E103_replay_max_error': predecessor_error,
        'mps_peak_bytes': peak, 'seconds': time.monotonic()-start, 'training_allowed': False}
    write_once(ROOT/'scores.json', result)
    write_once(EVIDENCE/'e114_context_dev_scores.json', {k:v for k,v in result.items() if k != 'branches'} |
        {'branches': c['branches'], 'scores_sha256': digest(ROOT/'scores.json'), 'metrics_opened': False})
    return {'locked_branches': c['branches'], 'views_each': len(rows), 'max_aggregate_error': max_error}


def report():
    c = validate()
    if digest(ROOT/'scores.json') != read(EVIDENCE/'e114_context_dev_scores.json')['scores_sha256']:
        raise ValueError('Locked scores differ')
    locked = read(ROOT/'scores.json'); results = {}; summary = {}
    for name, rows in locked['branches'].items():
        validate_pairs(rows, read(c['manifest'])['rows']); result = comparison(rows); results[name] = result
        summary[name] = {'passes_consumed_DEV_screen': all(v['passed'] for v in result.values()),
            'by_condition': {k: {'REAL_false_alerts': v['metrics']['binary_metrics']['confusion']['fp'],
                'AI_caught': v['metrics']['binary_metrics']['confusion']['tp'], 'checks': v['checks']}
                for k, v in result.items()}}
    result = {'state': 'E114_paired_consumed_DEV_complete', 'contract_sha256': digest(CONTRACT),
        'scores_sha256': digest(ROOT/'scores.json'), 'summary': summary, 'results': results,
        'independent_final_passed': False, 'serving_changed': False, 'limits': c['limits']}
    write_once(ROOT/'report.json', result); write_once(EVIDENCE/'e114_context_development.json', result)
    text = ('\n### E114 automatic consumed-DEV result\n\n'+json.dumps(summary, sort_keys=True)+
        '\n\nFull report: evidence/e114_context_development.json. This is consumed development; no independent proof or serving change. '
        'Only passing branches may proceed to a separately registered gallery regression. Failed branches require a new hypothesis, not relaxed gates.\n')
    for path in [ML_ROOT.parent/'PLAN.md', ML_ROOT.parent/'HISTORY.md', ML_ROOT/'EXPERIMENTS.md']:
        with path.open('a') as f:
            fcntl.flock(f, fcntl.LOCK_EX); f.write(text)
    return summary


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('stage', choices=['freeze', 'score', 'report'])
    ROOT.mkdir(exist_ok=True)
    with (ROOT/'execution.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        print(json.dumps({'freeze': freeze, 'score': score, 'report': report}[p.parse_args().stage](), indent=2))
