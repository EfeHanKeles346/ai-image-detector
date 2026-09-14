"""One gated, paired consumed-DEV comparison of the E125 full-frame/center ablation."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import socket
import time
if __name__ == '__main__':
    def denied(*a, **kw): raise RuntimeError('E126 DEV is offline')
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
from experiments.e71_development import validate_pairs, check_reference
from experiments.e83_development import validate as validate_old_cache, FEATURES, CONTRACT as OLD_CONTRACT
from experiments.e83_diagnostic import check_cache
from experiments.e112_context_features import pooled
from experiments.e125_fullframe_fit import validate as validate_fit, BRANCHES
from experiments.e113_context_fit import predict, BASE
from experiments.e114_context_development import comparison, pair_identity
from experiments.e123_fullframe_probe import fullframe_array
from pixelproof.e32_candidate import standardized_array
from experiments.e49_evaluation import CONDITIONS
from pixelproof.project_paths import DATA_ROOT, ML_ROOT

ROOT = DATA_ROOT/'e126'; EVIDENCE = ML_ROOT.parent/'evidence'; CONTRACT = ROOT/'contract.json'


def eligible(report, *, allow_none=False):
    if report.get('state') != 'E125_paired_TRAIN_ablation_complete' or set(report.get('reports', {})) != set(BRANCHES):
        raise ValueError('Both complete preregistered TRAIN branch reports required')
    passing = []
    for name in BRANCHES:
        r = report['reports'][name]
        permission = r.get('dev_scoring_permitted')
        expected = 'E125_TRAIN_guard_passed' if permission is True else 'E125_TRAIN_guard_failed'
        if type(permission) is not bool or r.get('state') != expected:
            raise ValueError('Inconsistent TRAIN permission')
        if r.get('dev_scoring_permitted') is True:
            if r['state'] != 'E125_TRAIN_guard_passed' or not r['solver']['success'] or \
                    not np.isfinite(r['solver']['max_constraint_violation']) or \
                    r['solver']['max_constraint_violation'] > 1e-8 or \
                    not r['runtime'].get('passed') or not r['retention']['passes_train_retention'] or \
                    not np.isfinite(r['solver']['minimum_ai_logit_shift']) or \
                    r['solver']['minimum_ai_logit_shift'] < -1e-8:
                raise ValueError('Inconsistent TRAIN permission')
            passing.append(name)
    if not passing and not allow_none: raise ValueError('Neither TRAIN branch passed; DEV access denied')
    return passing


def paired_views(image):
    # DEV paths already contain original/social transport; do not apply JPEG75 twice.
    transported = crops.transport_image(image, 'clean')
    center = standardized_array(transported)
    full = fullframe_array(transported)
    return [center, full, full]


def reference_raw(cache, rows, binding, clip):
    if str(cache['binding']) != binding or list(cache['roles']) != ['DEVELOPMENT']*len(rows) or \
            list(cache['record_ids']) != [r['record_id'] for r in rows] or \
            list(cache['image_sha256']) != [r['sha256'] for r in rows]:
        raise ValueError('Reference raw DEV identity/role/order differs')
    raw = cache['raw']
    if raw.shape != (len(rows), 3, 768) or raw.dtype != np.float32 or not np.isfinite(raw).all():
        raise ValueError('Invalid reference raw DEV vectors')
    if not np.allclose(pooled(raw), clip, atol=1e-5, rtol=0):
        raise ValueError('Reference raw aggregate differs from E83')
    return raw[:, 0].copy()


def verify_encoded(raw, reference, tolerance):
    if raw.shape != (3, 768) or not np.isfinite(raw).all():
        raise ValueError('Invalid encoded full-frame vectors')
    center_error = float(np.abs(raw[0]-reference).max())
    repeat_error = float(np.abs(raw[1]-raw[2]).max())
    if max(center_error, repeat_error) > tolerance:
        raise ValueError('Center/reference or duplicate full-frame parity failed')
    return center_error, repeat_error


def freeze():
    validate_fit()
    fit_path = DATA_ROOT/'e125/report.json'
    if digest(fit_path) != digest(EVIDENCE/'e125_context_fit.json'):
        raise ValueError('Paired TRAIN report differs')
    fit = read(fit_path); branches = eligible(fit)  # Must precede any DEV metadata or cache read.
    for name in branches:
        if digest(DATA_ROOT/'e125'/f'{name}.npz') != fit['reports'][name]['candidate_sha256']:
            raise ValueError('Candidate differs')
    prior_raw_report = DATA_ROOT/'e114/scores.json'
    if digest(prior_raw_report) != read(EVIDENCE/'e114_context_dev_scores.json')['scores_sha256'] or \
            digest(DATA_ROOT/'e114/raw.npz') != read(prior_raw_report)['raw_sha256'] or \
            digest(DATA_ROOT/'e114/contract.json') != read(prior_raw_report)['contract_sha256']:
        raise ValueError('Bound E114 raw reference differs')
    old = validate_old_cache()
    if digest(BACKBONE) != BACKBONE_SHA: raise ValueError('Pinned CLIP encoder differs')
    files = [Path(__file__), fit_path, DATA_ROOT/'e125/contract.json', BASE, BACKBONE,
        ML_ROOT/'experiments/e125_fullframe_fit.py', ML_ROOT/'experiments/e113_context_fit.py',
        ML_ROOT/'experiments/e114_context_development.py', ML_ROOT/'experiments/e123_fullframe_probe.py',
        ML_ROOT/'experiments/e112_context_features.py', DATA_ROOT/'e114/raw.npz',
        prior_raw_report, DATA_ROOT/'e114/contract.json',
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
    pair_identity(rows, read(prior_raw_report)['branches']['pooled_control'])
    files += [DATA_ROOT/'e125'/f'{name}.npz' for name in branches]
    c = {'state': 'E126_paired_consumed_DEV_registered',
        'inputs': read(DATA_ROOT/'e125/contract.json')['inputs'] | {str(p): digest(p) for p in files},
        'branches': branches, 'manifest': old['manifest'], 'reference': old['reference'],
        'views': 640, 'conditions': list(CONDITIONS), 'max_seconds': 2400,
        'parity_max_abs': 1e-5, 'reference_score_max_abs': 1e-6, 'mps_limit_bytes': 6*1024**3,
        'protocol': 'Complete original/social640-view E83 cache. For each already transported view encode batch3=[historical center,full frame,full frame] with the exact E123/E124 transforms. Replay center against SHA-bound E114 raw cache and repeat full-frame vector<=1e-5, all finite; verify reference aggregate against E83. Reuse DINO/pooled CLIP/DEAR and frozen E125 PCA/head; no fitting, new transport or threshold change.',
        'selection': 'Score every E125 branch that passed its prespecified complete TRAIN/runtime guards; compare both if both passed. No branch selected by DEV results. Lock all candidate scores before metrics.',
        'acceptance': 'All20 numerical gates plus zero lost E43/E92/E103 AI and zero new E92/E103 REAL errors per condition/source. No tradeoff of rescues against new errors. No gallery or E49 access.',
        'limits': old['limits']+' This comparison is consumed development, never independent final. 160 REAL photographs share10 SIDD scenes; AI families/prompt linkage are limited.',
        'training_allowed': False, 'promotion_allowed': False, 'downloads': 0}
    ROOT.mkdir(exist_ok=True); write_once(CONTRACT, c)
    write_once(EVIDENCE/'e126_fullframe_dev_contract.json', c | {'contract_sha256': digest(CONTRACT)})
    return {'branches': branches, 'views_per_branch': 640}


def validate():
    c = read(CONTRACT)
    if digest(CONTRACT) != read(EVIDENCE/'e126_fullframe_dev_contract.json')['contract_sha256']:
        raise ValueError('DEV contract differs')
    for p, sha in c['inputs'].items():
        if digest(p) != sha: raise ValueError('Bound DEV input differs')
    if eligible(read(DATA_ROOT/'e125/report.json')) != c['branches']:
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
    with np.load(DATA_ROOT/'e114/raw.npz', allow_pickle=False) as cache:
        reference = reference_raw(cache, rows, digest(DATA_ROOT/'e114/contract.json'), features['clip'])
    model, preprocess, device = load_encoder(); raw_parts = []; max_error = 0.; max_repeat = 0.; peak = 0
    with torch.inference_mode():
        for i, row in enumerate(rows):
            resource_check(deadline)
            if digest(row['path']) != row['sha256']: raise ValueError('DEV source/transport body differs')
            with Image.open(row['path']) as image:
                pack = paired_views(image)
            raw = model.encode_image(torch.stack([preprocess(Image.fromarray(crop)) for crop in pack]).to(device)).float().cpu().numpy()
            error, repeat = verify_encoded(raw, reference[i], c['parity_max_abs'])
            max_error = max(max_error, error); max_repeat = max(max_repeat, repeat); raw_parts.append(raw)
            peak = max(peak, torch.mps.driver_allocated_memory() if device.type == 'mps' else 0)
            if peak > c['mps_limit_bytes']: raise RuntimeError('DEV memory ceiling reached')
            if (i+1) % 64 == 0:
                print(json.dumps({'E126_encoded_views': i+1, 'max_center_error': max_error}), flush=True)
    raw = np.stack(raw_parts).astype(np.float32); full = raw[:, 1]; control = raw[:, 0]
    save_npz(ROOT/'raw.npz', raw=raw, binding=binding, roles=np.repeat('DEVELOPMENT', len(rows)),
        record_ids=np.array([r['record_id'] for r in rows]), image_sha256=np.array([r['sha256'] for r in rows]))
    locked = {}; ref_error = 0.; predecessor_error = 0.
    with threadpool_limits(limits=2):
        for name in c['branches']:
            with np.load(DATA_ROOT/'e125'/f'{name}.npz', allow_pickle=False) as a:
                if str(a['contract_sha256']) != digest(DATA_ROOT/'e125/contract.json') or str(a['base_sha256']) != digest(BASE):
                    raise ValueError('Candidate bindings differ')
                if str(a['branch']) != name: raise ValueError('Candidate branch differs')
                done = []; values = control if name == 'center_control' else full
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
    result = {'state': 'E126_all_passing_branch_scores_locked', 'branches': locked,
        'contract_sha256': binding, 'raw_sha256': digest(ROOT/'raw.npz'),
        'max_center_error': max_error, 'max_repeat_error': max_repeat, 'E43_replay_max_error': ref_error, 'E103_replay_max_error': predecessor_error,
        'mps_peak_bytes': peak, 'seconds': time.monotonic()-start, 'training_allowed': False}
    write_once(ROOT/'scores.json', result)
    write_once(EVIDENCE/'e126_fullframe_dev_scores.json', {k:v for k,v in result.items() if k != 'branches'} |
        {'branches': c['branches'], 'scores_sha256': digest(ROOT/'scores.json'), 'metrics_opened': False})
    return {'locked_branches': c['branches'], 'views_each': len(rows), 'max_center_error': max_error}


def report():
    c = validate()
    if digest(ROOT/'scores.json') != read(EVIDENCE/'e126_fullframe_dev_scores.json')['scores_sha256']:
        raise ValueError('Locked scores differ')
    locked = read(ROOT/'scores.json'); results = {}; summary = {}
    if set(locked['branches']) != set(c['branches']) or locked['contract_sha256'] != digest(CONTRACT) or \
            digest(ROOT/'raw.npz') != locked['raw_sha256']:
        raise ValueError('Complete bound paired scores required before metrics')
    for name, rows in locked['branches'].items():
        validate_pairs(rows, read(c['manifest'])['rows']); result = comparison(rows); results[name] = result
        summary[name] = {'passes_consumed_DEV_screen': all(v['passed'] for v in result.values()),
            'by_condition': {k: {'REAL_false_alerts': v['metrics']['binary_metrics']['confusion']['fp'],
                'AI_caught': v['metrics']['binary_metrics']['confusion']['tp'], 'checks': v['checks']}
                for k, v in result.items()}}
    result = {'state': 'E126_paired_consumed_DEV_complete', 'contract_sha256': digest(CONTRACT),
        'scores_sha256': digest(ROOT/'scores.json'), 'summary': summary, 'results': results,
        'independent_final_passed': False, 'serving_changed': False, 'limits': c['limits']}
    write_once(ROOT/'report.json', result); write_once(EVIDENCE/'e126_fullframe_development.json', result)
    text = ('\n### E126 automatic consumed-DEV result\n\n'+json.dumps(summary, sort_keys=True)+
        '\n\nFull report: evidence/e126_fullframe_development.json. This is consumed development; no independent proof or serving change. '
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
