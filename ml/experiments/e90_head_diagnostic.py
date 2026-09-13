"""Read the frozen E82 scalar-head sign on consumed E83 error bins; no candidate modification."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import socket
import time
import joblib
import numpy as np
import torch
from scipy.special import expit
from threadpoolctl import threadpool_limits
from experiments import e83_model as model
from experiments.e82_representation import latent_raw
from experiments.e83_development import validate as validate_dev, CONTRACT as DEV_CONTRACT, FEATURES, SCORES, REPORT as DEV_REPORT, CANDIDATE
from experiments.e83_diagnostic import check_cache
from experiments.e65_acquisition import digest, read, write_once
from experiments.e72_acquisition import resource_check
from pixelproof.project_paths import DATA_ROOT, ML_ROOT

ROOT = DATA_ROOT / 'e90'; EVIDENCE = ML_ROOT.parent / 'evidence'
CONTRACT = ROOT / 'diagnostic_contract.json'; REPORT = ROOT / 'head_agreement.json'


def scalar_logits(x, candidate):
    result = (latent_raw(x, candidate) @ np.asarray(candidate['nn_4_weight'], dtype=np.float64).T +
              np.asarray(candidate['nn_4_bias'], dtype=np.float64)).reshape(-1)
    if result.shape != (len(x),) or not np.isfinite(result).all():
        raise ValueError('finite aligned scalar-head logits required')
    return result


def summarize(values):
    return {'views': len(values), 'native_BCE_AI_sign_count': int(np.sum(values >= 0)),
            'native_BCE_REAL_sign_count': int(np.sum(values < 0)),
            'logit_min_median_max': np.quantile(values, [0, .5, 1]).tolist()}


def freeze():
    validate_dev(); report = read(DEV_REPORT); scores = read(SCORES)
    if digest(DEV_REPORT) != digest(EVIDENCE / 'e83_development.json') or report['passes_limited_dev_screen'] or \
            digest(SCORES) != report['scores_sha256'] or digest(CANDIDATE) != report['candidate_sha256'] or \
            digest(FEATURES) != scores['features_sha256']:
        raise ValueError('complete frozen rejected E83 diagnostic population required')
    files = [Path(__file__), Path(model.__file__), Path(__file__).with_name('e82_representation.py'),
             Path(__file__).with_name('e83_diagnostic.py'), DEV_CONTRACT, DEV_REPORT, FEATURES, SCORES, CANDIDATE]
    c = {'state': 'E90_read_only_E82_scalar_head_agreement_registered', 'inputs': {str(p): digest(p) for p in files},
         'views': 640, 'batch_size': 8, 'max_seconds': 600, 'score_replay_max_error': 1e-12,
         'analysis': 'Exact E83 runtime score replay and zero two-cut changes on all640 consumed DEV caches. '
                     'Compute saved E82 final scalar head in CPUfloat64 on the exact385 input coordinates. '
                     'Use native BCE sign0, not a selected threshold. Report sign counts and fixed min/median/max '
                     'per every source/condition and existing E43-to-E83 transition bin. Keep E83 detector scores unchanged.',
         'limits': 'Consumed E66, dependent REAL scenes and unknown AI prompt dependence. Agreement/disagreement '
                   'does not identify a causal mechanism or establish generalization. These are new diagnostic '
                   'logits from an existing trained head, not a new fitted/detection candidate or calibrated probability.',
         'new_fits': 0, 'new_detector_candidates': 0, 'image_reads': 0, 'e49_reads': 0,
         'threshold_selection_allowed': False, 'promotion_allowed': False}
    write_once(CONTRACT, c)
    write_once(EVIDENCE / 'e90_diagnostic_contract.json', c | {'contract_sha256': digest(CONTRACT)})
    return {'state': c['state'], 'views': c['views']}


def run():
    previous = validate_dev(); c = read(CONTRACT)
    if digest(CONTRACT) != read(EVIDENCE / 'e90_diagnostic_contract.json')['contract_sha256']:
        raise ValueError('E90 diagnostic contract changed')
    for p, sha in c['inputs'].items():
        if digest(p) != sha:
            raise ValueError('E90 input changed: ' + p)
    if REPORT.exists():
        raise FileExistsError('diagnostic already complete')
    torch.set_num_threads(2); started = time.monotonic(); deadline = started + c['max_seconds']; resource_check(deadline)
    rows = read(SCORES)['rows']
    if len(rows) != c['views']:
        raise ValueError('complete640-view diagnostic required')
    expected = np.array([r['score'] for r in rows]); old = np.array([r['reference_score'] for r in rows])
    head = joblib.load(previous['reference']['path'])['head']; actual, logits = [], []
    with np.load(FEATURES, allow_pickle=False) as cache, np.load(CANDIDATE, allow_pickle=False) as candidate, threadpool_limits(limits=2):
        check_cache(cache, rows, digest(DEV_CONTRACT))
        for start in range(0, len(rows), c['batch_size']):
            resource_check(deadline); part = slice(start, start + c['batch_size'])
            x = model.project(head, cache['dino'][part], cache['clip'][part], cache['dear'][part], candidate)
            actual.append(expit(head.decision_function(cache['dino'][part]) + x @ candidate['weights']))
            logits.append(scalar_logits(x[:, :385], candidate))
    actual = np.concatenate(actual); logits = np.concatenate(logits)
    error = float(np.max(np.abs(actual - expected)))
    changes = {str(cut): int(np.sum((actual >= cut) != (expected >= cut))) for cut in [model.AI_CUT, model.REAL_CUT]}
    if error > c['score_replay_max_error'] or any(changes.values()):
        raise ValueError('immutable E83 runtime replay differs')
    groups = {}
    for condition in sorted({r['condition'] for r in rows}):
        groups[condition] = {}
        for source in sorted({r['source'] for r in rows if r['condition'] == condition}):
            indices = np.array([i for i, r in enumerate(rows) if r['condition'] == condition and r['source'] == source])
            bins = {}
            for before in [False, True]:
                for after in [False, True]:
                    selected = indices[((old[indices] >= model.AI_CUT) == before) & ((expected[indices] >= model.AI_CUT) == after)]
                    if len(selected):
                        bins[f'old_ai_{int(before)}_new_ai_{int(after)}'] = summarize(logits[selected])
            groups[condition][source] = {'label': rows[indices[0]]['label'], 'views': len(indices), 'transitions': bins}
    result = {'state': 'E90_existing_scalar_head_agreement_complete', 'contract_sha256': digest(CONTRACT),
              'score_replay_max_error': error, 'decision_changes_by_cut': changes, 'groups': groups,
              'diagnostic_logits': len(logits), 'new_fits': 0, 'new_detector_candidates': 0,
              'image_reads': 0, 'e49_reads': 0, 'seconds': time.monotonic() - started, 'limits': c['limits']}
    write_once(REPORT, result); write_once(EVIDENCE / 'e90_head_agreement.json', result)
    return {'state': result['state'], 'score_replay_max_error': error, 'seconds': result['seconds']}


if __name__ == '__main__':
    def denied(*args, **kwargs):
        raise RuntimeError('E90 diagnostic offline')
    socket.socket.connect = denied; socket.socket.connect_ex = denied; socket.create_connection = denied
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('stage', choices=['freeze', 'run'])
    ROOT.mkdir(exist_ok=True)
    with (ROOT / 'diagnostic_execution.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        print(json.dumps({'freeze': freeze, 'run': run}[parser.parse_args().stage](), indent=2))
