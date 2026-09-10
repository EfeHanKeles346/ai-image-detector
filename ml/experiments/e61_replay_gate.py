"""Offline engineering verification of a strict AI replay gate; no fitting or E49 reads."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import socket


def disable_network():
    def denied(*args, **kwargs):
        raise RuntimeError('E61 network connections disabled')
    socket.socket.connect = denied
    socket.socket.connect_ex = denied
    socket.create_connection = denied
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_HUB_DISABLE_TELEMETRY='1')


if __name__ == '__main__':
    disable_network()

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

from experiments import e60_correction
from pixelproof import retention_gate
from pixelproof.project_paths import DATA_ROOT, ML_ROOT

EVIDENCE = ML_ROOT.parent / 'evidence'
CONTRACT = EVIDENCE / 'e61_replay_contract.json'
RESULT = EVIDENCE / 'e61_replay_gate.json'
REFERENCE_SHA = 'a3aec445926bcc8707b3775f01d2cdd9491ba8495ad8a8ec306840556ca47390'
CANDIDATE_SHA = '582d6c4f020ce309c8e44a88383ef9d69c559773d6332694bd203e1132415bc5'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write_once(path, value):
    with path.open('x') as f:
        f.write(json.dumps(value, indent=2, sort_keys=True) + '\n')


def freeze():
    previous = read(DATA_ROOT / 'e60/correction_contract.json')
    if digest(DATA_ROOT / 'e60/correction_contract.json') != read(EVIDENCE / 'e60_correction_contract.json')['contract_sha256']:
        raise ValueError('E60 contract changed')
    paths = {'manifest': DATA_ROOT / 'e54/data_contract_v2.json',
             'features': DATA_ROOT / 'e54/teacher.npz',
             'reference': DATA_ROOT / 'e43/e43_small_predev.joblib',
             'candidate': DATA_ROOT / 'e60/correction.npz',
             'correction_code': Path(e60_correction.__file__),
             'gate_code': Path(retention_gate.__file__), 'runner_code': Path(__file__)}
    inputs = {key: {'path': str(path), 'sha256': digest(path)} for key, path in paths.items()}
    for key in ('manifest', 'features', 'reference', 'correction_code'):
        if inputs[key]['sha256'] != previous['inputs'][str(paths[key])]:
            raise ValueError(f'E60 binding changed: {key}')
    if inputs['reference']['sha256'] != REFERENCE_SHA or inputs['candidate']['sha256'] != CANDIDATE_SHA:
        raise ValueError('frozen reference/candidate changed')
    config = {'state': 'E61_TRAIN_gate_engineering_preregistered', 'inputs': inputs,
              'ai_cut': e60_correction.AI_CUT, 'parents': 11630, 'ai_parents': 4595,
              'conditions': ['clean', 'assigned_transport', 'q75'], 'cpu_threads': 2,
              'test': 'E43 against itself must pass. Already-frozen E60 replay must reproduce '
                      'its recorded TRAIN transition counts and fail on any newly missed AI view.',
              'expected_e60_misses_by_condition': {'clean': 1, 'assigned_transport': 1, 'q75': 3},
              'scope': 'Known-result engineering verification, NOT a blind experiment or model optimization.',
              'fit_allowed': False, 'evaluation_reads_allowed': False,
              'promotion_allowed': False, 'downloads': 0}
    write_once(CONTRACT, config)
    return {'state': config['state'], 'contract_sha256': digest(CONTRACT)}


def run():
    if RESULT.exists():
        raise FileExistsError(RESULT)
    config = read(CONTRACT)
    for item in config['inputs'].values():
        if digest(item['path']) != item['sha256']:
            raise ValueError(f"changed bound input: {item['path']}")
    paths = {k: Path(v['path']) for k, v in config['inputs'].items()}
    rows = read(paths['manifest'])['rows']
    if len(rows) != config['parents'] or sum(r['label'] == 1 for r in rows) != config['ai_parents']:
        raise ValueError('incomplete TRAIN replay manifest')
    with np.load(paths['features'], allow_pickle=False) as a:
        if str(a['binding']) != digest(paths['manifest']) or a['features'].shape != (11630, 3, 3072):
            raise ValueError('feature/manifest ordering binding changed')
        features = a['features'].reshape(-1, 3072)
    with np.load(paths['candidate'], allow_pickle=False) as a:
        if str(a['reference_sha256']) != REFERENCE_SHA:
            raise ValueError('candidate reference binding changed')
        weights = a['weights']
    with threadpool_limits(limits=config['cpu_threads']):
        head = joblib.load(paths['reference'])['head']
        if not np.array_equal(head[-1].classes_, [0, 1]):
            raise ValueError('AI class orientation changed')
        baseline = head.predict_proba(features)[:, 1].reshape(-1, 3)
        candidate = e60_correction.predict(head, features, weights).reshape(-1, 3)
    kwargs = {'conditions': config['conditions'], 'ai_cut': config['ai_cut']}
    control = retention_gate.check_train_retention(rows, baseline, baseline.copy(), **kwargs)
    result = retention_gate.check_train_retention(rows, baseline, candidate, **kwargs)
    if not control['passes_train_retention'] or result['passes_train_retention']:
        raise ValueError('retention gate failed the known control/candidate distinction')
    if result['misses_by_condition'] != config['expected_e60_misses_by_condition']:
        raise ValueError('frozen E60 TRAIN replay does not match archived outcome')
    if digest(paths['reference']) != REFERENCE_SHA or digest(paths['candidate']) != CANDIDATE_SHA:
        raise ValueError('reference/candidate mutated during replay')
    report = {'state': 'E61_engineering_check_complete', 'contract_sha256': digest(CONTRACT),
              'control_passed': control['passes_train_retention'], 'candidate': result,
              'source_downloads': 0, 'new_models_fit': 0, 'evaluation_rows_read': 0,
              'serving_changed': False, 'promotion_allowed': False}
    write_once(RESULT, report)
    return {k: v for k, v in report.items() if k != 'candidate'} | {
        'candidate_state': result['state'], 'misses_by_condition': result['misses_by_condition'],
        'new_ai_miss_parents': result['new_ai_miss_parents']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['freeze', 'run'])
    print(json.dumps({'freeze': freeze, 'run': run}[parser.parse_args().stage](), indent=2))
