"""Registered native parity, consumed DEV guard replay and REAL-only E65 diagnostic.

Run freeze, then run, then report. No fitting, acquisitions or E49 access.
"""
from __future__ import annotations
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import socket
import time

os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_HUB_DISABLE_TELEMETRY='1',
                  OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')

def denied(*args, **kwargs):
    raise RuntimeError('E93 outgoing network denied')
socket.socket.connect = denied
socket.socket.connect_ex = denied
socket.create_connection = denied

import numpy as np
from PIL import Image
from pixelproof.e92_demo import E92Engine, guarded_outcome, social_view, digest, AI_CUT, REAL_CUT, MANIFEST
from pixelproof.project_paths import DATA_ROOT, ML_ROOT

OUT = ML_ROOT.parent / 'evidence'
ROOT = DATA_ROOT / 'e93'
CONTRACT = OUT / 'e93_diagnostic_contract.json'
SCORES = ROOT / 'diagnostic_scores.json'


def read(path):
    return json.loads(path.read_text())


def write_once(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write('\n')


def freeze():
    dev = read(DATA_ROOT / 'e92/dev_scores.json')['rows']
    selected = []
    for source in sorted({r['source'] for r in dev}):
        selected.append(min((r for r in dev if r['source'] == source and r['condition'] == 'publisher_original'),
                            key=lambda r: (r['sha256'], r['parent_id']))['parent_id'])
    paths = [Path(__file__), ML_ROOT / 'src/pixelproof/e92_demo.py',
             ML_ROOT / 'src/pixelproof/research_serve.py', MANIFEST,
             DATA_ROOT / 'e92/dev_scores.json', DATA_ROOT / 'e65/audit.json',
             DATA_ROOT / 'e65/diagnostic_scores.json']
    c = {'schema_version': 1, 'state': 'E93_registered_before_native_scores',
         'inputs': {str(p): digest(p) for p in paths}, 'parity_parents': selected,
         'parity_tolerance': 1e-5, 'no_cut_changes': True,
         'diagnostic_parents': 83, 'diagnostic_views': 166, 'max_seconds': 3600,
         'guard_id': 'e92-stability-v1', 'rule': 'Both E92 >= AI cut: ai_signal; both E92 < REAL cut and both E43 < AI cut: no_clear_signal; else uncertain.',
         'ai_cut': AI_CUT, 'real_cut': REAL_CUT, 'fit_allowed': False, 'e49_read_allowed': False,
         'independent_final': False, 'new_acquisition_allowed': False,
         'population': 'All admitted E65 REAL-only DIAGNOSTIC_DEV_ONLY parents. E66 consumed paired guard replay. Native scores below API upload/geometry restrictions are reported separately from demo acceptance.'}
    write_once(CONTRACT, c)
    return {'state': c['state'], 'parity_parents': selected}


def verify():
    c = read(CONTRACT)
    for path, sha in c['inputs'].items():
        if digest(Path(path)) != sha:
            raise ValueError('E93 registered input changed: ' + path)
    return c


def replay_guard(rows, original_name):
    grouped = {}
    for row in rows:
        grouped.setdefault(row['parent_id'], {})[row['condition']] = row
    output = []
    for parent, pairs in sorted(grouped.items()):
        if set(pairs) != {original_name, 'social_q75'}:
            raise ValueError('Incomplete diagnostic pair')
        pair = [pairs[original_name], pairs['social_q75']]
        outcome, reason = guarded_outcome([r['score'] for r in pair], [r['reference_score'] for r in pair])
        output.append({'parent_id': parent, 'source': pair[0]['source'], 'label': pair[0]['label'],
                       'scene_group': pair[0].get('scene_group', 'unknown'), 'outcome': outcome, 'reason': reason})
    return output


def run():
    c = verify()
    if SCORES.exists():
        raise FileExistsError('E93 scores already locked')
    import subprocess
    power = subprocess.check_output(['pmset', '-g', 'batt'], text=True)
    if 'AC Power' not in power:
        raise RuntimeError('Native diagnostic requires AC power')
    start = time.monotonic()
    engine = E92Engine()
    load_seconds = time.monotonic() - start
    dev = read(DATA_ROOT / 'e92/dev_scores.json')['rows']
    parity, latencies = [], []
    for parent in c['parity_parents']:
        pair = sorted([r for r in dev if r['parent_id'] == parent], key=lambda r: r['condition'])
        if len(pair) != 2:
            raise ValueError('Missing parity pair')
        images = []
        for row in pair:
            if digest(Path(row['path'])) != row['sha256']:
                raise ValueError('Native parity image changed')
            with Image.open(row['path']) as im:
                images.append(im.copy())
        # Verify the demo's in-memory social convention against the old derivative.
        if not np.array_equal(np.asarray(social_view(images[0])), np.asarray(images[1].convert('RGB'))):
            raise ValueError('Social transport pixel parity failed')
        tick = time.monotonic()
        scores, ref = engine.score_views(images)
        latencies.append(time.monotonic() - tick)
        for index, row in enumerate(pair):
            error = max(abs(float(scores[index]) - row['score']), abs(float(ref[index]) - row['reference_score']))
            changes = any((float(value) >= cut) != (old >= cut)
                          for value, old in [(scores[index], row['score']), (ref[index], row['reference_score'])]
                          for cut in [AI_CUT, REAL_CUT])
            parity.append({'parent_id': parent, 'condition': row['condition'], 'max_error': error, 'cut_changes': changes})
            if error > c['parity_tolerance'] or changes:
                raise ValueError('Native parity failed; E65 scoring denied')
    write_once(OUT / 'e93_native_parity.json', {'state': 'passed', 'contract_sha256': digest(CONTRACT),
        'views': len(parity), 'results': parity, 'load_seconds': load_seconds,
        'pair_latency_seconds': latencies, 'device': str(engine.device),
        'latency_scope': '14 native views, first pair cold; includes crops/all encoders/heads, excludes HTTP/upload. Not production p95.'})
    old = read(DATA_ROOT / 'e65/diagnostic_scores.json')['rows']
    sources = read(DATA_ROOT / 'e65/audit.json')['records']
    if len(sources) != 83 or len({r['parent_id'] for r in sources}) != 83 or any(
            r['label'] != 0 or r['role'] != 'DIAGNOSTIC_DEV_ONLY' or r['training_allowed'] for r in sources):
        raise ValueError('E65 admitted diagnostic population changed')
    rows = []
    for i, parent in enumerate(sorted(sources, key=lambda r: r['parent_id'])):
        if time.monotonic() - start > c['max_seconds']:
            raise TimeoutError('E93 native diagnostic deadline exceeded')
        pair = sorted([r for r in old if r['parent_id'] == parent['parent_id']], key=lambda r: r['condition'])
        if len(pair) != 2:
            raise ValueError('E65 expected two views')
        images = []
        for row in pair:
            if digest(Path(row['path'])) != row['sha256']:
                raise ValueError('E65 diagnostic image changed')
            with Image.open(row['path']) as im:
                images.append(im.copy())
        scores, reference = engine.score_views(images)
        for j, row in enumerate(pair):
            if abs(float(reference[j]) - row['score']) > c['parity_tolerance'] or any(
                    (float(reference[j]) >= cut) != (row['score'] >= cut) for cut in [AI_CUT, REAL_CUT]):
                raise ValueError('E65 E43 native reference parity failed')
            rows.append({key: row[key] for key in ['parent_id', 'source', 'label', 'condition', 'camera',
                'scene_group', 'path', 'sha256', 'role']} | {'score': float(scores[j]),
                'reference_score': float(reference[j]), 'status': 'ok'})
        if (i + 1) % 10 == 0:
            print(json.dumps({'diagnostic_parents': i + 1, 'total': 83}), flush=True)
    if len(rows) != 166:
        raise ValueError('Incomplete E93 diagnostic')
    verify()
    write_once(SCORES, {'state': 'scores_locked_before_metrics', 'contract_sha256': digest(CONTRACT),
                        'rows': rows, 'seconds': time.monotonic() - start})
    write_once(OUT / 'e93_scores_receipt.json', {'scores_sha256': digest(SCORES), 'views': len(rows),
                                              'contract_sha256': digest(CONTRACT)})
    return {'state': 'E93_scores_locked', 'views': len(rows)}


def summarize(rows, original_name):
    guarded = replay_guard(rows, original_name)
    groups = {}
    for source in sorted({r['source'] for r in rows}):
        pair_rows = [r for r in guarded if r['source'] == source]
        condition_metrics = {}
        for condition in [original_name, 'social_q75']:
            part = [r for r in rows if r['source'] == source and r['condition'] == condition]
            condition_metrics[condition] = {'n': len(part),
                'e92_ai': sum(r['score'] >= AI_CUT for r in part),
                'e43_ai': sum(r['reference_score'] >= AI_CUT for r in part)}
        groups[source] = {'label': pair_rows[0]['label'], 'parents': len(pair_rows),
                         'guarded_outcomes': dict(Counter(r['outcome'] for r in pair_rows)),
                         'raw_conditions': condition_metrics}
    scenes = {}
    for row in guarded:
        if row['label'] == 0:
            key = row['scene_group']
            scenes.setdefault(key, Counter()).update([row['outcome']])
    return {'sources': groups, 'guarded_real_scenes': {k: dict(v) for k, v in scenes.items()},
            'outcomes_by_label': {str(label): dict(Counter(r['outcome'] for r in guarded if r['label'] == label)) for label in [0, 1]},
            'parents': len(guarded), 'guarded_rows': guarded}


def report():
    verify()
    if digest(SCORES) != read(OUT / 'e93_scores_receipt.json')['scores_sha256']:
        raise ValueError('Locked E93 scores changed')
    diagnostic = summarize(read(SCORES)['rows'], 'original')
    dev = summarize(read(DATA_ROOT / 'e92/dev_scores.json')['rows'], 'publisher_original')
    result = {'state': 'E93_complete', 'contract_sha256': digest(CONTRACT), 'scores_sha256': digest(SCORES),
              'native_parity': read(OUT / 'e93_native_parity.json'), 'consumed_e66': dev, 'e65_real_diagnostic': diagnostic,
              'independent_final': False, 'new_fit': False, 'e92_acceptance_changed': False,
              'limits': 'Guard evaluated for ORIGINAL submissions plus their Q75 view. Re-uploading an already compressed image causes another Q75 pass and is not covered by this cached replay. Native scientific populations include images too large for the API; these counts are not API acceptance or production accuracy. E65 has no AI; unseen-generator retention remains unknown. Stability is not OOD detection.'}
    write_once(OUT / 'e93_demo_validation.json', result)
    return {k: {key: val for key, val in v.items() if key != 'guarded_rows'} if isinstance(v, dict) else v
            for k, v in result.items() if k != 'native_parity'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['freeze', 'run', 'report'])
    print(json.dumps({'freeze': freeze, 'run': run, 'report': report}[parser.parse_args().stage](), indent=2))
