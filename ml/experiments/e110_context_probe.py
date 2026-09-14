"""Offline, score-blind feasibility probe for missing crop-role CLIP TRAIN views."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import time
if __name__ == '__main__':
    def denied(*a, **kw): raise RuntimeError('E110 encoding is offline')
    socket.socket.connect = denied; socket.create_connection = denied; socket.socket.connect_ex = denied
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
import numpy as np
from PIL import Image
from experiments.e65_acquisition import digest, read, write_once
from experiments.e71_features import load_encoder, array_sha, save_npz, BACKBONE, BACKBONE_SHA
from experiments.e72_acquisition import resource_check
from experiments.e75_features import crop_views
from experiments.e84_features import social_crops
from experiments.e104_context_audit import context_coordinates
from pixelproof.project_paths import DATA_ROOT, ML_ROOT

ROOT = DATA_ROOT/'e110'; EVIDENCE = ML_ROOT.parent/'evidence'; CONTRACT = ROOT/'contract.json'
MANIFESTS = ['e54/data_contract_v2.json', 'e72/training_manifest.json',
             'e88/training_manifest.json', 'e100/training_manifest.json']
CACHES = ['e84b/social_features.npz', 'e75/midd_features.npz', 'e89/sid_features.npz', 'e101/midd_features.npz']


def population():
    parts = [read(DATA_ROOT/p)['rows'] for p in MANIFESTS]
    if [len(p) for p in parts] != [11630, 511, 128, 256]:
        raise ValueError('Complete four TRAIN cohorts required')
    rows = [r | {'cohort': j, 'cohort_index': i} for j, p in enumerate(parts) for i, r in enumerate(p)]
    if len({r['parent_id'] for r in rows}) != 12525 or any(r['role'].upper() != 'TRAIN' for r in rows):
        raise ValueError('Unique admitted TRAIN only')
    return rows


def select(rows):
    groups = sorted({(r['cohort'], r['label'], r['source']) for r in rows})
    return [min((r for r in rows if (r['cohort'], r['label'], r['source']) == group),
        key=lambda r: hashlib.sha256(('E110|'+r['parent_id']).encode()).digest()) for group in groups]


def freeze():
    from experiments.e103_fit import validate
    validate()
    if digest(BACKBONE) != BACKBONE_SHA:
        raise ValueError('CLIP backbone identity differs')
    rows = select(population())
    files = [Path(__file__), Path(load_encoder.__code__.co_filename), Path(crop_views.__code__.co_filename),
             Path(social_crops.__code__.co_filename), Path(context_coordinates.__code__.co_filename),
             ML_ROOT/'experiments/e42_features.py', ML_ROOT/'experiments/e65_diagnostic.py',
             ML_ROOT/'src/pixelproof/e32_candidate.py'] + [DATA_ROOT/p for p in MANIFESTS+CACHES]
    c = {'state': 'E110_missing_raw_TRAIN_probe_registered',
         'inputs': {str(p): digest(p) for p in files}, 'backbone_sha256': BACKBONE_SHA,
         'selected': rows, 'selection': 'One SHA256(E110|parent)-first TRAIN parent per cohort/label/source; legacy social only, newer cohorts all4 conditions.',
         'views': sum(1 if r['cohort'] == 0 else 4 for r in rows),
         'parity_max_abs': 1e-5, 'max_seconds': 1200, 'mps_limit_bytes': 6*1024**3,
         'operations': 'Exact frozen CLIP batch3 per image condition, RGB crops/preprocess unchanged. Save raw center/local vectors and verify reconstructed mean/std against historical aggregate.',
         'fit_allowed': False, 'model_scores': 0, 'dev_final_reads': 0,
         'next': 'If all cohort parity passes, separately register complete missing15210-view extraction with bound images and durable chunks; no subset classifier.'}
    ROOT.mkdir(exist_ok=True); write_once(CONTRACT, c)
    write_once(EVIDENCE/'e110_context_contract.json', {k:v for k,v in c.items() if k != 'selected'} |
        {'parents': len(rows), 'contract_sha256': digest(CONTRACT)})
    return {'parents': len(rows), 'views': c['views']}


def run():
    import torch
    c = read(CONTRACT)
    if digest(CONTRACT) != read(EVIDENCE/'e110_context_contract.json')['contract_sha256'] or digest(BACKBONE) != c['backbone_sha256']:
        raise ValueError('Probe/backbone identity differs')
    for p, sha in c['inputs'].items():
        if digest(p) != sha: raise ValueError('Probe input differs')
    torch.set_num_threads(2); start = time.monotonic(); deadline = start+c['max_seconds']
    resource_check(deadline)
    cache = []
    for p in CACHES:
        with np.load(DATA_ROOT/p, allow_pickle=False) as a:
            cache.append((list(a['parents']), a['clip'].copy()))
    model, preprocess, device = load_encoder(); records = []; raws = []; peak = 0
    with torch.inference_mode():
        for row in c['selected']:
            resource_check(deadline)
            if digest(row['path']) != row['sha256']: raise ValueError('TRAIN image differs')
            body = Path(row['path']).read_bytes(); social, transport_sha = social_crops(body)
            cohort, idx = row['cohort'], row['cohort_index']
            if cohort == 0:
                crops = social[None]; expected = cache[0][1][idx:idx+1]
                if cache[0][0][idx] != row['parent_id']: raise ValueError('Parent order differs')
            else:
                crops = np.concatenate([crop_views(body, row['parent_id']), social[None]])
                if cache[cohort][0][idx] != row['parent_id']: raise ValueError('Parent order differs')
                expected = cache[cohort][1][idx]
                if cohort == 1:
                    social_index = 11630+idx
                    if cache[0][0][social_index] != row['parent_id']: raise ValueError('Social parent differs')
                    expected = np.concatenate([expected, cache[0][1][social_index:social_index+1]])
            raw = np.stack([model.encode_image(torch.stack([preprocess(Image.fromarray(crop))
                for crop in view]).to(device)).float().cpu().numpy() for view in crops]).astype(np.float32)
            actual = np.concatenate([raw.mean(axis=1), raw.std(axis=1)], axis=1).astype(np.float32)
            error = float(np.abs(actual-expected).max()); context_coordinates(raw)
            peak = max(peak, torch.mps.driver_allocated_memory() if device.type == 'mps' else 0)
            records.append({'cohort': cohort, 'source': row['source'], 'views': len(raw),
                'aggregate_max_abs_error': error, 'raw_sha256': array_sha(raw),
                'crops_sha256': hashlib.sha256(crops.tobytes()).hexdigest(), 'transport_sha256': transport_sha})
            raws.append(raw)
            if peak > c['mps_limit_bytes']: raise RuntimeError('Probe GPU memory ceiling exceeded')
            print(json.dumps({'E110_parents': len(records), 'seconds': round(time.monotonic()-start), 'parity_error': error}), flush=True)
    raw = np.concatenate(raws)
    save_npz(ROOT/'raw.npz', raw=raw, raw_sha256=array_sha(raw), binding=digest(CONTRACT),
             parents=np.array([r['parent_id'] for r in c['selected']]), views=np.array([r['views'] for r in records]))
    elapsed = time.monotonic()-start
    result = {'state': 'E110_missing_raw_TRAIN_probe_complete', 'contract_sha256': digest(CONTRACT),
        'parents': len(records), 'views': len(raw), 'seconds_including_load': elapsed,
        'all_aggregate_parity_passed': all(r['aggregate_max_abs_error'] <= c['parity_max_abs'] for r in records),
        'max_aggregate_error': max(r['aggregate_max_abs_error'] for r in records),
        'mps_peak_bytes': peak, 'raw_cache_sha256': digest(ROOT/'raw.npz'), 'by_source': records,
        'rough_full_missing_view_seconds_including_amortized_load': elapsed/len(raw)*15210,
        'fit_allowed': False, 'model_scores': 0, 'dev_final_reads': 0,
        'limitations': 'Stratified feasibility subset, not a fitted model. Cost estimate is approximate and source-dependent; full population and old+new representations still required.'}
    write_once(ROOT/'probe.json', result); write_once(EVIDENCE/'e110_context_probe.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('stage', choices=['freeze', 'run'])
    print(json.dumps({'freeze': freeze, 'run': run}[parser.parse_args().stage](), indent=2))
