"""Complete every missing TRAIN CLIP crop-role view, with immutable resumable chunks."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import socket
import time
if __name__ == '__main__':
    def denied(*a, **kw): raise RuntimeError('E112 extraction is offline')
    socket.socket.connect = denied; socket.socket.connect_ex = denied; socket.create_connection = denied
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
import numpy as np
from PIL import Image
from experiments.e65_acquisition import digest, read, write_once
from experiments.e71_features import load_encoder, array_sha, save_npz, BACKBONE, BACKBONE_SHA, read_chunk
from experiments.e72_acquisition import resource_check
from experiments.e75_features import crop_views
from experiments.e84_features import social_crops
from experiments.e104_context_audit import context_coordinates
from experiments.e110b_context_probe import population, CACHES
from pixelproof.project_paths import DATA_ROOT, ML_ROOT

ROOT = DATA_ROOT/'e112'; EVIDENCE = ML_ROOT.parent/'evidence'; CONTRACT = ROOT/'contract.json'


def pooled(raw):
    if raw.ndim != 3 or raw.shape[1:] != (3, 768) or raw.dtype != np.float32 or not np.isfinite(raw).all():
        raise ValueError('Finite ordered float32 CLIP crop vectors required')
    return np.concatenate([raw.mean(axis=1), raw.std(axis=1)], axis=1).astype(np.float32)


def verify_chunk(a, row, binding, expected):
    raw = a['raw']
    if str(a['binding']) != binding or str(a['parent']) != row['parent_id'] or \
            str(a['source_sha256']) != row['sha256'] or str(a['role']) != 'TRAIN' or \
            str(a['raw_sha256']) != array_sha(raw) or raw.shape != ((1 if row['cohort'] == 0 else 4), 3, 768):
        raise ValueError('Incomplete or mismatched raw chunk')
    error = float(np.abs(pooled(raw)-expected).max())
    if error > 1e-5: raise ValueError('Historical aggregate parity exceeds 1e-5')
    return raw, error


def caches():
    result = []
    for p in CACHES:
        with np.load(DATA_ROOT/p, allow_pickle=False) as a:
            result.append((list(a['parents']), a['clip'].copy()))
    return result


def expected_for(row, cache):
    cohort, i = row['cohort'], row['cohort_index']
    if cache[cohort][0][i] != row['parent_id']: raise ValueError('Cache parent order differs')
    if cohort == 0: return cache[0][1][i:i+1]
    expected = cache[cohort][1][i]
    if cohort == 1:
        j = 11630+i
        if cache[0][0][j] != row['parent_id']: raise ValueError('Social cache parent differs')
        expected = np.concatenate([expected, cache[0][1][j:j+1]])
    return expected


def freeze():
    probe_contract = DATA_ROOT/'e110b/contract.json'; probe = DATA_ROOT/'e110b/probe.json'
    if digest(probe_contract) != read(EVIDENCE/'e110b_context_contract.json')['contract_sha256'] or \
            digest(probe) != digest(EVIDENCE/'e110b_context_probe.json') or \
            not read(probe)['all_aggregate_parity_passed'] or read(probe)['views'] != 70:
        raise ValueError('Passing frozen complete E110B probe required')
    previous = read(probe_contract)
    for p, sha in previous['inputs'].items():
        if digest(p) != sha: raise ValueError('E110B bound input differs')
    if digest(BACKBONE) != BACKBONE_SHA: raise ValueError('Backbone differs')
    rows = population()
    if sum(1 if r['cohort'] == 0 else 4 for r in rows) != 15210:
        raise ValueError('Full missing-view population required')
    files = [Path(__file__), probe_contract, probe, DATA_ROOT/'e71/features_contract.json',
        DATA_ROOT/'e71/features.json', DATA_ROOT/'e71/legacy_inventory.json',
        DATA_ROOT/'e106/contract.json', DATA_ROOT/'e106/audit.json',
        ML_ROOT/'experiments/e65_acquisition.py', ML_ROOT/'experiments/e72_acquisition.py']
    inputs = previous['inputs'] | {str(p): digest(p) for p in files}
    c = {'state': 'E112_complete_missing_context_features_registered', 'inputs': inputs,
        'rows': rows, 'backbone_sha256': BACKBONE_SHA, 'parents': 12525, 'missing_views': 15210,
        'final_views': 50100, 'parity_max_abs': 1e-5, 'max_seconds_per_execution': 14400,
        'mps_limit_bytes': 6*1024**3,
        'operations': 'Identical E110B native source transforms and CLIP batch3; all legacy social and all4 newer cohort conditions. Verify every reconstructed aggregate, including on resume. Preserve all3 legacy raw views from SHA-bound E71 chunks.',
        'resume': 'Per-parent raw vectors + body/array/source/role/condition bindings, atomic NPZ and immutable receipt. No refills. Incomplete or mismatched chunks fail closed.',
        'outputs': 'Complete ordered raw[12525,4,3,768] and context[12525,4,1536] arrays. No fitting, DEV/final or gallery reads.',
        'fit_allowed': False, 'model_scores': 0, 'downloads': 0}
    ROOT.mkdir(exist_ok=True); write_once(CONTRACT, c)
    write_once(EVIDENCE/'e112_context_contract.json', {k:v for k,v in c.items() if k != 'rows'} |
        {'contract_sha256': digest(CONTRACT)})
    return {'registered_parents': len(rows), 'missing_views': 15210}


def validate():
    c = read(CONTRACT)
    if digest(CONTRACT) != read(EVIDENCE/'e112_context_contract.json')['contract_sha256'] or digest(BACKBONE) != c['backbone_sha256']:
        raise ValueError('E112 contract/backbone differs')
    for p, sha in c['inputs'].items():
        if digest(p) != sha: raise ValueError('E112 frozen input differs: '+p)
    return c


def extract():
    import torch
    c = validate(); binding = digest(CONTRACT); rows = c['rows']
    if (ROOT/'report.json').exists(): raise FileExistsError('Extraction already complete')
    start = time.monotonic(); deadline = start+c['max_seconds_per_execution']
    resource_check(deadline); torch.set_num_threads(2)
    cache = caches(); model, preprocess, device = load_encoder(); peak = 0; created = 0; resumed = 0; max_error = 0.
    folder = ROOT/'chunks'; folder.mkdir(exist_ok=True)
    raw_all = np.empty((len(rows), 4, 3, 768), dtype=np.float32)
    old_c = read(DATA_ROOT/'e71/features_contract.json'); old_report = read(DATA_ROOT/'e71/features.json')
    old_index = read(DATA_ROOT/'e71/legacy_inventory.json')['cached']; receipts = {}
    with torch.inference_mode():
        for i, row in enumerate(rows):
            resource_check(deadline)
            if digest(row['path']) != row['sha256']: raise ValueError('TRAIN source body changed')
            expected = expected_for(row, cache); path = folder/f'{i:05d}.npz'; receipt = path.with_suffix('.json')
            if path.exists() and not receipt.exists():
                raise ValueError('Unreceipted chunk; preserve for explicit integrity review')
            if path.exists():
                info = read(receipt)
                if info['binding'] != binding or info['sha256'] != digest(path): raise ValueError('Chunk receipt differs')
                with np.load(path, allow_pickle=False) as a: raw, error = verify_chunk(a, row, binding, expected)
                resumed += 1
            else:
                body = Path(row['path']).read_bytes(); social, transport_sha = social_crops(body)
                crops = social[None] if row['cohort'] == 0 else np.concatenate([crop_views(body, row['parent_id']), social[None]])
                raw = np.stack([model.encode_image(torch.stack([preprocess(Image.fromarray(crop))
                    for crop in view]).to(device)).float().cpu().numpy() for view in crops]).astype(np.float32)
                a = {'raw': raw, 'raw_sha256': array_sha(raw), 'binding': binding, 'parent': row['parent_id'],
                     'source_sha256': row['sha256'], 'role': 'TRAIN', 'social_sha256': transport_sha,
                     'crops_sha256': hashlib.sha256(crops.tobytes()).hexdigest()}
                raw, error = verify_chunk(a, row, binding, expected)
                save_npz(path, **a)
                write_once(receipt, {'binding': binding, 'sha256': digest(path), 'parent': row['parent_id']})
                created += 1
            max_error = max(max_error, error); receipts[str(path)] = digest(path)
            if row['cohort'] == 0:
                if str(i) in old_index:
                    old_path = Path(old_index[str(i)]['path']); old_binding = old_c['legacy_contract_sha256']
                else:
                    old_path = DATA_ROOT/'e71/chunks'/f'{i:05d}.npz'; old_binding = digest(DATA_ROOT/'e71/features_contract.json')
                old_raw, _ = read_chunk(old_path, old_binding, row['parent_id'], old_report['chunks'][str(old_path)])
                raw_all[i, :3] = old_raw; raw_all[i, 3:] = raw
            else:
                raw_all[i] = raw
            peak = max(peak, torch.mps.driver_allocated_memory() if device.type == 'mps' else 0)
            if peak > c['mps_limit_bytes']: raise RuntimeError('MPS memory ceiling reached')
            if (i+1) % 100 == 0:
                progress = {'E112_completed_parents': i+1, 'total_parents': len(rows), 'created': created,
                    'resumed': resumed, 'seconds': round(time.monotonic()-start), 'max_aggregate_error': max_error}
                (ROOT/'progress.json').write_text(json.dumps(progress, indent=2)+'\n')
                print(json.dumps(progress), flush=True)
    context = context_coordinates(raw_all.reshape(-1, 3, 768)).reshape(len(rows), 4, 1536)
    output = ROOT/'features.npz'
    if output.exists(): raise FileExistsError('Complete output exists without report; explicit integrity review required')
    save_npz(output, raw=raw_all, raw_sha256=array_sha(raw_all), context=context,
        context_sha256=hashlib.sha256(context.tobytes()).hexdigest(), binding=binding,
        parents=np.array([r['parent_id'] for r in rows]), roles=np.array(['TRAIN']*len(rows)),
        conditions=np.array(['clean', 'assigned_transport', 'q75', 'social_q75']))
    detail = {'chunks': receipts, 'contract_sha256': binding}; write_once(ROOT/'chunks.json', detail)
    r = {'state': 'E112_complete_context_features_ready', 'contract_sha256': binding,
        'feature_sha256': digest(output), 'chunks_sha256': digest(ROOT/'chunks.json'),
        'parents': len(rows), 'views': 50100, 'missing_views_completed': 15210,
        'created_parent_chunks': created, 'resumed_parent_chunks': resumed,
        'max_aggregate_error': max_error, 'mps_peak_bytes': peak, 'seconds_this_execution': time.monotonic()-start,
        'model_scores': 0, 'dev_final_reads': 0, 'downloads': 0, 'fit_allowed': False,
        'next': 'A separate frozen complete-population paired context/pooled representation fit is required.'}
    write_once(ROOT/'report.json', r); write_once(EVIDENCE/'e112_context_features.json', r)
    return r


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('stage', choices=['freeze', 'extract'])
    ROOT.mkdir(exist_ok=True)
    with (ROOT/'execution.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        print(json.dumps({'freeze': freeze, 'extract': extract}[p.parse_args().stage](), indent=2))
