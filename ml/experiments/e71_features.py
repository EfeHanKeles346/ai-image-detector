"""E71 TRAIN CLIP cache completion; immutable E59 inputs stay read-only."""
from __future__ import annotations
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time

if __name__ == '__main__':
    def denied(*args, **kwargs): raise RuntimeError('E71 feature network disabled')
    socket.socket.connect = denied; socket.socket.connect_ex = denied; socket.create_connection = denied
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_HUB_DISABLE_TELEMETRY='1',
                      OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')

import numpy as np
from experiments.e65_acquisition import digest, read, write_once
from pixelproof.project_paths import DATA_ROOT, ML_ROOT

ROOT = DATA_ROOT / 'e71'
LEGACY = DATA_ROOT / 'e59'
EVIDENCE = ML_ROOT.parent / 'evidence'
CONTRACT = ROOT / 'features_contract.json'
INVENTORY = ROOT / 'legacy_inventory.json'
OUTPUT = ROOT / 'clip_features.npz'
REPORT = ROOT / 'features.json'
MANIFEST = DATA_ROOT / 'e54/data_contract_v2.json'
CROP_INDEX = DATA_ROOT / 'e54/crop_index.json'
CHECKOUT = ML_ROOT / 'external/UniversalFakeDetect'
BACKBONE = DATA_ROOT / 'e47/models/ViT-L-14.pt'
BACKBONE_SHA = 'b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836'


def resource_check(deadline):
    if time.monotonic() >= deadline: raise TimeoutError('E71 stage deadline reached; completed chunks preserved')
    if not Path('/Volumes/LaCie').is_mount() or shutil.disk_usage(ROOT).free < 20*1024**3:
        raise RuntimeError('external volume or20GiB reserve unavailable')
    if 'AC Power' not in subprocess.check_output(['pmset', '-g', 'batt'], text=True):
        raise RuntimeError('E71 requires AC power')


def aggregate(raw):
    raw = np.asarray(raw)
    if raw.shape != (3, 3, 768) or raw.dtype != np.float32 or not np.isfinite(raw).all():
        raise ValueError('finite float32 three-condition/three-crop CLIP vectors required')
    return np.concatenate([raw.mean(axis=-2), raw.std(axis=-2, ddof=0)], axis=-1).astype(np.float32)


def array_sha(value):
    value = np.ascontiguousarray(value, dtype=np.float32)
    return hashlib.sha256(str(value.shape).encode() + value.tobytes()).hexdigest()


def read_chunk(path, binding, parent, expected_sha=None):
    if expected_sha is not None and digest(path) != expected_sha: raise ValueError('frozen chunk body changed')
    with np.load(path, allow_pickle=False) as a:
        if str(a['binding']) != binding or str(a['parent_id']) != parent: raise ValueError('chunk parent/binding differs')
        raw, features = a['raw'], a['features']
        if features.shape != (3, 1536) or features.dtype != np.float32:
            raise ValueError('chunk aggregate shape/dtype differs')
        if array_sha(raw) != str(a['raw_sha256']) or array_sha(features) != str(a['features_sha256']):
            raise ValueError('chunk array digest differs')
        if not np.array_equal(aggregate(raw), features): raise ValueError('chunk aggregation differs')
        return raw, features


def check_population(rows):
    if len(rows) != 11630 or len({r['parent_id'] for r in rows}) != 11630 or \
            any(str(r['role']).upper() != 'TRAIN' for r in rows):
        raise ValueError('complete admitted TRAIN population required')
    if sum(r['label'] == 1 for r in rows) != 4595 or sum(r['label'] == 0 for r in rows) != 7035:
        raise ValueError('both full TRAIN classes required')


def choose_parity(rows, cached):
    available = [r for i, r in enumerate(rows) if str(i) in cached]
    return [min((r for r in available if r['source'] == source),
                key=lambda r: hashlib.sha256(('E71|' + r['parent_id']).encode()).hexdigest())['parent_id']
            for source in sorted({r['source'] for r in available})]


def freeze():
    if CONTRACT.exists() or INVENTORY.exists(): raise FileExistsError('E71 inventory/contract already frozen')
    deadline = time.monotonic() + 900; resource_check(deadline)
    legacy = read(LEGACY / 'feature_contract.json'); legacy_binding = digest(LEGACY / 'feature_contract.json')
    if legacy_binding != read(EVIDENCE / 'e59_feature_contract.json')['contract_sha256'] or \
            digest(Path(__file__).with_name('e59_features.py')) != legacy['code_sha256']:
        raise ValueError('legacy protocol changed')
    inputs = dict(legacy['inputs'])
    for p, sha in inputs.items():
        if digest(p) != sha: raise ValueError('legacy input changed: ' + p)
    if digest(BACKBONE) != BACKBONE_SHA: raise ValueError('cached CLIP identity changed')
    rows = read(MANIFEST)['rows']; check_population(rows)
    order = hashlib.sha256(json.dumps([r['parent_id'] for r in rows]).encode()).hexdigest()
    if order != legacy['parent_order_sha256']: raise ValueError('legacy TRAIN order differs')
    crop_index = read(CROP_INDEX)['records']; cached = {}
    for i, r in enumerate(rows):
        if i % 64 == 0: resource_check(deadline)
        p = LEGACY / 'chunks' / f'{i:05d}.npz'
        if p.exists():
            read_chunk(p, legacy_binding, r['parent_id'])
            crop = crop_index[r['parent_id']]
            if digest(crop['path']) != crop['sha256']: raise ValueError('legacy source crop changed')
            cached[str(i)] = {'path': str(p), 'sha256': digest(p), 'parent_id': r['parent_id']}
        if (i+1) % 500 == 0:
            print(json.dumps({'stage': 'E71_inventory', 'parents_checked': i+1, 'cached_verified': len(cached)}), flush=True)
    if len(cached) != 9599: raise ValueError('observed legacy inventory changed before freeze')
    inventory = {'legacy_contract_sha256': legacy_binding, 'cached': cached, 'source_crop_bodies_verified': len(cached)}
    write_once(INVENTORY, inventory)
    files = [Path(__file__), Path(__file__).with_name('e65_acquisition.py'), INVENTORY,
             LEGACY/'feature_contract.json', EVIDENCE/'e59_feature_contract.json',
             DATA_ROOT/'e70/dev_report.json', EVIDENCE/'e70_development.json']
    files += sorted((CHECKOUT/'models').rglob('*.py'))
    inputs.update({str(p): digest(p) for p in files})
    c = {'state': 'E71_TRAIN_CLIP_completion_registered', 'inputs': inputs, 'parents': 11630,
         'cached_parents': len(cached), 'missing_parents': 11630-len(cached), 'parent_order_sha256': order,
         'legacy_contract_sha256': legacy_binding, 'parity_parents': choose_parity(rows, cached),
         'parity_max_error': 1e-5, 'max_seconds': 3600, 'cpu_threads': 2,
         'features': 'Exact official CLIP preprocessing on E54 RGB224 crops, float32 frozen ViT-L/14, '
                     'batch3 crops; raw3x3x768 and mean/std3x1536. Reuse verified legacy chunks, '
                     'create only missing parents under E71. Re-encode cached source samples for parity.',
         'policy': 'All11630 admitted TRAIN parents in existing order, all4595 AI. No E59 writes, '
                   'old fits, handoff, automation, classifier score, DEV pixels or downloads. '
                   'E66 is consumed DEVELOPMENT, excluded from TRAIN/final.',
         'backbone_sha256': BACKBONE_SHA, 'candidate_fit_allowed': False, 'downloads': 0}
    write_once(CONTRACT, c)
    write_once(EVIDENCE/'e71_features_contract.json', c | {'contract_sha256': digest(CONTRACT)})
    return {'state': c['state'], 'cached': len(cached), 'missing': 11630-len(cached), 'parity_parents': len(c['parity_parents'])}


def validate():
    c = read(CONTRACT)
    if digest(CONTRACT) != read(EVIDENCE/'e71_features_contract.json')['contract_sha256']:
        raise ValueError('E71 feature contract changed')
    for p, sha in c['inputs'].items():
        if digest(p) != sha: raise ValueError('bound E71 input changed: ' + p)
    return c


def load_encoder():
    import types
    import torch
    from packaging import version
    sys.modules.setdefault('pkg_resources', types.SimpleNamespace(packaging=types.SimpleNamespace(version=version)))
    sys.modules.setdefault('ftfy', types.SimpleNamespace(fix_text=lambda value: value))
    sys.path.insert(0, str(CHECKOUT))
    from models.clip import clip
    model, preprocess = clip.load(str(BACKBONE), device='cpu', jit=False)
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    model = model.to(device).float().eval()
    for parameter in model.parameters(): parameter.requires_grad_(False)
    return model, preprocess, device


def load_crops(entry, data_binding):
    if digest(entry['path']) != entry['sha256']: raise ValueError('source crop body changed')
    with np.load(entry['path'], allow_pickle=False) as a:
        if str(a['binding']) != data_binding: raise ValueError('crop population binding differs')
        crops = a['crops']
    if crops.shape != (3, 3, 224, 224, 3) or crops.dtype != np.uint8: raise ValueError('invalid RGB crop tensor')
    return crops


def save_npz(path, **arrays):
    if path.exists(): raise FileExistsError('immutable array output exists')
    temporary = path.with_suffix('.npz.part')
    with temporary.open('wb') as f: np.savez_compressed(f, **arrays)
    temporary.replace(path)


def extract():
    c = validate()
    if REPORT.exists(): raise FileExistsError('E71 features already complete')
    import torch
    from PIL import Image
    torch.set_num_threads(c['cpu_threads']); started = time.monotonic(); deadline = started+c['max_seconds']
    resource_check(deadline)
    rows = read(MANIFEST)['rows']; check_population(rows); index = read(CROP_INDEX)['records']
    cached = read(INVENTORY)['cached']; binding = digest(CONTRACT); data_binding = digest(MANIFEST)
    model, preprocess, device = load_encoder()
    def encode(r):
        crops = load_crops(index[r['parent_id']], data_binding)
        raw = []
        for view in crops:
            tensor = torch.stack([preprocess(Image.fromarray(crop)) for crop in view]).to(device)
            raw.append(model.encode_image(tensor).float().cpu().numpy())
        return np.stack(raw).astype(np.float32)
    (ROOT/'chunks').mkdir(exist_ok=True); parts = []; chunks = {}; created = 0; max_error = 0.
    with torch.inference_mode():
        for i, r in enumerate(rows):
            if r['parent_id'] not in c['parity_parents']: continue
            resource_check(deadline)
            old, _ = read_chunk(cached[str(i)]['path'], c['legacy_contract_sha256'], r['parent_id'], cached[str(i)]['sha256'])
            new = encode(r); error = float(np.abs(old-new).max()); max_error = max(max_error, error)
            if error > c['parity_max_error']: raise ValueError('historical CLIP raw-vector parity failed')
        print(json.dumps({'stage': 'E71_legacy_parity_passed', 'parents': len(c['parity_parents']), 'max_error': max_error}), flush=True)
        for i, r in enumerate(rows):
            if i % 32 == 0: resource_check(deadline)
            if str(i) in cached:
                item = cached[str(i)]; p = Path(item['path'])
                _, features = read_chunk(p, c['legacy_contract_sha256'], r['parent_id'], item['sha256'])
            else:
                p = ROOT/'chunks'/f'{i:05d}.npz'
                if p.exists(): _, features = read_chunk(p, binding, r['parent_id'])
                else:
                    resource_check(deadline); raw = encode(r); features = aggregate(raw)
                    save_npz(p, binding=np.array(binding), parent_id=np.array(r['parent_id']), raw=raw,
                             features=features, raw_sha256=np.array(array_sha(raw)), features_sha256=np.array(array_sha(features)))
                    read_chunk(p, binding, r['parent_id']); created += 1
            parts.append(features); chunks[str(p)] = digest(p)
            if (i+1) % 100 == 0 or (str(i) not in cached and created % 25 == 0):
                print(json.dumps({'stage': 'E71_features', 'parents': i+1, 'total': len(rows),
                                  'created_this_run': created, 'seconds': round(time.monotonic()-started)}), flush=True)
    resource_check(deadline); features = np.stack(parts); parents = np.array([r['parent_id'] for r in rows])
    if OUTPUT.exists():
        with np.load(OUTPUT, allow_pickle=False) as a:
            if str(a['binding']) != binding or not np.array_equal(a['parents'], parents) or \
                    not np.array_equal(a['features'], features) or set(a['roles']) != {'TRAIN'}:
                raise ValueError('unreceipted feature archive differs')
    else: save_npz(OUTPUT, features=features, parents=parents, roles=np.array(['TRAIN']*len(rows)), binding=np.array(binding))
    result = {'state': 'E71_TRAIN_CLIP_features_complete', 'contract_sha256': binding,
              'feature_sha256': digest(OUTPUT), 'shape': list(features.shape), 'cached_parents': len(cached),
              'new_parent_chunks': len(rows)-len(cached), 'created_this_run': created,
              'parity_parents': len(c['parity_parents']), 'legacy_raw_max_error': max_error,
              'seconds': time.monotonic()-started, 'chunks': chunks, 'dev_or_final_rows_read': 0,
              'classifier_scores': 0, 'downloads': 0, 'legacy_outputs_modified': False}
    write_once(REPORT, result)
    write_once(EVIDENCE/'e71_features.json', {k:v for k,v in result.items() if k != 'chunks'} | {'report_sha256': digest(REPORT)})
    return {k:v for k,v in result.items() if k != 'chunks'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('stage', choices=('freeze', 'extract'))
    ROOT.mkdir(parents=True, exist_ok=True)
    with (ROOT/'execution.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        print(json.dumps({'freeze': freeze, 'extract': extract}[parser.parse_args().stage](), indent=2))
