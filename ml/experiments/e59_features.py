"""Resumable frozen CLIP features for existing TRAIN only; no fitting or scoring."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import numpy as np

from experiments.e53_offline import EVIDENCE, digest, fixed_write
from experiments.e54_data import load as data_load, CONTRACT as DATA_CONTRACT, INDEX, TEACHER
from experiments.e54_adapt import CropCache
from experiments.e57_data import safe
from experiments.e59_clip_probe import CONTRACT as PROBE_CONTRACT, RESULT as PROBE_RESULT
from experiments.e47_univfd_diagnostic import BACKBONE, BACKBONE_SHA256, CHECKOUT, CHECKOUT_COMMIT, _git_commit
from experiments.e42_features import _save_npz
from pixelproof.project_paths import DATA_ROOT

ROOT = DATA_ROOT / 'e59'
CONTRACT = ROOT / 'feature_contract.json'
FEATURES = ROOT / 'clip_features.npz'
RECEIPT = ROOT / 'features.json'


def aggregate(raw):
    raw = np.asarray(raw)
    if raw.shape[-2:] != (3, 768) or not np.isfinite(raw).all():
        raise ValueError('finite three-crop 768-D vectors required')
    return np.concatenate([raw.mean(axis=-2), raw.std(axis=-2, ddof=0)], axis=-1).astype(np.float32)


def array_sha(value):
    value = np.ascontiguousarray(value, dtype=np.float32)
    return hashlib.sha256(str(value.shape).encode() + value.tobytes()).hexdigest()


def read_chunk(path, binding, parent):
    with np.load(path, allow_pickle=False) as a:
        if str(a['binding']) != binding or str(a['parent_id']) != parent:
            raise ValueError('chunk role/parent binding changed')
        raw, features = a['raw'], a['features']
        if raw.shape != (3, 3, 768) or features.shape != (3, 1536):
            raise ValueError('chunk shape changed')
        if raw.dtype != np.float32 or features.dtype != np.float32:
            raise ValueError('chunk dtype changed')
        if array_sha(raw) != str(a['raw_sha256']) or array_sha(features) != str(a['features_sha256']):
            raise ValueError('chunk contents changed')
        if not np.array_equal(aggregate(raw), features):
            raise ValueError('chunk aggregation changed')
        return features


def freeze():
    safe(time.monotonic() + 3600)
    data = data_load()
    probe = json.loads(PROBE_RESULT.read_text())
    if probe['embedding_replay_max_error'] > 1e-5 or probe['contract_sha256'] != digest(PROBE_CONTRACT):
        raise ValueError('probe not verified')
    if digest(BACKBONE) != BACKBONE_SHA256 or _git_commit(CHECKOUT) != CHECKOUT_COMMIT:
        raise ValueError('cached encoder identity changed')
    if len(data['rows']) != 11630 or len({r['parent_id'] for r in data['rows']}) != 11630:
        raise ValueError('existing TRAIN population changed')
    paths = [DATA_CONTRACT, INDEX, TEACHER, PROBE_CONTRACT, PROBE_RESULT, BACKBONE,
             Path(__file__).with_name('e54_data.py'), Path(__file__).with_name('e54_adapt.py'),
             Path(__file__).with_name('e42_features.py'), Path(__file__).with_name('e57_data.py'),
             Path(__file__).with_name('e56_fivek_pilot.py'), Path(__file__).with_name('e47_univfd_diagnostic.py')]
    paths += sorted((CHECKOUT / 'models/clip').rglob('*.py'))
    paths += sorted(CHECKOUT.glob('LICENSE*'))
    value = {'state': 'E59_full_features_frozen_before_extraction', 'code_sha256': digest(__file__),
             'inputs': {str(p): digest(p) for p in paths}, 'parents': 11630, 'views': 34890,
             'parent_order_sha256': hashlib.sha256(json.dumps([r['parent_id'] for r in data['rows']]).encode()).hexdigest(),
             'encoder': 'CLIP ViT-L/14 cached pretrained image encoder, frozen float32 eval',
             'crop_preprocessing': 'Official CLIP transform applied to existing E54 RGB224 crop bytes',
             'aggregation': 'raw 768-D crop mean concatenated with population std, float32; no L2 normalization',
             'shape_per_parent': [3, 1536], 'raw_shape_per_parent': [3, 3, 768],
             'batch_crops': 3, 'replay_tolerance': 1e-5, 'download_bytes': 0,
             'training': {'arms': ['dino3072', 'clip1536', 'dino_clip4608'], 'folds': 3,
                          'C': .01, 'tol': 1e-8, 'max_iter': 1000, 'seed': 53, 'cpu_threads': 2,
                          'scaler': 'weighted StandardScaler fitted on FIT only', 'fit_dtype': 'float64',
                          'roles_and_loss_mass': 'unchanged E54 roles and original base FIT view mass; no FiveK',
                          'restriction': 'Separate implementation/input contract must freeze before any fit'}}
    fixed_write(CONTRACT, value)
    fixed_write(EVIDENCE / 'e59_feature_contract.json', value | {'contract_sha256': digest(CONTRACT)})
    return {'parents': value['parents'], 'views': value['views'], 'contract_sha256': digest(CONTRACT)}


def load():
    value = json.loads(CONTRACT.read_text())
    if digest(__file__) != value['code_sha256'] or digest(CONTRACT) != json.loads((EVIDENCE / 'e59_feature_contract.json').read_text())['contract_sha256']:
        raise ValueError('frozen feature protocol changed')
    for p, sha in value['inputs'].items():
        if digest(p) != sha:
            raise ValueError('bound input changed: ' + p)
    data = data_load()
    actual = hashlib.sha256(json.dumps([r['parent_id'] for r in data['rows']]).encode()).hexdigest()
    if actual != value['parent_order_sha256']:
        raise ValueError('TRAIN parent ordering changed')
    return data, value


def finalize(data, config, binding, deadline):
    parts, chunks = [], {}
    for i, row in enumerate(data['rows']):
        safe(deadline)
        path = ROOT / 'chunks' / f'{i:05d}.npz'
        parts.append(read_chunk(path, binding, row['parent_id']))
        chunks[str(path)] = digest(path)
    features = np.stack(parts)
    parents = np.asarray([r['parent_id'] for r in data['rows']])
    if FEATURES.exists():
        with np.load(FEATURES, allow_pickle=False) as a:
            if str(a['binding']) != binding or not np.array_equal(a['parents'], parents) or not np.array_equal(a['features'], features):
                raise ValueError('incompatible completed archive; no overwrite')
    else:
        _save_npz(FEATURES, {'binding': np.asarray(binding), 'parents': parents, 'features': features})
    safe(deadline)
    value = {'state': 'E59_CLIP_features_complete', 'contract_sha256': binding,
             'parents': len(parents), 'views': len(parents)*3, 'features_sha256': digest(FEATURES),
             'chunks': chunks, 'new_detector_scores': 0, 'fits': 0, 'downloads': 0}
    fixed_write(RECEIPT, value)
    fixed_write(EVIDENCE / 'e59_features.json', {k: v for k, v in value.items() if k != 'chunks'} | {'receipt_sha256': digest(RECEIPT)})
    return {'state': value['state'], 'parents': len(parents)}


def extract(minutes):
    import types
    import torch
    from PIL import Image
    from packaging import version
    deadline = time.monotonic() + minutes*60
    safe(deadline)
    data, config = load()
    binding = digest(CONTRACT)
    sys.modules.setdefault('pkg_resources', types.SimpleNamespace(packaging=types.SimpleNamespace(version=version)))
    sys.modules.setdefault('ftfy', types.SimpleNamespace(fix_text=lambda value: value))
    sys.path.insert(0, str(CHECKOUT))
    from models.clip import clip
    model, preprocess = clip.load(str(BACKBONE), device='cpu', jit=False)
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    model = model.to(device).float().eval()
    for p in model.parameters():
        p.requires_grad_(False)
    cache = CropCache(data)
    def encode(view):
        rgb = cache.batch([view])
        x = torch.stack([preprocess(Image.fromarray(crop)) for crop in rgb]).to(device)
        out = model.encode_image(x).float().cpu().numpy()
        if out.shape != (3, 768) or not np.isfinite(out).all():
            raise ValueError('invalid CLIP features')
        return out
    reused, created = 0, 0
    try:
        with torch.inference_mode():
            first = encode(0)
            parity = float(np.abs(first - encode(0)).max())
            if parity > config['replay_tolerance']:
                raise ValueError('encoder replay failed')
            print(json.dumps({'phase': 'E59_start', 'replay_error': parity, 'device': str(device)}), flush=True)
            for i, row in enumerate(data['rows']):
                safe(deadline)
                path = ROOT / 'chunks' / f'{i:05d}.npz'
                if path.exists():
                    read_chunk(path, binding, row['parent_id'])
                    reused += 1
                else:
                    raw = np.stack([encode(i*3+v) for v in range(3)]).astype(np.float32)
                    features = aggregate(raw)
                    safe(deadline)
                    _save_npz(path, {'binding': np.asarray(binding), 'parent_id': np.asarray(row['parent_id']),
                                     'raw': raw, 'features': features, 'raw_sha256': np.asarray(array_sha(raw)),
                                     'features_sha256': np.asarray(array_sha(features))})
                    read_chunk(path, binding, row['parent_id'])
                    created += 1
                if (i+1) % 100 == 0:
                    print(json.dumps({'phase': 'E59_features', 'parents': i+1, 'total': len(data['rows']),
                                      'new_this_run': created, 'reused': reused}), flush=True)
            return finalize(data, config, binding, deadline)
    finally:
        cache.pool.shutdown()


def run(minutes):
    import fcntl
    if not 1 <= minutes <= 60:
        raise ValueError('bounded run requires 1-60 minutes')
    deadline = time.monotonic() + minutes*60
    safe(deadline)
    work = Path(__file__).resolve().parents[1] / 'work'
    with (work / 'e59_features.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if RECEIPT.exists() and (EVIDENCE / 'e59_features.json').exists():
            raise FileExistsError('features complete; do not rerun extraction')
        log = work / ('e59_features_' + time.strftime('%Y%m%dT%H%M%S') + '.log')
        env = dict(os.environ, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
        with log.open('x') as stream:
            child = subprocess.Popen([sys.executable, '-m', 'experiments.e59_features', 'extract', '--minutes', str(minutes)],
                                     stdout=stream, stderr=subprocess.STDOUT, start_new_session=True, env=env)
            print(json.dumps({'worker_pid': child.pid, 'log': str(log), 'minutes': minutes}), flush=True)
            try:
                while child.poll() is None:
                    safe(deadline)
                    time.sleep(2)
                if child.returncode:
                    raise RuntimeError('feature worker stopped; inspect bound log before resuming')
            finally:
                if child.poll() is None:
                    os.killpg(child.pid, signal.SIGTERM)
                    try:
                        child.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid, signal.SIGKILL)
                        child.wait()
    return {'state': 'E59_guard_complete'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=('freeze', 'run', 'extract'))
    parser.add_argument('--minutes', type=int, default=60)
    args = parser.parse_args()
    if not 1 <= args.minutes <= 60:
        parser.error('minutes must be between 1 and 60')
    if args.phase == 'run':
        def stop(signum, frame):
            raise KeyboardInterrupt('feature guard interrupted')
        signal.signal(signal.SIGTERM, stop)
    value = freeze() if args.phase == 'freeze' else globals()[args.phase](args.minutes)
    print(json.dumps(value, indent=2))
