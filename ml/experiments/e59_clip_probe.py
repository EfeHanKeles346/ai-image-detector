"""Bounded cached-CLIP engineering probe; no detector head, fit or accuracy."""
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
from experiments.e54_data import load as data_load, CONTRACT as DATA_CONTRACT, INDEX
from experiments.e54_adapt import CropCache
from experiments.e57_data import safe
from experiments.e47_univfd_diagnostic import BACKBONE, BACKBONE_SHA256, CHECKOUT, CHECKOUT_COMMIT, _git_commit

CONTRACT = EVIDENCE / 'e59_clip_probe_contract.json'
RESULT = EVIDENCE / 'e59_clip_probe.json'


def select(rows, roles):
    chosen = []
    if len(rows) != len(roles):
        raise ValueError('role cardinality mismatch')
    for label in (0, 1):
        ids = [i for i, (r, role) in enumerate(zip(rows, roles, strict=True)) if role == 'FIT' and r['label'] == label]
        ids.sort(key=lambda i: hashlib.sha256(('E59_RESOURCE|' + rows[i]['parent_id']).encode()).hexdigest())
        if len(ids) < 4:
            raise ValueError('four FIT parents per class required')
        chosen.extend(ids[:4])
    return chosen


def freeze():
    safe(time.monotonic() + 900)
    data = data_load()
    if digest(BACKBONE) != BACKBONE_SHA256 or _git_commit(CHECKOUT) != CHECKOUT_COMMIT:
        raise ValueError('pinned backbone/checkout changed')
    paths = [BACKBONE, DATA_CONTRACT, INDEX, EVIDENCE / 'e47_univfd_acquisition.json',
             Path(__file__).with_name('e54_adapt.py'), Path(__file__).with_name('e54_data.py')]
    paths += sorted((CHECKOUT / 'models/clip').rglob('*.py'))
    paths += sorted(CHECKOUT.glob('LICENSE*'))
    if not list(CHECKOUT.glob('LICENSE*')):
        raise ValueError('licence missing')
    ids = select(data['rows'], data['folds'][0]['roles'])
    config = {'state': 'E59_resource_probe_frozen', 'code_sha256': digest(__file__),
              'inputs': {str(p): digest(p) for p in paths},
              'parents': [{'index': i, 'parent_id': data['rows'][i]['parent_id'], 'label': data['rows'][i]['label']} for i in ids],
              'views': 24, 'crops': 72, 'batch_crops': 3, 'replays': 2, 'tolerance': 1e-5,
              'policy': 'Unmodified cached RGB224, official CLIP normalization, raw 768-D image embeddings; no head.'}
    fixed_write(CONTRACT, config)
    return config


def worker():
    import types
    import torch
    from PIL import Image
    from packaging import version
    config = json.loads(CONTRACT.read_text())
    if digest(__file__) != config['code_sha256']:
        raise ValueError('probe code changed')
    for p, sha in config['inputs'].items():
        if digest(p) != sha:
            raise ValueError('probe input changed: ' + p)
    deadline = time.monotonic() + 900
    safe(deadline)
    data = data_load()
    if select(data['rows'], data['folds'][0]['roles']) != [r['index'] for r in config['parents']]:
        raise ValueError('selection changed')
    # Same image-only compatibility shims as the pinned E47 integration. No text inference.
    sys.modules.setdefault('pkg_resources', types.SimpleNamespace(packaging=types.SimpleNamespace(version=version)))
    sys.modules.setdefault('ftfy', types.SimpleNamespace(fix_text=lambda value: value))
    sys.path.insert(0, str(CHECKOUT))
    from models.clip import clip
    model, preprocess = clip.load(str(BACKBONE), device='cpu', jit=False)
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    cache = CropCache(data)
    ids = [r['index'] * 3 + v for r in config['parents'] for v in range(3)]
    def encode(view):
        rgb = cache.batch([view])
        x = torch.stack([preprocess(Image.fromarray(crop)) for crop in rgb]).to(device)
        out = model.encode_image(x).float().cpu().numpy()
        if out.shape != (3, 768) or not np.isfinite(out).all():
            raise ValueError('invalid image features')
        return out
    try:
        with torch.inference_mode():
            encode(ids[0])
            passes = []
            durations = []
            for _ in range(2):
                features = []
                start = time.monotonic()
                for view in ids:
                    safe(deadline)
                    features.append(encode(view))
                durations.append(time.monotonic() - start)
                passes.append(np.stack(features))
            error = float(np.abs(passes[0] - passes[1]).max())
            if error > config['tolerance']:
                raise ValueError('embedding replay exceeds frozen tolerance')
            memory = int(torch.mps.driver_allocated_memory()) if device.type == 'mps' else None
    finally:
        cache.pool.shutdown()
    value = {'state': 'E59_cached_CLIP_resource_probe_complete', 'contract_sha256': digest(CONTRACT),
             'parents': 8, 'views': 24, 'crops_per_pass': 72, 'passes': 2, 'device': str(device),
             'seconds_per_pass': durations, 'embedding_replay_max_error': error,
             'sampled_driver_bytes': memory, 'accuracy_measured': False, 'candidate_saved': False,
             'detector_scores': 0, 'downloads': 0, 'fits': 0,
             'limitations': ['Eight FIT parents only, not quality or full-dataset throughput proof.',
                             'Sampled driver allocation is not peak memory; includes cached model allocations.',
                             'No trained UnivFD head used; 768-D embeddings remain unsaved probe intermediates.']}
    fixed_write(RESULT, value)
    return value


def run():
    import fcntl
    deadline = time.monotonic() + 900
    safe(deadline)
    work = Path(__file__).resolve().parents[1] / 'work'
    with (work / 'e59_probe.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if RESULT.exists():
            raise FileExistsError('completed probe must not be repeated')
        child = subprocess.Popen([sys.executable, '-m', 'experiments.e59_clip_probe', 'worker'], start_new_session=True)
        try:
            while child.poll() is None:
                safe(deadline)
                time.sleep(2)
            if child.returncode:
                raise RuntimeError('CLIP probe failed; no further step authorized')
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
    parser.add_argument('phase', choices=('freeze', 'run', 'worker'))
    args = parser.parse_args()
    if args.phase == 'run':
        def stop(signum, frame):
            raise KeyboardInterrupt('probe guard stopped')
        signal.signal(signal.SIGTERM, stop)
    print(json.dumps(globals()[args.phase](), indent=2))
