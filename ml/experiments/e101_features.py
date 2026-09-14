"""Four fixed encoder conditions for audited MIDD extension TRAIN originals; no new-image classifier scores."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import socket
import time
import numpy as np

from experiments.e65_acquisition import digest, read, write_once
from experiments.e71_features import array_sha, save_npz
from experiments.e72_acquisition import resource_check
from experiments.e75_features import crop_views
from experiments.e84_features import Encoders, social_crops, parity, WIDTHS
from experiments.e84b_features import validate as validate_encoders, population as parity_population
from experiments.e85_data import CONDITIONS
from experiments.e99_acquisition import validate as validate_acquisition
from pixelproof.project_paths import DATA_ROOT, ML_ROOT
AUDIT = DATA_ROOT / 'e100/audit.json'
MANIFEST = DATA_ROOT / 'e100/training_manifest.json'


def validate_audit():
    validate_acquisition()
    c = read(DATA_ROOT / 'e100/audit_contract.json')
    if digest(DATA_ROOT / 'e100/audit_contract.json') != read(ML_ROOT.parent / 'evidence/e100_audit_contract.json')['contract_sha256']:
        raise ValueError('E100 admission contract changed')
    for path, expected in c['inputs'].items():
        if digest(path) != expected:
            raise ValueError('E100 admission input changed')
    return c


def training_rows():
    return [r | {'original_sha256': r['sha256'], 'original_path': r['path']}
            for r in read(MANIFEST)['rows']]


ROOT = DATA_ROOT / 'e101'; EVIDENCE = ML_ROOT.parent / 'evidence'
CONTRACT = ROOT / 'features_contract.json'; REPORT = ROOT / 'features.json'; OUTPUT = ROOT / 'midd_features.npz'


def four_crops(body, parent):
    old = crop_views(body, parent); social, qsha = social_crops(body)
    if old.shape != (3, 3, 224, 224, 3) or social.shape != (3, 224, 224, 3):
        raise ValueError('exact old3/new1 crop layout required')
    packs = np.concatenate([old, social[None]], axis=0)
    if packs.dtype != np.uint8 or not np.array_equal(packs[:3], old):
        raise ValueError('old three image conditions changed')
    return packs, qsha


def check_chunk(a, rows, binding):
    if str(a['binding']) != binding or list(a['parents']) != [r['parent_id'] for r in rows] or \
            list(a['source_sha256']) != [r['sha256'] for r in rows] or \
            list(a['original_sha256']) != [r['original_sha256'] for r in rows] or \
            list(a['roles']) != ['TRAIN'] * len(rows) or list(a['conditions']) != CONDITIONS:
        raise ValueError('complete ordered MIDD extension TRAIN identities/conditions required')
    for field in ['social_sha256', 'crops_sha256']:
        if a[field].shape != (len(rows),) or any(len(v) != 64 for v in a[field]):
            raise ValueError('complete derived image/crop hashes required')
    features = {}
    for key, width in WIDTHS.items():
        value = a[key]
        if value.shape != (len(rows), 4, width) or value.dtype != np.float32 or \
                not np.isfinite(value).all() or str(a[key + '_sha256']) != array_sha(value):
            raise ValueError('finite four-condition feature bodies required')
        features[key] = value
    return features


def freeze():
    validate_audit(); previous = validate_encoders(); audit = read(AUDIT)
    receipt = read(EVIDENCE / 'e100_audit.json')
    if digest(AUDIT) != receipt['report_sha256'] or digest(MANIFEST) != receipt['manifest_sha256'] or \
            audit['admitted'] < 1 or audit['model_scores']:
        raise ValueError('complete audited MIDD extension cohort required')
    manifest = read(MANIFEST); rows = training_rows()
    if len(rows) != audit['admitted'] or len({r['parent_id'] for r in rows}) != len(rows) or \
            any(r['role'] != 'TRAIN' or r['label'] != 0 or not r['training_allowed'] for r in rows) or \
            manifest['whole_publisher_role'] != 'RESEARCH_TRAIN_ONLY':
        raise ValueError('exact new REAL TRAIN cohort required')
    files = [Path(__file__), Path(__file__).with_name('e75_features.py'), Path(__file__).with_name('e84_features.py'),
             Path(__file__).with_name('e84b_features.py'), Path(__file__).with_name('e85_data.py'),
             Path(__file__).with_name('e100_audit.py'), MANIFEST, AUDIT, DATA_ROOT / 'e100/audit_contract.json',
             DATA_ROOT / 'e84b/features_contract.json']
    c = {'state': 'E101_MIDD_extension_four_condition_TRAIN_features_registered', 'inputs': {str(p): digest(p) for p in files},
         'parents': len(rows), 'views': len(rows)*4, 'conditions': CONDITIONS, 'parity_parents': previous['parity_parents'],
         'max_seconds': 5400, 'mps_driver_limit_bytes': 6 * 1024**3, 'batch_parents': 2,
         'population': 'All E100-admitted original MIDD JPEG parents, four registered sensors; unchanged source '
                       'order/identity, no selection/refill or model scores. Whole publisher research TRAIN only.',
         'transforms': 'Exact E75 old clean/assigned_transport/source-Q75 crops plus E84 social1080->JPEG75 crops. '
                       'Preserve all3 old-condition arrays. Verify original JPEG identities on every '
                       'execution. No per-image transform choice; assigned transport hashes existing parent ID.',
         'encoders': 'Exact E84/E83 DINO3072, CLIP1536, DEAR1640 float32 operations. Two parents x4views '
                     'per fixed8-view batch; immutable parent chunks. Same34-source old-clean parity and fixed '
                     'two-cut/feature tolerances. Repeat first MIDD extension batch feature error<=1e-5. No MIDD extension classifier head.',
         'resource': '90min/6GiB ceiling for up to1024 views with frozen encoder operations. AC/20GiB reserve. '
                     'Start after the completed E92 comparison; no concurrent GPU extraction.',
         'resume': 'Verify complete parent/order/source/original-JPEG/condition/role/feature bindings. '
                   'Reuse only complete immutable2-parent chunks; final NPZ/report written once.',
         'next': 'This cache alone does not authorize a fit. After E101 completes, separately register '
                 'any justified data-coverage extension; preserve all old AI/REAL and transport guards.',
         'limits': read(DATA_ROOT / 'e100/audit_contract.json')['limits'],
         'downloads': 0, 'new_image_classifier_scores': 0, 'dev_final_image_reads': 0, 'fit_allowed': False}
    write_once(CONTRACT, c)
    write_once(EVIDENCE / 'e101_features_contract.json', c | {'contract_sha256': digest(CONTRACT)})
    return {'state': c['state'], 'parents': len(rows), 'views': len(rows)*4}


def validate():
    validate_audit(); validate_encoders(); c = read(CONTRACT)
    if digest(CONTRACT) != read(EVIDENCE / 'e101_features_contract.json')['contract_sha256']:
        raise ValueError('E101 contract changed')
    for p, sha in c['inputs'].items():
        if digest(p) != sha:
            raise ValueError('E101 input changed: ' + p)
    return c


def require_previous_comparison():
    for local, public in [('e92/fit.json', 'e92_fit.json'), ('e92/dev_report.json', 'e92_development.json')]:
        if digest(DATA_ROOT / local) != digest(EVIDENCE / public):
            raise ValueError('Complete unchanged E92 comparison required')


def extract():
    require_previous_comparison(); c = validate()
    if REPORT.exists():
        raise FileExistsError('E101 feature receipt already complete')
    started = time.monotonic(); deadline = started + c['max_seconds']; resource_check(deadline)
    rows = training_rows(); binding = digest(CONTRACT); folder = ROOT / 'chunks'; folder.mkdir(exist_ok=True)
    enc = Encoders(); check = parity(enc, parity_population(), c, deadline)
    if not check['passed']:
        result = {'state': 'E101_old_source_parity_failed', 'parity': check, 'feature_extraction_complete': False,
                  'contract_sha256': binding, 'new_image_classifier_scores': 0}
        write_once(REPORT, result); write_once(EVIDENCE / 'e101_features.json', result)
        return result
    peak = enc.memory(); parts = {k: [] for k in WIDTHS}; chunks = {}; created = 0; repeat_error = None
    for start in range(0, len(rows), c['batch_parents']):
        resource_check(deadline); batch = rows[start:start + c['batch_parents']]; path = folder / f'{start:05d}.npz'
        for row in batch:
            if digest(row['path']) != row['sha256'] or digest(row['original_path']) != row['original_sha256']:
                raise ValueError('audited original JPEG identity changed')
        if not path.exists() or start == 0:
            packs, transport = [], []
            for row in batch:
                crops, qsha = four_crops(Path(row['path']).read_bytes(), row['parent_id'])
                packs.append(crops); transport.append(qsha)
            flat = np.stack(packs).reshape(-1, 3, 224, 224, 3)
            actual = {k: v.reshape(len(batch), 4, WIDTHS[k]) for k, v in enc.encode(flat).items()}
            crop_hashes = [hashlib.sha256(v.tobytes()).hexdigest() for v in packs]
            if start == 0:
                repeated = enc.encode(flat)
                repeat_error = max(float(np.max(np.abs(v.reshape(repeated[k].shape) - repeated[k]))) for k, v in actual.items())
                if repeat_error > 1e-5:
                    raise ValueError('fixed MIDD extension feature repeat differs')
            if path.exists():
                with np.load(path, allow_pickle=False) as old:
                    previous = check_chunk(old, batch, binding)
                    if list(old['crops_sha256']) != crop_hashes or list(old['social_sha256']) != transport or \
                            any(not np.array_equal(previous[k], actual[k]) for k in WIDTHS):
                        raise ValueError('resumed first MIDD extension chunk replay differs')
            else:
                save_npz(path, **actual, **{k + '_sha256': np.array(array_sha(v)) for k, v in actual.items()},
                    parents=np.array([r['parent_id'] for r in batch]), source_sha256=np.array([r['sha256'] for r in batch]),
                    original_sha256=np.array([r['original_sha256'] for r in batch]), roles=np.repeat('TRAIN', len(batch)),
                    conditions=np.array(CONDITIONS), social_sha256=np.array(transport), crops_sha256=np.array(crop_hashes),
                    binding=np.array(binding))
                created += len(batch)
        with np.load(path, allow_pickle=False) as a:
            features = check_chunk(a, batch, binding)
        for key, values in features.items():
            parts[key].append(values)
        chunks[path.name] = digest(path); peak = max(peak, enc.memory())
        if peak > c['mps_driver_limit_bytes']:
            raise RuntimeError('E101 MPS budget exceeded')
        if (start + len(batch)) % 16 == 0:
            print(json.dumps({'E101_parents': start + len(batch), 'total': len(rows),
                              'seconds': round(time.monotonic() - started)}), flush=True)
    arrays = {k: np.concatenate(v) for k, v in parts.items()}
    if any(v.shape != (len(rows), 4, WIDTHS[k]) for k, v in arrays.items()):
        raise ValueError('complete MIDD extension four-condition feature archive required')
    if OUTPUT.exists():
        with np.load(OUTPUT, allow_pickle=False) as old:
            if str(old['binding']) != binding or list(old['parents']) != [r['parent_id'] for r in rows] or \
                    list(old['conditions']) != CONDITIONS or list(old['roles']) != ['TRAIN'] * len(rows) or \
                    any(not np.array_equal(old[k], arrays[k]) for k in WIDTHS):
                raise ValueError('existing complete MIDD extension feature archive differs')
    else:
        save_npz(OUTPUT, **arrays, parents=np.array([r['parent_id'] for r in rows]), roles=np.repeat('TRAIN', len(rows)),
                 conditions=np.array(CONDITIONS), binding=np.array(binding))
    result = {'state': 'E101_MIDD_extension_four_condition_TRAIN_features_complete', 'parents': len(rows), 'views': len(rows) * 4,
              'conditions': CONDITIONS, 'shapes': {k: list(v.shape) for k, v in arrays.items()}, 'chunks': chunks,
              'contract_sha256': binding, 'feature_sha256': digest(OUTPUT), 'parity': check,
              'first_batch_repeat_max_error': repeat_error, 'mps_driver_peak_bytes': peak,
              'created_parents_this_execution': created, 'seconds': time.monotonic() - started,
              'new_image_classifier_scores': 0, 'dev_final_image_reads': 0, 'fit_allowed': False}
    write_once(REPORT, result)
    write_once(EVIDENCE / 'e101_features.json', {k: v for k, v in result.items() if k != 'chunks'} |
               {'report_sha256': digest(REPORT), 'chunks': len(chunks)})
    return {k: v for k, v in result.items() if k != 'chunks'}


if __name__ == '__main__':
    def denied(*args, **kwargs):
        raise RuntimeError('E101 feature extraction offline')
    socket.socket.connect = denied; socket.socket.connect_ex = denied; socket.create_connection = denied
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('stage', choices=['freeze', 'extract'])
    ROOT.mkdir(exist_ok=True)
    with (ROOT / 'features_execution.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        print(json.dumps({'freeze': freeze, 'extract': extract}[parser.parse_args().stage](), indent=2))
