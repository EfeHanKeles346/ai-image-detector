"""Native MIDD image admission, never model scoring or calibration."""
import argparse
from collections import Counter
import fcntl
import hashlib
import json
from pathlib import Path
import re
import shutil
import socket
import time

import numpy as np
import PIL
from PIL import Image

from experiments.e65_acquisition import digest, read, write_once
from experiments.e65_audit import fingerprint, cross_role_matches, VERSION
from experiments.e72_acquisition import resource_check
from experiments.e72_audit import resolve
from experiments.e99_acquisition import validate as validate_download, ROOT as SOURCE
from pixelproof.project_paths import DATA_ROOT, ML_ROOT

ROOT = DATA_ROOT / 'e100'
EVIDENCE = ML_ROOT.parent / 'evidence'
CONTRACT = ROOT / 'audit_contract.json'


def references(c):
    rows = []
    for path in c['reference_files']:
        doc = read(path)
        rows.extend(doc['rows'] if path.endswith('development_manifest.json') else doc['records'])
    if any(r['version'] != VERSION for r in rows):
        raise ValueError('Fingerprint convention changed')
    return rows


def freeze():
    validate_download()
    receipt = SOURCE/'download.json'
    if digest(receipt) != read(EVIDENCE/'e99_download.json')['receipt_sha256']:
        raise ValueError('Download receipt changed')
    result = read(receipt)
    if result['images'] != 256 or result['training_allowed'] or result['model_scores']:
        raise ValueError('Complete unscored quarantine required')
    prior = DATA_ROOT/'e88/audit_contract.json'; c = read(prior)
    for path in c['reference_files']:
        if digest(path) != c['inputs'][path]:
            raise ValueError('Protected snapshot changed')
    sid = DATA_ROOT/'e88/audit.json'
    if digest(sid) != read(EVIDENCE/'e88_audit.json')['report_sha256']:
        raise ValueError('SID reference changed')
    ref_files = c['reference_files'] + [str(sid)]
    gallery = ML_ROOT/'work/e95_owner_gallery/contract.json'
    if digest(gallery) != '4753e1877ac8b89a9ec9440ee396eae5807b9198dce7040aa94e99f72dcb85a0':
        raise ValueError('Gallery identities changed')
    # Only identities used. No gallery predictions are consulted for selection.
    gallery_sha = {r['sha256'] for r in read(gallery)['files']}
    refs = references({'reference_files': ref_files})
    if len(gallery_sha) != 206 or not gallery_sha <= {r['sha256'] for r in refs}:
        raise ValueError('Complete gallery reference coverage required')
    paths = [Path(__file__), Path(__file__).with_name('e99_acquisition.py'),
        Path(__file__).with_name('e72_audit.py'), Path(__file__).with_name('e65_audit.py'),
        Path(__file__).with_name('e65_acquisition.py'), prior, receipt, gallery] + list(map(Path, ref_files))
    contract = {'state': 'E100_native_admission_registered', 'inputs': {str(p): digest(p) for p in paths},
        'reference_files': ref_files, 'reference_records': len(refs), 'gallery_unique_covered': len(gallery_sha),
        'pillow_version': PIL.__version__, 'numpy_version': np.__version__, 'max_pixels': 100000000,
        'max_seconds': 3600, 'policy': 'Native E65 fingerprint;224px floor; body/RGB or dHash<=4 AND pHash63<=4 '
            'protected overlap exclusion. Internal transitive perceptual and sensor/capture-second groups, '
            'frozen E72 deterministic representative selection. No score/brightness filter or refill.',
        'limits': 'Perceptual/time grouping is not proof of independent scenes. MIDD remains research '
            'TRAIN only; originals are publisher JPEGs, not RAW. No CAL/test/serving promotion.',
        'license': 'CC BY-NC-SA4.0 academic/research only', 'model_scores': 0}
    write_once(CONTRACT, contract)
    write_once(EVIDENCE/'e100_audit_contract.json', {k: v for k, v in contract.items() if k not in ['inputs', 'reference_files']} |
               {'contract_sha256': digest(CONTRACT)})
    return {'state': contract['state'], 'reference_records': len(refs), 'gallery_unique_covered': len(gallery_sha)}


def audit():
    validate_download(); c = read(CONTRACT)
    if digest(CONTRACT) != read(EVIDENCE/'e100_audit_contract.json')['contract_sha256']:
        raise ValueError('Audit contract changed')
    if PIL.__version__ != c['pillow_version'] or np.__version__ != c['numpy_version']:
        raise ValueError('Decoder runtime changed')
    for path, sha in c['inputs'].items():
        if digest(path) != sha:
            raise ValueError('Bound audit input changed')
    if (ROOT/'audit.json').exists() or (ROOT/'training_manifest.json').exists():
        raise FileExistsError('Audit already recorded')
    start = time.monotonic(); deadline = start+c['max_seconds']; facts = []; failures = []
    for i, row in enumerate(read(SOURCE/'download.json')['rows']):
        resource_check(deadline)
        if shutil.disk_usage(ROOT).free < 30*1024**3:
            raise RuntimeError('30GiB reserve reached')
        raw = Path(row['path']).read_bytes()
        if len(raw) != row['bytes'] or hashlib.sha256(raw).hexdigest() != row['sha256']:
            raise ValueError('Acquired image changed')
        try:
            f = fingerprint(raw, max_pixels=c['max_pixels'])
            if min(f['width'], f['height']) < 224:
                raise ValueError('Image below224px')
            with Image.open(row['path']) as image:
                exif = image.getexif(); nested = exif.get_ifd(34665) if 34665 in exif else {}
                stamp = str(nested.get(36867, exif.get(36867, '')))
                detail = {str(k): str(exif[k]) for k in [271, 272, 305] if k in exif}
                if 34855 in nested:
                    detail['ISO'] = str(nested[34855])
            stamp = stamp if re.fullmatch(r'\d{4}:\d{2}:\d{2} \d{2}:\d{2}:\d{2}', stamp) else None
            facts.append(row | f | {'capture_second': stamp, 'exif': detail})
        except (ValueError, OSError, Image.DecompressionBombError) as error:
            failures.append({'parent_id': row['parent_id'], 'reason': str(error)})
        if (i+1) % 16 == 0:
            print(json.dumps({'processed': i+1, 'decoded': len(facts), 'failed': len(failures)}), flush=True)
    refs = references(c)
    cross = cross_role_matches(facts, refs)
    internal = [m for m in cross_role_matches(facts, facts) if m['train_parent'] < m['cal_parent']]
    selection = resolve(facts, internal, {r['parent_id'] for r in failures} | {m['train_parent'] for m in cross})
    keep = set(selection['kept'])
    admitted = [r | {'role': 'TRAIN', 'training_allowed': True, 'scene_group': selection['lineage'][r['parent_id']],
                    'scene_independence_verified': False} for r in facts if r['parent_id'] in keep]
    report = {'state': 'E100_native_TRAIN_audit_complete', 'contract_sha256': digest(CONTRACT), 'records': admitted,
        'selected': 256, 'admitted': len(admitted), 'sensor_counts': dict(Counter(r['sensor'] for r in admitted)),
        'decoded': len(facts), 'decode_failures': failures, 'reference_records': len(refs),
        'cross_matches': cross, 'internal_matches': internal, 'selection': selection,
        'seconds': time.monotonic()-start, 'model_scores': 0, 'limits': c['limits']}
    write_once(ROOT/'audit.json', report)
    manifest = {'state': 'E100_MIDD_additional_research_TRAIN', 'rows': admitted, 'publisher': 'MIDD',
        'whole_publisher_role': 'RESEARCH_TRAIN_ONLY', 'audit_sha256': digest(ROOT/'audit.json'),
        'class_counts': {'0': len(admitted), '1': 0}, 'model_scores': 0,
        'independent_scenes_verified': False, 'license': c['license']}
    write_once(ROOT/'training_manifest.json', manifest)
    summary = {k: v for k, v in report.items() if k not in ['records', 'decode_failures', 'cross_matches', 'internal_matches', 'selection']}
    summary.update(decode_failures=len(failures), cross_match_observations=len(cross),
        internal_match_pairs=len(internal), quarantined=len(selection['quarantined']),
        duplicate_members_removed=len(selection['duplicate_members_removed']),
        report_sha256=digest(ROOT/'audit.json'), manifest_sha256=digest(ROOT/'training_manifest.json'))
    write_once(EVIDENCE/'e100_audit.json', summary)
    return summary


if __name__ == '__main__':
    def denied(*args, **kwargs):
        raise RuntimeError('E100 admission audit is offline')
    socket.socket.connect = denied; socket.socket.connect_ex = denied; socket.create_connection = denied
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['freeze', 'audit'])
    ROOT.mkdir(parents=True, exist_ok=True)
    with (ROOT/'execution.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        print(json.dumps({'freeze': freeze, 'audit': audit}[parser.parse_args().stage](), indent=2))
