"""Fixed existing RAW rendering and protected-overlap audit for the128 E87 SID originals."""
import argparse
from collections import Counter
import fcntl
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import time

from experiments.e65_acquisition import digest, read, write_once
from experiments.e65_audit import fingerprint, cross_role_matches, VERSION, DECODER, PYTHON, decoder_versions
from experiments.e72_audit import components, validate as validate_prior
from experiments.e72_acquisition import resource_check
from experiments.e87_acquisition import validate as validate_acquisition, RECEIPT as DOWNLOAD
from pixelproof.project_paths import DATA_ROOT, ML_ROOT

ROOT = DATA_ROOT / 'e88'
EVIDENCE = ML_ROOT.parent / 'evidence'
CONTRACT = ROOT / 'audit_contract.json'
REPORT = ROOT / 'audit.json'
MANIFEST = ROOT / 'training_manifest.json'


def resolve(rows, matches, rejected):
    groups = components(rows, matches); kept, discarded, lineage = [], [], {}; bad = set(rejected)
    for group in groups:
        identity = 'SID:component:' + hashlib.sha256('|'.join(group).encode()).hexdigest()
        lineage.update({p: identity for p in group})
        if set(group) & bad:
            bad.update(group)
        else:
            chosen = min(group, key=lambda p: hashlib.sha256(('E88|' + p).encode()).hexdigest())
            kept.append(chosen); discarded.extend(p for p in group if p != chosen)
    return {'kept': sorted(kept), 'quarantined': sorted(bad), 'duplicate_members_removed': sorted(discarded),
            'components': groups, 'lineage': lineage}


def raw_body_matches(rows, refs):
    index = {}
    for ref in refs:
        for field in ['sha256', 'original_sha256']:
            if ref.get(field):
                index.setdefault(ref[field], set()).add(ref['parent_id'])
    return [{'train_parent': row['parent_id'], 'cal_parent': parent, 'match': 'original_RAW_body_sha256'}
            for row in rows for parent in sorted(index.get(row['sha256'], set()))]


def freeze():
    validate_acquisition(); old = validate_prior()
    if digest(DOWNLOAD) != read(EVIDENCE / 'e87_download.json')['receipt_sha256']:
        raise ValueError('complete verified RAW acquisition required')
    receipt = read(DOWNLOAD)
    if receipt['images'] != 128 or receipt['model_scores'] or receipt['RAW_decodes'] or receipt['training_allowed']:
        raise ValueError('exact128 unscored quarantined RAW originals required')
    prior = DATA_ROOT / 'e72/audit.json'
    if digest(prior) != read(EVIDENCE / 'e72_audit.json')['report_sha256']:
        raise ValueError('all admitted MIDD reference fingerprints required')
    reference_files = old['reference_files'] + [str(prior)]
    paths = [Path(__file__), Path(__file__).with_name('e87_acquisition.py'), Path(__file__).with_name('e72_audit.py'),
             Path(__file__).with_name('e65_audit.py'), Path(__file__).with_name('e65_acquisition.py'),
             DECODER, DOWNLOAD, DATA_ROOT / 'e72/audit_contract.json'] + [Path(p) for p in reference_files]
    versions = decoder_versions()
    if versions != read(DATA_ROOT / 'e65/audit_contract.json')['decoder_versions']:
        raise ValueError('reuse exact existing E65 isolated RAW runtime required')
    c = {'state': 'E88_SID_fixed_RAW_render_and_TRAIN_audit_registered', 'inputs': {str(p): digest(p) for p in paths},
         'reference_files': reference_files, 'decoder_versions': versions, 'fingerprint_version': VERSION,
         'selected': 128, 'max_pixels': 100000000, 'max_seconds': 3600, 'raw_timeout_seconds': 120,
         'rendering': 'Reuse frozen e56_raw_decode.py and its existing isolated environment exactly: full-size '
                      'as-shot white balance, AHD request (LibRaw may handle X-Trans internally), sRGB8, gamma2.4/12.92, '
                      'no auto brightness/auto WB, bright1, clip highlights, no median/FBDD noise reduction, PNG6. '
                      'No embedded JPEG preview, learned denoising or per-image brightness adjustment. This is our '
                      'existing RAW rendering convention, not reproduction of SID authors16-bit/default-gamma references.',
         'policy': 'Verify each original RAW length/SHA before decoding. Reject decode failure/below224px and '
                   'any original-RAW body, rendered body/RGB or dHash<=4 AND pHash63<=4 protected match. Include '
                   'complete E72-protected snapshot/AI reserves, consumed E66 and all511 MIDD admitted parents. '
                   'Internal perceptual components propagate rejection transitively; keep one SHA256(E88|parent_id) '
                   'representative per survivor. No score/brightness filter, quota refill or role reassignment of protected data.',
         'metadata': 'Only existing decoder numeric RGB/white-balance summaries and published camera/filename scene '
                     'identities. No GPS/serial extraction, source metadata upload or detector scores.',
         'limits': 'Incomplete scene detection; filename identities and perceptual screening cannot prove independent '
                   'scenes or distinguish all shared scenes across cameras. Older two-camera low-light coverage only. '
                   'Whole SID publisher research TRAIN only, including unselected/upstream validation/test; never fresh '
                   'DEV/final. Existing E84B/E85/E86 inputs stay unchanged. No guarantee of performance or AI retention.',
         'license': read(DATA_ROOT / 'e87/acquisition_contract.json')['license'],
         'role_after_audit': 'TRAIN', 'model_scoring_allowed': False, 'promotion_allowed': False}
    write_once(CONTRACT, c)
    write_once(EVIDENCE / 'e88_audit_contract.json', c | {'contract_sha256': digest(CONTRACT)})
    return {'state': c['state'], 'selected': 128}


def validate():
    validate_acquisition(); validate_prior(); c = read(CONTRACT)
    if digest(CONTRACT) != read(EVIDENCE / 'e88_audit_contract.json')['contract_sha256']:
        raise ValueError('E88 audit contract changed')
    for p, sha in c['inputs'].items():
        if digest(p) != sha:
            raise ValueError('E88 bound input changed: ' + p)
    if decoder_versions() != c['decoder_versions']:
        raise ValueError('isolated RAW runtime changed')
    return c


def audit():
    c = validate()
    if REPORT.exists() or MANIFEST.exists():
        raise FileExistsError('E88 audit already recorded')
    started = time.monotonic(); deadline = started + c['max_seconds']; rows = read(DOWNLOAD)['rows']
    facts, failed = [], []; (ROOT / 'decoded').mkdir(exist_ok=True)
    for index, row in enumerate(rows):
        resource_check(deadline)
        original = Path(row['path'])
        if original.stat().st_size != row['bytes'] or digest(original) != row['sha256']:
            raise ValueError('acquired RAW identity changed')
        target = ROOT / 'decoded' / (row['camera'] + '_' + original.stem + '.png')
        try:
            process = subprocess.run([str(PYTHON), str(DECODER), str(original), str(target)],
                text=True, capture_output=True, timeout=c['raw_timeout_seconds'],
                env=dict(os.environ, OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2'))
            if process.returncode:
                raise ValueError(process.stderr[-1500:])
            details = json.loads(process.stdout)
            f = fingerprint(target.read_bytes(), max_pixels=c['max_pixels'])
            if details['versions'] != c['decoder_versions'] or details['pixel_sha256'] != f['pixel_sha256'] or min(f['width'], f['height']) < 224:
                raise ValueError('rendered identity/version/dimensions differ')
            facts.append(row | {'original_path': str(original), 'original_sha256': row['sha256'],
                'path': str(target), 'label': 0, 'condition': 'original', 'source': 'SID:' + row['camera'],
                'sensor': row['camera'], 'capture_second': None, 'decode': details, 'decoded_format': 'PNG'} | f)
        except (ValueError, OSError, subprocess.TimeoutExpired) as error:
            failed.append({'parent_id': row['parent_id'], 'reason': str(error)})
        if (index + 1) % 8 == 0:
            print(json.dumps({'E88_decoded': len(facts), 'failed': len(failed), 'processed': index + 1,
                              'seconds': round(time.monotonic() - started)}), flush=True)
    refs = []
    for path in c['reference_files']:
        item = read(path); refs.extend(item['rows'] if path.endswith('development_manifest.json') else item['records'])
    if any(r['version'] != VERSION for r in refs):
        raise ValueError('reference fingerprint convention differs')
    cross = cross_role_matches(facts, refs); raw_cross = raw_body_matches(rows, refs)
    internal = [m for m in cross_role_matches(facts, facts) if m['train_parent'] < m['cal_parent']]
    rejected = {r['parent_id'] for r in failed} | {r['train_parent'] for r in cross + raw_cross}
    selection = resolve(facts, internal, rejected); keep = set(selection['kept'])
    admitted = [r | {'role': 'TRAIN', 'training_allowed': True, 'filename_scene_group': r['scene_group'],
                    'scene_group': selection['lineage'][r['parent_id']], 'scene_independence_verified': False}
                for r in facts if r['parent_id'] in keep]
    admitted.sort(key=lambda r: r['parent_id'])
    result = {'state': 'E88_SID_RAW_TRAIN_audit_complete', 'contract_sha256': digest(CONTRACT), 'records': admitted,
              'selected': 128, 'admitted': len(admitted), 'camera_counts': dict(Counter(r['camera'] for r in admitted)),
              'decode_failures': failed, 'reference_records': len(refs), 'cross_matches': cross, 'raw_cross_matches': raw_cross,
              'internal_matches': internal, 'selection': selection, 'seconds': time.monotonic() - started,
              'dimensions': dict(Counter(f"{r['width']}x{r['height']}" for r in facts)),
              'model_scores': 0, 'training_allowed': bool(admitted), 'independent_final_admitted': False, 'limits': c['limits']}
    write_once(REPORT, result)
    manifest = {'state': 'E88_SID_audited_research_TRAIN', 'rows': admitted, 'audit_sha256': digest(REPORT),
                'class_counts': {'0': len(admitted), '1': 0}, 'publisher': 'SID', 'license': c['license'],
                'whole_publisher_role': 'RESEARCH_TRAIN_ONLY', 'independent_scenes_verified': False, 'model_scores': 0}
    write_once(MANIFEST, manifest)
    summary = {k: v for k, v in result.items() if k not in ['records', 'cross_matches', 'raw_cross_matches', 'internal_matches', 'selection']}
    summary.update(report_sha256=digest(REPORT), manifest_sha256=digest(MANIFEST), cross_match_observations=len(cross),
                   raw_cross_match_observations=len(raw_cross), internal_match_pairs=len(internal),
                   quarantined=len(selection['quarantined']), duplicate_members_removed=len(selection['duplicate_members_removed']))
    write_once(EVIDENCE / 'e88_audit.json', summary)
    return summary


if __name__ == '__main__':
    def denied(*args, **kwargs):
        raise RuntimeError('E88 RAW audit is offline')
    socket.socket.connect = denied; socket.socket.connect_ex = denied; socket.create_connection = denied
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('stage', choices=['freeze', 'audit'])
    ROOT.mkdir(exist_ok=True)
    with (ROOT / 'audit_execution.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        print(json.dumps({'freeze': freeze, 'audit': audit}[parser.parse_args().stage](), indent=2))
