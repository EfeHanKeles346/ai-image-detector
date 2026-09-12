"""Small, score-blind WIFD/RawNIND acquisition with pinned metadata and restartable files."""
from __future__ import annotations
import argparse
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import json
from pathlib import Path
import re
import shutil
import time
from urllib.parse import quote

import requests
import yaml
from pixelproof.project_paths import DATA_ROOT, ML_ROOT

ROOT = DATA_ROOT / 'e65'
EVIDENCE = ML_ROOT.parent / 'evidence'
CONTRACT = ROOT / 'acquisition_contract.json'
RECEIPT = ROOT / 'download.json'
REVISION = '3f577edf0b14c686aa08e8d0d8ae07a83ba44f26'


def digest(path, algorithm='sha256', prefix=b''):
    h = hashlib.new(algorithm); h.update(prefix)
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write_once(path, value):
    with Path(path).open('x') as f:
        f.write(json.dumps(value, indent=2, sort_keys=True) + '\n')


def ranked(values, key):
    return sorted(values, key=lambda r: hashlib.sha256(('E65|' + key(r)).encode()).hexdigest())


def choose_wifd(tree):
    if tree.get('truncated') or tree.get('sha') != REVISION:
        raise ValueError('complete pinned WIFD tree required')
    groups = defaultdict(lambda: defaultdict(list))
    for item in tree['tree']:
        path = item['path']
        if item['type'] != 'blob' or '/sdr_image/' not in path or not path.lower().endswith('.jpg'):
            continue
        iso = int(re.search(r'_ISO(\d+)_', path).group(1))
        groups[path.split('/')[1]][iso].append(item)
    selected = []
    for camera, by_iso in sorted(groups.items()):
        for iso in sorted({min(by_iso), max(by_iso)}):
            # Sequence numbers are not scene identities; avoid repeated setting tuples.
            settings = {}
            for item in ranked(by_iso[iso], lambda r: r['path']):
                setting = re.sub(r'_\d+\.jpg$', '', item['path'], flags=re.I)
                settings.setdefault(setting, item)
            for item in ranked(list(settings.values()), lambda r: r['path'])[:4]:
                selected.append({'source': 'WIFD', 'camera': camera, 'iso': iso,
                    'name': Path(item['path']).name, 'remote_path': item['path'],
                    'url': f"https://raw.githubusercontent.com/CSCRC-SCREED/WIFD/{REVISION}/{quote(item['path'])}",
                    'bytes': item['size'], 'checksum_type': 'git_blob_sha1', 'checksum': item['sha'],
                    'license': 'MIT', 'condition': 'publisher_sdr_jpeg',
                    'scene_group': 'WIFD:unknown_sdr_scenes_no_independence_claim',
                    'label': 0, 'role': 'DIAGNOSTIC_DEV_ONLY', 'training_allowed': False})
    return selected


def choose_rawnind(metadata, dataset, permissions, reserves):
    files = {r['dataFile']['filename']: r for r in metadata['data']['latestVersion']['files']}
    rights = dict(line.split(': ', 1) for line in permissions.splitlines() if ': ' in line)
    blocked = set(reserves['test_reserve'])
    selected = []
    for cfa in ('Bayer', 'X-Trans'):
        eligible = []
        for scene, info in dataset[cfa].items():
            if info.get('test_reserve') or scene in blocked or not rights.get(scene, '').startswith('CC0,'):
                continue
            clean = [r for r in info['clean_images'] if r['filename'] in files
                     and re.search(r'_ISO(\d+)_', r['filename'])]
            if not clean:
                continue
            clean = ranked(clean, lambda r: r['filename'])[0]
            clean_iso = int(re.search(r'_ISO(\d+)_', clean['filename']).group(1))
            noisy = [r for r in info['noisy_images'] if r['filename'] in files
                     and re.search(r'_ISO(\d+)_', r['filename']) and
                     clean_iso < int(re.search(r'_ISO(\d+)_', r['filename']).group(1)) <= 6400]
            if not noisy:
                continue
            high = max(int(re.search(r'_ISO(\d+)_', r['filename']).group(1)) for r in noisy)
            noisy = ranked([r for r in noisy if f'_ISO{high}_' in r['filename']], lambda r: r['filename'])[0]
            if any(files[r['filename']].get('restricted') for r in (clean, noisy)):
                continue
            eligible.append((scene, clean, noisy))
        for scene, clean, noisy in ranked(eligible, lambda r: cfa + ':' + r[0])[:4]:
            for condition, item in (('low_iso_raw', clean), ('high_iso_raw', noisy)):
                f = files[item['filename']]['dataFile']
                if f['checksum']['type'] != 'MD5':
                    raise ValueError('unexpected RawNIND checksum convention')
                selected.append({'source': 'RawNIND', 'camera': 'pending_RAW_metadata', 'cfa': cfa,
                    'name': f['filename'], 'scene_group': f'RawNIND:{scene}', 'condition': condition,
                    'url': f"https://dataverse.uclouvain.be/api/access/datafile/{f['id']}",
                    'bytes': f['filesize'], 'checksum_type': 'md5', 'checksum': f['checksum']['value'],
                    'body_sha1': item['sha1'], 'license': rights[scene],
                    'dataset_license': 'CC-BY-SA-4.0', 'label': 0,
                    'role': 'DIAGNOSTIC_DEV_ONLY', 'training_allowed': False})
    return selected


def verify_file(path, row):
    if path.stat().st_size != row['bytes']:
        raise ValueError('payload size mismatch')
    algorithm, prefix = ('sha1', f"blob {row['bytes']}\0".encode()) if row['checksum_type'] == 'git_blob_sha1' else ('md5', b'')
    if digest(path, algorithm, prefix) != row['checksum']:
        raise ValueError('publisher payload checksum mismatch')
    if 'body_sha1' in row and digest(path, 'sha1') != row['body_sha1']:
        raise ValueError('RAW filename SHA1 mismatch')
    return digest(path)


def freeze():
    names = ('wifd_tree.response.json', 'rawnind_metadata.json', 'dataset.yaml', 'permissions.txt',
             'test_reserve.yaml', 'wifd_LICENSE', 'wifd_README.md')
    inputs = {str(ROOT / name): digest(ROOT / name) for name in names}
    tree = read(ROOT / names[0]); metadata = read(ROOT / names[1])
    rows = choose_wifd(tree) + choose_rawnind(metadata,
        yaml.safe_load((ROOT / 'dataset.yaml').read_text()), (ROOT / 'permissions.txt').read_text(),
        yaml.safe_load((ROOT / 'test_reserve.yaml').read_text()))
    for row in rows:
        row['parent_id'] = 'e65:' + hashlib.sha256((row['source'] + ':' + row['name']).encode()).hexdigest()
        row['path'] = str(ROOT / 'originals' / row['source'] / row['name'])
    if not rows or sum(r['bytes'] for r in rows) > 2 * 1024**3:
        raise ValueError('empty or over-budget pilot')
    if len({r['parent_id'] for r in rows}) != len(rows):
        raise ValueError('duplicate selected identity')
    value = {'state': 'E65_score_blind_acquisition_registered', 'code_sha256': digest(__file__),
             'inputs': inputs, 'rows': rows, 'bytes': sum(r['bytes'] for r in rows),
             'selection': 'WIFD SDR: up to4 distinct setting tuples at min/max ISO per available camera; '
                          'RawNIND:4 CC0 non-test-reserve scenes per CFA, one clean and high ISO<=6400 pair; '
                          'all choices hash-ranked seedE65, no scores/pixels inspected.',
             'workers': 2, 'max_seconds': 2400, 'byte_budget': 2 * 1024**3,
             'all_publishers_reserved': True, 'training_allowed': False, 'model_scoring_allowed': False,
             'limitation': 'REAL-only pilot, not a balanced final or proof of scene/publisher independence. '
                           'Overlap/decoding/provenance checks must precede a separate diagnostic score contract.'}
    write_once(CONTRACT, value)
    write_once(EVIDENCE / 'e65_acquisition_contract.json', value | {'contract_sha256': digest(CONTRACT)})
    return {'rows': len(rows), 'bytes': value['bytes'], 'sources': dict(Counter(r['source'] for r in rows))}


def fetch_one(row, deadline):
    path = Path(row['path']); path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return row | {'sha256': verify_file(path, row), 'download_status': 'verified_existing'}
    part = path.with_suffix(path.suffix + '.part')
    for attempt in range(3):
        if time.monotonic() >= deadline:
            raise TimeoutError('pilot download runtime budget reached')
        try:
            offset = part.stat().st_size if part.exists() else 0
            if offset == row['bytes']:
                sha = verify_file(part, row); part.replace(path)
                return row | {'sha256': sha, 'download_status': 'verified_resumed'}
            if offset > row['bytes']:
                raise ValueError('partial file exceeds expected payload')
            with requests.get(row['url'], headers={'Range': f'bytes={offset}-'} if offset else {},
                              stream=True, timeout=(15, 45)) as response:
                response.raise_for_status()
                if response.status_code == 206:
                    if not response.headers.get('Content-Range', '').startswith(f'bytes {offset}-'):
                        raise ValueError('resumed range does not match partial payload')
                    mode = 'ab'
                elif response.status_code == 200:
                    mode = 'wb'  # Server ignored Range; restart safely rather than append duplicate bytes.
                    offset = 0
                else:
                    raise ValueError('unexpected download response')
                received = offset
                with part.open(mode) as stream:
                    for chunk in response.iter_content(1024**2):
                        if time.monotonic() >= deadline:
                            raise TimeoutError('pilot download runtime budget reached')
                        received += len(chunk)
                        if received > row['bytes']:
                            raise ValueError('response larger than pinned payload')
                        stream.write(chunk)
            sha = verify_file(part, row); part.replace(path)
            return row | {'sha256': sha, 'download_status': 'downloaded_verified'}
        except requests.RequestException:
            if attempt == 2:
                raise
    raise RuntimeError('unreachable download state')


def download():
    c = read(CONTRACT)
    if digest(CONTRACT) != read(EVIDENCE / 'e65_acquisition_contract.json')['contract_sha256'] or digest(__file__) != c['code_sha256']:
        raise ValueError('acquisition contract/code changed')
    for path, sha in c['inputs'].items():
        if digest(path) != sha:
            raise ValueError('source metadata changed')
    if RECEIPT.exists():
        raise FileExistsError('complete receipt exists; do not replace')
    if not Path('/Volumes/LaCie').is_mount() or shutil.disk_usage(ROOT).free < c['byte_budget'] + 10 * 1024**3:
        raise RuntimeError('external volume missing or insufficient space')
    deadline = time.monotonic() + c['max_seconds']; done = []
    with ThreadPoolExecutor(max_workers=c['workers']) as pool:
        for result in pool.map(lambda row: fetch_one(row, deadline), c['rows']):
            done.append(result)
            print(json.dumps({'verified_files': len(done), 'total': len(c['rows']),
                              'verified_bytes': sum(r['bytes'] for r in done)}), flush=True)
    receipt = {'state': 'E65_payloads_verified_unscored', 'contract_sha256': digest(CONTRACT),
               'rows': done, 'bytes': sum(r['bytes'] for r in done), 'training_allowed': False,
               'model_scores_created': 0, 'balanced_final_admitted': False}
    write_once(RECEIPT, receipt)
    summary = {k: v for k, v in receipt.items() if k != 'rows'} | {'files': len(done), 'receipt_sha256': digest(RECEIPT)}
    write_once(EVIDENCE / 'e65_download.json', summary)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('freeze', 'download'))
    with (ROOT / 'execution.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        print(json.dumps({'freeze': freeze, 'download': download}[parser.parse_args().stage](), indent=2))
