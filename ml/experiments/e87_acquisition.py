"""One bounded SID original long-RAW acquisition, quarantined for possible research TRAIN."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import re
import shutil
import time
import zipfile

from experiments.e65_acquisition import digest, read, write_once
from experiments.e72_acquisition import PinnedRanges, head_identity, member_identity, verify_body, resource_check
from pixelproof.project_paths import DATA_ROOT, ML_ROOT

ROOT = DATA_ROOT / 'e87'
EVIDENCE = ML_ROOT.parent / 'evidence'
CONTRACT = ROOT / 'acquisition_contract.json'
RECEIPT = ROOT / 'download.json'
METADATA = DATA_ROOT / 'source_research/sid_metadata'


def select(camera, catalog, body):
    if camera not in ['Sony', 'Fuji'] or catalog['camera'] != camera:
        raise ValueError('registered camera required')
    if catalog['image_members_opened'] or catalog['url'] != f'https://storage.googleapis.com/isl-datasets/SID/{camera}2025.zip':
        raise ValueError('official metadata-only archive required')
    names = {line.split()[1].removeprefix('./') for line in body.splitlines() if line.strip()}
    extension = 'ARW' if camera == 'Sony' else 'RAF'
    if any(not re.fullmatch(camera + r'/long/0\d{4}_00_\d+(?:\.\d+)?s\.' + extension, n) for n in names):
        raise ValueError('only original published TRAIN long references allowed')
    if len({Path(n).name[:5] for n in names}) != len(names) or len(names) < 64:
        raise ValueError('one RAW per published filename scene required')
    by_name = {r['filename']: r for r in catalog['entries']}
    if len(by_name) != len(catalog['entries']):
        raise ValueError('duplicate archive names')
    chosen = sorted(names, key=lambda n: hashlib.sha256(('E87|' + n).encode()).hexdigest())[:64]
    rows = [by_name[n] for n in chosen]
    if any(r['directory'] or not 0 < r['bytes'] <= 64 * 1024**2 or
           not 0 < r['compressed_bytes'] <= 64 * 1024**2 or
           r['compression'] not in [zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED] for r in rows):
        raise ValueError('unsupported or oversized RAW member')
    return rows


def freeze():
    metadata = read(EVIDENCE / 'sid_metadata_inventory.json')
    lead = read(EVIDENCE / 'sid_archive_inventory.json')
    if lead['image_members_opened'] or lead['data_admission'] or metadata['image_payload_bytes']:
        raise ValueError('unscored source metadata required')
    inputs = {str(p): digest(p) for p in [Path(__file__), Path(__file__).with_name('e72_acquisition.py'),
        Path(__file__).with_name('e65_acquisition.py'), EVIDENCE / 'sid_metadata_inventory.json',
        EVIDENCE / 'sid_archive_inventory.json', METADATA / 'LICENSE.md', METADATA / 'README.md']}
    for name in ['LICENSE.md', 'README.md']:
        if digest(METADATA / name) != metadata['inputs'][name]['sha256']:
            raise ValueError('pinned publisher documentation changed')
    packages = []
    for item in lead['archives']:
        camera = item['camera']; path = Path(item['catalog_path']); listing = METADATA / f'{camera}_train_list.txt'
        if digest(path) != item['catalog_sha256'] or digest(listing) != metadata['inputs'][listing.name]['sha256']:
            raise ValueError('source catalog/list changed')
        catalog = read(path); rows = select(camera, catalog, listing.read_text())
        inputs[str(path)] = digest(path); inputs[str(listing)] = digest(listing)
        packages.append({'camera': camera, 'url': catalog['url'], 'identity': catalog['identity'], 'rows': rows,
                         'range_budget_bytes': sum(r['compressed_bytes'] for r in rows) + 4 * 1024**2})
    if sorted(p['camera'] for p in packages) != ['Fuji', 'Sony']:
        raise ValueError('exactly two cameras required')
    raw_bytes = sum(r['bytes'] for p in packages for r in p['rows'])
    transfer = sum(p['range_budget_bytes'] for p in packages)
    if raw_bytes > 5 * 1024**3 or transfer > 4 * 1024**3:
        raise ValueError('bounded RAW/transfer budget exceeded')
    c = {'state': 'E87_SID_original_RAW_acquisition_registered', 'inputs': inputs, 'packages': packages,
         'selected': 128, 'raw_bytes': raw_bytes, 'range_budget_bytes': transfer, 'max_seconds': 5400,
         'selection': '64 per camera by SHA256(E87|filename), one published TRAIN long reference per filename scene. '
                      'No image content/scores, short bursts, official validation/test or network outputs. No refill.',
         'resource': 'One download worker, exact206/If-Match and ZIP CRC/length/SHA. AC,30GiB free reserve, '
                     '90min per execution. Reuse verified completed members; no full archive fallback.',
         'role': 'QUARANTINE_FOR_POSSIBLE_RESEARCH_TRAIN', 'training_allowed': False, 'model_scoring_allowed': False,
         'license': 'Official pinned SID README states MIT; LICENSE refers to software/documentation. '
                    'Keep source/citation and explicit scope limitation. Research acquisition only, no redistribution/serving.',
         'next': 'Separately freeze deterministic original-RAW rendering and complete protected-overlap/scene audit '
                 'before TRAIN admission. Whole SID publisher reserved from fresh DEV/final including unselected '
                 'and upstream val/test members. Existing E84B/E85/E86 remain unchanged.',
         'limits': 'RAW originals still have camera processing and may embed JPEG previews. A future decoded RGB '
                   'must come from RAW sensor data with documented rendering, never embedded preview or model output. '
                   'Filename groups do not establish scene independence across cameras or source transfer.',
         'image_scores_used_in_selection': 0, 'e49_reads': 0}
    write_once(CONTRACT, c)
    write_once(EVIDENCE / 'e87_acquisition_contract.json', {k: v for k, v in c.items() if k != 'packages'} |
               {'contract_sha256': digest(CONTRACT), 'camera_counts': {p['camera']: len(p['rows']) for p in packages}})
    return {'selected': c['selected'], 'raw_bytes': raw_bytes, 'range_budget_bytes': transfer}


def validate():
    c = read(CONTRACT)
    if digest(CONTRACT) != read(EVIDENCE / 'e87_acquisition_contract.json')['contract_sha256']:
        raise ValueError('E87 contract changed')
    for p, expected in c['inputs'].items():
        if digest(p) != expected:
            raise ValueError('E87 input changed: ' + p)
    return c


def download():
    c = validate()
    if RECEIPT.exists():
        raise FileExistsError('E87 acquisition already complete')
    deadline = time.monotonic() + c['max_seconds']; binding = digest(CONTRACT)
    def check():
        resource_check(deadline)
        if shutil.disk_usage(ROOT).free < 30 * 1024**3:
            raise RuntimeError('30GiB storage reserve required')
    done, transferred = [], 0
    for package in c['packages']:
        check(); camera = package['camera']; folder = ROOT / 'raw' / camera; folder.mkdir(parents=True, exist_ok=True)
        if head_identity(package['url']) != package['identity']:
            raise ValueError('remote RAW archive changed')
        source = PinnedRanges(package['url'], package['identity'], package['range_budget_bytes'], deadline)
        with zipfile.ZipFile(source) as archive:
            for index, row in enumerate(package['rows']):
                check(); info = archive.getinfo(row['filename']); member_identity(info, row)
                path = folder / Path(row['filename']).name; receipt = path.with_suffix(path.suffix + '.receipt.json')
                if path.exists():
                    sha = verify_body(path.read_bytes(), row)
                else:
                    if receipt.exists():
                        raise ValueError('receipt without RAW body')
                    body = archive.read(info); sha = verify_body(body, row)
                    temporary = path.with_suffix(path.suffix + '.part'); temporary.write_bytes(body); temporary.replace(path)
                record = {'parent_id': 'SID:' + camera + ':' + path.stem, 'camera': camera,
                          'scene_group': 'SID:' + camera + ':' + path.name[:5], 'member': row['filename'],
                          'path': str(path), 'sha256': sha, 'bytes': row['bytes'], 'crc32': row['crc32'],
                          'contract_sha256': binding, 'role': c['role'], 'training_allowed': False}
                if receipt.exists():
                    if read(receipt) != record:
                        raise ValueError('verified completed RAW receipt differs')
                else:
                    write_once(receipt, record)
                done.append(record)
                if (index + 1) % 8 == 0:
                    print(json.dumps({'E87_camera': camera, 'verified': index + 1, 'total': 64, 'range_bytes': source.used}), flush=True)
        transferred += source.used
        if head_identity(package['url']) != package['identity']:
            raise ValueError('archive changed after acquisition')
    if len(done) != c['selected'] or sum(r['bytes'] for r in done) != c['raw_bytes']:
        raise ValueError('incomplete fixed acquisition')
    result = {'state': 'E87_SID_RAW_acquisition_complete_quarantine', 'contract_sha256': binding, 'rows': done,
              'images': len(done), 'raw_bytes': c['raw_bytes'], 'range_bytes_this_execution': transferred,
              'model_scores': 0, 'RAW_decodes': 0, 'training_allowed': False}
    write_once(RECEIPT, result)
    summary = {k: v for k, v in result.items() if k != 'rows'} | {'receipt_sha256': digest(RECEIPT)}
    write_once(EVIDENCE / 'e87_download.json', summary)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('stage', choices=['freeze', 'download'])
    ROOT.mkdir(exist_ok=True)
    with (ROOT / 'acquisition_execution.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        print(json.dumps({'freeze': freeze, 'download': download}[parser.parse_args().stage](), indent=2))
