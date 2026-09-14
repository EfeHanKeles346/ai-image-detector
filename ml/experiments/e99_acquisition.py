"""Additional MIDD capture pipelines; original TRAIN bytes only, initially quarantined."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
from pathlib import Path, PurePosixPath
import json
import re
import shutil
import time
import zipfile

from experiments.e65_acquisition import digest, read, write_once
from experiments.e72_acquisition import (BASE_URL, LICENSE_SHA, PinnedRanges,
    head_identity, member_identity, verify_body, resource_check)
from pixelproof.project_paths import DATA_ROOT, ML_ROOT

SENSORS = ('ISOCELL_GN1', 'ISOCELL_HM3', 'OmniVision_OV64B', 'Sony_IMX766')
ROOT = DATA_ROOT / 'e99'
EVIDENCE = ML_ROOT.parent / 'evidence'
CONTRACT = ROOT / 'acquisition_contract.json'


def select(sensor, entries):
    if sensor not in SENSORS:
        raise ValueError('Unregistered sensor')
    names = [r['filename'] for r in entries]
    if len(names) != len(set(names)):
        raise ValueError('Duplicate archive names')
    if any(PurePosixPath(n).is_absolute() or '..' in PurePosixPath(n).parts or '\\' in n for n in names):
        raise ValueError('Unsafe archive path')
    rows = [r for r in entries if re.fullmatch(re.escape(sensor) + r'/training_set/original/[0-9]+\.jpg', r['filename'])]
    if len(rows) < 64:
        raise ValueError('Insufficient original TRAIN capacity; no replacement sensor')
    rows = sorted(rows, key=lambda r: hashlib.sha256(('E99|' + r['filename']).encode()).hexdigest())[:64]
    if any(r['directory'] or not 0 < r['bytes'] <= 32*1024**2 or
           not 0 < r['compressed_bytes'] <= 32*1024**2 or
           r['compression'] not in [zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED] for r in rows):
        raise ValueError('Selected member unsupported; no refill')
    return rows


def freeze():
    if CONTRACT.exists():
        raise FileExistsError('Contract already frozen')
    packages = []
    inputs = [Path(__file__), Path(__file__).with_name('e72_acquisition.py'),
              Path(__file__).with_name('e65_acquisition.py'), DATA_ROOT/'e98/midd_listing.xml']
    for sensor in SENSORS:
        path = ROOT / f'{sensor}_catalog.json'
        if not path.exists():
            url = BASE_URL + sensor + '.zip'; identity = head_identity(url)
            source = PinnedRanges(url, identity, 8*1024**2, time.monotonic()+300)
            with zipfile.ZipFile(source) as archive:
                entries = [{'filename': x.filename, 'bytes': x.file_size, 'compressed_bytes': x.compress_size,
                    'compression': x.compress_type, 'crc32': f'{x.CRC:08x}',
                    'header_offset': x.header_offset, 'directory': x.is_dir()} for x in archive.infolist()]
                license_info = archive.getinfo(sensor + '/LICENSE.txt')
                if license_info.file_size > 65536:
                    raise ValueError('Oversized license')
                license_body = archive.read(license_info)
                if hashlib.sha256(license_body).hexdigest() != LICENSE_SHA:
                    raise ValueError('Package terms differ from reviewed research licence')
                select(sensor, entries)
            if head_identity(url) != identity:
                raise ValueError('Remote archive changed during inventory')
            write_once(path, {'sensor': sensor, 'url': url, 'identity': identity, 'entries': entries,
                'license_sha256': LICENSE_SHA, 'metadata_bytes_received': source.used, 'image_members_read': 0})
        catalog = read(path); inputs.append(path)
        if catalog['image_members_read'] or catalog['license_sha256'] != LICENSE_SHA:
            raise ValueError('Unscored licensed metadata required')
        rows = select(sensor, catalog['entries'])
        packages.append({k: catalog[k] for k in ['sensor', 'url', 'identity']} | {'rows': rows,
            'range_budget_bytes': sum(r['compressed_bytes'] for r in rows) + 16*1024**2})
        print(json.dumps({'sensor': sensor, 'selected': len(rows), 'bytes': sum(r['bytes'] for r in rows)}), flush=True)
    total = sum(r['bytes'] for p in packages for r in p['rows'])
    if total > 4*1024**3:
        raise ValueError('Predeclared body budget exceeded; no score-dependent shrink')
    contract = {'state': 'E99_MIDD_extra_sensors_registered', 'packages': packages,
        'inputs': {str(p): digest(p) for p in inputs}, 'selected': 256, 'body_bytes': total,
        'selection': '64 per registered new sensor, SHA256(E99|filename), official TRAIN/original JPG only; no refill.',
        'license': 'CC BY-NC-SA4.0 academic/research; archive LICENSE bytes pinned',
        'license_sha256': LICENSE_SHA, 'max_seconds': 5400, 'workers': 2,
        'role': 'QUARANTINE_FOR_POSSIBLE_RESEARCH_TRAIN', 'training_allowed': False,
        'model_scores': 0, 'image_scores_used_in_selection': 0,
        'limits': 'MIDD already belongs to research TRAIN. New sensors are not independent publishers. '
            'Original means publisher original noisy capture, not unprocessed RAW. No denoised/test data. '
            'Pixel/protected/scene audit and separate experiment required before TRAIN admission.'}
    write_once(CONTRACT, contract)
    summary = {k: v for k, v in contract.items() if k not in ['packages', 'inputs']}
    summary.update(contract_sha256=digest(CONTRACT), sensors={p['sensor']: {'selected': len(p['rows']),
        'body_bytes': sum(r['bytes'] for r in p['rows']), 'identity': p['identity']} for p in packages})
    write_once(EVIDENCE / 'e99_acquisition_contract.json', summary)
    return summary


def validate():
    c = read(CONTRACT)
    if digest(CONTRACT) != read(EVIDENCE/'e99_acquisition_contract.json')['contract_sha256']:
        raise ValueError('Acquisition contract changed')
    for path, sha in c['inputs'].items():
        if digest(path) != sha:
            raise ValueError('Bound acquisition input changed')
    return c


def download():
    c = validate(); binding = digest(CONTRACT); deadline = time.monotonic()+c['max_seconds']
    if (ROOT/'download.json').exists():
        raise FileExistsError('Acquisition already complete')

    def fetch(p):
        folder = ROOT/'images'/p['sensor']; folder.mkdir(parents=True, exist_ok=True)
        if head_identity(p['url']) != p['identity']:
            raise ValueError('Archive identity changed')
        source = PinnedRanges(p['url'], p['identity'], p['range_budget_bytes'], deadline)
        result = []
        with zipfile.ZipFile(source) as archive:
            for row in p['rows']:
                resource_check(deadline)
                if shutil.disk_usage(ROOT).free < 30*1024**3:
                    raise RuntimeError('30GiB storage reserve reached')
                info = archive.getinfo(row['filename']); member_identity(info, row)
                target = folder/Path(row['filename']).name; receipt = target.with_suffix('.receipt.json')
                if target.exists():
                    body = target.read_bytes()
                else:
                    body = archive.read(info)
                sha = verify_body(body, row)
                record = {'parent_id': 'MIDD:'+p['sensor']+':'+target.stem, 'sensor': p['sensor'],
                    'source': 'MIDD:'+p['sensor'], 'member': row['filename'], 'path': str(target),
                    'bytes': len(body), 'sha256': sha, 'crc32': row['crc32'], 'label': 0,
                    'condition': 'original', 'role': c['role'], 'training_allowed': False,
                    'contract_sha256': binding}
                if receipt.exists() and read(receipt) != record:
                    raise ValueError('Completed receipt differs')
                if not target.exists():
                    temp = target.with_suffix('.jpg.part'); temp.write_bytes(body); temp.replace(target)
                if not receipt.exists():
                    write_once(receipt, record)
                result.append(record)
                if len(result) % 8 == 0:
                    print(json.dumps({'sensor': p['sensor'], 'verified': len(result), 'total': 64}), flush=True)
        if head_identity(p['url']) != p['identity']:
            raise ValueError('Remote archive changed after transfer')
        return result, source.used

    with ThreadPoolExecutor(max_workers=c['workers']) as pool:
        outputs = list(pool.map(fetch, c['packages']))
    rows = sorted([r for output, _ in outputs for r in output], key=lambda r: r['parent_id'])
    if len(rows) != c['selected'] or sum(r['bytes'] for r in rows) != c['body_bytes']:
        raise ValueError('Incomplete acquisition')
    report = {'state': 'E99_download_complete_quarantine', 'contract_sha256': binding, 'rows': rows,
        'images': len(rows), 'body_bytes': sum(r['bytes'] for r in rows),
        'range_bytes_this_execution': sum(b for _, b in outputs), 'model_scores': 0,
        'training_allowed': False, 'limits': c['limits']}
    write_once(ROOT/'download.json', report)
    summary = {k: v for k, v in report.items() if k != 'rows'} | {'receipt_sha256': digest(ROOT/'download.json')}
    write_once(EVIDENCE/'e99_download.json', summary)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['freeze', 'download'])
    ROOT.mkdir(parents=True, exist_ok=True)
    with (ROOT/'execution.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        print(json.dumps({'freeze': freeze, 'download': download}[parser.parse_args().stage](), indent=2))
