"""Acquire author CocoGlide package and recover compiled identities by exact pixels."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
from io import BytesIO, StringIO
import json
from pathlib import Path
import re
import shutil
import time
import zipfile
import numpy as np
from PIL import Image
import requests
from experiments.e65_acquisition import digest, read, write_once
from experiments.e65_audit import fingerprint
from experiments.e115_cocoglide_metadata import URL, EXPECTED, head
from pixelproof.project_paths import DATA_ROOT, ML_ROOT, WORK_ROOT

ROOT = DATA_ROOT/'e116'; EVIDENCE = ML_ROOT.parent/'evidence'; CONTRACT = ROOT/'contract.json'
ARCHIVE = ROOT/'CocoGlide.zip'


def unique_join(author, compiled):
    lookup = defaultdict(list)
    for row in author:
        lookup[(row['auth_pixel_sha256'], row['edit_pixel_sha256'], row['mask_pixel_sha256'])].append(row)
    joined = []
    for row in compiled:
        key = (row['auth_pixel_sha256'], row['edit_pixel_sha256'], row['mask_pixel_sha256'])
        matches = lookup[key]
        if len(matches) != 1: raise ValueError('Missing or ambiguous exact author triple; no filename/order guess')
        joined.append(row | matches[0])
    if len({r['coco_id'] for r in joined}) != len(joined):
        raise ValueError('Repeated original parents require grouping before any split')
    return joined


def mask_sha(body):
    with Image.open(BytesIO(body)) as image:
        if image.width*image.height > 32_000_000: raise ValueError('Mask decode limit')
        image.load(); a = np.asarray(image)
        if a.ndim != 2 or not set(np.unique(a)).issubset({0, 255}):
            raise ValueError('Unexpected mask encoding')
        return hashlib.sha256(str(a.shape).encode()+np.ascontiguousarray(a).tobytes()).hexdigest()


def freeze():
    if head() != EXPECTED: raise ValueError('Author package identity differs')
    inventory = WORK_ROOT/'model2_local_inventory_2026-09-14.json'
    if digest(inventory) != read(EVIDENCE/'model2_local_inventory_2026-09-14.json')['inventory_sha256'] or \
            digest(DATA_ROOT/'e111/audit.json') != read(EVIDENCE/'e111_model2_lineage.json')['report_sha256'] or \
            digest(DATA_ROOT/'e115/catalog.json') != read(EVIDENCE/'e115_cocoglide_metadata.json')['catalog_sha256']:
        raise ValueError('Prior metadata/inventory receipt differs')
    files = [Path(__file__), Path(fingerprint.__code__.co_filename), inventory,
        DATA_ROOT/'e111/audit.json', DATA_ROOT/'e115/catalog.json',
        ML_ROOT/'experiments/e115_cocoglide_metadata.py']
    c = {'state': 'E116_author_lineage_recovery_registered', 'url': URL, 'identity': EXPECTED,
        'inputs': {str(p): digest(p) for p in files}, 'requested_archive_bytes': EXPECTED['bytes'],
        'role': 'PROVENANCE_AUDIT_ONLY', 'max_seconds_download': 900,
        'selection': 'Entire author512-pair package, no prediction-based selection. Validate real/edit/mask triples by canonical exact pixel digests against existing compiled512 pairs; never join by row number or basename.',
        'purpose': 'Recover original COCO val2017 IDs, prompts and image-specific licences from author table.csv/licenses.json; these were absent from compilation metadata.',
        'limits': 'Public author ETag/size/Last-Modified, TLS, ZIP CRC and local full SHA; no publisher full SHA advertised. Matching author ancestry is not automatic TRAIN/CAL/final admission. Keep E111 overlap quarantine.',
        'training_allowed': False, 'model_scores': 0}
    ROOT.mkdir(exist_ok=True); write_once(CONTRACT, c)
    write_once(EVIDENCE/'e116_cocoglide_contract.json', c | {'contract_sha256': digest(CONTRACT)})
    return {'archive_bytes': EXPECTED['bytes'], 'purpose': c['purpose']}


def validate():
    c = read(CONTRACT)
    if digest(CONTRACT) != read(EVIDENCE/'e116_cocoglide_contract.json')['contract_sha256']:
        raise ValueError('Contract differs')
    for p, sha in c['inputs'].items():
        if digest(p) != sha: raise ValueError('Frozen input differs')
    return c


def download():
    c = validate()
    if ARCHIVE.exists(): raise FileExistsError('Archive already exists; inspect receipt before resuming')
    if shutil.disk_usage(ROOT).free < 20*1024**3: raise RuntimeError('Disk reserve')
    if head() != c['identity']: raise ValueError('Author archive changed')
    start = time.monotonic(); size = 0; temp = ARCHIVE.with_suffix('.zip.part')
    with requests.get(URL, headers={'If-Match': EXPECTED['etag']}, stream=True, timeout=(10, 40)) as response:
        response.raise_for_status()
        if response.status_code != 200 or response.headers.get('ETag') != EXPECTED['etag']:
            raise ValueError('Author response identity differs')
        with temp.open('wb') as stream:
            for chunk in response.iter_content(1024**2):
                size += len(chunk)
                if size > EXPECTED['bytes'] or time.monotonic()-start > c['max_seconds_download']:
                    raise RuntimeError('Download bound exceeded')
                stream.write(chunk)
    if size != EXPECTED['bytes'] or head() != EXPECTED: raise ValueError('Incomplete/changed archive')
    temp.replace(ARCHIVE)
    r = {'state': 'E116_author_archive_downloaded', 'archive_bytes': size, 'archive_sha256': digest(ARCHIVE),
        'seconds': time.monotonic()-start, 'contract_sha256': digest(CONTRACT),
        'training_allowed': False, 'model_scores': 0, 'purpose': c['purpose']}
    write_once(ROOT/'download.json', r); write_once(EVIDENCE/'e116_cocoglide_download.json', r)
    return r


def audit():
    c = validate(); receipt = read(ROOT/'download.json')
    if digest(ROOT/'download.json') != digest(EVIDENCE/'e116_cocoglide_download.json') or digest(ARCHIVE) != receipt['archive_sha256']:
        raise ValueError('Downloaded package receipt differs')
    catalog = read(DATA_ROOT/'e115/catalog.json'); by_name = {r['name']: r for r in catalog['records']}
    authors = []
    with zipfile.ZipFile(ARCHIVE) as archive:
        if set(archive.namelist()) != set(by_name) or archive.testzip() is not None:
            raise ValueError('Catalog or archive CRC differs')
        for info in archive.infolist():
            if info.file_size != by_name[info.filename]['bytes'] or f'{info.CRC:08x}' != by_name[info.filename]['crc32']:
                raise ValueError('Member identity differs')
        table = list(csv.DictReader(StringIO(archive.read('table.csv').decode('utf-8-sig'))))
        license_doc = json.loads(archive.read('licenses.json'))
        image_license = {int(r['id']): r['license'] for r in license_doc['images']}
        license_names = {r['id']: {'name': r['name'], 'url': r['url']} for r in license_doc['licenses']}
        if len(table) != 512 or len(image_license) != 512: raise ValueError('Author metadata count differs')
        for row in table:
            match = re.fullmatch(r'fake/glide_inpainting_val2017_(\d+)_up.png', row['fake'])
            if not match: raise ValueError('Unrecognized author COCO naming convention')
            coco_id = int(match.group(1)); license_id = image_license[coco_id]
            auth = fingerprint(archive.read(row['real']), max_pixels=32_000_000)
            edit = fingerprint(archive.read(row['fake']), max_pixels=32_000_000)
            authors.append({'coco_id': coco_id, 'coco_split': 'val2017', 'prompt': row['prompt'],
                'license_id': license_id, 'license': license_names[license_id],
                'author_real_member': row['real'], 'author_edit_member': row['fake'],
                'auth_pixel_sha256': auth['pixel_sha256'], 'edit_pixel_sha256': edit['pixel_sha256'],
                'mask_pixel_sha256': mask_sha(archive.read(row['mask']))})
    facts = read(DATA_ROOT/'e111/audit.json'); indexed = {r['parent_id']: r for r in facts['records']}
    blocked = {m['train_parent'].rsplit(':', 1)[0] for m in facts['cross_matches'] if m['train_parent'].startswith('CocoGlide:')}
    compiled = []
    for i, row in enumerate(read(WORK_ROOT/'model2_local_inventory_2026-09-14.json')['rows']):
        path = Path(row['image']).with_suffix('.mask.png')
        if digest(path) != row['mask_sha256']: raise ValueError('Compiled mask differs')
        compiled.append({'compiled_index': i, 'auth_pixel_sha256': indexed[f'CocoGlide:{i}:auth']['pixel_sha256'],
            'edit_pixel_sha256': indexed[f'CocoGlide:{i}:edit']['pixel_sha256'], 'mask_pixel_sha256': mask_sha(path.read_bytes()),
            'overlap_quarantined': f'CocoGlide:{i}' in blocked, 'historically_exposed_first120': i < 120})
    joined = unique_join(authors, compiled)
    write_once(ROOT/'lineage.json', {'rows': joined, 'contract_sha256': digest(CONTRACT), 'archive_sha256': receipt['archive_sha256']})
    result = {'state': 'E116_exact_author_lineage_recovered', 'contract_sha256': digest(CONTRACT),
        'lineage_sha256': digest(ROOT/'lineage.json'), 'matched_triples': len(joined),
        'unique_original_COCO_parents': len({r['coco_id'] for r in joined}), 'COCO_split': 'val2017',
        'image_license_counts': dict(Counter(r['license']['name'] for r in joined)),
        'E111_quarantined_parent_groups_retained': sum(r['overlap_quarantined'] for r in joined),
        'training_allowed': False, 'model_scores': 0,
        'next': 'Use recovered COCO IDs to audit original ancestry across protected manifests, then separately register whole-parent research roles. These are512 same-source originals, not newly independent images.'}
    write_once(ROOT/'audit.json', result); write_once(EVIDENCE/'e116_cocoglide_lineage.json', result)
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('stage', choices=['freeze', 'download', 'audit'])
    print(json.dumps({'freeze': freeze, 'download': download, 'audit': audit}[p.parse_args().stage](), indent=2))
