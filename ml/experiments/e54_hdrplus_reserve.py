"""Small generation-pinned camera-derived HDR+ reserve; no detector or training."""
import argparse
import base64
import hashlib
from io import BytesIO
import json
from pathlib import Path
import re
from urllib.parse import quote, urlencode

from PIL import Image
import requests

from experiments.e51_prefit_audit import fingerprint
from experiments.e53_offline import EVIDENCE, digest, fixed_write
from experiments.e54_mnw_reserve import bounded_get, dino_atomic
from pixelproof.project_paths import DATA_ROOT

ROOT = DATA_ROOT/'e52/hdrplus_reserve_v1'
MANIFEST = ROOT/'manifest.json'
PREFIX = '20171106/results_20171023/'
API = 'https://storage.googleapis.com/storage/v1/b/hdrplusdata/o'
MAX_TRANSFER = 768*1024**2
MAX_IMAGE = 12*1024**2
SOURCE = 'https://www.hdrplusdata.org/dataset.html'
LICENSE = 'https://creativecommons.org/licenses/by-sa/4.0/'


def select(records):
    images = [r for r in records if r['name'].endswith('/final.jpg')]
    synthetic = [r for r in images if r['name'].split('/')[-2].startswith('synthetic_')]
    photographic = [r for r in images if r not in synthetic]
    for row in photographic:
        if not re.fullmatch(r'[A-Za-z0-9]+_[0-9]{8}_[0-9]{6}_[0-9]{3}', row['name'].split('/')[-2]):
            raise ValueError('unknown capture naming/provenance; review before selection')
    if len({r['name'] for r in images}) != len(images) or len(photographic) < 100:
        raise ValueError('duplicate/incomplete photographic inventory')
    ranked = sorted(photographic, key=lambda r: hashlib.sha256(('HDRPLUS_E52|'+r['name']).encode()).hexdigest())[:100]
    if any(not 0 < int(r['size']) <= MAX_IMAGE for r in ranked):
        raise ValueError('selected body violates fixed image cap')
    if sum(int(r['size']) for r in ranked) > MAX_TRANSFER:
        raise ValueError('selection exceeds fixed total budget')
    return ranked, len(images), [r['name'] for r in synthetic]


def freeze():
    if MANIFEST.exists():
        raise FileExistsError('HDR+ selection already frozen')
    records = []; pages = []; token = None
    for _ in range(20):
        params = {'prefix': PREFIX, 'maxResults': 1000,
                  'fields': 'nextPageToken,items(name,size,md5Hash,generation)'}
        if token:
            params['pageToken'] = token
        raw = bounded_get(API+'?'+urlencode(params), 512*1024)
        value = json.loads(raw); records.extend(value.get('items', []))
        pages.append({'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw)})
        token = value.get('nextPageToken')
        if not token:
            break
    else:
        raise ValueError('complete listing exceeds metadata ceiling')
    ranked, total, synthetic = select(records)
    source = bounded_get(SOURCE, 128*1024).decode()
    if LICENSE not in source:
        raise ValueError('source licence changed')
    rows = []
    for row in ranked:
        if len(base64.b64decode(row['md5Hash'], validate=True)) != 16 or not row['generation'].isdigit():
            raise ValueError('invalid publisher checksum/generation')
        burst = row['name'].split('/')[-2]
        rows.append({**row, 'bytes': int(row['size']), 'parent_id': 'HDRPLUS:'+burst,
                     'source': 'Google-HDRplus-2017', 'label': 0,
                     'role': 'RESERVED_EXTERNAL_EVALUATION',
                     'capture_session_proxy': '_'.join(burst.split('_')[:2]),
                     'url': API+'/'+quote(row['name'], safe='')+'?alt=media&generation='+row['generation']})
    terms = {'source_url': SOURCE, 'license': 'CC-BY-SA-4.0', 'license_url': LICENSE,
             'attribution': 'Hasinoff, Sharlet, Geiss, Adams, Barron, Kainz, Chen, Levoy. Burst photography for high dynamic range and low-light imaging on mobile cameras. ACM TOG / SIGGRAPH Asia 2016.',
             'processing': 'Publisher 2017-10-23 HDR+ camera-burst finishing, Q95 JPEG. No local change to acquired originals.',
             'scope': 'Local research evaluation only by project policy. No image redistribution or endorsement.'}
    result = {'state': 'HDRPLUS_reserve_frozen_before_images', 'code_sha256': digest(__file__),
              'helper_sha256': digest(Path(__file__).with_name('e54_mnw_reserve.py')),
              'rows': rows, 'parents': len(rows), 'source_final_jpegs': total,
              'synthetic_folders_excluded': synthetic, 'listing_pages': pages,
              'image_bytes_expected': sum(r['bytes'] for r in rows), 'image_transfer_cap': MAX_TRANSFER,
              'source_html': source, 'source_html_sha256': hashlib.sha256(source.encode()).hexdigest(),
              'terms': terms, 'model_scores_created': 0, 'training_allowed_by_project': False,
              'balanced_final_admitted': False,
              'limits': ['Older Nexus/Pixel computational photographs, not new phones or untouched sensor JPEGs.',
                         'One burst is one parent; same-day/session names do not prove scene independence.',
                         'Reserve entire publisher; no TRAIN/CAL split or threshold tuning.']}
    fixed_write(MANIFEST, result); fixed_write(ROOT/'ATTRIBUTION.json', terms)
    summary = {k: v for k, v in result.items() if k not in {'rows', 'source_html'}}
    fixed_write(EVIDENCE/'e54_hdrplus_manifest.json', summary | {'manifest_sha256': digest(MANIFEST)})
    return summary


def verify_bytes(raw, row):
    if len(raw) != row['bytes'] or base64.b64encode(hashlib.md5(raw).digest()).decode() != row['md5Hash']:
        raise ValueError('publisher body size/MD5 mismatch')


def download():
    if (ROOT/'download.json').exists():
        raise FileExistsError('HDR+ acquisition already complete')
    manifest = json.loads(MANIFEST.read_text())
    receipt = json.loads((EVIDENCE/'e54_hdrplus_manifest.json').read_text())
    if digest(MANIFEST) != receipt['manifest_sha256'] or manifest['code_sha256'] != digest(__file__):
        raise ValueError('frozen selection/code changed')
    if digest(Path(__file__).with_name('e54_mnw_reserve.py')) != manifest['helper_sha256']:
        raise ValueError('transfer helper changed')
    ledger = ROOT/'transfer_ledger.json'
    spent = json.loads(ledger.read_text())['charged_image_transfer_bytes'] if ledger.exists() else 0
    records = []
    for row in manifest['rows']:
        path = ROOT/'images'/(row['parent_id'].removeprefix('HDRPLUS:')+'.jpg')
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            part = path.with_suffix('.part'); offset = part.stat().st_size if part.exists() else 0
            if offset > row['bytes']:
                raise ValueError('partial body exceeds expected size')
            if offset < row['bytes']:
                spent += row['bytes']-offset
                if spent > MAX_TRANSFER:
                    raise ValueError('conservative retry budget exhausted')
                dino_atomic(ledger, {'charged_image_transfer_bytes': spent})
                headers = {'Range': f'bytes={offset}-'} if offset else {}
                with requests.get(row['url'], headers=headers, stream=True, timeout=(15,60)) as response:
                    response.raise_for_status()
                    if offset and (response.status_code != 206 or not response.headers.get('Content-Range','').startswith(f'bytes {offset}-')):
                        raise ValueError('server refused exact partial resume')
                    with part.open('ab' if offset else 'wb') as stream:
                        for block in response.iter_content(64*1024):
                            offset += len(block)
                            if offset > row['bytes']:
                                raise ValueError('body byte cap exceeded')
                            stream.write(block)
            verify_bytes(part.read_bytes(), row); part.replace(path)
        raw = path.read_bytes(); verify_bytes(raw, row)
        facts = fingerprint(raw)
        with Image.open(BytesIO(raw)) as im:
            exif = im.getexif()
            device = {'make': str(exif.get(271, 'unknown')), 'model': str(exif.get(272, 'unknown'))}
        records.append({**row, **facts, **device, 'path': str(path), 'condition': 'original'})
        if len(records) % 10 == 0:
            print(json.dumps({'phase': 'HDRPLUS_reserve_download', 'images': len(records),
                              'total': len(manifest['rows']), 'charged_image_transfer_bytes': spent}), flush=True)
    result = {'state': 'HDRPLUS_downloaded_unscored_not_final_admitted',
              'manifest_sha256': digest(MANIFEST), 'records': records, 'parents': len(records),
              'verified_image_bytes': sum(r['bytes'] for r in records), 'charged_image_transfer_bytes': spent,
              'training_allowed_by_project': False, 'model_scores_created': 0, 'overlap_audit_pending': True}
    fixed_write(ROOT/'download.json', result)
    summary = {k:v for k,v in result.items() if k != 'records'}
    fixed_write(EVIDENCE/'e54_hdrplus_download.json', summary | {'report_sha256': digest(ROOT/'download.json')})
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('phase', choices=['freeze','download'])
    args = parser.parse_args()
    print(json.dumps({'freeze':freeze,'download':download}[args.phase](), indent=2))
