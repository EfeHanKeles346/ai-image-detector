"""Pinned author DiffSeg30k TRAIN subset: automatic acquisition into quarantine."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import time
import zipfile
import requests
from experiments.e65_acquisition import digest, read, write_once
from experiments.e72_acquisition import PinnedRanges, member_identity, verify_body, resource_check
from pixelproof.project_paths import DATA_ROOT, ML_ROOT

REVISION = '8049819755ff430dcf4701636b14e6f1df6c4542'
REPO = 'Chaos2629/Diffseg30k'
ARCHIVE_SHA = '9ef27f4c1c35ef21bce150435312de18fa35cf00df6d622271219455a4ee3d02'
ARCHIVE_BYTES = 21563023972
ROOT = DATA_ROOT/'e105'; EVIDENCE = ML_ROOT.parent/'evidence'
CONTRACT = ROOT/'acquisition_contract.json'


def head(url):
    with requests.head(url, allow_redirects=True, timeout=(10, 40)) as r:
        r.raise_for_status()
        return {'bytes': int(r.headers['Content-Length']), 'etag': r.headers['ETag'],
                'last_modified': r.headers.get('Last-Modified')}


def select(entries, count=512):
    names = [r['filename'] for r in entries]
    if len(names) != len(set(names)) or any(PurePosixPath(n).is_absolute() or
            '..' in PurePosixPath(n).parts or '\\' in n for n in names):
        raise ValueError('Unsafe or duplicated archive names')
    by_name = {r['filename']: r for r in entries}
    images = [n for n in names if re.fullmatch(r'train_copy/[0-9]{8}\.image\.png', n)]
    if len(images) < count:
        raise ValueError('Insufficient fixed TRAIN subset')
    selected = sorted(images, key=lambda n: hashlib.sha256(('E105|'+n).encode()).digest())[:count]
    result = []
    for name in selected:
        mask = name.replace('.image.png', '.mask.png')
        if mask not in by_name:
            raise ValueError('Selected mask missing; no refill')
        pair = [by_name[name], by_name[mask]]
        if any(r['directory'] or not 0 < r['bytes'] <= 16*1024**2 or
               not 0 < r['compressed_bytes'] <= 16*1024**2 or
               r['compression'] not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED) for r in pair):
            raise ValueError('Unsupported selected member; no refill')
        result.append({'id': PurePosixPath(name).name.split('.')[0], 'members': pair})
    return result


def freeze():
    if CONTRACT.exists(): raise FileExistsError('E105 already registered')
    catalog = read(ROOT/'catalog.json')
    url = f'https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/train.zip'
    api = requests.get(f'https://huggingface.co/api/datasets/{REPO}/revision/{REVISION}',
                       params={'blobs': 'true'}, timeout=30)
    api.raise_for_status(); metadata = api.json()
    archive = next(x for x in metadata['siblings'] if x['rfilename'] == 'train.zip')
    if metadata['sha'] != REVISION or metadata.get('gated') or \
            archive['lfs']['sha256'] != ARCHIVE_SHA or archive['size'] != ARCHIVE_BYTES or \
            catalog['url'] != url or catalog['identity'] != head(url) or \
            catalog['identity']['bytes'] != ARCHIVE_BYTES or catalog['image_members_read']:
        raise ValueError('Pinned public author archive metadata changed')
    card = requests.get(f'https://huggingface.co/datasets/{REPO}/raw/{REVISION}/README.md', timeout=30)
    card.raise_for_status()
    if len(card.content) > 65536 or 'license: apache-2.0' not in card.text:
        raise ValueError('Reviewed dataset-card licence differs')
    card_path = ROOT/'source_card.txt'
    if card_path.exists() and card_path.read_bytes() != card.content:
        raise ValueError('Existing source card differs')
    if not card_path.exists(): card_path.write_bytes(card.content)
    pairs = select(catalog['rows']); total = sum(m['bytes'] for p in pairs for m in p['members'])
    if total > 2*1024**3: raise ValueError('2GiB source-body cap exceeded')
    c = {'state': 'E105_DiffSeg30k_TRAIN_subset_registered', 'repo': REPO, 'revision': REVISION,
        'url': url, 'identity': catalog['identity'], 'publisher_archive_sha256': ARCHIVE_SHA,
        'pairs': pairs, 'images': len(pairs), 'masks': len(pairs), 'body_bytes': total,
        'inputs': {str(p): digest(p) for p in [Path(__file__), ROOT/'catalog.json', card_path,
            Path(__file__).with_name('e72_acquisition.py'), Path(__file__).with_name('e65_acquisition.py')]},
        'selection': '512 SHA256(E105|official TRAIN image filename)-first images and corresponding masks; no refill.',
        'role': 'QUARANTINE_MODEL2_RESEARCH', 'workers': 2, 'max_seconds': 7200,
        'license': 'Author dataset card Apache-2.0; upstream image/model rights and ancestry still require audit.',
        'limits': 'COCO-derived multi-turn edits; no automatic TRAIN/CAL/final admission or independent-publisher '
            'claim. ZIP CRC and local SHA verified; whole-archive publisher SHA is metadata only for range subsets.',
        'training_allowed': False, 'classifier_scores': 0, 'validation_image_reads': 0}
    write_once(CONTRACT, c)
    summary = {k:v for k,v in c.items() if k not in ['pairs', 'inputs']}
    write_once(EVIDENCE/'e105_acquisition_contract.json', summary | {'contract_sha256': digest(CONTRACT)})
    return summary


def validate():
    c = read(CONTRACT)
    if digest(CONTRACT) != read(EVIDENCE/'e105_acquisition_contract.json')['contract_sha256']:
        raise ValueError('E105 contract changed')
    for p, sha in c['inputs'].items():
        if digest(p) != sha: raise ValueError('E105 bound input changed')
    return c


def download():
    c = validate(); binding = digest(CONTRACT); start = time.monotonic(); deadline = start+c['max_seconds']
    if (ROOT/'download.json').exists(): raise FileExistsError('E105 download already complete')
    def worker(pairs):
        if head(c['url']) != c['identity']: raise ValueError('Archive identity changed')
        budget = sum(m['compressed_bytes'] for p in pairs for m in p['members'])+24*1024**2
        source = PinnedRanges(c['url'], c['identity'], budget, deadline); records = []
        with zipfile.ZipFile(source) as archive:
            for pair in pairs:
                resource_check(deadline)
                if shutil.disk_usage(ROOT).free < 30*1024**3: raise RuntimeError('30GiB reserve reached')
                receipt_path = ROOT/'pairs'/f'{pair["id"]}.json'
                result = {'id': pair['id'], 'role': c['role'], 'contract_sha256': binding, 'files': []}
                for row in pair['members']:
                    info = archive.getinfo(row['filename']); member_identity(info, row)
                    target = ROOT/'files'/PurePosixPath(row['filename']).name
                    target.parent.mkdir(exist_ok=True); receipt_path.parent.mkdir(exist_ok=True)
                    body = target.read_bytes() if target.exists() else archive.read(info)
                    sha = verify_body(body, row)
                    if not target.exists():
                        temp = target.with_suffix(target.suffix+'.part'); temp.write_bytes(body); temp.replace(target)
                    result['files'].append({'path': str(target), 'member': row['filename'], 'bytes': len(body),
                                            'sha256': sha, 'crc32': row['crc32']})
                if receipt_path.exists() and read(receipt_path) != result: raise ValueError('Pair receipt differs')
                if not receipt_path.exists(): write_once(receipt_path, result)
                records.append(result)
                if len(records) % 16 == 0:
                    print(json.dumps({'E105_worker_pairs': len(records), 'worker_total': len(pairs),
                                      'seconds': round(time.monotonic()-start)}), flush=True)
        if head(c['url']) != c['identity']: raise ValueError('Archive identity changed after transfer')
        return records, source.used
    with ThreadPoolExecutor(max_workers=2) as pool:
        outputs = list(pool.map(worker, [c['pairs'][::2], c['pairs'][1::2]]))
    rows = sorted([r for records,_ in outputs for r in records], key=lambda r:r['id'])
    total = sum(f['bytes'] for r in rows for f in r['files'])
    if len(rows) != c['images'] or total != c['body_bytes']: raise ValueError('Incomplete download')
    report = {'state': 'E105_download_complete_quarantine', 'contract_sha256': binding,
        'rows': rows, 'images': len(rows), 'masks': len(rows), 'body_bytes': total,
        'range_bytes_this_execution': sum(n for _,n in outputs), 'seconds': time.monotonic()-start,
        'training_allowed': False, 'classifier_scores': 0, 'limits': c['limits']}
    write_once(ROOT/'download.json', report)
    summary = {k:v for k,v in report.items() if k != 'rows'} | {'receipt_sha256': digest(ROOT/'download.json')}
    write_once(EVIDENCE/'e105_download.json', summary)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('stage', choices=['freeze','download'])
    ROOT.mkdir(parents=True, exist_ok=True)
    with (ROOT/'execution.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        print(json.dumps({'freeze':freeze,'download':download}[parser.parse_args().stage](), indent=2))
