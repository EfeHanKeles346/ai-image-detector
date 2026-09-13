"""Bounded MIDD original-TRAIN member acquisition; no detector scores or role admission."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import io
import json
from pathlib import Path
import re
import shutil
import subprocess
import time
import zipfile
import zlib
import requests
from experiments.e65_acquisition import digest, read, write_once
from pixelproof.project_paths import DATA_ROOT, ML_ROOT

ROOT = DATA_ROOT / 'e72'
EVIDENCE = ML_ROOT.parent / 'evidence'
LEAD = EVIDENCE / 'midd_catalog_lead.json'
CONTRACT = ROOT / 'acquisition_contract.json'
RECEIPT = ROOT / 'download.json'
SENSORS = ('Hynix_SL846', 'ISOCELL_3P9', 'Sony_IMX258', 'OmniVision_OV32A')
LICENSE_SHA = 'e66c269d4819aaab34b49ef5220c4ddab6756f21bb5180761a4eb8561f2b7bbd'
BASE_URL = 'https://download.ai-benchmark.com/public.php/dav/files/Gq3n2cS7QkH7ZMz/'


def resource_check(deadline):
    if time.monotonic() >= deadline: raise TimeoutError('E72 bounded stage deadline reached')
    if not Path('/Volumes/LaCie').is_mount() or shutil.disk_usage(ROOT).free < 20*1024**3:
        raise RuntimeError('external volume / 20GiB reserve unavailable')
    if 'AC Power' not in subprocess.check_output(['pmset', '-g', 'batt'], text=True):
        raise RuntimeError('E72 requires AC power')


def select(catalog, n=128):
    sensor = catalog['sensor']
    if sensor not in SENSORS or catalog['url'] != BASE_URL + sensor + '.zip':
        raise ValueError('unregistered sensor or public archive URL')
    rows = [r for r in catalog['entries'] if not r['directory'] and
            re.fullmatch(re.escape(sensor) + r'/training_set/original/[0-9]+\.jpg', r['filename'])]
    if len({r['filename'] for r in rows}) != len(rows) or len(rows) < n:
        raise ValueError('duplicate or insufficient original TRAIN members')
    if any(not 0 < r['bytes'] <= 32*1024**2 or not 0 < r['compressed_bytes'] <= 32*1024**2
           or r['compression'] not in [zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED] for r in rows):
        raise ValueError('unsupported or oversized original member')
    return sorted(rows, key=lambda r: hashlib.sha256(('E72|' + r['filename']).encode()).hexdigest())[:n]


def head_identity(url):
    with requests.head(url, timeout=(10, 40)) as response:
        response.raise_for_status()
        return {'bytes': int(response.headers['Content-Length']), 'etag': response.headers['ETag'],
                'last_modified': response.headers.get('Last-Modified')}


class PinnedRanges(io.RawIOBase):
    """Seekable exact-range source; refuses bulk fallback before consuming the response."""
    def __init__(self, url, identity, budget, deadline):
        self.url=url; self.identity=identity; self.size=identity['bytes']
        self.budget=budget; self.deadline=deadline; self.used=0; self.position=0
    def readable(self): return True
    def seekable(self): return True
    def tell(self): return self.position
    def seek(self, offset, whence=0):
        p = offset if whence==0 else self.position+offset if whence==1 else self.size+offset if whence==2 else -1
        if not 0 <= p <= self.size: raise ValueError('invalid archive seek')
        self.position=p; return p
    def read(self, size=-1):
        if time.monotonic() >= self.deadline: raise TimeoutError('range deadline reached')
        amount=self.size-self.position if size < 0 else min(size, self.size-self.position)
        if amount==0: return b''
        if self.used+amount > self.budget: raise ValueError('bounded range transfer exceeded')
        start=self.position; end=start+amount-1
        with requests.get(self.url, headers={'Range': f'bytes={start}-{end}',
                          'If-Match': self.identity['etag'], 'Accept-Encoding': 'identity'},
                          stream=True, timeout=(10, 45)) as r:
            r.raise_for_status()
            if r.status_code != 206 or r.headers.get('Content-Range') != f'bytes {start}-{end}/{self.size}':
                raise ValueError('server did not honor exact range; bulk fallback refused')
            if r.headers.get('ETag', self.identity['etag']) != self.identity['etag']:
                raise ValueError('archive ETag changed')
            body=bytearray()
            for chunk in r.iter_content(65536):
                if time.monotonic() >= self.deadline: raise TimeoutError('range deadline reached')
                body.extend(chunk)
                if len(body)>amount: raise ValueError('overlong range body')
            if len(body)!=amount: raise ValueError('truncated range body')
        self.position+=amount; self.used+=amount; return bytes(body)


def member_identity(info, expected):
    observed={'filename': info.filename, 'bytes': info.file_size, 'compressed_bytes': info.compress_size,
              'compression': info.compress_type, 'crc32': f'{info.CRC:08x}', 'header_offset': info.header_offset}
    if any(expected[k]!=v for k,v in observed.items()) or info.flag_bits & 1 or info.is_dir():
        raise ValueError('frozen ZIP member identity changed')


def verify_body(body, row):
    if len(body)!=row['bytes'] or f'{zlib.crc32(body):08x}'!=row['crc32']:
        raise ValueError('member length or CRC differs')
    return hashlib.sha256(body).hexdigest()


def freeze():
    ROOT.mkdir(exist_ok=True)
    lead=read(LEAD)
    if lead['image_members_read'] or {r['sensor'] for r in lead['sensors']}!=set(SENSORS):
        raise ValueError('score-blind four-sensor metadata required')
    inputs={str(LEAD):digest(LEAD), str(Path(__file__)):digest(__file__),
            str(Path(__file__).with_name('e65_acquisition.py')):digest(Path(__file__).with_name('e65_acquisition.py'))}
    packages=[]
    for item in lead['sensors']:
        p=Path(item['catalog_path'])
        if digest(p)!=item['catalog_sha256']: raise ValueError('frozen directory changed')
        catalog=read(p)
        if catalog['license_sha256']!=LICENSE_SHA or catalog['identity']!=item['identity']:
            raise ValueError('license/archive identity differs')
        rows=select(catalog)
        inputs[str(p)]=digest(p)
        packages.append({'sensor':catalog['sensor'], 'url':catalog['url'], 'identity':catalog['identity'],
            'rows':rows, 'range_budget_bytes':sum(r['compressed_bytes'] for r in rows)+4*1024**2})
    total=sum(r['bytes'] for p in packages for r in p['rows'])
    if total>4*1024**3: raise ValueError('bounded payload exceeds4GiB')
    c={'state':'E72_MIDD_original_TRAIN_acquisition_registered', 'inputs':inputs, 'packages':packages,
       'selected':512, 'selected_original_bytes':total, 'license_sha256':LICENSE_SHA,
       'selection':'128 per each of four preselected sensor vendors by SHA256(E72|filename). '
                   'Original official TRAIN JPG only, no image score/content selection or replacement.',
       'resource':'Two download workers, exact HTTP206 ranges with If-Match, CRC/length and body SHA. '
                  '20GiB storage reserve, AC,90min per execution; verified completed member resume.',
       'max_seconds':5400, 'max_workers':2, 'role':'QUARANTINE_FOR_POSSIBLE_RESEARCH_TRAIN',
       'training_allowed':False, 'model_scoring_allowed':False, 'image_scores_used_in_selection':0,
       'license':'CC BY-NC-SA4.0 research only; no automatic serving promotion.',
       'next':'Separately freeze canonical decode/protected-overlap and connected-component audit. '
              'Include complete prior reserves plus E65 and consumed E66. No quota refill after quarantine.',
       'limits':'Filename numbers are not verified scenes; sensor labels are not independent camera instances. '
                'Whole MIDD publisher reserved from DEV/fresh final, including unselected/official-test/denoised '
                'members. E71 TRAIN population unchanged; original means publisher original, not RAW.'}
    write_once(CONTRACT,c); write_once(EVIDENCE/'e72_acquisition_contract.json',
        {k:v for k,v in c.items() if k!='packages'} | {'contract_sha256':digest(CONTRACT),
        'sensor_counts':{p['sensor']:len(p['rows']) for p in packages}})
    return {'selected':512, 'bytes':total, 'contract_sha256':digest(CONTRACT)}


def validate():
    c=read(CONTRACT)
    if digest(CONTRACT)!=read(EVIDENCE/'e72_acquisition_contract.json')['contract_sha256']:
        raise ValueError('E72 acquisition contract changed')
    for p,s in c['inputs'].items():
        if digest(p)!=s: raise ValueError('E72 bound input changed: '+p)
    return c


def download():
    c=validate()
    if RECEIPT.exists(): raise FileExistsError('E72 download already complete')
    deadline=time.monotonic()+c['max_seconds']; resource_check(deadline)
    binding=digest(CONTRACT)
    def fetch(package):
        sensor=package['sensor']; folder=ROOT/'images'/sensor; folder.mkdir(parents=True,exist_ok=True)
        if head_identity(package['url'])!=package['identity']: raise ValueError('remote archive changed')
        source=PinnedRanges(package['url'],package['identity'],package['range_budget_bytes'],deadline)
        rows=[]
        with zipfile.ZipFile(source) as archive:
            names=archive.namelist()
            if len(names)!=len(set(names)): raise ValueError('duplicate ZIP names')
            for i,row in enumerate(package['rows']):
                resource_check(deadline)
                info=archive.getinfo(row['filename']); member_identity(info,row)
                path=folder/Path(row['filename']).name; receipt=path.with_suffix('.receipt.json')
                if path.exists():
                    body=path.read_bytes(); sha=verify_body(body,row)
                    if receipt.exists():
                        old=read(receipt)
                        if old['sha256']!=sha or old['contract_sha256']!=binding or old['member']!=row['filename']:
                            raise ValueError('completed member receipt/body changed')
                else:
                    if receipt.exists(): raise ValueError('receipt exists without completed image')
                    body=archive.read(info); sha=verify_body(body,row)
                    temp=path.with_suffix('.jpg.part'); temp.write_bytes(body); temp.replace(path)
                record={'parent_id':'MIDD:'+sensor+':'+path.stem, 'path':str(path), 'member':row['filename'],
                    'sha256':sha, 'bytes':len(body), 'crc32':row['crc32'], 'contract_sha256':binding,
                    'sensor':sensor, 'source':'MIDD:'+sensor, 'label':0, 'condition':'original',
                    'role':c['role'], 'training_allowed':False}
                if not receipt.exists(): write_once(receipt,record)
                rows.append(record)
                if (i+1)%16==0: print(json.dumps({'sensor':sensor,'verified':i+1,'total':len(package['rows'])}),flush=True)
        if head_identity(package['url'])!=package['identity']: raise ValueError('archive changed after download')
        return rows,source.used
    with ThreadPoolExecutor(max_workers=c['max_workers']) as pool: results=list(pool.map(fetch,c['packages']))
    rows=sorted([r for batch,_ in results for r in batch],key=lambda r:r['parent_id'])
    if len(rows)!=c['selected'] or sum(r['bytes'] for r in rows)!=c['selected_original_bytes']:
        raise ValueError('incomplete bounded acquisition')
    result={'state':'E72_MIDD_download_complete_quarantine','contract_sha256':binding,'rows':rows,
        'range_bytes_this_execution':sum(size for _,size in results),'images':len(rows),
        'bytes':sum(r['bytes'] for r in rows),'model_scores':0,'training_allowed':False}
    write_once(RECEIPT,result); summary={k:v for k,v in result.items() if k!='rows'}
    summary['receipt_sha256']=digest(RECEIPT); write_once(EVIDENCE/'e72_download.json',summary); return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('stage',choices=['freeze','download'])
    ROOT.mkdir(exist_ok=True)
    with (ROOT/'download_execution.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        print(json.dumps({'freeze':freeze,'download':download}[parser.parse_args().stage](),indent=2))
