"""Bounded, source-stratified FiveK TRAIN supplement. No detector scores."""
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

import certifi
import requests

from experiments.e53_offline import EVIDENCE, digest, fixed_write
from experiments.e51_prefit_audit import fingerprint, cross_role_matches
from experiments.e54_reserve_reference import SNAPSHOT
from experiments.e54_mnw_reserve import ROOT as MNW
from experiments.e54_hdrplus_reserve import ROOT as HDR
from experiments.e56_fivek_metadata import ROOT as META, BASE
from experiments import e56_fivek_pilot as pilot

ROOT = META.parent.parent/'e57'
CONTRACT = ROOT/'data_contract.json'
MANIFEST = ROOT/'admitted.json'
MAX_IMAGE = 48*1024**2


def select(rows):
    if len({r['filename'] for r in rows}) != len(rows): raise ValueError('duplicate identity')
    groups = defaultdict(list)
    for row in rows: groups[(row['subject'], row['light'])].append(row)
    if len(groups) != 18: raise ValueError('subject/light cells changed')
    return [r for key in sorted(groups) for r in sorted(groups[key],
        key=lambda r: hashlib.sha256(('FIVEK_E57_TRAIN_V1|'+r['filename']).encode()).hexdigest())[:12]]


def safe(deadline):
    pilot.safe(deadline)
    if shutil.disk_usage(ROOT.parent).free < 20*1024**3: raise RuntimeError('less than 20 GiB free')


def verify_headers(headers, row, offset=0, status=200):
    if offset and (status != 206 or headers.get('Content-Range') != f"bytes {offset}-{row['bytes']-1}/{row['bytes']}"):
        raise ValueError('resume range mismatch')
    if not offset and status != 200: raise ValueError('expected complete body response')
    if int(headers.get('Content-Length', '0')) != row['bytes']-offset: raise ValueError('body size changed')
    for name, field in (('ETag', 'etag'), ('Last-Modified', 'last_modified')):
        if row[field] and headers.get(name) != row[field]: raise ValueError('body validator changed')


def freeze():
    deadline = time.monotonic()+3600; safe(deadline)
    metadata = META/'manifest_v2.json'
    if digest(metadata) != json.loads((EVIDENCE/'e56_fivek_metadata.json').read_text())['manifest_sha256']:
        raise ValueError('metadata changed')
    runtime = pilot.decoder(['--versions'], deadline)
    pc = pilot.load()
    if runtime != pc['runtime']: raise ValueError('pilot decoder changed')
    rows = select(json.loads(metadata.read_text())['rows'])
    if len(rows) != 201: raise ValueError('selected population changed')
    def head(row):
        safe(deadline)
        with requests.head(row['original_url'], timeout=(10,30), allow_redirects=True,
                headers={'Accept-Encoding':'identity'}, verify=certifi.where()) as response:
            response.raise_for_status()
            size = int(response.headers.get('Content-Length','0'))
            etag, modified = response.headers.get('ETag'), response.headers.get('Last-Modified')
            if not response.url.startswith(BASE) or not 0 < size <= MAX_IMAGE or not (etag or modified):
                raise ValueError('invalid/oversize selected resource: '+row['filename'])
        return row | {'parent_id':'FiveK:'+row['filename'], 'source':'MIT-Adobe-FiveK', 'label':0,
                      'role':'QUARANTINED_TRAIN_CANDIDATE', 'bytes':size,'etag':etag,'last_modified':modified}
    with ThreadPoolExecutor(max_workers=4) as pool: selected = list(pool.map(head, rows))
    total = sum(r['bytes'] for r in selected)
    if total > 4*1024**3: raise ValueError('total original payload exceeds 4 GiB')
    inputs = [metadata, pilot.CONTRACT, pilot.ROOT/'download.json', pilot.DECODER, pilot.ENV/'install_report.json',
              Path(pilot.__file__), Path(__file__).with_name('e51_prefit_audit.py'),
              Path(__file__).with_name('e55_resume.py'), META/'LicenseAdobe.txt', META/'LicenseAdobeMIT.txt']
    refs = [(SNAPSHOT,'e54_reserve_reference.json','snapshot_sha256'),
            (MNW/'download.json','e54_mnw_reserve_download.json','report_sha256'),
            (HDR/'download.json','e54_hdrplus_download.json','report_sha256'),
            (pilot.ROOT/'download.json','e56_fivek_pilot_download.json','report_sha256')]
    for path, receipt, key in refs:
        if digest(path) != json.loads((EVIDENCE/receipt).read_text())[key]: raise ValueError('reference changed')
        inputs.append(path)
    value = {'state':'E57_data_frozen_before_GET','code_sha256':digest(__file__),
        'inputs':{str(p):digest(p) for p in inputs}, 'rows':selected, 'image_bytes_expected':total,
        'image_cap':MAX_IMAGE,'total_cap':4*1024**3,'max_runtime_seconds':3600,'runtime':runtime,
        'decoder_recipe':pc['recipe'],'role':'FIT-only after protected audit; never CAL/validation/final',
        'policy':'No replacements for excluded matches or failed resources; both internal endpoints excluded.',
        'source_integrity_limit':'HTTP validators and local SHA; no official per-file SHA available.',
        'model_scores':0,'training_admitted':False}
    fixed_write(CONTRACT,value);fixed_write(EVIDENCE/'e57_data_contract.json',value|{'contract_sha256':digest(CONTRACT)})
    return {'parents':len(selected),'bytes':total,'contract_sha256':digest(CONTRACT)}


def load():
    value = json.loads(CONTRACT.read_text())
    if digest(__file__) != value['code_sha256'] or digest(CONTRACT) != json.loads((EVIDENCE/'e57_data_contract.json').read_text())['contract_sha256']:
        raise ValueError('E57 frozen acquisition changed')
    for path, expected in value['inputs'].items():
        if digest(path) != expected: raise ValueError('E57 acquisition input changed: '+path)
    return value


def acquire_raw(row, binding, deadline):
    safe(deadline)
    folder = ROOT/'raw'; folder.mkdir(exist_ok=True)
    target = folder/row['filename']; part = target.with_suffix('.dng.part'); receipt = target.with_suffix('.receipt.json')
    if not target.exists() and part.exists() and part.stat().st_size == row['bytes']:
        fixed_write(receipt,{'sha256':digest(part),'contract_sha256':binding,'bytes':row['bytes']});part.replace(target)
    if target.exists():
        result = json.loads(receipt.read_text())
        if result['contract_sha256'] != binding or digest(target) != result['sha256'] or target.stat().st_size != row['bytes']:
            raise ValueError('cached original changed')
        return target, result['sha256']
    offset = part.stat().st_size if part.exists() else 0
    headers = {'Accept-Encoding':'identity'}
    if row['etag']: headers['If-Match'] = row['etag']
    else: headers['If-Unmodified-Since'] = row['last_modified']
    if offset: headers['Range'] = f'bytes={offset}-'
    with requests.get(row['original_url'], headers=headers, stream=True, timeout=(10,30), verify=certifi.where()) as response:
        response.raise_for_status()
        if not response.url.startswith(BASE): raise ValueError('off-source image redirect')
        verify_headers(response.headers,row,offset,response.status_code)
        checked = time.monotonic()
        with part.open('ab' if part.exists() else 'xb') as stream:
            for chunk in response.iter_content(65536):
                offset += len(chunk)
                if offset > row['bytes']: raise ValueError('image exceeds frozen size')
                if time.monotonic()-checked >= 5: safe(deadline); checked=time.monotonic()
                stream.write(chunk)
            stream.flush();os.fsync(stream.fileno())
    if offset != row['bytes']: raise ValueError('truncated selected image')
    sha = digest(part);fixed_write(receipt,{'sha256':sha,'contract_sha256':binding,'bytes':offset});part.replace(target)
    return target,sha


def download():
    config=load();binding=digest(CONTRACT);deadline=time.monotonic()+3600;safe(deadline)
    (ROOT/'png').mkdir(exist_ok=True);(ROOT/'records').mkdir(exist_ok=True)
    for name in ('LicenseAdobe.txt','LicenseAdobeMIT.txt'):
        target=ROOT/name
        if not target.exists():target.write_bytes((META/name).read_bytes())
        if digest(target)!=config['inputs'][str(META/name)]:raise ValueError('licence copy mismatch')
    old={r['filename']:r for r in json.loads((pilot.ROOT/'download.json').read_text())['records']}
    records=[];reused=0
    for row in config['rows']:
        safe(deadline);cached=ROOT/'records'/(row['filename']+'.json')
        if cached.exists():
            record=json.loads(cached.read_text())
            if record['contract_sha256']!=binding or digest(record['path'])!=record['sha256'] or digest(record['original_path'])!=record['original_sha256']:
                raise ValueError('completed parent changed')
        else:
            if row['filename'] in old:
                previous=old[row['filename']];rawpath=pilot.ROOT/row['filename'];png=Path(previous['path'])
                sha=previous['original_sha256'];result=previous['decode']
                if (digest(rawpath)!=sha or digest(png)!=previous['sha256'] or row['bytes']!=previous['original_bytes']
                    or row['etag']!=previous['etag'] or row['last_modified']!=previous['last_modified']):
                    raise ValueError('selected pilot reuse differs')
                reused+=1
            else:
                rawpath,sha=acquire_raw(row,binding,deadline);png=ROOT/'png'/Path(row['filename']).with_suffix('.png')
                result=pilot.decoder([rawpath,png],deadline)
            if result['versions']!=config['runtime']:raise ValueError('decoder runtime changed')
            facts=fingerprint(png.read_bytes())
            if facts['pixel_sha256']!=result['pixel_sha256']:raise ValueError('decoded pixel mismatch')
            record=row|facts|{'path':str(png),'original_path':str(rawpath),'original_sha256':sha,
                'original_bytes':row['bytes'],'decode':result,'condition':'fixed_raw_development','contract_sha256':binding}
            fixed_write(cached,record)
        records.append(record)
        if len(records)%10==0:print(json.dumps({'phase':'E57_acquire_decode','parents':len(records),'total':len(config['rows'])}),flush=True)
    result={'state':'E57_acquired_quarantine','contract_sha256':binding,'records':records,'parents':len(records),
        'original_bytes':sum(r['original_bytes'] for r in records),'png_bytes':sum(r['bytes'] for r in records),
        'pilot_reused_parents':sum(Path(r['original_path']).parent==pilot.ROOT for r in records),
        'new_original_bytes':sum(r['original_bytes'] for r in records if Path(r['original_path']).parent!=pilot.ROOT),
        'model_scores':0,'training_admitted':False}
    fixed_write(ROOT/'download.json',result);fixed_write(EVIDENCE/'e57_download.json',{k:v for k,v in result.items() if k!='records'}|{'report_sha256':digest(ROOT/'download.json')})
    return {k:v for k,v in result.items() if k!='records'}


def audit():
    load();deadline=time.monotonic()+3600;safe(deadline)
    path=ROOT/'download.json'
    if digest(path)!=json.loads((EVIDENCE/'e57_download.json').read_text())['report_sha256']:raise ValueError('download receipt changed')
    rows=json.loads(path.read_text())['records']
    for row in rows:
        safe(deadline)
        if digest(row['path'])!=row['sha256'] or digest(row['original_path'])!=row['original_sha256']:raise ValueError('source body changed')
    refs=json.loads(SNAPSHOT.read_text())['records']+json.loads((MNW/'download.json').read_text())['records']+json.loads((HDR/'download.json').read_text())['records']
    cross=cross_role_matches(rows,refs)
    internal=[r for r in cross_role_matches(rows,rows) if r['train_parent']<r['cal_parent']]
    raw_hashes={r['sha256'] for r in refs}
    excluded={r['parent_id'] for r in rows if r['original_sha256'] in raw_hashes}
    excluded.update(r['train_parent'] for r in cross)
    for r in internal:excluded.update((r['train_parent'],r['cal_parent']))
    admitted=[r|{'role':'FIT_ONLY_RESEARCH'} for r in rows if r['parent_id'] not in excluded]
    if not admitted:raise ValueError('no eligible REAL supplement')
    value={'state':'E57_supplement_admitted_for_fixed_FIT_research','contract_sha256':digest(CONTRACT),
        'download_sha256':digest(path),'rows':admitted,'excluded_parents':sorted(excluded),
        'cross_matches':cross,'internal_pairs':internal,'reference_observations':len(refs),
        'model_scores':0,'serving_allowed':False,'final_evaluation_allowed':False,
        'limits':['Heuristic screen is not exhaustive semantic deduplication.','Old SLR local RAW renders, not modern phone camera JPEGs.']}
    fixed_write(MANIFEST,value);fixed_write(EVIDENCE/'e57_admission.json',{k:v for k,v in value.items() if k not in {'rows','cross_matches','internal_pairs'}}|
        {'parents':len(admitted),'cross_count':len(cross),'internal_count':len(internal),'manifest_sha256':digest(MANIFEST)})
    return {'parents':len(admitted),'excluded':len(excluded),'references':len(refs)}


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['freeze','download','audit']);args=parser.parse_args()
    # Local lock cannot fabricate an absent external mount and prevents duplicate acquisitions.
    work=EVIDENCE.parent/'ml/work';work.mkdir(exist_ok=True)
    with (work/'e57.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        print(json.dumps(globals()[args.phase](),indent=2))
