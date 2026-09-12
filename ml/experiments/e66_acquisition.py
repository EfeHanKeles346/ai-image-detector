"""Pinned SIDD Small sRGB acquisition for grouped development, no model inference."""
from __future__ import annotations
import argparse
import fcntl
import json
from pathlib import Path
import shutil
import time
import requests
from experiments.e65_acquisition import digest, read, write_once, verify_file
from pixelproof.project_paths import DATA_ROOT, ML_ROOT

ROOT=DATA_ROOT/'e66'
EVIDENCE=ML_ROOT.parent/'evidence'
CONTRACT=ROOT/'acquisition_contract.json'
ARCHIVE=ROOT/'SIDD_Small_sRGB_Only.zip'
RECEIPT=ROOT/'download.json'
URL='http://130.63.97.225/share/SIDD_Small_sRGB_Only.zip'


def freeze():
    metadata=[ROOT/'sidd_site.html',ROOT/'sidd_download.html',ROOT/'sidd_headers.json']
    header=next(r for r in read(metadata[-1]) if r['url']==URL)
    if header['status']!=200 or int(header['headers']['Content-Length'])!=6615978508:
        raise ValueError('publisher archive size/status differs')
    checksum='796971867583bf14677dcae510e52538';sha1='5a4aa6aa7abcf7b0b56c88ad578fea9a4ff77935'
    if checksum not in metadata[1].read_text() or sha1 not in metadata[1].read_text():
        raise ValueError('published checksum missing')
    c={'state':'E66_SIDD_Small_acquisition_registered','inputs':{str(p):digest(p) for p in
        [*metadata,Path(__file__),Path(__file__).with_name('e65_acquisition.py')]},
        'row':{'url':URL,'path':str(ARCHIVE),'bytes':6615978508,'checksum_type':'md5',
               'checksum':checksum,'body_sha1':sha1},'max_seconds':7200,'workers':1,
        'expected_pairs':160,'underlying_scenes_at_most':10,'publisher_license':'MIT',
        'role':'QUARANTINED_DEVELOPMENT_CANDIDATE','whole_publisher_train_excluded':True,
        'training_allowed':False,'model_scoring_allowed':False,'balanced_final_admitted':False,
        'policy':'Download complete published Small sRGB archive only, verify size+MD5+SHA1 and record SHA256. '
                 'No archive member execution. ZIP path/size/decode/group/protected-overlap audit required '
                 'before separate DEV admission. Prefer noisy camera-derived sRGB for DEV; GT remains '
                 'processed companion evidence, not an additional independent photo. Official benchmark untouched.',
        'limits':'Only10 underlying scenes and5 older phones; no modern-camera/fresh-final coverage claim. '
                 'Codalab mirror HEAD returned403; official direct mirror supports ranges. '
                 'HTTP transport checksums are publisher-integrity checks, not an independent authenticated signature.'}
    write_once(CONTRACT,c);write_once(EVIDENCE/'e66_acquisition_contract.json',c|{'contract_sha256':digest(CONTRACT)})
    return {'state':c['state'],'bytes':c['row']['bytes']}


def validate():
    c=read(CONTRACT)
    if digest(CONTRACT)!=read(EVIDENCE/'e66_acquisition_contract.json')['contract_sha256']:
        raise ValueError('frozen acquisition contract differs')
    for p,s in c['inputs'].items():
        if digest(p)!=s:raise ValueError(f'bound input changed: {p}')
    return c


def download():
    c=validate();r=c['row'];deadline=time.monotonic()+c['max_seconds']
    if RECEIPT.exists():raise FileExistsError('complete acquisition receipt exists')
    if not Path('/Volumes/LaCie').is_mount() or shutil.disk_usage(ROOT).free<30*1024**3:
        raise RuntimeError('volume missing or below30GiB reserve')
    if not ARCHIVE.exists():
        partial=ARCHIVE.with_suffix('.zip.part');last_progress=0
        for attempt in range(3):
            offset=partial.stat().st_size if partial.exists() else 0
            if offset>r['bytes']:raise ValueError('partial larger than expected archive')
            if offset==r['bytes']:
                verify_file(partial,r);partial.replace(ARCHIVE);break
            try:
                with requests.get(r['url'],headers={'Range':f'bytes={offset}-'} if offset else {},
                                  stream=True,timeout=(15,60)) as response:
                    response.raise_for_status()
                    if response.status_code==206:
                        if not response.headers.get('Content-Range','').startswith(f'bytes {offset}-'):
                            raise ValueError('incorrect resumed range')
                        mode='ab'
                    elif response.status_code==200: mode='wb';offset=0
                    else:raise ValueError('unexpected HTTP status')
                    with partial.open(mode) as out:
                        for chunk in response.iter_content(1024**2):
                            now=time.monotonic()
                            if now>=deadline:raise TimeoutError('archive runtime ceiling reached; partial preserved')
                            offset+=len(chunk)
                            if offset>r['bytes']:raise ValueError('body exceeds pinned length')
                            out.write(chunk)
                            if now-last_progress>=15:
                                print(json.dumps({'downloaded_bytes':offset,'total_bytes':r['bytes'],
                                                  'percent':round(100*offset/r['bytes'],1)}),flush=True)
                                last_progress=now
                verify_file(partial,r);partial.replace(ARCHIVE);break
            except requests.RequestException as error:
                print(json.dumps({'attempt_failed':attempt+1,'error':type(error).__name__}),flush=True)
                if attempt==2:raise
    sha=verify_file(ARCHIVE,r)
    result={'state':'E66_SIDD_archive_verified_quarantine','archive_sha256':sha,
            'archive_bytes':r['bytes'],'contract_sha256':digest(CONTRACT),'training_allowed':False,
            'model_scores_created':0,'decoded_images':0,'balanced_final_admitted':False}
    write_once(RECEIPT,result);write_once(EVIDENCE/'e66_download.json',result|{'receipt_sha256':digest(RECEIPT)})
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=('freeze','download'))
    with (ROOT/'execution.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        print(json.dumps({'freeze':freeze,'download':download}[parser.parse_args().stage](),indent=2))
