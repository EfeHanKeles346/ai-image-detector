"""Acquire one pinned DEAR-r research checkpoint; no image/model execution."""
from __future__ import annotations
import argparse
import fcntl
import json
from pathlib import Path
import time
import requests
from experiments.e65_acquisition import digest,read,write_once
from experiments.e72_acquisition import resource_check
from pixelproof.project_paths import DATA_ROOT,ML_ROOT
ROOT=DATA_ROOT/'e78';EVIDENCE=ML_ROOT.parent/'evidence';RESEARCH=DATA_ROOT/'research/dear'
CONTRACT=ROOT/'acquisition_contract.json';WEIGHT=ROOT/'dear_r.pth';REPORT=ROOT/'acquisition.json'
REVISION='5b57350b0ee75553109b3844f3c8a9341fc7e707'
SHA='430fde11debe1850ab24af43945b4d68f2bb3ee52a837d84cf525eb603ccde97'
SIZE=94372114


def select(metadata):
    if metadata['sha']!=REVISION or metadata['gated'] or metadata['private'] or metadata['disabled']:
        raise ValueError('pinned public checkpoint required')
    rows=[r for r in metadata['siblings'] if r['rfilename']=='dear_r/model_best.pth']
    if len(rows)!=1 or rows[0]['size']!=SIZE or rows[0]['lfs']['size']!=SIZE or rows[0]['lfs']['sha256']!=SHA:
        raise ValueError('DEAR-r payload metadata differs')
    return {'url':f'https://huggingface.co/k-aisi-anti-deepfake/dear-checkpoints/resolve/{REVISION}/dear_r/model_best.pth',
            'bytes':SIZE,'sha256':SHA,'path':str(WEIGHT)}


def write_stream(response,part,expected,deadline):
    if response.status_code!=200 or response.headers.get('Content-Encoding','identity') not in ('identity',''):
        raise ValueError('uncompressed complete HTTP200 required')
    length=response.headers.get('Content-Length')
    if length is not None and int(length)!=expected:raise ValueError('unexpected payload length')
    count=0
    with part.open('xb') as f:
        for chunk in response.iter_content(1024**2):
            resource_check(deadline)
            if not chunk:continue
            count+=len(chunk)
            if count>expected:raise ValueError('payload exceeded cap')
            f.write(chunk)
    if count!=expected:raise ValueError('incomplete payload')
    return count


def freeze():
    ROOT.mkdir(exist_ok=True)
    files=[Path(__file__),RESEARCH/'model_metadata.json',RESEARCH/'code_tree.json',
           RESEARCH/'inference_dependency_metadata.json',RESEARCH/'weights/LICENSE',RESEARCH/'weights/NOTICE',
           RESEARCH/'weights/README.md',Path(__file__).with_name('e72_acquisition.py')]
    for receipt in [read(EVIDENCE/'dear_metadata_lead.json'),read(RESEARCH/'inference_dependency_metadata.json')]:
        for item in receipt['files']:
            path=Path(item['path'])
            if digest(path)!=item['sha256']:raise ValueError('reviewed upstream text changed')
            files.append(path)
    row=select(read(RESEARCH/'model_metadata.json'))
    c={'state':'E78_DEAR_r_single_checkpoint_acquisition_registered','inputs':{str(p):digest(p) for p in files},
       'checkpoint':row,'max_seconds':900,'image_bytes':0,'image_scores':0,'third_party_execution_allowed':False,
       'rationale':'Fixed aligned-pair pretrained forensic alternative to semantic DINO/CLIP features. '
                   'DEAR-r selected from author training design before any local DEAR scores; no c/r local sweep.',
       'license':'Weights CC-BY-NC4.0 plus NOTICE SD1.5 OpenRAIL use restrictions; upstream AlignedForensics '
                 'has no explicit license. Local research only; no serving or redistribution of weights.',
       'next':'Only separately registered synthetic inference/parity/resource checks after strict weights_only load. '
              'Any image-feature extraction or classifier fit needs a separate contract.',
       'transfer_limits':'Author postprocessing robustness is not proof of low phone REAL FPR or modern AI retention.',
       'promotion_allowed':False}
    write_once(CONTRACT,c);write_once(EVIDENCE/'e78_acquisition_contract.json',c|{'contract_sha256':digest(CONTRACT)})
    return {'state':c['state'],'bytes':SIZE}


def validate():
    c=read(CONTRACT)
    if digest(CONTRACT)!=read(EVIDENCE/'e78_acquisition_contract.json')['contract_sha256']:raise ValueError('E78 contract changed')
    for p,s in c['inputs'].items():
        if digest(p)!=s:raise ValueError('E78 source changed: '+p)
    return c


def download():
    c=validate()
    if REPORT.exists():raise FileExistsError('E78 acquisition already complete')
    start=time.monotonic();deadline=start+c['max_seconds'];resource_check(deadline);row=c['checkpoint'];part=WEIGHT.with_suffix('.pth.part')
    reused=WEIGHT.exists()
    if not reused:
        if part.exists():
            if part.stat().st_size==SIZE and digest(part)==SHA:part.replace(WEIGHT);reused=True
            else:raise ValueError('incomplete prior payload retained; explicit engineering retry required')
        if not reused:
            with requests.get(row['url'],headers={'Accept-Encoding':'identity'},stream=True,timeout=(15,45)) as response:
                write_stream(response,part,SIZE,deadline)
            if digest(part)!=SHA:raise ValueError('publisher SHA256 mismatch')
            part.replace(WEIGHT)
    if WEIGHT.stat().st_size!=SIZE or digest(WEIGHT)!=SHA:raise ValueError('checkpoint bytes differ')
    result={'state':'E78_DEAR_r_weights_verified_unexecuted','contract_sha256':digest(CONTRACT),
        'weight_sha256':SHA,'bytes':SIZE,'downloaded_bytes':0 if reused else SIZE,'seconds':time.monotonic()-start,
        'image_bytes':0,'image_scores':0,'third_party_code_executed':False,'promotion_allowed':False}
    write_once(REPORT,result);write_once(EVIDENCE/'e78_acquisition.json',result);return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=['freeze','download'])
    ROOT.mkdir(exist_ok=True)
    with (ROOT/'acquisition_execution.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        print(json.dumps({'freeze':freeze,'download':download}[parser.parse_args().stage](),indent=2))
