"""Bounded official MNW evaluation-only acquisition. Never trains or scores a model."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
from io import BytesIO
import json
from pathlib import Path
import re
from urllib.parse import quote

import requests
from PIL import Image

from experiments.e53_offline import EVIDENCE,digest,fixed_write
from experiments.e51_prefit_audit import fingerprint
from pixelproof.project_paths import DATA_ROOT

ROOT=DATA_ROOT/'e52/mnw_reserve_v1'
MANIFEST=ROOT/'manifest.json'
REVISION='c93abf43e8157558a0e60aab7df4278b2c539253'
TREES={'Adobe_Firefly_v4':'508d04f9c2eb6c9408e36ada63385f24f6fe7967',
       'Flux_2_pro':'1c1b05d60d33b3994075c7cdd23dce2eb45158de',
       'Google_Imagen4':'aef7b6c6969279533f8bb2aa86d499cafad5e84c',
       'MAI_image2':'c3ece93bf41edde04c2364e0706e5effc3756508',
       'Midjourney_v8':'bb71d7d7bda5067193bdd770a0f466b1587ed839',
       'OpenAI_GPTimage2':'bca8a9c2786f3d9d6317e09cc3f01dfa6fb4e1ef'}
MAX_IMAGE=12*1024**2
MAX_TRANSFER=768*1024**2


def bounded_get(url,maximum):
    with requests.get(url,timeout=(15,60),stream=True) as response:
        response.raise_for_status();parts=[];size=0
        for part in response.iter_content(64*1024):
            size+=len(part)
            if size>maximum:raise ValueError('response exceeds declared byte cap')
            parts.append(part)
    return b''.join(parts)


def parse_pointer(raw, git_blob):
    if hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()!=git_blob:
        raise ValueError('Git pointer blob changed')
    match=re.fullmatch(rb'version https://git-lfs.github.com/spec/v1\noid sha256:([0-9a-f]{64})\nsize ([0-9]+)\n',raw)
    if not match:raise ValueError('not an exact expected LFS pointer')
    size=int(match[2])
    if not 0<size<=MAX_IMAGE:raise ValueError('selected image violates fixed size cap')
    return match[1].decode(),size


def freeze():
    if MANIFEST.exists():raise FileExistsError('MNW reserve already frozen')
    selected=[];metadata_bytes=0
    readme=bounded_get(f'https://raw.githubusercontent.com/microsoft/MNW/{REVISION}/README.md',128*1024)
    if b'cannot be used for training' not in readme:
        raise ValueError('evaluation-only source terms require review')
    for family,tree in TREES.items():
        raw=bounded_get(f'https://api.github.com/repos/microsoft/MNW/git/trees/{tree}',2*1024**2)
        metadata_bytes+=len(raw);value=json.loads(raw)
        if value['sha']!=tree or value.get('truncated'):raise ValueError('incomplete folder metadata')
        rows=[r for r in value['tree'] if r['type']=='blob' and r['mode']=='100644' and Path(r['path']).suffix.lower() in {'.jpg','.jpeg','.png','.webp'}]
        if len(rows)<50:raise ValueError('fixed family quota unavailable')
        ranked=sorted(rows,key=lambda r:hashlib.sha256(f"MNW_E52|{family}|{r['path']}".encode()).hexdigest())[:50]
        selected.extend({'family':family,'path':f"AI_Images/{family}/{r['path']}",'git_blob':r['sha']} for r in ranked)
    def resolve(row):
        raw=bounded_get(f"https://raw.githubusercontent.com/microsoft/MNW/{REVISION}/{quote(row['path'])}",1024)
        sha,size=parse_pointer(raw,row['git_blob'])
        return {**row,'sha256':sha,'bytes':size,'pointer_bytes':len(raw),'label':1,'role':'RESERVED_EXTERNAL_EVALUATION',
                'parent_id':'MNW:'+sha,'source':'Microsoft-Northwestern-WITNESS',
                'prompt_group':'unknown; reserve whole publisher','url':f"https://media.githubusercontent.com/media/microsoft/MNW/{REVISION}/{quote(row['path'])}"}
    with ThreadPoolExecutor(max_workers=4) as pool:rows=list(pool.map(resolve,selected))
    if len({r['sha256'] for r in rows})!=len(rows):raise ValueError('duplicate selected image; no silent replacement')
    total=sum(r['bytes'] for r in rows)
    if total>MAX_TRANSFER:raise ValueError('fixed image selection exceeds total cap')
    value={'state':'MNW_exact_reserve_frozen_before_images','revision':REVISION,'code_sha256':digest(__file__),
        'trees':TREES,'rows':rows,'parents':len(rows),'image_bytes_expected':total,'image_transfer_cap':MAX_TRANSFER,
        'metadata_bytes_this_freeze':metadata_bytes+len(readme)+sum(r['pointer_bytes'] for r in rows),
        'source_terms':readme.decode(),'source_terms_sha256':hashlib.sha256(readme).hexdigest(),
        'training_allowed':False,'threshold_calibration_allowed':False,'noncommercial_evaluation_only':True,
        'model_scores_created':0,'balanced_final_admitted':False}
    fixed_write(MANIFEST,value)
    fixed_write(EVIDENCE/'e54_mnw_reserve_manifest.json',{k:v for k,v in value.items() if k not in {'rows','source_terms'}}|
                {'manifest_sha256':digest(MANIFEST)})
    return {k:v for k,v in value.items() if k not in {'rows','source_terms'}}


def download():
    value=json.loads(MANIFEST.read_text());receipt=json.loads((EVIDENCE/'e54_mnw_reserve_manifest.json').read_text())
    if digest(MANIFEST)!=receipt['manifest_sha256'] or value['code_sha256']!=digest(__file__):
        raise ValueError('MNW frozen selection/code changed')
    if (EVIDENCE/'e54_mnw_reserve_download.json').exists():raise FileExistsError('download already complete')
    ledger=ROOT/'transfer_ledger.json'
    spent=json.loads(ledger.read_text())['charged_image_transfer_bytes'] if ledger.exists() else 0
    records=[]
    for row in value['rows']:
        path=ROOT/'images'/(row['sha256']+'.image');path.parent.mkdir(parents=True,exist_ok=True)
        if not path.exists():
            part=path.with_suffix('.part');offset=part.stat().st_size if part.exists() else 0
            if offset>row['bytes']:raise ValueError('partial image exceeds expected size')
            if offset<row['bytes']:
                # Reserve the whole remaining request before network access. Failed requests
                # remain charged, bounding retries without claiming all charged bytes arrived.
                spent+=row['bytes']-offset
                if spent>MAX_TRANSFER:raise ValueError('conservative transfer budget exhausted')
                dino_atomic(ledger,{'charged_image_transfer_bytes':spent})
                headers={'Range':f'bytes={offset}-'} if offset else {}
                with requests.get(row['url'],headers=headers,stream=True,timeout=(15,60)) as response:
                    response.raise_for_status()
                    if offset and (response.status_code!=206 or not response.headers.get('Content-Range','').startswith(f'bytes {offset}-')):
                        raise ValueError('server refuses exact partial resume; preserve bytes')
                    with part.open('ab' if offset else 'wb') as stream:
                        for block in response.iter_content(64*1024):
                            offset+=len(block)
                            if offset>row['bytes']:raise ValueError('image byte cap exceeded')
                            stream.write(block)
            if part.stat().st_size!=row['bytes'] or digest(part)!=row['sha256']:
                raise ValueError('download SHA/size mismatch')
            part.replace(path)
        raw=path.read_bytes()
        if len(raw)!=row['bytes'] or hashlib.sha256(raw).hexdigest()!=row['sha256']:raise ValueError('saved body changed')
        facts=fingerprint(raw)
        records.append({**row,**facts,'path':str(path),'condition':'original'})
        if len(records)%25==0:print(json.dumps({'phase':'MNW_reserve_download','images':len(records),'total':len(value['rows']),'charged_image_transfer_bytes':spent}),flush=True)
    result={'state':'MNW_reserve_downloaded_not_final_admitted','manifest_sha256':digest(MANIFEST),
            'parents':len(records),'charged_image_transfer_bytes':spent,
            'verified_image_bytes':sum(r['bytes'] for r in value['rows']),'records':records,
            'model_scores_created':0,'training_allowed':False,'protected_overlap_audit_pending':True}
    fixed_write(ROOT/'download.json',result)
    fixed_write(EVIDENCE/'e54_mnw_reserve_download.json',{k:v for k,v in result.items() if k!='records'}|
                {'report_sha256':digest(ROOT/'download.json')})
    return {k:v for k,v in result.items() if k!='records'}


def dino_atomic(path,value):
    part=path.with_suffix('.json.part');part.write_text(json.dumps(value));part.replace(path)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['freeze','download']);args=parser.parse_args()
    print(json.dumps({'freeze':freeze,'download':download}[args.phase](),indent=2))
