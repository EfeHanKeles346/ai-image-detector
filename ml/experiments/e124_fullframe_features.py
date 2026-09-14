"""Complete resumable full-frame CLIP features after E123 parity; TRAIN only."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import socket
import time
import numpy as np
from PIL import Image
from experiments import e123_fullframe_probe as pilot
from experiments.e65_acquisition import digest,read,write_once
from experiments.e71_features import load_encoder,save_npz,array_sha
from experiments.e72_acquisition import resource_check
from pixelproof.project_paths import DATA_ROOT,ML_ROOT

ROOT=DATA_ROOT/'e124';EVIDENCE=ML_ROOT.parent/'evidence';CONTRACT=ROOT/'contract.json'
CONDITIONS=['clean','assigned_transport','q75','social_q75']


def verify_chunk(a,rows,binding):
    full=a['full']
    if str(a['binding'])!=binding or list(a['parents'])!=[r['parent_id'] for r in rows] or \
            list(a['sources_sha256'])!=[r['sha256'] for r in rows] or list(a['roles'])!=['TRAIN']*len(rows) or \
            list(a['conditions'])!=CONDITIONS or full.shape!=(len(rows),4,768) or full.dtype!=np.float32 or \
            not np.isfinite(full).all() or str(a['array_sha256'])!=array_sha(full):
        raise ValueError('Full-frame chunk identity/coverage/integrity differs')
    return full


def freeze():
    c=read(pilot.CONTRACT);report=read(pilot.ROOT/'report.json')
    if digest(pilot.CONTRACT)!=read(EVIDENCE/'e123_fullframe_contract.json')['contract_sha256'] or \
            digest(pilot.ROOT/'report.json')!=digest(EVIDENCE/'e123_fullframe_probe.json') or \
            report['state']!='E123_fullframe_feasibility_passed' or report['views']!=160 or \
            max(report['max_center_vector_error'],report['max_repeat_error'])>1e-5 or \
            digest(pilot.ROOT/'features.npz')!=report['features_sha256']:
        raise ValueError('Complete passing full-frame probe required')
    for p,s in c['inputs'].items():
        if digest(p)!=s:raise ValueError('Bound feasibility input differs')
    files=[Path(__file__),pilot.CONTRACT,pilot.ROOT/'report.json',pilot.ROOT/'features.npz',Path(pilot.__file__)]
    contract={'state':'E124_complete_fullframe_TRAIN_features_registered','inputs':c['inputs']|{str(p):digest(p) for p in files},
        'parents':12525,'views':50100,'conditions':CONDITIONS,'window_parents':3,'encoder_batch':3,
        'representation':'Exact E123 uncropped available-frame warp224 and matched JPEG90, all four unchanged transports. Encode12 full-frame vectors per3-parent window in four batches of3; append no labels or model scores. Preserve E112 old caches unchanged.',
        'acceptance':'Complete original parent/source/role/condition ordering; finite float32 vectors; every E123 selected source/condition must reproduce its full-frame vector within1e-5. New chunks and resumed chunks need body/array/source/role hashes; no orphan adoption.',
        'max_seconds_per_execution':14400,'mps_limit_bytes':6*1024**3,'parity_max_abs':1e-5,
        'downloads':0,'model_scores':0,'fit_allowed':False,
        'next':'Separately register a full-frame versus center-only PCA128 correction comparison on all50100 TRAIN views using frozen E103 base and unchanged all-AI/correct-REAL guards. No DEV before complete passing TRAIN.',
        'limits':c['limits']}
    ROOT.mkdir(exist_ok=True);write_once(CONTRACT,contract)
    write_once(EVIDENCE/'e124_fullframe_contract.json',contract|{'contract_sha256':digest(CONTRACT)})
    return {'parents':12525,'views':50100}


def validate():
    c=read(CONTRACT)
    if digest(CONTRACT)!=read(EVIDENCE/'e124_fullframe_contract.json')['contract_sha256']:raise ValueError('Full extraction contract differs')
    for p,s in c['inputs'].items():
        if digest(p)!=s:raise ValueError('Frozen full extraction input differs')
    return c


def extract():
    import torch
    c=validate();binding=digest(CONTRACT)
    if (ROOT/'report.json').exists():raise FileExistsError('Full-frame extraction already complete')
    rows=read(DATA_ROOT/'e112/contract.json')['rows']
    if len(rows)!=12525 or any(str(r['role']).upper()!='TRAIN' for r in rows):raise ValueError('Complete TRAIN population required')
    start=time.monotonic();deadline=start+c['max_seconds_per_execution'];resource_check(deadline);torch.set_num_threads(2)
    with np.load(pilot.ROOT/'features.npz',allow_pickle=False) as a:
        if str(a['binding'])!=digest(pilot.CONTRACT):raise ValueError('Pilot cache binding differs')
        probe={str(p):x.copy() for p,x in zip(a['parents'],a['full'],strict=True)}
    model,preprocess,device=load_encoder();chunks=ROOT/'chunks';chunks.mkdir(exist_ok=True)
    full=np.empty((12525,4,768),dtype=np.float32);receipts={};created=0;resumed=0;checked=0;error=0.;peak=0
    with torch.inference_mode():
        for offset in range(0,len(rows),3):
            resource_check(deadline);group=rows[offset:offset+3]
            for r in group:
                if digest(r['path'])!=r['sha256']:raise ValueError('TRAIN original changed')
            path=chunks/f'{offset:05d}.npz';receipt=path.with_suffix('.json')
            if path.exists() and not receipt.exists():raise ValueError('Unreceipted full-frame chunk; explicit review needed')
            if path.exists():
                r=read(receipt)
                if r['binding']!=binding or r['sha256']!=digest(path):raise ValueError('Full-frame receipt differs')
                with np.load(path,allow_pickle=False) as a:values=verify_chunk(a,group,binding).copy()
                resumed+=len(group)
            else:
                pixels=[]
                for r in group:
                    body=Path(r['path']).read_bytes()
                    pixels.extend(pilot.fullframe_array(im) for im in pilot.transported_images(body,r['parent_id']))
                values=np.concatenate([model.encode_image(torch.stack([preprocess(Image.fromarray(x)) for x in pixels[j:j+3]]).to(device)).float().cpu().numpy()
                    for j in range(0,len(pixels),3)]).reshape(len(group),4,768).astype(np.float32)
                a={'full':values,'binding':binding,'parents':np.array([r['parent_id'] for r in group]),
                    'sources_sha256':np.array([r['sha256'] for r in group]),'roles':np.repeat('TRAIN',len(group)),
                    'conditions':np.array(CONDITIONS),'array_sha256':array_sha(values)}
                verify_chunk(a,group,binding);save_npz(path,**a)
                write_once(receipt,{'binding':binding,'sha256':digest(path)});created+=len(group)
            for r,x in zip(group,values,strict=True):
                if r['parent_id'] in probe:
                    error=max(error,float(np.abs(x-probe[r['parent_id']]).max()));checked+=4
                    if error>c['parity_max_abs']:raise ValueError('Full-frame batch/reference parity differs')
            full[offset:offset+len(group)]=values;receipts[str(path)]=digest(path)
            peak=max(peak,torch.mps.driver_allocated_memory() if device.type=='mps' else 0)
            if peak>c['mps_limit_bytes']:raise RuntimeError('Full-frame MPS memory bound')
            if (offset+len(group))%150==0:
                progress={'completed_parents':offset+len(group),'total_parents':12525,'seconds':round(time.monotonic()-start),'max_pilot_error':error}
                (ROOT/'progress.json').write_text(json.dumps(progress,indent=2)+'\n');print(json.dumps(progress),flush=True)
    if checked!=160:raise ValueError('Incomplete full-frame feasibility replay')
    output=ROOT/'features.npz'
    if output.exists():raise FileExistsError('Complete output lacks receipt; inspect integrity')
    save_npz(output,full=full,parents=np.array([r['parent_id'] for r in rows]),roles=np.repeat('TRAIN',len(rows)),
        conditions=np.array(CONDITIONS),binding=binding,array_sha256=array_sha(full))
    write_once(ROOT/'chunks.json',{'binding':binding,'chunks':receipts})
    result={'state':'E124_fullframe_TRAIN_cache_complete','contract_sha256':binding,'feature_sha256':digest(output),
        'chunks_sha256':digest(ROOT/'chunks.json'),'parents':len(rows),'views':50100,'created_parents':created,'resumed_parents':resumed,
        'pilot_views_replayed':checked,'max_pilot_error':error,'peak_mps_bytes':peak,'seconds':time.monotonic()-start,
        'model_scores':0,'downloads':0,'fit_allowed':False,'next':c['next']}
    write_once(ROOT/'report.json',result);write_once(EVIDENCE/'e124_fullframe_features.json',result)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['freeze','extract']);args=p.parse_args()
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    def denied(*a,**kw):raise RuntimeError('Full-frame extraction is offline')
    socket.socket.connect=denied;socket.socket.connect_ex=denied;socket.create_connection=denied
    ROOT.mkdir(exist_ok=True)
    with (ROOT/'execution.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        print(json.dumps({'freeze':freeze,'extract':extract}[args.stage](),indent=2))
