"""TRAIN-only E67 fixed-crop blur response features; restartable and offline."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import json
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
import time

if __name__=='__main__':
    def denied(*args, **kwargs):raise RuntimeError('E67 feature network disabled')
    socket.socket.connect=denied;socket.socket.connect_ex=denied;socket.create_connection=denied
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HUB_DISABLE_TELEMETRY='1')

import joblib
import numpy as np
from PIL import Image,ImageFilter
from threadpoolctl import threadpool_limits
from experiments.e65_acquisition import digest,read,write_once
from experiments import e42_features as dino
from pixelproof.project_paths import DATA_ROOT,ML_ROOT

ROOT=DATA_ROOT/'e67'
EVIDENCE=ML_ROOT.parent/'evidence'
CONTRACT=ROOT/'features_contract.json'
OUTPUT=ROOT/'blur_features.npz'
REPORT=ROOT/'features.json'
AI_CUT=0.07940196245908739
REAL_CUT=0.011505939625203613


def blur_crops(crops):
    if crops.dtype!=np.uint8 or crops.shape!=(3,3,224,224,3):
        raise ValueError('expected three conditions x three uint8 RGB crops')
    return np.stack([np.asarray(Image.fromarray(c).filter(ImageFilter.GaussianBlur(radius=.8)))
                     for c in crops.reshape(-1,224,224,3)]).reshape(crops.shape)


def resource_check(deadline):
    if time.monotonic()>deadline:raise TimeoutError('E67 feature runtime budget reached')
    if not Path('/Volumes/LaCie').is_mount() or shutil.disk_usage(ROOT).free<10*1024**3:
        raise RuntimeError('external volume missing/space floor')
    power=subprocess.check_output(['pmset','-g','batt'],text=True)
    match=re.search(r'(\d+)%;',power)
    if 'Battery Power' in power and (not match or int(match.group(1))<30):
        raise RuntimeError('battery below30%')


def freeze():
    previous=read(EVIDENCE/'e61_replay_contract.json')
    inputs={k:previous['inputs'][k] for k in ('manifest','features','reference')}
    receipt=read(EVIDENCE/'e54_crop_cache.json')
    inputs['crop_index']={'path':str(DATA_ROOT/'e54/crop_index.json'),'sha256':receipt['index_sha256']}
    for k,p in {'runner':Path(__file__),'feature_code':Path(dino.__file__),
                'preprocess_code':ML_ROOT/'src/pixelproof/e32_candidate.py',
                'helpers':Path(__file__).with_name('e65_acquisition.py')}.items():
        inputs[k]={'path':str(p),'sha256':digest(p)}
    for item in inputs.values():
        if digest(item['path'])!=item['sha256']:raise ValueError('bound input changed')
    rows=read(inputs['manifest']['path'])['rows'];sources=sorted({r['source'] for r in rows})
    if len(rows)!=11630 or any(str(r['role']).upper()!='TRAIN' for r in rows):raise ValueError('TRAIN population differs')
    parity=[min((r for r in rows if r['source']==s),key=lambda r:digest_string(r['parent_id']))['parent_id'] for s in sources]
    c={'state':'E67_fixed_crop_blur_TRAIN_features_registered','inputs':inputs,'parents':11630,
       'conditions':['clean','assigned_transport','q75'],'crop_shape':[3,3,224,224,3],
       'transform':'Pillow GaussianBlur radius0.8 on each already-selected uint8 RGB224 crop, '
                   'all labels and all3 cached E54 conditions identically. No crop reselection.',
       'feature':'Frozen DINOv2S blocks2/5/8/11 CLS, mean/std over3 crops; store blurred3072D. '
                 'Candidate may use original features and original-minus-blurred response, TRAIN only.',
       'batch_parents':8,'chunk_parents':128,'max_seconds':14400,'cpu_threads':2,
       'parity_parents':parity,'reference_max_score_error':5e-5,'decision_changes_allowed':0,
       'known_train_reference_only':True,'dev_or_final_reads':False,'candidate_fit_allowed':False,
       'policy':'Verify every cached crop identity/binding even on resume. Freeze feature chunks with '
                'hash receipts; no classifier fitting or DEV/E49 scoring. E54 q75 means JPEG75 at '
                'source resolution before2048 cap; do not misname it the1080px E49/SIDD transport.'}
    ROOT.mkdir(exist_ok=True);write_once(CONTRACT,c)
    write_once(EVIDENCE/'e67_features_contract.json',c|{'contract_sha256':digest(CONTRACT)})
    return {'state':c['state'],'parents':len(rows),'reference_parity_parents':len(parity)}


def digest_string(value):
    import hashlib
    return hashlib.sha256(('E67|'+value).encode()).hexdigest()


def validate():
    c=read(CONTRACT)
    if digest(CONTRACT)!=read(EVIDENCE/'e67_features_contract.json')['contract_sha256']:
        raise ValueError('feature contract changed')
    for v in c['inputs'].values():
        if digest(v['path'])!=v['sha256']:raise ValueError('bound feature input changed')
    return c


def load_crops(record,binding,blur=False):
    if digest(record['path'])!=record['sha256']:raise ValueError('crop cache body changed')
    with np.load(record['path'],allow_pickle=False) as a:
        if str(a['binding'])!=binding:raise ValueError('crop population binding changed')
        crops=a['crops']
    if crops.shape!=(3,3,224,224,3) or crops.dtype!=np.uint8:raise ValueError('invalid crop tensor')
    return blur_crops(crops) if blur else crops


def extract():
    c=validate()
    if REPORT.exists() or OUTPUT.exists():raise FileExistsError('E67 feature extraction already complete')
    import torch
    torch.set_num_threads(c['cpu_threads'])
    started=time.monotonic();deadline=started+c['max_seconds'];resource_check(deadline)
    paths={k:Path(v['path']) for k,v in c['inputs'].items()}
    rows=read(paths['manifest'])['rows'];index=read(paths['crop_index'])['records']
    binding=digest(CONTRACT);data_binding=digest(paths['manifest'])
    with np.load(paths['features'],allow_pickle=False) as a:
        if str(a['binding'])!=data_binding or a['features'].shape!=(11630,3,3072):raise ValueError('teacher cache differs')
        original=a['features']
    model,means,stds,weight_sha=dino._load_small()
    device=torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    model=model.to(device).eval();mean=torch.tensor(means,device=device).view(1,3,1,1)
    std=torch.tensor(stds,device=device).view(1,3,1,1)
    def infer(crops):
        n=len(crops);arrays=np.stack(crops).reshape(-1,224,224,3)
        tensor=torch.from_numpy(arrays).to(device).permute(0,3,1,2).float().div_(255)
        blocks=model.forward_intermediates((tensor-mean)/std,indices=list(dino.BLOCKS['small']),
            return_prefix_tokens=True,norm=True,intermediates_only=True)
        tokens=torch.stack([b[1][:,0,:] for b in blocks],dim=1).float().cpu().numpy()
        return dino.aggregate_tokens(tokens,n*3).reshape(n,3,3072)
    head=joblib.load(paths['reference'])['head'];parity=[]
    chosen=[i for i,r in enumerate(rows) if r['parent_id'] in set(c['parity_parents'])]
    (ROOT/'chunks').mkdir(exist_ok=True)
    output=[];chunks=[]
    with torch.inference_mode(),ThreadPoolExecutor(max_workers=2) as pool,threadpool_limits(limits=2):
        for start in range(0,len(chosen),c['batch_parents']):
            resource_check(deadline);ids=chosen[start:start+c['batch_parents']]
            fresh=infer([load_crops(index[rows[i]['parent_id']],data_binding) for i in ids])
            old=head.predict_proba(original[ids].reshape(-1,3072))[:,1]
            new=head.predict_proba(fresh.reshape(-1,3072))[:,1]
            parity.append({'max_score_error':float(np.max(np.abs(old-new))),
                'decision_changes':sum(int(np.sum((old>=cut)!=(new>=cut))) for cut in (AI_CUT,REAL_CUT))})
        max_error=max(p['max_score_error'] for p in parity);changes=sum(p['decision_changes'] for p in parity)
        if max_error>c['reference_max_score_error'] or changes:raise ValueError('E54 original-crop reference replay drift')
        print(json.dumps({'stage':'TRAIN_original_crop_parity_passed','parents':len(chosen),
                          'max_score_error':max_error,'decision_changes':changes}),flush=True)
        del original
        for start in range(0,len(rows),c['chunk_parents']):
            resource_check(deadline);group=rows[start:start+c['chunk_parents']]
            number=start//c['chunk_parents'];out=ROOT/'chunks'/f'{number:04d}.npz';receipt=out.with_suffix('.json')
            parent_ids=[r['parent_id'] for r in group]
            if receipt.exists():
                saved=read(receipt)
                if saved['contract_sha256']!=binding or saved['parent_ids']!=parent_ids or digest(out)!=saved['sha256']:
                    raise ValueError('resumed chunk binding/hash differs')
                # Explicit body check prevents a valid feature receipt hiding a changed crop input.
                for r in group:
                    entry=index[r['parent_id']]
                    if digest(entry['path'])!=entry['sha256']:raise ValueError('resumed crop changed')
                with np.load(out,allow_pickle=False) as a:
                    if str(a['binding'])!=binding or a['features'].shape!=(len(group),3,3072):raise ValueError('chunk shape differs')
                    values=a['features']
            else:
                collected=[]
                for offset in range(0,len(group),c['batch_parents']):
                    resource_check(deadline);batch=group[offset:offset+c['batch_parents']]
                    crop_arrays=list(pool.map(lambda r:load_crops(index[r['parent_id']],data_binding,True),batch))
                    collected.append(infer(crop_arrays))
                    if offset%32==0:print(json.dumps({'stage':'blur_features','parents':start+offset+len(batch),
                            'total':len(rows),'seconds':round(time.monotonic()-started)}),flush=True)
                values=np.concatenate(collected)
                if not np.isfinite(values).all():raise ValueError('nonfinite feature')
                if out.exists():
                    with np.load(out,allow_pickle=False) as a:
                        if str(a['binding'])!=binding or not np.array_equal(a['features'],values):raise ValueError('orphan chunk replay differs')
                else:
                    temporary=out.with_suffix('.npz.part')
                    with temporary.open('wb') as f:np.savez_compressed(f,features=values,binding=np.array(binding))
                    temporary.replace(out)
                saved={'contract_sha256':binding,'parent_ids':parent_ids,'sha256':digest(out)}
                write_once(receipt,saved)
            output.append(values);chunks.append(saved)
    with OUTPUT.open('xb') as f:np.savez_compressed(f,features=np.concatenate(output),binding=np.array(binding),
        parent_ids=np.array([r['parent_id'] for r in rows]),roles=np.array(['TRAIN']*len(rows)))
    result={'state':'E67_TRAIN_blur_features_complete','contract_sha256':binding,'feature_sha256':digest(OUTPUT),
        'shape':[len(rows),3,3072],'chunks':len(chunks),'reference_parity_parents':len(chosen),
        'reference_max_score_error':max_error,'reference_decision_changes':changes,'backbone_sha256':weight_sha,
        'seconds':time.monotonic()-started,'dev_or_final_rows_read':0,'candidate_created':False}
    write_once(REPORT,result);write_once(EVIDENCE/'e67_features.json',result|{'report_sha256':digest(REPORT)})
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=('freeze','extract'))
    ROOT.mkdir(exist_ok=True)
    with (ROOT/'execution.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        print(json.dumps({'freeze':freeze,'extract':extract}[parser.parse_args().stage](),indent=2))
