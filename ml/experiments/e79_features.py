"""Complete TRAIN-only frozen DEAR-r crop features under an explicit overnight resource budget."""
from __future__ import annotations
import argparse
import fcntl
import json
import os
from pathlib import Path
import socket
import time
if __name__=='__main__':
    def denied(*a,**kw):raise RuntimeError('E79 features are offline')
    socket.socket.connect=denied;socket.socket.connect_ex=denied;socket.create_connection=denied
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
import numpy as np
import torch
from experiments import e78_model as model
from experiments.e78_batch_probe import validate as validate_probe
from experiments.e77_fit import validate as validate_training
from experiments.e76_fit import combine_rows
from experiments.e75_features import crop_views
from experiments.e71_features import load_crops,array_sha,save_npz
from experiments.e65_acquisition import digest,read,write_once
from experiments.e72_acquisition import resource_check
from pixelproof.project_paths import DATA_ROOT,ML_ROOT
ROOT=DATA_ROOT/'e79';EVIDENCE=ML_ROOT.parent/'evidence';CONTRACT=ROOT/'features_contract.json'
REPORT=ROOT/'features.json';OUTPUT=ROOT/'dear_features.npz';MANIFEST=DATA_ROOT/'e54/data_contract_v2.json'
NEW_MANIFEST=DATA_ROOT/'e72/training_manifest.json';CROP_INDEX=DATA_ROOT/'e54/crop_index.json'


def aggregate(raw):
    if raw.shape!=(3,3,820) or raw.dtype!=np.float32 or not np.isfinite(raw).all():
        raise ValueError('finite three-condition/three-crop DEAR vectors required')
    return np.concatenate([raw.mean(axis=1),raw.std(axis=1,ddof=0)],axis=1).astype(np.float32)


def read_chunk(path,binding,row):
    with np.load(path,allow_pickle=False) as a:
        if str(a['binding'])!=binding or str(a['parent_id'])!=row['parent_id'] or str(a['source_sha256'])!=row['sha256']:
            raise ValueError('chunk population/source binding differs')
        raw=a['raw'];features=a['features']
        if str(a['raw_sha256'])!=array_sha(raw) or str(a['features_sha256'])!=array_sha(features) or \
                not np.array_equal(features,aggregate(raw)):
            raise ValueError('chunk body/aggregation differs')
        return raw,features


def encode(net,crops):
    if crops.shape!=(3,3,224,224,3) or crops.dtype!=np.uint8:raise ValueError('complete aligned RGB crop tensor required')
    with torch.inference_mode():
        x=model.normalize(crops.reshape(9,224,224,3)).to(next(net.parameters()).device)
        values=net.pooled(x)[:,net.gate.gate.bool()].cpu().numpy()
    raw=values.reshape(3,3,820).astype(np.float32);aggregate(raw)
    return raw


def freeze():
    ROOT.mkdir(exist_ok=True);validate_probe();validate_training()
    probe=read(DATA_ROOT/'e78/batch_probe.json');failed=read(DATA_ROOT/'e77/fit.json')
    if digest(DATA_ROOT/'e78/batch_probe.json')!=digest(EVIDENCE/'e78_batch_probe.json') or \
            any(v['feature_max_abs']>5e-4 or v['logit_max_abs']>1e-4 or v['decision_changes'] for v in probe['parity'].values()) or \
            probe['mps_driver_bytes']>6*1024**3 or probe['projected_extraction_seconds']>9000 or \
            digest(DATA_ROOT/'e77/fit.json')!=digest(EVIDENCE/'e77_fit.json') or failed['dev_scoring_permitted']:
        raise ValueError('unchanged numeric/memory parity and failed E77 required')
    rows=combine_rows(read(MANIFEST)['rows'],read(NEW_MANIFEST)['rows'])
    paths=[Path(__file__),Path(model.__file__),MANIFEST,NEW_MANIFEST,CROP_INDEX,
           DATA_ROOT/'e77/fit.json',DATA_ROOT/'e78/batch_probe.json',DATA_ROOT/'e78/batch_probe_contract.json',
           Path(__file__).with_name('e71_features.py'),Path(__file__).with_name('e75_features.py'),
           Path(__file__).with_name('e76_fit.py'),Path(__file__).with_name('e72_acquisition.py')]
    c={'state':'E79_DEAR_r_complete_TRAIN_features_registered','inputs':{str(p):digest(p) for p in paths},
       'parents':12141,'old_parents':11630,'new_parents':511,'real_parents':7546,'ai_parents':4595,
       'conditions':['clean','assigned_transport','q75'],
       'population':'Exact E76/E77 ordered TRAIN parents, no omissions/refill/relabel. Old E54 crop bodies; '
                    'MIDD exact E75 crop_views after body SHA verification. No DEV/final.',
       'representation':'Frozen official DEAR-r stride0=1 backbone and820 active gate channels. '
                        'All nine RGB224 crops per parent, batch9 float32 MPS, exact ImageNet normalization. '
                        'Raw3x3x820; population mean/std over3 crops gives3x1640. No L2 or learned map.',
       'adaptation':'E54 global+2texture crops and mean/std are local adaptations; not official native-full-image inference.',
       'max_seconds':9000,'cpu_threads':2,'mps_driver_limit_bytes':6*1024**3,
       'resource_revision':'Both E78 probes passed numeric/memory checks but failed the original7200s cost estimate. '
                           'Explicitly allocate9000s under user overnight authorization before any image scoring. '
                           'This changes only wall-time budget; failed probes remain failed, no detector quality gate changes.',
       'resume':'Immutable per-parent chunks bound to contract,parent,source body and array hashes. '
                'Resume all missing chunks under same recipe; never select by score. Same-device replays<=1e-5.',
       'parity_parents':[min((r['parent_id'] for r in rows if r['source']==source)) for source in sorted({r['source'] for r in rows})],
       'classifier_scores':0,'upstream_head_executed':False,'classifier_fit_allowed':False,
       'next':'Only complete verified features permit a separately registered full-AI-confidence constrained fit.',
       'downloads':0,'promotion_allowed':False,'license':'DEAR-r and MIDD research-only restrictions persist.'}
    write_once(CONTRACT,c);write_once(EVIDENCE/'e79_features_contract.json',c|{'contract_sha256':digest(CONTRACT)})
    return {'state':c['state'],'parents':len(rows),'parity_parents':len(c['parity_parents'])}


def validate():
    validate_probe();validate_training();c=read(CONTRACT)
    if digest(CONTRACT)!=read(EVIDENCE/'e79_features_contract.json')['contract_sha256']:raise ValueError('E79 contract changed')
    for p,s in c['inputs'].items():
        if digest(p)!=s:raise ValueError('E79 input changed: '+p)
    return c


def extract():
    c=validate()
    if REPORT.exists():raise FileExistsError('E79 extraction already complete')
    started=time.monotonic();deadline=started+c['max_seconds'];resource_check(deadline);torch.set_num_threads(2)
    rows=combine_rows(read(MANIFEST)['rows'],read(NEW_MANIFEST)['rows']);index=read(CROP_INDEX)['records']
    binding=digest(CONTRACT);data_binding=digest(MANIFEST);device='mps' if torch.backends.mps.is_available() else 'cpu'
    net=model.load(DATA_ROOT/'e78/dear_r.pth',device);(ROOT/'chunks').mkdir(exist_ok=True)
    def crops_for(i,row):
        if i<c['old_parents']:return load_crops(index[row['parent_id']],data_binding)
        path=Path(row['path'])
        if digest(path)!=row['sha256']:raise ValueError('MIDD source body differs')
        return crop_views(path.read_bytes(),row['parent_id'])
    parts=[];chunks={};created=0;parity=0.;peak=0
    for i,row in enumerate(rows):
        resource_check(deadline);path=ROOT/'chunks'/f'{i:05d}.npz'
        if path.exists():raw,features=read_chunk(path,binding,row)
        else:
            raw=encode(net,crops_for(i,row));features=aggregate(raw)
            save_npz(path,binding=np.array(binding),parent_id=np.array(row['parent_id']),source_sha256=np.array(row['sha256']),
                raw=raw,features=features,raw_sha256=np.array(array_sha(raw)),features_sha256=np.array(array_sha(features)))
            read_chunk(path,binding,row);created+=1
        if row['parent_id'] in c['parity_parents']:
            error=float(np.max(np.abs(raw-encode(net,crops_for(i,row)))));parity=max(parity,error)
            if error>1e-5:raise ValueError('same-device source replay differs')
        if device=='mps':
            peak=max(peak,torch.mps.driver_allocated_memory())
            if peak>c['mps_driver_limit_bytes']:raise RuntimeError('E79 MPS driver memory limit exceeded')
        parts.append(features);chunks[str(path)]=digest(path)
        if (i+1)%50==0 or i==0:
            elapsed=time.monotonic()-started
            print(json.dumps({'stage':'E79_features','parents':i+1,'total':len(rows),'created_this_run':created,
                              'seconds':round(elapsed),'mps_driver_peak_bytes':peak}),flush=True)
    resource_check(deadline);features=np.stack(parts);parents=np.array([r['parent_id'] for r in rows])
    if features.shape!=(12141,3,1640):raise ValueError('complete DEAR TRAIN archive required')
    if OUTPUT.exists():
        with np.load(OUTPUT,allow_pickle=False) as a:
            if str(a['binding'])!=binding or not np.array_equal(a['parents'],parents) or \
                    not np.array_equal(a['features'],features) or set(a['roles'])!={'TRAIN'}:
                raise ValueError('unreceipted feature archive differs')
    else:save_npz(OUTPUT,features=features,parents=parents,roles=np.array(['TRAIN']*len(rows)),binding=np.array(binding))
    result={'state':'E79_DEAR_r_TRAIN_features_complete','contract_sha256':binding,'shape':list(features.shape),
        'feature_sha256':digest(OUTPUT),'chunks':chunks,'created_this_run':created,'parity_parents':len(c['parity_parents']),
        'same_device_max_error':parity,'mps_driver_peak_bytes':peak,'seconds':time.monotonic()-started,
        'classifier_scores':0,'upstream_head_executed':False,'dev_or_final_rows_read':0,'downloads':0,'promotion_allowed':False}
    write_once(REPORT,result);compact={k:v for k,v in result.items() if k!='chunks'}
    write_once(EVIDENCE/'e79_features.json',compact|{'report_sha256':digest(REPORT)});return compact


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=['freeze','extract'])
    ROOT.mkdir(exist_ok=True)
    with (ROOT/'features_execution.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        print(json.dumps({'freeze':freeze,'extract':extract}[parser.parse_args().stage](),indent=2))
