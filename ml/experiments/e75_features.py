"""Frozen DINO/CLIP feature extension for separately audited E72 MIDD TRAIN images."""
from __future__ import annotations
import argparse
import fcntl
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
import socket
import time
if __name__=='__main__':
    def denied(*a,**kw):raise RuntimeError('E75 feature extraction is offline')
    socket.socket.connect=denied;socket.socket.connect_ex=denied;socket.create_connection=denied
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
import joblib
import numpy as np
from PIL import Image,ImageOps
from threadpoolctl import threadpool_limits
from experiments import e42_features as dino
from experiments.e65_acquisition import digest,read,write_once
from experiments.e71_features import load_encoder,aggregate,load_crops,array_sha,save_npz
from experiments.e71_fit import validate as validate_previous
from experiments.e72_audit import validate as validate_admission
from experiments.e72_acquisition import resource_check
from pixelproof.project_paths import DATA_ROOT,ML_ROOT
from experiments.e64_constrained import AI_CUT,REAL_CUT
ROOT=DATA_ROOT/'e75'
EVIDENCE=ML_ROOT.parent/'evidence'
CONTRACT=ROOT/'features_contract.json'
REPORT=ROOT/'features.json'
OUTPUT=ROOT/'midd_features.npz'


def crop_views(raw,parent):
    # Exact E54 transport order; this Q75 precedes the2048 cap, not social1080px Q75.
    with Image.open(BytesIO(raw)) as opened:
        image=ImageOps.exif_transpose(opened).convert('RGB')
    stream=BytesIO();image.save(stream,format='JPEG',quality=75,subsampling=2,optimize=False)
    with Image.open(BytesIO(stream.getvalue())) as opened:q75=opened.convert('RGB')
    views=[dino.transport_image(image,'clean'),dino.transport_image(image,dino.assigned_transport(parent)),
           dino.transport_image(q75,'clean')]
    return np.stack([np.stack(dino.texture_crops(view)) for view in views])


def validate_population(rows):
    if not rows or len(rows)>512 or len({r['parent_id'] for r in rows})!=len(rows):
        raise ValueError('nonempty unique bounded MIDD TRAIN population required')
    if any(r['role']!='TRAIN' or r['label']!=0 or not r['training_allowed'] or
           not r['source'].startswith('MIDD:') for r in rows):raise ValueError('inadmissible MIDD TRAIN role/source')


def read_chunk(path,binding,row):
    with np.load(path,allow_pickle=False) as a:
        if str(a['binding'])!=binding or str(a['parent_id'])!=row['parent_id'] or str(a['source_sha256'])!=row['sha256']:
            raise ValueError('chunk population/source binding differs')
        values=[]
        for key,width in [('dino',3072),('clip',1536)]:
            x=a[key]
            if x.shape!=(3,width) or x.dtype!=np.float32 or not np.isfinite(x).all() or str(a[key+'_sha256'])!=array_sha(x):
                raise ValueError('chunk feature shape/body differs')
            values.append(x)
        return values


def freeze():
    ROOT.mkdir(exist_ok=True);validate_admission();previous=validate_previous()
    manifest=DATA_ROOT/'e72/training_manifest.json';audit=DATA_ROOT/'e72/audit.json'
    receipt=read(EVIDENCE/'e72_audit.json')
    if digest(manifest)!=receipt['manifest_sha256'] or digest(audit)!=receipt['report_sha256']:
        raise ValueError('completed audited TRAIN admission required')
    rows=read(manifest)['rows'];validate_population(rows)
    if read(manifest)['audit_sha256']!=digest(audit):raise ValueError('admission report binding differs')
    train=read(previous['inputs']['manifest']['path'])['rows'];old_ids={r['parent_id'] for r in train};old_sha={r['sha256'] for r in train}
    if any(r['parent_id'] in old_ids or r['sha256'] in old_sha for r in rows):raise ValueError('existing TRAIN overlap')
    inputs={str(manifest):digest(manifest),str(audit):digest(audit)}
    for p in [Path(__file__),Path(dino.__file__),Path(__file__).with_name('e54_data.py'),
              Path(__file__).with_name('e71_features.py'),Path(__file__).with_name('e65_acquisition.py'),
              ML_ROOT/'src/pixelproof/e32_candidate.py',DATA_ROOT/'e54/crop_index.json',DATA_ROOT/'e71/features_contract.json']:
        inputs[str(p)]=digest(p)
    if digest(DATA_ROOT/'e54/crop_index.json')!=read(EVIDENCE/'e54_crop_cache.json')['index_sha256']:
        raise ValueError('historical crop index changed')
    parity=read(EVIDENCE/'e67_features_contract.json')['parity_parents']
    if len(parity)!=30 or not set(parity)<=old_ids:raise ValueError('registered source parity population differs')
    c={'state':'E75_audited_MIDD_TRAIN_feature_extension_registered','inputs':inputs,'manifest':str(manifest),
       'parents':len(rows),'conditions':['clean','assigned_transport','q75'],'parity_parents':parity,
       'features':'Exact E54 crop_views source JPEG75-before2048-cap, unchanged global+2texture224 crops. '
                  'DINOv2S blocks2/5/8/11 CLS mean/std3072 and official frozen CLIPViT-L/14 '
                  'three-crop mean/std1536, CLIPbatch3 and DINO9 crops per parent. No feature adaptation.',
       'parity':'Before MIDD encoding, re-encode30 previously registered TRAIN source representatives '
                'in all3 cached crop conditions; E43 max score error<=5e-5, zero decision changes at '
                'both cuts, CLIP mean/std max error<=1e-5 versus complete E71 feature archive.',
       'reference_max_score_error':5e-5,'clip_max_feature_error':1e-5,'max_seconds':7200,'cpu_threads':2,
       'resource':'AC,20GiB reserve,2h execution budget, one parent at a time with immutable resume chunks.',
       'policy':'All audited MIDD TRAIN representatives, no new image download or score-based selection. '
                'New MIDD classifier scores0; old TRAIN reference parity90 scores per replay only. '
                'Whole MIDD publisher remains research TRAIN only, E66 consumed DEV excluded. '
                'No change to E54/E71 features and no automatic fit or DEV/E49 access.',
       'candidate_fit_allowed':False,'dev_or_final_rows_read':0,'downloads':0}
    write_once(CONTRACT,c);write_once(EVIDENCE/'e75_features_contract.json',c|{'contract_sha256':digest(CONTRACT)})
    return {'state':c['state'],'parents':len(rows)}


def validate():
    validate_previous();validate_admission();c=read(CONTRACT)
    if digest(CONTRACT)!=read(EVIDENCE/'e75_features_contract.json')['contract_sha256']:
        raise ValueError('feature contract changed')
    for p,s in c['inputs'].items():
        if digest(p)!=s:raise ValueError('E75 input changed: '+p)
    return c


def extract():
    c=validate()
    if REPORT.exists():raise FileExistsError('E75 features already complete')
    import torch
    torch.set_num_threads(2);started=time.monotonic();deadline=started+c['max_seconds'];resource_check(deadline)
    previous=read(DATA_ROOT/'e71/fit_contract.json');paths={k:Path(v['path']) for k,v in previous['inputs'].items()}
    rows=read(c['manifest'])['rows'];validate_population(rows);binding=digest(CONTRACT)
    backbone,means,stds,weight_sha=dino._load_small();device=torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    backbone=backbone.to(device).eval();mean=torch.tensor(means,device=device).view(1,3,1,1)
    std=torch.tensor(stds,device=device).view(1,3,1,1)
    clip_model,preprocess,clip_device=load_encoder()
    def encode(crops):
        if crops.shape!=(3,3,224,224,3) or crops.dtype!=np.uint8:raise ValueError('invalid canonical RGB crop tensor')
        tensor=torch.from_numpy(crops.reshape(-1,224,224,3)).to(device).permute(0,3,1,2).float().div_(255)
        blocks=backbone.forward_intermediates((tensor-mean)/std,indices=list(dino.BLOCKS['small']),
            return_prefix_tokens=True,norm=True,intermediates_only=True)
        tokens=torch.stack([b[1][:,0,:] for b in blocks],dim=1).float().cpu().numpy()
        original=dino.aggregate_tokens(tokens,3).astype(np.float32)
        raw=[]
        for view in crops:
            pack=torch.stack([preprocess(Image.fromarray(crop)) for crop in view]).to(clip_device)
            raw.append(clip_model.encode_image(pack).float().cpu().numpy())
        return original,aggregate(np.stack(raw).astype(np.float32))
    old_rows=read(paths['manifest'])['rows'];index=read(DATA_ROOT/'e54/crop_index.json')['records']
    ids=[i for i,r in enumerate(old_rows) if r['parent_id'] in c['parity_parents']]
    with np.load(paths['features'],allow_pickle=False) as a:old_dino=a['features'][ids]
    with np.load(paths['clip'],allow_pickle=False) as a:old_clip=a['features'][ids]
    head=joblib.load(paths['reference'])['head'];score_error=0.;clip_error=0.;changes=0
    parts=[];chunks={};(ROOT/'chunks').mkdir(exist_ok=True)
    with torch.inference_mode(),threadpool_limits(limits=2):
        for j,i in enumerate(ids):
            resource_check(deadline);parent=old_rows[i]['parent_id']
            fresh,semantic=encode(load_crops(index[parent],digest(paths['manifest'])))
            old=head.predict_proba(old_dino[j])[:,1];new=head.predict_proba(fresh)[:,1]
            score_error=max(score_error,float(np.max(np.abs(new-old))))
            clip_error=max(clip_error,float(np.max(np.abs(semantic-old_clip[j]))))
            changes+=sum(int(np.sum((old>=cut)!=(new>=cut))) for cut in [AI_CUT,REAL_CUT])
        if score_error>c['reference_max_score_error'] or clip_error>c['clip_max_feature_error'] or changes:
            raise ValueError('historical DINO/CLIP source parity failed')
        print(json.dumps({'stage':'E75_encoder_parity_passed','parents':len(ids),'score_error':score_error,'clip_error':clip_error}),flush=True)
        for i,row in enumerate(rows):
            resource_check(deadline)
            if digest(row['path'])!=row['sha256']:raise ValueError('admitted source body changed')
            path=ROOT/'chunks'/f'{i:05d}.npz'
            if not path.exists():
                crops=crop_views(Path(row['path']).read_bytes(),row['parent_id']);original,semantic=encode(crops)
                save_npz(path,dino=original,clip=semantic,dino_sha256=np.array(array_sha(original)),clip_sha256=np.array(array_sha(semantic)),
                    binding=np.array(binding),parent_id=np.array(row['parent_id']),source_sha256=np.array(row['sha256']),
                    crop_sha256=np.array(hashlib.sha256(crops.tobytes()).hexdigest()))
            parts.append(read_chunk(path,binding,row));chunks[str(path)]=digest(path)
            if (i+1)%16==0:print(json.dumps({'E75_parents':i+1,'total':len(rows),'seconds':round(time.monotonic()-started)}),flush=True)
    originals=np.stack([p[0] for p in parts]);semantics=np.stack([p[1] for p in parts]);parents=np.array([r['parent_id'] for r in rows])
    if OUTPUT.exists():
        with np.load(OUTPUT,allow_pickle=False) as a:
            if str(a['binding'])!=binding or not np.array_equal(a['parents'],parents) or not np.array_equal(a['dino'],originals) or \
                    not np.array_equal(a['clip'],semantics) or set(a['roles'])!={'TRAIN'}:raise ValueError('unreceipted final archive differs')
    else:save_npz(OUTPUT,dino=originals,clip=semantics,parents=parents,roles=np.array(['TRAIN']*len(rows)),binding=np.array(binding))
    result={'state':'E75_MIDD_TRAIN_DINO_CLIP_features_complete','parents':len(rows),'dino_shape':list(originals.shape),
        'clip_shape':list(semantics.shape),'feature_sha256':digest(OUTPUT),'contract_sha256':binding,
        'parity_parents':len(ids),'reference_max_score_error':score_error,'clip_max_feature_error':clip_error,
        'reference_decision_changes':changes,'new_MIDD_classifier_scores':0,'dev_or_final_rows_read':0,
        'seconds':time.monotonic()-started,'chunks':chunks,'downloads':0,'backbone_sha256':weight_sha}
    write_once(REPORT,result);summary={k:v for k,v in result.items() if k!='chunks'}
    summary['report_sha256']=digest(REPORT);write_once(EVIDENCE/'e75_features.json',summary);return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=['freeze','extract'])
    ROOT.mkdir(exist_ok=True)
    with (ROOT/'features_execution.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        print(json.dumps({'freeze':freeze,'extract':extract}[parser.parse_args().stage](),indent=2))
