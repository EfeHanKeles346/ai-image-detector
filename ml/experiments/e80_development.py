"""One consumed E66 DEV comparison after E80 TRAIN feasibility; never an independent final."""
from __future__ import annotations
import argparse
from collections import defaultdict
import fcntl
import json
import os
from pathlib import Path
import socket
import time

if __name__=='__main__':
    def denied(*args,**kwargs):raise RuntimeError('E80 DEV network disabled')
    socket.socket.connect=denied;socket.socket.connect_ex=denied;socket.create_connection=denied
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HUB_DISABLE_TELEMETRY='1')

import joblib
import numpy as np
from PIL import Image
from threadpoolctl import threadpool_limits
from experiments import e42_features as dino,e80_model as model
from experiments.e65_acquisition import digest,read,write_once
from experiments.e65_diagnostic import social_q75_bytes
from experiments.e80_fit import validate as validate_fit,CONTRACT as FIT_CONTRACT,CANDIDATE,REPORT as FIT_REPORT
from experiments.e80_fit import ROOT,EVIDENCE
from experiments.e71_features import resource_check,load_encoder
from experiments import e78_model as dear_model
from experiments.e71_development import clip_aggregate,check_reference,validate_pairs,transitions,digest_string
from pixelproof.project_paths import DATA_ROOT
from experiments.e49_evaluation import CONDITIONS,evaluate_condition
from pixelproof.project_paths import ML_ROOT

CONTRACT=ROOT/'dev_contract.json'
SCORES=ROOT/'dev_scores.json'
REPORT=ROOT/'dev_report.json'


def dear_features(net,crops):
    import torch
    if crops.ndim!=5 or crops.shape[1:]!=(3,224,224,3) or crops.dtype!=np.uint8 or not len(crops):
        raise ValueError('complete DEV three-crop RGB view batches required')
    flat=crops.reshape(-1,224,224,3);parts=[]
    with torch.inference_mode():
        for start in range(0,len(flat),9):
            batch=flat[start:start+9];n=len(batch)
            if n<9:batch=np.concatenate([batch,np.repeat(batch[-1:],9-n,axis=0)])
            x=dear_model.normalize(batch).to(next(net.parameters()).device)
            raw=net.pooled(x)[:,net.gate.gate.bool()].cpu().numpy()
            parts.append(raw[:n])
    raw=np.concatenate(parts).reshape(len(crops),3,820)
    result=np.concatenate([raw.mean(axis=1),raw.std(axis=1,ddof=0)],axis=1).astype(np.float32)
    if result.shape!=(len(crops),1640) or not np.isfinite(result).all():raise ValueError('invalid forensic DEV features')
    return result


def freeze():
    validate_fit();result=read(FIT_REPORT);previous_fit=read(DATA_ROOT/'e71/fit_contract.json')
    if digest(FIT_REPORT)!=digest(EVIDENCE/'e80_fit.json') or not result['dev_scoring_permitted']:
        raise ValueError('E80 TRAIN guard failed: DEV scoring denied')
    if digest(CANDIDATE)!=result['candidate_sha256']:raise ValueError('candidate changed')
    files=[Path(__file__),Path(model.__file__),Path(dino.__file__),Path(__file__).with_name('e65_diagnostic.py'),
        Path(__file__).with_name('e49_evaluation.py'),ML_ROOT/'src/pixelproof/benchmark_metrics.py',
        ML_ROOT/'src/pixelproof/e32_candidate.py',FIT_CONTRACT,FIT_REPORT,CANDIDATE,
        Path(__file__).with_name('e71_features.py'), DATA_ROOT/'e71/dev_scores.json', DATA_ROOT/'e71/dev_report.json',
        Path(dear_model.__file__),Path(__file__).with_name('e71_development.py'),DATA_ROOT/'e78/dear_r.pth']
    if digest(DATA_ROOT/'e71/dev_scores.json')!=read(EVIDENCE/'e71_dev_scores.json')['scores_sha256']:
        raise ValueError('prior consumed reference scores changed')
    c={'state':'E80_single_consumed_E66_DEV_comparison_registered','inputs':{str(p):digest(p) for p in files},
       'manifest':previous_fit['dev_manifest'],'manifest_sha256':previous_fit['dev_manifest_sha256'],
       'reference':previous_fit['inputs']['reference'],'views':640,'batch_views':8,'max_seconds':2400,
       'conditions':list(CONDITIONS),'ai_cut':model.AI_CUT,'real_cut':model.REAL_CUT,
       'features':'Unchanged E43 global+2texture crops for each image/transport, alongside exact official '
                  'CLIP preprocessing on the same RGB224 crops; batch3 CLIP crops, frozen float32 '
                  'ViT-L/14 mean/std1536. Frozen E80 original/CLIP/bilinear/REAL residual/DEAR map. '
                  'DEAR batch9 padded only for the last partial microbatch; discard padding before three-crop '
                  'mean/std1640. No input resize beyond frozen E43 crop pipeline; float32.',
       'dev_status':'CONSUMED DEVELOPMENT: E70/E71 scores and reports already exposed. Not fresh validation.',
       'prior_reference_max_error':1e-6,
       'comparison':'All20 original E49 numeric gates applied to E66 DEV population via evaluate_condition; '
                    'do not call E49 final validator or bootstrap. Additionally zero newly missed E43-caught '
                    'AI in each transport/source and non-increased pooled REAL FPR in both transports.',
       'limits':'160 REAL observations are only10 dependent SIDD scenes across5 cameras; AI160 from2 previously '
               'seen families with unknown prompt linkage. Descriptive source/scene counts only; no independent '
               'view intervals or final/promotion claim. Scores locked before metrics; no sweep.',
       'e49_read_allowed':False,'promotion_allowed':False,'training_allowed':False}
    write_once(CONTRACT,c);write_once(EVIDENCE/'e80_dev_contract.json',c|{'contract_sha256':digest(CONTRACT)})
    return {'state':c['state'],'views':640}


def validate():
    validate_fit();c=read(CONTRACT)
    if digest(CONTRACT)!=read(EVIDENCE/'e80_dev_contract.json')['contract_sha256']:raise ValueError('DEV contract changed')
    for p,s in c['inputs'].items():
        if digest(p)!=s:raise ValueError('DEV input changed')
    if digest(c['manifest'])!=c['manifest_sha256'] or digest(c['reference']['path'])!=c['reference']['sha256']:
        raise ValueError('DEV/reference identity changed')
    return c


def score():
    c=validate()
    if SCORES.exists():raise FileExistsError('DEV scores already complete')
    import torch
    torch.set_num_threads(2);start_time=time.monotonic();deadline=start_time+c['max_seconds'];resource_check(deadline)
    head=joblib.load(c['reference']['path'])['head']
    with np.load(CANDIDATE,allow_pickle=False) as a:
        if str(a['contract_sha256'])!=digest(FIT_CONTRACT) or str(a['reference_sha256'])!=c['reference']['sha256']:
            raise ValueError('candidate binding differs')
        arrays={k:a[k] for k in a.files if k not in ['contract_sha256','reference_sha256']}
    backbone,means,stds,_=dino._load_small();device=torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    backbone=backbone.to(device).eval();mean=torch.tensor(means,device=device).view(1,3,1,1)
    std=torch.tensor(stds,device=device).view(1,3,1,1)
    clip_model,clip_preprocess,clip_device=load_encoder()
    dear_net=dear_model.load(DATA_ROOT/'e78/dear_r.pth',device)
    prior_rows=read(DATA_ROOT/'e71/dev_scores.json')['rows']
    manifest=read(c['manifest'])['rows'];validate_pairs(prior_rows,manifest)
    prior_reference={(r['parent_id'],r['condition']):r for r in prior_rows}
    reference_max_error=0.
    rows=[];(ROOT/'dev_q75').mkdir(exist_ok=True)
    for r in manifest:
        if digest(r['path'])!=r['sha256']:raise ValueError('audited DEV image changed')
        body=social_q75_bytes(r['path']);q=ROOT/'dev_q75'/(digest_string(r['parent_id'])+'.jpg')
        if q.exists():
            if q.read_bytes()!=body:raise ValueError('Q75 derivative replay changed')
        else:
            with q.open('xb') as f:f.write(body)
        for cond,p in [(CONDITIONS[0],Path(r['path'])),(CONDITIONS[1],q)]:
            rows.append(r|{'condition':cond,'path':str(p),'sha256':digest(p),'record_id':r['parent_id']+'|'+cond})
    validate_pairs(rows,manifest);done=[]
    with torch.inference_mode(),threadpool_limits(limits=2):
        for start in range(0,len(rows),c['batch_views']):
            resource_check(deadline);batch=rows[start:start+c['batch_views']];normal=[];clip_features=[]
            for r in batch:
                if digest(r['path'])!=r['sha256']:raise ValueError('DEV scoring image changed')
                with Image.open(r['path']) as im:pack=dino.texture_crops(dino.transport_image(im,'clean'))
                normal.extend(pack)
                clip_tensor=torch.stack([clip_preprocess(Image.fromarray(crop)) for crop in pack]).to(clip_device)
                raw=clip_model.encode_image(clip_tensor).float().cpu().numpy()
                # One DEV condition, same three-crop aggregation as the TRAIN feature contract.
                clip_features.append(clip_aggregate(raw))
            tensor=torch.from_numpy(np.stack(normal)).to(device).permute(0,3,1,2).float().div_(255)
            blocks=backbone.forward_intermediates((tensor-mean)/std,indices=list(dino.BLOCKS['small']),
                return_prefix_tokens=True,norm=True,intermediates_only=True)
            tokens=torch.stack([b[1][:,0,:] for b in blocks],dim=1).float().cpu().numpy()
            original=dino.aggregate_tokens(tokens,len(batch));clip_features=np.stack(clip_features)
            forensic=dear_features(dear_net,np.stack(normal).reshape(len(batch),3,224,224,3))
            old=head.predict_proba(original)[:,1];new=model.predict(head,original,clip_features,forensic,arrays)
            for i,r in enumerate(batch):
                previous=prior_reference[(r['parent_id'],r['condition'])]
                error=check_reference(r,old[i],previous,c['prior_reference_max_error'])
                reference_max_error=max(reference_max_error,error)
            if not np.isfinite(new).all() or np.any((new<0)|(new>1)):raise ValueError('invalid DEV score')
            done.extend(r|{'reference_score':float(old[i]),'score':float(new[i]),'status':'ok'} for i,r in enumerate(batch))
            if len(done)%64==0:print(json.dumps({'DEV_views':len(done),'total':len(rows)}),flush=True)
    validate_pairs(done,manifest)
    result={'state':'E80_DEV_scores_locked_before_metrics','contract_sha256':digest(CONTRACT),
        'rows':done,'seconds':time.monotonic()-start_time,'prior_reference_max_error':reference_max_error,
        'previously_consumed_by':['E70','E71'],'reference_decision_changes':0}
    write_once(SCORES,result);write_once(EVIDENCE/'e80_dev_scores.json',{'views':len(done),'scores_sha256':digest(SCORES),
        'contract_sha256':digest(CONTRACT),'seconds':result['seconds'],'metrics_opened':False,
        'prior_reference_max_error':reference_max_error,'reference_decision_changes':0})
    return {'views':len(done),'seconds':result['seconds']}


def report():
    c=validate()
    if digest(SCORES)!=read(EVIDENCE/'e80_dev_scores.json')['scores_sha256']:raise ValueError('locked DEV scores differ')
    rows=read(SCORES)['rows'];validate_pairs(rows,read(c['manifest'])['rows']);reports={};checks={}
    for cond in CONDITIONS:
        selected=[r for r in rows if r['condition']==cond]
        old=evaluate_condition([r|{'score':r['reference_score']} for r in selected]);new=evaluate_condition(selected)
        change=transitions(selected);scenes={}
        for scene in sorted({r['scene_group'] for r in selected if r['label']==0}):
            subset=[r for r in selected if r['label']==0 and r['scene_group']==scene]
            scenes[scene]={'observations':len(subset),'old_false_ai':sum(r['reference_score']>=model.AI_CUT for r in subset),
                           'new_false_ai':sum(r['score']>=model.AI_CUT for r in subset)}
        reports[cond]={'old':old,'new':new,'transitions':change,'real_scenes':scenes}
        checks[cond]={'absolute_gates_passed':new['gate']['passed'],
            'zero_new_ai_misses':all(v['new_ai_misses']==0 for v in change.values()),
            'real_pooled_not_increased':new['binary_rates']['pooled_real_false_ai']<=old['binary_rates']['pooled_real_false_ai']}
    passed=all(all(v.values()) for v in checks.values())
    result={'state':'E80_limited_DEV_comparison_complete','checks':checks,'passes_limited_dev_screen':passed,
        'consumed_regression_may_be_registered':passed,'reports':reports,'contract_sha256':digest(CONTRACT),
        'scores_sha256':digest(SCORES),'candidate_sha256':digest(CANDIDATE),
        'independent_final_passed':False,'serving_changed':False,'limitations':c['limits']}
    write_once(REPORT,result);write_once(EVIDENCE/'e80_development.json',result)
    return {'checks':checks,'passes_limited_dev_screen':passed}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=('freeze','score','report'))
    with (ROOT/'dev_execution.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        print(json.dumps({'freeze':freeze,'score':score,'report':report}[parser.parse_args().stage](),indent=2))
