"""One E66 DEV comparison after E70 TRAIN feasibility; never an independent final."""
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
    def denied(*args,**kwargs):raise RuntimeError('E70 DEV network disabled')
    socket.socket.connect=denied;socket.socket.connect_ex=denied;socket.create_connection=denied
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HUB_DISABLE_TELEMETRY='1')

import joblib
import numpy as np
from PIL import Image,ImageFilter
from threadpoolctl import threadpool_limits
from experiments import e42_features as dino,e70_model as model
from experiments.e65_acquisition import digest,read,write_once
from experiments.e65_diagnostic import social_q75_bytes
from experiments.e70_fit import validate as validate_fit,CONTRACT as FIT_CONTRACT,CANDIDATE,REPORT as FIT_REPORT
from experiments.e70_fit import ROOT,EVIDENCE
from experiments.e67_features import resource_check
from experiments.e49_evaluation import CONDITIONS,evaluate_condition
from pixelproof.project_paths import ML_ROOT

CONTRACT=ROOT/'dev_contract.json'
SCORES=ROOT/'dev_scores.json'
REPORT=ROOT/'dev_report.json'


def blur_view(crops):
    if np.asarray(crops).shape!=(3,224,224,3) or np.asarray(crops).dtype!=np.uint8:
        raise ValueError('three uint8 RGB224 crops required')
    return [np.asarray(Image.fromarray(c).filter(ImageFilter.GaussianBlur(radius=.8))) for c in crops]


def validate_pairs(rows,manifest):
    expected={r['parent_id']:r for r in manifest}
    if len(expected)!=len(manifest) or len(rows)!=2*len(manifest):raise ValueError('DEV coverage differs')
    seen=set()
    for r in rows:
        key=(r['parent_id'],r['condition'])
        if key in seen or r['parent_id'] not in expected or r['condition'] not in CONDITIONS:
            raise ValueError('duplicate or unexpected DEV observation')
        seen.add(key);m=expected[r['parent_id']]
        if r['role']!='DEVELOPMENT' or any(r[k]!=m[k] for k in ['source','label']):
            raise ValueError('DEV label/source/role differs')
    if seen!={(p,c) for p in expected for c in CONDITIONS}:raise ValueError('incomplete DEV pairing')


def transitions(rows):
    groups=defaultdict(list)
    for r in rows:groups[f"{r['label']}:{r['source']}"] .append(r)
    result={}
    for group,items in sorted(groups.items()):
        old=np.array([r['reference_score']>=model.AI_CUT for r in items])
        new=np.array([r['score']>=model.AI_CUT for r in items])
        label=items[0]['label']
        result[group]={'views':len(items),'old_ai_rate':float(old.mean()),'new_ai_rate':float(new.mean()),
            'new_ai_misses':int(np.sum(old&~new)) if label==1 else 0,
            'real_rescues':int(np.sum(old&~new)) if label==0 else 0,
            'new_real_false_ai':int(np.sum(~old&new)) if label==0 else 0}
    return result


def freeze():
    fit=validate_fit();result=read(FIT_REPORT)
    if digest(FIT_REPORT)!=digest(EVIDENCE/'e70_fit.json') or not result['dev_scoring_permitted']:
        raise ValueError('E70 TRAIN guard failed: DEV scoring denied')
    if digest(CANDIDATE)!=result['candidate_sha256']:raise ValueError('candidate changed')
    files=[Path(__file__),Path(model.__file__),Path(dino.__file__),Path(__file__).with_name('e65_diagnostic.py'),
        Path(__file__).with_name('e49_evaluation.py'),ML_ROOT/'src/pixelproof/benchmark_metrics.py',
        ML_ROOT/'src/pixelproof/e32_candidate.py',FIT_CONTRACT,FIT_REPORT,CANDIDATE]
    c={'state':'E70_single_E66_DEV_comparison_registered','inputs':{str(p):digest(p) for p in files},
       'manifest':fit['dev_manifest'],'manifest_sha256':fit['dev_manifest_sha256'],
       'reference':fit['inputs']['reference'],'views':640,'batch_views':8,'max_seconds':1800,
       'conditions':list(CONDITIONS),'ai_cut':model.AI_CUT,'real_cut':model.REAL_CUT,
       'features':'Unchanged E43 global+2texture crops for each image/transport, plus GaussianBlur0.8 '
                  'on exactly those crops; no crop reselection. Frozen E70 original/response bilinear map.',
       'comparison':'All20 original E49 numeric gates applied to E66 DEV population via evaluate_condition; '
                    'do not call E49 final validator or bootstrap. Additionally zero newly missed E43-caught '
                    'AI in each transport/source and non-increased pooled REAL FPR in both transports.',
       'limits':'160 REAL observations are only10 dependent SIDD scenes across5 cameras; AI160 from2 previously '
               'seen families with unknown prompt linkage. Descriptive source/scene counts only; no independent '
               'view intervals or final/promotion claim. Scores locked before metrics; no sweep.',
       'e49_read_allowed':False,'promotion_allowed':False,'training_allowed':False}
    write_once(CONTRACT,c);write_once(EVIDENCE/'e70_dev_contract.json',c|{'contract_sha256':digest(CONTRACT)})
    return {'state':c['state'],'views':640}


def validate():
    validate_fit();c=read(CONTRACT)
    if digest(CONTRACT)!=read(EVIDENCE/'e70_dev_contract.json')['contract_sha256']:raise ValueError('DEV contract changed')
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
    manifest=read(c['manifest'])['rows'];rows=[];(ROOT/'dev_q75').mkdir(exist_ok=True)
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
            resource_check(deadline);batch=rows[start:start+c['batch_views']];normal=[];blur=[]
            for r in batch:
                if digest(r['path'])!=r['sha256']:raise ValueError('DEV scoring image changed')
                with Image.open(r['path']) as im:pack=dino.texture_crops(dino.transport_image(im,'clean'))
                normal.extend(pack);blur.extend(blur_view(pack))
            tensor=torch.from_numpy(np.stack(normal+blur)).to(device).permute(0,3,1,2).float().div_(255)
            blocks=backbone.forward_intermediates((tensor-mean)/std,indices=list(dino.BLOCKS['small']),
                return_prefix_tokens=True,norm=True,intermediates_only=True)
            tokens=torch.stack([b[1][:,0,:] for b in blocks],dim=1).float().cpu().numpy()
            features=dino.aggregate_tokens(tokens,2*len(batch));original=features[:len(batch)];blurred=features[len(batch):]
            old=head.predict_proba(original)[:,1];new=model.predict(head,original,blurred,arrays)
            if not np.isfinite(new).all() or np.any((new<0)|(new>1)):raise ValueError('invalid DEV score')
            done.extend(r|{'reference_score':float(old[i]),'score':float(new[i]),'status':'ok'} for i,r in enumerate(batch))
            if len(done)%64==0:print(json.dumps({'DEV_views':len(done),'total':len(rows)}),flush=True)
    validate_pairs(done,manifest)
    result={'state':'E70_DEV_scores_locked_before_metrics','contract_sha256':digest(CONTRACT),
        'rows':done,'seconds':time.monotonic()-start_time}
    write_once(SCORES,result);write_once(EVIDENCE/'e70_dev_scores.json',{'views':len(done),'scores_sha256':digest(SCORES),
        'contract_sha256':digest(CONTRACT),'seconds':result['seconds'],'metrics_opened':False})
    return {'views':len(done),'seconds':result['seconds']}


def digest_string(value):
    import hashlib
    return hashlib.sha256(value.encode()).hexdigest()


def report():
    c=validate()
    if digest(SCORES)!=read(EVIDENCE/'e70_dev_scores.json')['scores_sha256']:raise ValueError('locked DEV scores differ')
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
    result={'state':'E70_limited_DEV_comparison_complete','checks':checks,'passes_limited_dev_screen':passed,
        'consumed_regression_may_be_registered':passed,'reports':reports,'contract_sha256':digest(CONTRACT),
        'scores_sha256':digest(SCORES),'candidate_sha256':digest(CANDIDATE),
        'independent_final_passed':False,'serving_changed':False,'limitations':c['limits']}
    write_once(REPORT,result);write_once(EVIDENCE/'e70_development.json',result)
    return {'checks':checks,'passes_limited_dev_screen':passed}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=('freeze','score','report'))
    with (ROOT/'dev_execution.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        print(json.dumps({'freeze':freeze,'score':score,'report':report}[parser.parse_args().stage](),indent=2))
