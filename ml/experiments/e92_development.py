"""One consumed E66 comparison from frozen encoder caches after full E92 TRAIN/runtime pass."""
from __future__ import annotations
import argparse
import fcntl
import json
import os
from pathlib import Path
import socket
import time
if __name__=='__main__':
    def denied(*a,**kw):raise RuntimeError('E92 consumed DEV offline')
    socket.socket.connect=denied;socket.socket.connect_ex=denied;socket.create_connection=denied
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
import joblib
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from experiments import e92_model as model
from experiments.e92_fit import validate as validate_fit,CONTRACT as FIT_CONTRACT,CANDIDATE,REPORT as FIT_REPORT,ROOT,EVIDENCE
from experiments.e83_development import validate as validate_cache,FEATURES,CONTRACT as CACHE_CONTRACT,SCORES as CACHE_SCORES
from experiments.e83_diagnostic import check_cache
from experiments.e71_development import check_reference,validate_pairs,transitions
from experiments.e49_evaluation import CONDITIONS,evaluate_condition
from experiments.e65_acquisition import digest,read,write_once
from experiments.e72_acquisition import resource_check
from pixelproof.project_paths import DATA_ROOT,ML_ROOT
CONTRACT=ROOT/'dev_contract.json';SCORES=ROOT/'dev_scores.json';REPORT=ROOT/'dev_report.json'
PREVIOUS_SCORES=DATA_ROOT/'e86/dev_scores.json';PREVIOUS_REPORT=DATA_ROOT/'e86/dev_report.json'


def validate_predecessor(rows, cached, manifest):
    validate_pairs(rows,manifest);validate_pairs(cached,manifest)
    fields=('parent_id','condition','sha256','role','source','label')
    if any(any(a[k]!=b[k] for k in fields) for a,b in zip(rows,cached,strict=True)):
        raise ValueError('E86 predecessor and E83 cache identities/order differ')


def freeze():
    validate_fit();fit=read(FIT_REPORT)
    if digest(FIT_REPORT)!=digest(EVIDENCE/'e92_fit.json') or not fit['dev_scoring_permitted']:
        raise ValueError('E92 complete TRAIN/runtime pass required before DEV cache access')
    if digest(CANDIDATE)!=fit['candidate_sha256']:raise ValueError('candidate changed')
    cache=validate_cache();old_scores=read(CACHE_SCORES);old_report=read(DATA_ROOT/'e83/dev_report.json')
    if digest(CACHE_SCORES)!=old_report['scores_sha256'] or digest(FEATURES)!=old_scores['features_sha256'] or \
            digest(DATA_ROOT/'e83/dev_report.json')!=digest(EVIDENCE/'e83_development.json'):
        raise ValueError('immutable E83 cache/report required')
    previous=read(PREVIOUS_REPORT)
    if digest(PREVIOUS_REPORT)!=digest(EVIDENCE/'e86_development.json') or digest(PREVIOUS_SCORES)!=previous['scores_sha256']:
        raise ValueError('immutable E86 predecessor report/scores required')
    validate_predecessor(read(PREVIOUS_SCORES)['rows'],old_scores['rows'],read(cache['manifest'])['rows'])
    paths=[Path(__file__),Path(model.__file__),PREVIOUS_REPORT,PREVIOUS_SCORES,FIT_CONTRACT,FIT_REPORT,CANDIDATE,FEATURES,CACHE_CONTRACT,CACHE_SCORES,
        DATA_ROOT/'e83/dev_report.json',Path(__file__).with_name('e83_model.py'),Path(__file__).with_name('e83_diagnostic.py'),
        Path(__file__).with_name('e83_development.py'),Path(__file__).with_name('e71_development.py'),
        Path(__file__).with_name('e49_evaluation.py'),ML_ROOT/'src/pixelproof/benchmark_metrics.py']
    c={'state':'E92_single_consumed_E66_cache_comparison_registered','inputs':{str(p):digest(p) for p in paths},
       'manifest':cache['manifest'],'manifest_sha256':cache['manifest_sha256'],'reference':cache['reference'],
       'views':640,'batch_views':8,'max_seconds':600,'conditions':list(CONDITIONS),'ai_cut':model.AI_CUT,'real_cut':model.REAL_CUT,
       'features':'Exact complete E83 DEVELOPMENT DINO3072/CLIP1536/DEAR1640 arrays, ordered original/Q75 image '
                  'identities. Verify roles, source-body hashes and contract binding; no new encoder/image operations. '
                  'Frozen E92 map in runtime batch8. Replay every E43 reference score<=1e-6 and both cuts unchanged.',
       'dev_status':'CONSUMED DEVELOPMENT already used by E70/E71/E83/E86. Neither fresh nor independent final.',
       'comparison':'All20 original numeric gates, zero newly missed E43-caught AI in every source/condition, '
                    'non-increased pooled REAL FPR in both conditions. Also report all paired source transitions '
                    'against rejected E86 so losing its rescues cannot be hidden. Scores locked before metrics.',
       'limits':cache['limits']+' E83/E86 DEV labels/features remain excluded from TRAIN; cached comparison does not reset consumption.',
       'prior_reference_max_error':1e-6,'image_reads':0,'downloads':0,'training_allowed':False,'e49_read_allowed':False,'promotion_allowed':False}
    write_once(CONTRACT,c);write_once(EVIDENCE/'e92_dev_contract.json',c|{'contract_sha256':digest(CONTRACT)})
    return {'state':c['state'],'views':c['views']}


def validate():
    validate_fit();validate_cache();c=read(CONTRACT)
    if digest(CONTRACT)!=read(EVIDENCE/'e92_dev_contract.json')['contract_sha256']:raise ValueError('E92 DEV contract changed')
    for p,s in c['inputs'].items():
        if digest(p)!=s:raise ValueError('E92 DEV input changed: '+p)
    if digest(c['manifest'])!=c['manifest_sha256'] or digest(c['reference']['path'])!=c['reference']['sha256']:
        raise ValueError('reference or DEV population changed')
    return c


def score():
    c=validate()
    if SCORES.exists():raise FileExistsError('E92 DEV scores already locked')
    torch.set_num_threads(2);start=time.monotonic();deadline=start+c['max_seconds'];resource_check(deadline)
    rows=read(PREVIOUS_SCORES)['rows'];validate_pairs(rows,read(c['manifest'])['rows'])
    if len(rows)!=c['views']:raise ValueError('complete640-view cache required')
    head=joblib.load(c['reference']['path'])['head'];done=[];max_error=0.
    with np.load(FEATURES,allow_pickle=False) as cached,np.load(CANDIDATE,allow_pickle=False) as a,threadpool_limits(limits=2):
        check_cache(cached,rows,digest(CACHE_CONTRACT))
        if str(a['contract_sha256'])!=digest(FIT_CONTRACT) or str(a['reference_sha256'])!=c['reference']['sha256']:
            raise ValueError('candidate bindings differ')
        for start_index in range(0,len(rows),c['batch_views']):
            resource_check(deadline);selected=slice(start_index,start_index+c['batch_views']);batch=rows[selected]
            original=cached['dino'][selected];old=head.predict_proba(original)[:,1]
            new=model.predict(head,original,cached['clip'][selected],cached['dear'][selected],a)
            if not np.isfinite(new).all() or np.any((new<0)|(new>1)):raise ValueError('invalid DEV scores')
            for i,row in enumerate(batch):
                max_error=max(max_error,check_reference(row,old[i],row,c['prior_reference_max_error']))
                done.append(row|{'predecessor_score':row['score'],'reference_score':float(old[i]),'score':float(new[i]),'status':'ok'})
    validate_pairs(done,read(c['manifest'])['rows'])
    result={'state':'E92_DEV_scores_locked_before_metrics','contract_sha256':digest(CONTRACT),'rows':done,
            'prior_reference_max_error':max_error,'reference_decision_changes':0,'features_sha256':digest(FEATURES),
            'image_reads':0,'seconds':time.monotonic()-start,'previously_consumed_by':['E70','E71','E83','E86']}
    write_once(SCORES,result);write_once(EVIDENCE/'e92_dev_scores.json',{k:v for k,v in result.items() if k!='rows'}|
        {'scores_sha256':digest(SCORES),'views':len(done),'metrics_opened':False})
    return {'views':len(done),'seconds':result['seconds'],'prior_reference_max_error':max_error}


def report():
    c=validate()
    if digest(SCORES)!=read(EVIDENCE/'e92_dev_scores.json')['scores_sha256']:raise ValueError('locked E92 scores changed')
    rows=read(SCORES)['rows'];validate_pairs(rows,read(c['manifest'])['rows']);reports={};checks={}
    for condition in CONDITIONS:
        selected=[r for r in rows if r['condition']==condition]
        old=evaluate_condition([r|{'score':r['reference_score']} for r in selected]);new=evaluate_condition(selected)
        change=transitions(selected)
        predecessor=evaluate_condition([r|{'score':r['predecessor_score']} for r in selected])
        previous_changes=transitions([r|{'reference_score':r['predecessor_score']} for r in selected])
        scenes={}
        for scene in sorted({r['scene_group'] for r in selected if r['label']==0}):
            subset=[r for r in selected if r['label']==0 and r['scene_group']==scene]
            scenes[scene]={'observations':len(subset),'old_false_ai':sum(r['reference_score']>=model.AI_CUT for r in subset),
                           'new_false_ai':sum(r['score']>=model.AI_CUT for r in subset)}
        reports[condition]={'old':old,'new':new,'transitions':change,'real_scenes':scenes,
                            'E86_predecessor':predecessor,'transitions_vs_E86':previous_changes}
        checks[condition]={'absolute_gates_passed':new['gate']['passed'],
            'zero_new_ai_misses':all(v['new_ai_misses']==0 for v in change.values()),
            'real_pooled_not_increased':new['binary_rates']['pooled_real_false_ai']<=old['binary_rates']['pooled_real_false_ai']}
    passed=all(all(v.values()) for v in checks.values())
    result={'state':'E92_limited_DEV_comparison_complete','checks':checks,'passes_limited_dev_screen':passed,
        'consumed_regression_may_be_registered':passed,'reports':reports,'contract_sha256':digest(CONTRACT),
        'scores_sha256':digest(SCORES),'candidate_sha256':digest(CANDIDATE),
        'independent_final_passed':False,'serving_changed':False,'limitations':c['limits']}
    write_once(REPORT,result);write_once(EVIDENCE/'e92_development.json',result)
    return {'checks':checks,'passes_limited_dev_screen':passed}

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=('freeze','score','report'))
    with (ROOT/'dev_execution.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        print(json.dumps({'freeze':freeze,'score':score,'report':report}[parser.parse_args().stage](),indent=2))
