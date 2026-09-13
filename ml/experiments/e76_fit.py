"""One fixed-map, full-confidence TRAIN fit after audited native MIDD expansion."""
from __future__ import annotations
import argparse
import fcntl
import json
import os
from pathlib import Path
import socket
import time
if __name__=='__main__':
    def denied(*a,**kw):raise RuntimeError('E76 fit is offline')
    socket.socket.connect=denied;socket.socket.connect_ex=denied;socket.create_connection=denied
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
import joblib
import numpy as np
from threadpoolctl import threadpool_limits
from experiments import e74_model as model
from experiments.e73_fit import fit_confidence
from experiments.e74_fit import validate as validate_map
from experiments.e75_features import validate as validate_features
from experiments.e65_acquisition import digest,read,write_once
from experiments.e72_acquisition import resource_check
from pixelproof.project_paths import DATA_ROOT,ML_ROOT
from pixelproof.training_weights import balanced_parent_weights
from pixelproof.retention_gate import check_train_retention
ROOT=DATA_ROOT/'e76';EVIDENCE=ML_ROOT.parent/'evidence'
CONTRACT=ROOT/'fit_contract.json';CANDIDATE=ROOT/'correction.npz';REPORT=ROOT/'fit.json'
CONDITIONS=['clean','assigned_transport','q75']


def combine_rows(old,new):
    rows=old+new
    if len(old)!=11630 or len(new)!=511 or len({r['parent_id'] for r in rows})!=len(rows) or \
            len({r['sha256'] for r in rows})!=len(rows) or any(r['role'].upper()!='TRAIN' for r in rows) or \
            any(r['label']!=0 or not r['source'].startswith('MIDD:') for r in new) or \
            sum(r['label']==1 for r in rows)!=4595:
        raise ValueError('complete disjoint admitted expansion required')
    return rows


def real_slice_gates(rows,scores,old_count):
    if scores.shape!=(len(rows),3) or not np.isfinite(scores).all() or not 0<old_count<len(rows):
        raise ValueError('finite paired expanded score coverage required')
    real=np.array([r['label']==0 for r in rows]);is_new=np.arange(len(rows))>=old_count
    if not np.all(real[is_new]) or not (real&~is_new).any():raise ValueError('both old and new REAL required')
    detected=scores>=model.AI_CUT;source=np.array([r['source'] for r in rows]);result={}
    for i,condition in enumerate(CONDITIONS):
        sensor={s:float(detected[is_new&(source==s),i].mean()) for s in sorted(set(source[is_new]))}
        rates={'old_real':float(detected[real&~is_new,i].mean()),'expanded_real':float(detected[real,i].mean()),
               'new_midd_real':float(detected[is_new,i].mean()),'new_midd_by_sensor':sensor}
        checks={k:rates[k]<=.10 for k in ['old_real','expanded_real','new_midd_real']}
        checks['worst_new_sensor_lte_20']=max(sensor.values())<=.20
        result[condition]={'rates':rates,'checks':checks,'passed':all(checks.values())}
    return result


def freeze():
    ROOT.mkdir(exist_ok=True);validate_map();validate_features()
    paths=[DATA_ROOT/'e75/features.json',DATA_ROOT/'e75/midd_features.npz',DATA_ROOT/'e75/features_contract.json',
        DATA_ROOT/'e72/training_manifest.json',DATA_ROOT/'e74/fit.json',DATA_ROOT/'e74/correction.npz',
        Path(__file__),Path(model.__file__),Path(__file__).with_name('e73_fit.py'),
        Path(__file__).with_name('e64_constrained.py'),Path(__file__).with_name('e65_acquisition.py'),
        ML_ROOT/'src/pixelproof/training_weights.py']
    features=read(paths[0]);failed=read(DATA_ROOT/'e74/fit.json')
    if digest(paths[0])!=read(EVIDENCE/'e75_features.json')['report_sha256'] or \
            digest(paths[1])!=features['feature_sha256'] or features['dino_shape']!=[511,3,3072] or \
            features['clip_shape']!=[511,3,1536] or features['reference_decision_changes'] or \
            features['new_MIDD_classifier_scores'] or features['reference_max_score_error']>5e-5 or \
            features['clip_max_feature_error']>1e-5:
        raise ValueError('complete parity-verified MIDD features required')
    if digest(DATA_ROOT/'e74/fit.json')!=digest(EVIDENCE/'e74_fit.json') or failed['dev_scoring_permitted'] or \
            digest(DATA_ROOT/'e74/correction.npz')!=failed['candidate_sha256']:
        raise ValueError('frozen failed E74 map required')
    previous=read(DATA_ROOT/'e71/fit_contract.json')
    rows=combine_rows(read(previous['inputs']['manifest']['path'])['rows'],read(DATA_ROOT/'e72/training_manifest.json')['rows'])
    c={'state':'E76_fixed_map_MIDD_expanded_TRAIN_fit_registered','inputs':{str(p):digest(p) for p in paths},
       'representation':'Exact E74 original64/CLIP64/bilinear128 coordinates and old-TRAIN scales, no refit; '
                        'discard correction weights, zero257 coefficients. Only new input is511 audited MIDD TRAIN parents.',
       'objective':'Unchanged E64 cut-centered BCE, class/source/parent balance, hard REAL2x, class mass.5, L2.01.',
       'constraints':'Unchanged E73 non-decreasing logits for all13,785 AI TRAIN views; E64 correct-REAL '
                     'decision guard applies to old and new REAL views. No margin/weight/cut sweep.',
       'optimizer':'SLSQP200,ftol1e-9,CPUfloat64/two threads,zero start,final iterate only.',
       'parents':len(rows),'old_parents':11630,'new_parents':511,'real_parents':7546,'ai_parents':4595,
       'ai_views':13785,'conditions':CONDITIONS,'max_seconds':1800,'ai_cut':model.AI_CUT,'real_cut':model.REAL_CUT,
       'train_guard':'Solver success/violation<=1e-8, minimum AI shift>=-1e-8, exact saved replay, zero new AI/REAL '
                     'errors. All3 conditions: <=10% REAL FPR in old/expanded/new-MIDD slices, <=20% worst MIDD sensor.',
       'evaluation':'Only complete TRAIN pass permits one separately registered consumed E66 screen with all20 '
                    'numeric gates and zero new AI misses per source/condition; no E49 until that screen passes.',
       'dev_consumed_by':['E70','E71'],'dev_fresh_or_independent':False,'dev_features_or_scores_in_fit':0,
       'e49_read_allowed':False,'promotion_allowed':False,'downloads':0,
       'limits':'Research-only MIDD license; unknown scene independence. Whole publisher TRAIN only. '
                'TRAIN score protection is not external confidence/recall evidence.'}
    write_once(CONTRACT,c);write_once(EVIDENCE/'e76_fit_contract.json',c|{'contract_sha256':digest(CONTRACT)})
    return {'state':c['state'],'parents':len(rows)}


def validate():
    validate_map();validate_features();c=read(CONTRACT)
    if digest(CONTRACT)!=read(EVIDENCE/'e76_fit_contract.json')['contract_sha256']:raise ValueError('E76 contract changed')
    for p,s in c['inputs'].items():
        if digest(p)!=s:raise ValueError('E76 input changed: '+p)
    return c


def fit():
    c=validate()
    if CANDIDATE.exists() or REPORT.exists():raise FileExistsError('E76 candidate already fitted')
    start=time.monotonic();deadline=start+c['max_seconds'];resource_check(deadline)
    previous=read(DATA_ROOT/'e71/fit_contract.json');p={k:Path(v['path']) for k,v in previous['inputs'].items()}
    old_rows=read(p['manifest'])['rows'];new_rows=read(DATA_ROOT/'e72/training_manifest.json')['rows'];rows=combine_rows(old_rows,new_rows)
    with np.load(p['features'],allow_pickle=False) as a:original=a['features']
    with np.load(p['clip'],allow_pickle=False) as a:clip=a['features']
    with np.load(DATA_ROOT/'e75/midd_features.npz',allow_pickle=False) as a:
        if str(a['binding'])!=digest(DATA_ROOT/'e75/features_contract.json') or set(a['roles'])!={'TRAIN'} or \
                list(a['parents'])!=[r['parent_id'] for r in new_rows]:raise ValueError('new TRAIN alignment differs')
        original=np.concatenate([original,a['dino']]).reshape(-1,3072);clip=np.concatenate([clip,a['clip']]).reshape(-1,1536)
    with np.load(DATA_ROOT/'e74/correction.npz',allow_pickle=False) as a:
        arrays={k:a[k] for k in a.files if k not in ['weights','contract_sha256','reference_sha256']}
    arrays['weights']=np.zeros(257);head=joblib.load(p['reference'])['head'];labels=np.repeat([r['label'] for r in rows],3)
    if int(np.sum(labels==1))!=c['ai_views']:raise ValueError('incomplete original AI replay')
    with threadpool_limits(limits=2):
        baseline=head.predict_proba(original)[:,1];logits=head.decision_function(original);x=model.project(head,original,clip,arrays)
        if not np.array_equal(baseline,model.predict(head,original,clip,arrays)):raise ValueError('zero-init replay differs')
        weights=balanced_parent_weights(labels,np.repeat([r['source'] for r in rows],3),np.repeat([r['parent_id'] for r in rows],3))
        weights[(labels==0)&(baseline>=model.AI_CUT)]*=2
        for y in (0,1):weights[labels==y]*=.5/weights[labels==y].sum()
        counter=[0]
        def check():
            resource_check(deadline);counter[0]+=1
            if counter[0]%10==0:print(json.dumps({'E76_iteration':counter[0],'seconds':round(time.monotonic()-start)}),flush=True)
        arrays['weights'],solver=fit_confidence(x,logits,labels,weights,check)
        candidate=model.predict(head,original,clip,arrays)
        with CANDIDATE.open('xb') as f:np.savez_compressed(f,**arrays,contract_sha256=np.array(digest(CONTRACT)),
            reference_sha256=np.array(previous['inputs']['reference']['sha256']))
        with np.load(CANDIDATE,allow_pickle=False) as a:
            if not np.array_equal(candidate,model.predict(head,original,clip,a)):raise ValueError('serialized replay differs')
    gate=check_train_retention(rows,baseline.reshape(-1,3),candidate.reshape(-1,3),conditions=CONDITIONS,ai_cut=model.AI_CUT)
    slices=real_slice_gates(rows,candidate.reshape(-1,3),c['old_parents'])
    passed=solver['success'] and solver['max_constraint_violation']<=1e-8 and solver['minimum_ai_logit_shift']>=-1e-8 and \
        gate['passes_train_retention'] and all(gate['comparisons'][k]['real']['non_ai_to_ai']==0 and slices[k]['passed'] for k in CONDITIONS)
    result={'state':'E76_TRAIN_guard_passed' if passed else 'E76_TRAIN_guard_failed','dev_scoring_permitted':bool(passed),
        'solver':solver,'train_gate':gate,'real_population_gates':slices,
        'reference_real_slices':real_slice_gates(rows,baseline.reshape(-1,3),c['old_parents']),
        'candidate_sha256':digest(CANDIDATE),'contract_sha256':digest(CONTRACT),'seconds':time.monotonic()-start,
        'dev_features_or_scores_in_fit':0,'e49_rows_read':0,'promotion_allowed':False,'independent_final_passed':False}
    write_once(REPORT,result);write_once(EVIDENCE/'e76_fit.json',result)
    return {'state':result['state'],'seconds':result['seconds'],'real_population_gates':slices,'solver_success':solver['success']}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=['freeze','fit'])
    ROOT.mkdir(exist_ok=True)
    with (ROOT/'fit_execution.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        print(json.dumps({'freeze':freeze,'fit':fit}[parser.parse_args().stage](),indent=2))
