"""One full-AI-confidence-preserving fit in the exact frozen E71 TRAIN coordinates."""
from __future__ import annotations
import argparse
import fcntl
import json
import os
from pathlib import Path
import socket
import time
if __name__=='__main__':
    def denied(*a,**kw):raise RuntimeError('E73 fit is offline')
    socket.socket.connect=denied;socket.socket.connect_ex=denied;socket.create_connection=denied
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
import joblib
import numpy as np
from scipy.optimize import LinearConstraint,minimize
from scipy.special import logit
from threadpoolctl import threadpool_limits
from experiments import e71_model as model
from experiments.e64_constrained import objective,GUARD_MARGIN,MAX_ITER,FTOL
from experiments.e71_fit import validate as validate_previous
from experiments.e71_features import resource_check
from experiments.e65_acquisition import digest,read,write_once
from pixelproof.project_paths import DATA_ROOT,ML_ROOT
from pixelproof.training_weights import balanced_parent_weights
from pixelproof.retention_gate import check_train_retention
ROOT=DATA_ROOT/'e73'
EVIDENCE=ML_ROOT.parent/'evidence'
CONTRACT=ROOT/'fit_contract.json'
CANDIDATE=ROOT/'correction.npz'
REPORT=ROOT/'fit.json'


def constraints_for(x,baseline_logits,labels):
    if x.ndim!=2 or baseline_logits.shape!=(len(x),) or labels.shape!=(len(x),) or \
            not np.isfinite(x).all() or not np.isfinite(baseline_logits).all() or set(labels)!={0,1}:
        raise ValueError('finite aligned two-class TRAIN arrays required')
    ai=labels==1;real=(labels==0)&(baseline_logits<logit(model.AI_CUT))
    slack=np.maximum(logit(model.AI_CUT)-baseline_logits[real]-GUARD_MARGIN,0)
    bounds=[LinearConstraint(x[ai],0,np.inf)]
    if real.any():bounds.append(LinearConstraint(x[real],-np.inf,slack))
    return bounds,ai,real,slack


def fit_confidence(x,baseline_logits,labels,weights,check=lambda:None):
    bounds,ai,real,slack=constraints_for(x,baseline_logits,labels)
    trace=[]
    def callback(w):
        check();trace.append(float(objective(w,x,baseline_logits,labels,weights)[0]))
    result=minimize(objective,np.zeros(x.shape[1]),args=(x,baseline_logits,labels,weights),
        method='SLSQP',jac=True,constraints=bounds,callback=callback,
        options={'maxiter':MAX_ITER,'ftol':FTOL,'disp':False})
    shift=x@result.x
    violation=max(0.,float(np.max(-shift[ai])),float(np.max(shift[real]-slack)) if real.any() else 0.)
    return result.x,{'success':bool(result.success),'message':str(result.message),'iterations':int(result.nit),
        'objective':float(result.fun),'max_constraint_violation':violation,'trace':trace,
        'all_ai_views_protected':int(ai.sum()),'minimum_ai_logit_shift':float(np.min(shift[ai]))}


def reset_basis(previous):
    keys=(*model.BASE_KEYS,'clip_center','clip_scale','clip_mean','clip_components','clip_scales')
    a={k:np.array(previous[k],copy=True) for k in keys}
    a['weights']=np.zeros(129)
    return a


def freeze():
    ROOT.mkdir(exist_ok=True);previous=validate_previous()
    files=[Path(__file__),Path(model.__file__),Path(__file__).with_name('e64_constrained.py'),
        Path(__file__).with_name('e65_acquisition.py'),ML_ROOT/'src/pixelproof/training_weights.py',
        DATA_ROOT/'e71/fit_contract.json',DATA_ROOT/'e71/fit.json',DATA_ROOT/'e71/correction.npz',
        DATA_ROOT/'e71/dev_report.json',DATA_ROOT/'e71/dev_scores.json']
    dev=read(DATA_ROOT/'e71/dev_report.json');fit=read(DATA_ROOT/'e71/fit.json')
    if digest(DATA_ROOT/'e71/dev_report.json')!=digest(EVIDENCE/'e71_development.json') or \
            dev['passes_limited_dev_screen'] or digest(DATA_ROOT/'e71/dev_scores.json')!=dev['scores_sha256'] or \
            digest(DATA_ROOT/'e71/correction.npz')!=fit['candidate_sha256']:
        raise ValueError('frozen rejected E71 identity required')
    c={'state':'E73_full_AI_confidence_TRAIN_fit_registered','inputs':{str(p):digest(p) for p in files},
       'previous_fit_contract_sha256':digest(DATA_ROOT/'e71/fit_contract.json'),
       'representation':'Exact E71 original64 and CLIP64 TRAIN coordinates, no basis refit; zero129 weights, '
                        'discard E71 learned correction. No E72 data.',
       'objective':'Unchanged E64 cut-centered BCE, class/source/parent balancing, hard REAL2x, class mass.5, L2.01.',
       'constraints':'Every TRAIN AI logit >= its E43 baseline logit (ALL13,785 views, including missed AI). '
                     'Unchanged E64 correct-REAL decision protection with min(existing margin,1e-7).',
       'optimizer':'SLSQP200,ftol1e-9,CPUfloat64/two threads,zero start,final iterate only; no penalty/margin sweep.',
       'max_seconds':1800,'parents':11630,'ai_parents':4595,'ai_views':13785,
       'conditions':['clean','assigned_transport','q75'],'ai_cut':model.AI_CUT,'real_cut':model.REAL_CUT,
       'train_guard':'Solver success,violation<=1e-8,minimum AI logit shift>=-1e-8,exact serialized replay, '
                     'zero new AI misses/REAL errors and REAL FPR<=10% in ALL3 TRAIN conditions.',
       'evaluation':'Only TRAIN pass permits one separately registered CONSUMED E66 comparison with all20 '
                    'numeric gates, zero newly missed AI per source/condition and non-increased REAL FPR. '
                    'No E49 until that screen passes. TRAIN constraints do not prove external confidence retention.',
       'dev_consumed_by':['E70','E71'],'dev_fresh_or_independent':False,'dev_features_or_scores_in_fit':0,
       'previous_dev_score_body_hashed_only':True,'e49_read_allowed':False,'promotion_allowed':False,'downloads':0}
    write_once(CONTRACT,c);write_once(EVIDENCE/'e73_fit_contract.json',c|{'contract_sha256':digest(CONTRACT)})
    return {'state':c['state'],'ai_views':13785}


def validate():
    validate_previous();c=read(CONTRACT)
    if digest(CONTRACT)!=read(EVIDENCE/'e73_fit_contract.json')['contract_sha256']:
        raise ValueError('E73 contract changed')
    for p,s in c['inputs'].items():
        if digest(p)!=s:raise ValueError('E73 input changed: '+p)
    return c


def fit():
    c=validate()
    if CANDIDATE.exists() or REPORT.exists():raise FileExistsError('E73 candidate already fitted')
    previous=read(DATA_ROOT/'e71/fit_contract.json');p={k:Path(v['path']) for k,v in previous['inputs'].items()}
    rows=read(p['manifest'])['rows'];start=time.monotonic();deadline=start+c['max_seconds'];resource_check(deadline)
    if len(rows)!=c['parents'] or any(r['role'].upper()!='TRAIN' for r in rows):raise ValueError('complete TRAIN required')
    with np.load(p['features'],allow_pickle=False) as a:
        if str(a['binding'])!=digest(p['manifest']) or a['features'].shape!=(11630,3,3072):raise ValueError('TRAIN DINO binding differs')
        original=a['features'].reshape(-1,3072)
    with np.load(p['clip'],allow_pickle=False) as a:
        if str(a['binding'])!=digest(p['feature_contract']) or list(a['parents'])!=[r['parent_id'] for r in rows] or \
                a['features'].shape!=(11630,3,1536) or set(a['roles'])!={'TRAIN'}:raise ValueError('TRAIN CLIP alignment differs')
        clip=a['features'].reshape(-1,1536)
    with np.load(DATA_ROOT/'e71/correction.npz',allow_pickle=False) as a:arrays=reset_basis(a)
    head=joblib.load(p['reference'])['head'];labels=np.repeat([r['label'] for r in rows],3)
    if int(np.sum(labels==1))!=c['ai_views']:raise ValueError('incomplete AI confidence replay')
    with threadpool_limits(limits=2):
        baseline=head.predict_proba(original)[:,1];logits=head.decision_function(original)
        x=model.project(head,original,clip,arrays)
        if not np.array_equal(baseline,model.predict(head,original,clip,arrays)):raise ValueError('zero-init parity failed')
        weights=balanced_parent_weights(labels,np.repeat([r['source'] for r in rows],3),np.repeat([r['parent_id'] for r in rows],3))
        weights[(labels==0)&(baseline>=model.AI_CUT)]*=2
        for y in (0,1):weights[labels==y]*=.5/weights[labels==y].sum()
        counter=[0]
        def check():
            resource_check(deadline);counter[0]+=1
            if counter[0]%10==0:print(json.dumps({'E73_iteration':counter[0],'seconds':round(time.monotonic()-start)}),flush=True)
        arrays['weights'],solver=fit_confidence(x,logits,labels,weights,check)
        candidate=model.predict(head,original,clip,arrays);shift=x@arrays['weights']
        with CANDIDATE.open('xb') as f:np.savez_compressed(f,**arrays,contract_sha256=np.array(digest(CONTRACT)),
            reference_sha256=np.array(previous['inputs']['reference']['sha256']))
        with np.load(CANDIDATE,allow_pickle=False) as a:
            if not np.array_equal(candidate,model.predict(head,original,clip,a)):raise ValueError('serialized replay differs')
    gate=check_train_retention(rows,baseline.reshape(-1,3),candidate.reshape(-1,3),conditions=c['conditions'],ai_cut=model.AI_CUT)
    passed=solver['success'] and solver['max_constraint_violation']<=1e-8 and solver['minimum_ai_logit_shift']>=-1e-8 and \
        gate['passes_train_retention'] and all(gate['comparisons'][k]['real']['non_ai_to_ai']==0 and
            gate['comparisons'][k]['real']['new_ai_rate']<=.10 for k in c['conditions'])
    result={'state':'E73_TRAIN_guard_passed' if passed else 'E73_TRAIN_guard_failed','dev_scoring_permitted':bool(passed),
        'solver':solver,'train_gate':gate,'candidate_sha256':digest(CANDIDATE),'contract_sha256':digest(CONTRACT),
        'ai_logit_shift_quantiles':dict(zip(['min','q01','q05','q50','q95','max'],np.quantile(shift[labels==1],[0,.01,.05,.5,.95,1]).tolist())),
        'seconds':time.monotonic()-start,'e49_rows_read':0,'dev_features_or_scores_in_fit':0,
        'promotion_allowed':False,'independent_final_passed':False}
    write_once(REPORT,result);write_once(EVIDENCE/'e73_fit.json',result)
    return {k:result[k] for k in ['state','solver','seconds','ai_logit_shift_quantiles']} | {
        'real_rates':{k:gate['comparisons'][k]['real']['new_ai_rate'] for k in c['conditions']}}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=['freeze','fit'])
    ROOT.mkdir(exist_ok=True)
    with (ROOT/'fit_execution.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        print(json.dumps({'freeze':freeze,'fit':fit}[parser.parse_args().stage](),indent=2))
