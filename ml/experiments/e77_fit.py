"""One full-confidence TRAIN fit with frozen REAL-only reconstruction residual."""
from __future__ import annotations
import argparse
import fcntl
import json
import os
from pathlib import Path
import socket
import time
if __name__=='__main__':
    def denied(*a,**kw):raise RuntimeError('E77 fit is offline')
    socket.socket.connect=denied;socket.socket.connect_ex=denied;socket.create_connection=denied
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
import joblib
import torch
import numpy as np
from threadpoolctl import threadpool_limits
from experiments import e77_model as model
from experiments.e73_fit import fit_confidence
from experiments.e77_representation import validate as validate_representation
from experiments.e76_fit import combine_rows,real_slice_gates
from experiments.e65_acquisition import digest,read,write_once
from experiments.e72_acquisition import resource_check
from pixelproof.project_paths import DATA_ROOT,ML_ROOT
from pixelproof.training_weights import balanced_parent_weights
from pixelproof.retention_gate import check_train_retention
ROOT=DATA_ROOT/'e77';EVIDENCE=ML_ROOT.parent/'evidence'
CONTRACT=ROOT/'fit_contract.json';CANDIDATE=ROOT/'correction.npz';REPORT=ROOT/'fit.json'
CONDITIONS=['clean','assigned_transport','q75']


def freeze():
    ROOT.mkdir(exist_ok=True);validate_representation()
    paths=[DATA_ROOT/'e77/representation_contract.json',DATA_ROOT/'e77/representation.json',
        DATA_ROOT/'e77/manifold.npz',DATA_ROOT/'e77/manifold_features.npz',
        DATA_ROOT/'e76/fit.json',DATA_ROOT/'e74/correction.npz',
        Path(__file__),Path(model.__file__),Path(__file__).with_name('e77_representation.py'),
        Path(__file__).with_name('e76_fit.py'),Path(__file__).with_name('e73_fit.py'),
        Path(__file__).with_name('e64_constrained.py'),ML_ROOT/'src/pixelproof/training_weights.py']
    representation=read(paths[1])
    if digest(paths[1])!=digest(EVIDENCE/'e77_representation.json') or \
            representation['state']!='E77_REAL_manifold_representation_complete' or \
            representation['epochs']!=100 or representation['real_views']!=22638 or \
            representation['ai_views_in_AE_loss'] or representation['classifier_scores'] or \
            representation['dev_or_final_rows_read'] or \
            representation['feature_shape']!=[12141,3,64] or \
            digest(paths[2])!=representation['artifact_sha256'] or \
            digest(paths[3])!=representation['feature_sha256'] or \
            not representation['serialized_residual_exact'] or not representation['serialized_coordinates_exact']:
        raise ValueError('complete REAL-only representation and parity required')
    previous=read(DATA_ROOT/'e71/fit_contract.json')
    rows=combine_rows(read(previous['inputs']['manifest']['path'])['rows'],read(DATA_ROOT/'e72/training_manifest.json')['rows'])
    c={'state':'E77_REAL_manifold_TRAIN_fit_registered','inputs':{str(p):digest(p) for p in paths},
       'representation':'Exact E74 original64/CLIP64/bilinear128 plus frozen E77 REAL-only residual64. '
                        'No representation refit; discard all previous correction weights, zero321 coefficients.',
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
    write_once(CONTRACT,c);write_once(EVIDENCE/'e77_fit_contract.json',c|{'contract_sha256':digest(CONTRACT)})
    return {'state':c['state'],'parents':len(rows)}


def validate():
    validate_representation();c=read(CONTRACT)
    if digest(CONTRACT)!=read(EVIDENCE/'e77_fit_contract.json')['contract_sha256']:raise ValueError('E77 contract changed')
    for p,s in c['inputs'].items():
        if digest(p)!=s:raise ValueError('E77 input changed: '+p)
    return c


def fit():
    c=validate();torch.set_num_threads(2)
    if CANDIDATE.exists() or REPORT.exists():raise FileExistsError('E77 candidate already fitted')
    start=time.monotonic();deadline=start+c['max_seconds'];resource_check(deadline)
    previous=read(DATA_ROOT/'e71/fit_contract.json');p={k:Path(v['path']) for k,v in previous['inputs'].items()}
    old_rows=read(p['manifest'])['rows'];new_rows=read(DATA_ROOT/'e72/training_manifest.json')['rows'];rows=combine_rows(old_rows,new_rows)
    with np.load(p['features'],allow_pickle=False) as a:original=a['features']
    with np.load(p['clip'],allow_pickle=False) as a:clip=a['features']
    with np.load(DATA_ROOT/'e75/midd_features.npz',allow_pickle=False) as a:
        if str(a['binding'])!=digest(DATA_ROOT/'e75/features_contract.json') or set(a['roles'])!={'TRAIN'} or \
                list(a['parents'])!=[r['parent_id'] for r in new_rows]:raise ValueError('new TRAIN alignment differs')
        original=np.concatenate([original,a['dino']]).reshape(-1,3072);clip=np.concatenate([clip,a['clip']]).reshape(-1,1536)
    with np.load(DATA_ROOT/'e74/correction.npz',allow_pickle=False) as old, \
            np.load(DATA_ROOT/'e77/manifold.npz',allow_pickle=False) as representation:
        if str(representation['binding'])!=digest(DATA_ROOT/'e77/representation_contract.json'):
            raise ValueError('representation binding differs')
        arrays=model.assemble(old,representation)
    head=joblib.load(p['reference'])['head'];labels=np.repeat([r['label'] for r in rows],3)
    if int(np.sum(labels==1))!=c['ai_views']:raise ValueError('incomplete original AI replay')
    with threadpool_limits(limits=2):
        baseline=head.predict_proba(original)[:,1];logits=head.decision_function(original);x=model.project(head,original,clip,arrays)
        with np.load(DATA_ROOT/'e77/manifold_features.npz',allow_pickle=False) as cached:
            if str(cached['binding'])!=digest(DATA_ROOT/'e77/representation_contract.json') or \
                    set(cached['roles'])!={'TRAIN'} or list(cached['parents'])!=[r['parent_id'] for r in rows] or \
                    not np.array_equal(x[:,256:320],cached['features'].reshape(-1,64)):
                raise ValueError('residual TRAIN alignment/replay differs')
        if not np.array_equal(baseline,model.predict(head,original,clip,arrays)):raise ValueError('zero-init replay differs')
        weights=balanced_parent_weights(labels,np.repeat([r['source'] for r in rows],3),np.repeat([r['parent_id'] for r in rows],3))
        weights[(labels==0)&(baseline>=model.AI_CUT)]*=2
        for y in (0,1):weights[labels==y]*=.5/weights[labels==y].sum()
        counter=[0]
        def check():
            resource_check(deadline);counter[0]+=1
            if counter[0]%10==0:print(json.dumps({'E77_iteration':counter[0],'seconds':round(time.monotonic()-start)}),flush=True)
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
    result={'state':'E77_TRAIN_guard_passed' if passed else 'E77_TRAIN_guard_failed','dev_scoring_permitted':bool(passed),
        'solver':solver,'train_gate':gate,'real_population_gates':slices,
        'reference_real_slices':real_slice_gates(rows,baseline.reshape(-1,3),c['old_parents']),
        'candidate_sha256':digest(CANDIDATE),'contract_sha256':digest(CONTRACT),'seconds':time.monotonic()-start,
        'dev_features_or_scores_in_fit':0,'e49_rows_read':0,'promotion_allowed':False,'independent_final_passed':False}
    write_once(REPORT,result);write_once(EVIDENCE/'e77_fit.json',result)
    return {'state':result['state'],'seconds':result['seconds'],'real_population_gates':slices,'solver_success':solver['success']}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=['freeze','fit'])
    ROOT.mkdir(exist_ok=True)
    with (ROOT/'fit_execution.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        print(json.dumps({'freeze':freeze,'fit':fit}[parser.parse_args().stage](),indent=2))
