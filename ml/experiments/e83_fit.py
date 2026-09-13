"""One worst-REAL constrained fit adding frozen supervised E82 features."""
from __future__ import annotations
import argparse
import fcntl
import json
import os
from pathlib import Path
import socket
import time
if __name__=='__main__':
    def denied(*a,**kw):raise RuntimeError('E83 fit is offline')
    socket.socket.connect=denied;socket.socket.connect_ex=denied;socket.create_connection=denied
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
import joblib
import torch
import numpy as np
from scipy.special import expit
from threadpoolctl import threadpool_limits
from experiments import e83_model as model
from experiments.e81_real_minimax import fit as fit_real
from experiments.e80_fit import full_training_gates,runtime_checks
from experiments.e82_representation import validate as validate_representation
from experiments.e76_fit import combine_rows,real_slice_gates
from experiments.e65_acquisition import digest,read,write_once
from experiments.e72_acquisition import resource_check
from pixelproof.project_paths import DATA_ROOT,ML_ROOT
from pixelproof.retention_gate import check_train_retention
ROOT=DATA_ROOT/'e83';EVIDENCE=ML_ROOT.parent/'evidence'
CONTRACT=ROOT/'fit_contract.json';CANDIDATE=ROOT/'correction.npz';REPORT=ROOT/'fit.json'
CONDITIONS=['clean','assigned_transport','q75']


def write_report(result):
    write_once(REPORT,result);write_once(EVIDENCE/'e83_fit.json',result)


def freeze():
    ROOT.mkdir(exist_ok=True);validate_representation()
    representation=read(DATA_ROOT/'e82/representation.json');failed=read(DATA_ROOT/'e81/fit.json')
    if digest(DATA_ROOT/'e82/representation.json')!=digest(EVIDENCE/'e82_representation.json') or \
            representation['state']!='E82_supervised_TRAIN_representation_complete' or \
            representation['epochs']!=100 or representation['TRAIN_views']!=36423 or \
            representation['ai_views']!=13785 or representation['feature_shape']!=[12141,3,64] or \
            not representation['final_weighted_BCE']<representation['initial_weighted_BCE'] or \
            not representation['serialized_latent_exact'] or not representation['serialized_coordinates_exact'] or \
            representation['fixed_input_batch8_max_error']>1e-10 or representation['dev_final_rows_read'] or \
            representation['new_image_inference'] or \
            digest(DATA_ROOT/'e82/supervised_map.npz')!=representation['artifact_sha256'] or \
            digest(DATA_ROOT/'e82/supervised_features.npz')!=representation['feature_sha256'] or \
            digest(DATA_ROOT/'e81/fit.json')!=digest(EVIDENCE/'e81_fit.json') or failed['dev_scoring_permitted']:
        raise ValueError('complete verified E82 and frozen failed E81 required')
    paths=[DATA_ROOT/'e82/representation_contract.json',DATA_ROOT/'e82/representation.json',
        DATA_ROOT/'e82/supervised_map.npz',DATA_ROOT/'e82/supervised_features.npz',
        DATA_ROOT/'e81/fit_contract.json',DATA_ROOT/'e81/fit.json',DATA_ROOT/'e80/correction.npz',
        Path(__file__),Path(model.__file__),Path(__file__).with_name('e82_representation.py'),
        Path(__file__).with_name('e80_fit.py'),Path(__file__).with_name('e81_real_minimax.py')]
    previous=read(DATA_ROOT/'e81/fit_contract.json')
    c={k:previous[k] for k in ['parents','old_parents','new_parents','real_parents','ai_parents','ai_views',
        'conditions','max_seconds','ai_cut','real_cut','train_guard','evaluation','dev_consumed_by',
        'dev_fresh_or_independent','dev_features_or_scores_in_fit','e49_read_allowed','promotion_allowed',
        'downloads','limits','objective','constraints','optimizer']}
    c.update(state='E83_supervised_feature_TRAIN_fit_registered',inputs={str(p):digest(p) for p in paths},
       representation='Exact E80 multimodal385 coordinates plus complete frozen E82 latent64 and intercept1. '
                      'No encoder, PCA, AE, scaler, MLP or direction refit; zero450 correction weights.',
       trigger='E80/E81 source and covered-accuracy TRAIN failures. E82 learned a class-discriminative nonlinear '
               'map on TRAIN only; low representation loss is not an external detector result.',
       replay='Require every cached E82 latent coordinate exactly reproduced in the combined map before optimization. '
              'Keep same E80 all-view runtime batch8 score/two-cut/all-AI-confidence guards after provisional TRAIN pass.',
       hypothesis='Supervised nonlinear coordinates may separate hard REAL from protected AI beyond prior fixed features. '
                  'One representation change with unchanged E81 objective; no rank/seed/weight/margin/cut sweep.')
    write_once(CONTRACT,c);write_once(EVIDENCE/'e83_fit_contract.json',c|{'contract_sha256':digest(CONTRACT)})
    return {'state':c['state'],'parents':c['parents']}


def validate():
    validate_representation();c=read(CONTRACT)
    if digest(CONTRACT)!=read(EVIDENCE/'e83_fit_contract.json')['contract_sha256']:raise ValueError('E83 contract changed')
    for p,s in c['inputs'].items():
        if digest(p)!=s:raise ValueError('E83 input changed: '+p)
    return c


def fit():
    c=validate();torch.set_num_threads(2)
    if CANDIDATE.exists() or REPORT.exists():raise FileExistsError('E83 candidate already fitted')
    start=time.monotonic();deadline=start+c['max_seconds'];resource_check(deadline)
    previous=read(DATA_ROOT/'e71/fit_contract.json');p={k:Path(v['path']) for k,v in previous['inputs'].items()}
    old_rows=read(p['manifest'])['rows'];new_rows=read(DATA_ROOT/'e72/training_manifest.json')['rows'];rows=combine_rows(old_rows,new_rows)
    with np.load(p['features'],allow_pickle=False) as a:original=a['features']
    with np.load(p['clip'],allow_pickle=False) as a:clip=a['features']
    with np.load(DATA_ROOT/'e75/midd_features.npz',allow_pickle=False) as a:
        if str(a['binding'])!=digest(DATA_ROOT/'e75/features_contract.json') or set(a['roles'])!={'TRAIN'} or \
                list(a['parents'])!=[r['parent_id'] for r in new_rows]:raise ValueError('new TRAIN alignment differs')
        original=np.concatenate([original,a['dino']]).reshape(-1,3072);clip=np.concatenate([clip,a['clip']]).reshape(-1,1536)
    with np.load(DATA_ROOT/'e79/dear_features.npz',allow_pickle=False) as cached:
        if str(cached['binding'])!=digest(DATA_ROOT/'e79/features_contract.json') or \
                set(cached['roles'])!={'TRAIN'} or list(cached['parents'])!=[r['parent_id'] for r in rows]:
            raise ValueError('forensic TRAIN alignment differs')
        dear=cached['features'].reshape(-1,1640)
    with np.load(DATA_ROOT/'e80/correction.npz',allow_pickle=False) as previous_arrays, \
            np.load(DATA_ROOT/'e82/supervised_map.npz',allow_pickle=False) as representation:
        if str(representation['binding'])!=digest(DATA_ROOT/'e82/representation_contract.json') or \
                str(representation['input_map_sha256'])!=digest(DATA_ROOT/'e80/correction.npz'):
            raise ValueError('supervised map bindings differ')
        arrays=model.assemble(previous_arrays,representation)
    head=joblib.load(p['reference'])['head'];labels=np.repeat([r['label'] for r in rows],3)
    if int(np.sum(labels==1))!=c['ai_views']:raise ValueError('incomplete original AI replay')
    with threadpool_limits(limits=2):
        baseline=head.predict_proba(original)[:,1];logits=head.decision_function(original);x=model.project(head,original,clip,dear,arrays)
        with np.load(DATA_ROOT/'e82/supervised_features.npz',allow_pickle=False) as cached:
            if str(cached['binding'])!=digest(DATA_ROOT/'e82/representation_contract.json') or \
                    set(cached['roles'])!={'TRAIN'} or list(cached['parents'])!=[r['parent_id'] for r in rows] or \
                    not np.array_equal(x[:,385:449],cached['features'].reshape(-1,64)):
                raise ValueError('complete supervised TRAIN feature replay differs')
        if not np.array_equal(baseline,model.predict(head,original,clip,dear,arrays)):raise ValueError('zero-init replay differs')
        counter=[0]
        def check():
            resource_check(deadline);counter[0]+=1
            if counter[0]%10==0:print(json.dumps({'E83_iteration':counter[0],'seconds':round(time.monotonic()-start)}),flush=True)
        arrays['weights'],solver=fit_real(x,logits,labels,np.repeat([r['source'] for r in rows],3),
            np.repeat([r['parent_id'] for r in rows],3),np.tile(CONDITIONS,len(rows)),check)
        candidate=model.predict(head,original,clip,dear,arrays)
        with CANDIDATE.open('xb') as f:np.savez_compressed(f,**arrays,contract_sha256=np.array(digest(CONTRACT)),
            reference_sha256=np.array(previous['inputs']['reference']['sha256']))
        with np.load(CANDIDATE,allow_pickle=False) as a:
            if not np.array_equal(candidate,model.predict(head,original,clip,dear,a)):raise ValueError('serialized replay differs')
    gate=check_train_retention(rows,baseline.reshape(-1,3),candidate.reshape(-1,3),conditions=CONDITIONS,ai_cut=model.AI_CUT)
    slices=real_slice_gates(rows,candidate.reshape(-1,3),c['old_parents'])
    absolute=full_training_gates(rows,candidate.reshape(-1,3),c['old_parents'])
    passed=all(v['gate']['passed'] for group in absolute.values() for v in group.values()) and solver['success'] and solver['max_constraint_violation']<=1e-8 and solver['minimum_ai_logit_shift']>=-1e-8 and \
        gate['passes_train_retention'] and all(gate['comparisons'][k]['real']['non_ai_to_ai']==0 and slices[k]['passed'] for k in CONDITIONS)
    runtime={'executed':False,'reason':'provisional TRAIN guards failed'}
    if passed:
        actual=[];shifts=[]
        with threadpool_limits(limits=2):
            for offset in range(0,len(labels),8):
                resource_check(deadline);end=offset+8
                correction=model.project(head,original[offset:end],clip[offset:end],dear[offset:end],arrays)@arrays['weights']
                actual.append(expit(head.decision_function(original[offset:end])+correction));shifts.append(correction)
        runtime={'executed':True,'batch_size':8}|runtime_checks(candidate,np.concatenate(actual),np.concatenate(shifts),labels)
        passed=runtime['passed']
    result={'state':'E83_TRAIN_guard_passed' if passed else 'E83_TRAIN_guard_failed','dev_scoring_permitted':bool(passed),
        'solver':solver,'train_gate':gate,'real_population_gates':slices,'full_TRAIN_metric_gates':absolute,
        'runtime_batch_replay':runtime,
        'reference_real_slices':real_slice_gates(rows,baseline.reshape(-1,3),c['old_parents']),
        'candidate_sha256':digest(CANDIDATE),'contract_sha256':digest(CONTRACT),'seconds':time.monotonic()-start,
        'dev_features_or_scores_in_fit':0,'e49_rows_read':0,'promotion_allowed':False,'independent_final_passed':False}
    write_report(result)
    return {'state':result['state'],'seconds':result['seconds'],'real_population_gates':slices,'solver_success':solver['success']}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=['freeze','fit'])
    ROOT.mkdir(exist_ok=True)
    with (ROOT/'fit_execution.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        print(json.dumps({'freeze':freeze,'fit':fit}[parser.parse_args().stage](),indent=2))
