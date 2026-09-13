"""One worst-REAL constrained fit adding frozen supervised E85 features."""
from __future__ import annotations
import argparse
import fcntl
import json
import os
from pathlib import Path
import socket
import time
if __name__=='__main__':
    def denied(*a,**kw):raise RuntimeError('E86 fit is offline')
    socket.socket.connect=denied;socket.socket.connect_ex=denied;socket.create_connection=denied
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
import joblib
import torch
import numpy as np
from scipy.special import expit
from threadpoolctl import threadpool_limits
from experiments import e86_model as model
from experiments.e81_real_minimax import fit as fit_real
from experiments.e80_fit import runtime_checks
from experiments.e86_gates import full_training_gates,population_gates as real_slice_gates
from experiments.e85_data import load as load_four,CONDITIONS
from experiments.e85_representation import validate as validate_representation
from experiments.e65_acquisition import digest,read,write_once
from experiments.e72_acquisition import resource_check
from pixelproof.project_paths import DATA_ROOT,ML_ROOT
from pixelproof.retention_gate import check_train_retention
ROOT=DATA_ROOT/'e86';EVIDENCE=ML_ROOT.parent/'evidence'
CONTRACT=ROOT/'fit_contract.json';CANDIDATE=ROOT/'correction.npz';REPORT=ROOT/'fit.json'


def write_report(result):
    write_once(REPORT,result);write_once(EVIDENCE/'e86_fit.json',result)


def freeze():
    ROOT.mkdir(exist_ok=True);validate_representation()
    representation=read(DATA_ROOT/'e85/representation.json');previous_result=read(DATA_ROOT/'e83/dev_report.json')
    if digest(DATA_ROOT/'e85/representation.json')!=digest(EVIDENCE/'e85_representation.json') or \
            representation['state']!='E85_supervised_TRAIN_representation_complete' or representation['epochs']!=100 or \
            representation['TRAIN_views']!=48564 or representation['ai_views']!=18380 or representation['feature_shape']!=[12141,4,64] or \
            representation['conditions']!=CONDITIONS or not representation['input_map_old_three_replay']['passed'] or \
            not representation['final_weighted_BCE']<representation['initial_weighted_BCE'] or \
            not representation['serialized_latent_exact'] or not representation['serialized_coordinates_exact'] or \
            representation['fixed_input_batch8_max_error']>1e-10 or representation['dev_final_rows_read'] or \
            representation['new_image_inference'] or digest(DATA_ROOT/'e85/supervised_map.npz')!=representation['artifact_sha256'] or \
            digest(DATA_ROOT/'e85/supervised_features.npz')!=representation['feature_sha256'] or \
            digest(DATA_ROOT/'e83/dev_report.json')!=digest(EVIDENCE/'e83_development.json') or previous_result['passes_limited_dev_screen']:
        raise ValueError('complete verified E85 and frozen rejected E83 required')
    paths=[DATA_ROOT/'e85/representation_contract.json',DATA_ROOT/'e85/representation.json',
        DATA_ROOT/'e85/supervised_map.npz',DATA_ROOT/'e85/supervised_features.npz',
        DATA_ROOT/'e83/fit_contract.json',DATA_ROOT/'e83/dev_report.json',DATA_ROOT/'e80/correction.npz',
        Path(__file__),Path(model.__file__),Path(__file__).with_name('e83_model.py'),
        Path(__file__).with_name('e85_representation.py'),Path(__file__).with_name('e85_data.py'),
        Path(__file__).with_name('e86_gates.py'),Path(__file__).with_name('e80_fit.py'),
        Path(__file__).with_name('e81_real_minimax.py')]
    old=read(DATA_ROOT/'e83/fit_contract.json')
    c={k:old[k] for k in ['parents','old_parents','new_parents','real_parents','ai_parents','max_seconds','ai_cut','real_cut',
        'dev_fresh_or_independent','dev_features_or_scores_in_fit','e49_read_allowed','promotion_allowed','downloads','limits','objective','optimizer']}
    c.update(state='E86_four_condition_supervised_TRAIN_fit_registered',inputs={str(p):digest(p) for p in paths},
        ai_views=18380,views=48564,conditions=CONDITIONS,dev_consumed_by=['E70','E71','E83'],
        representation='Same E83 functional450-column layout: exact E80 multimodal385, new frozen E85 supervised64, '
                       'intercept1. No feature/scaler/network refit here; zero450 weights. Same E81 worst-REAL objective.',
        constraints='Same per-view E43 confidence guard extended to all18,380 AI views, including missed AI, '
                    'and same correct-REAL decision guard for all four conditions and both old/new REAL parents.',
        train_guard='Solver success/violation<=1e-8/minimum AI shift>=-1e-8; exact saved replay; zero newly missed AI '
                    'and zero newly wrong REAL. For each of all four conditions, old/expanded/new-MIDD REAL<=10%, '
                    'worst MIDD sensor<=20%, and all10 original absolute numeric gates on both old and expanded TRAIN '
                    '(80 checks). New-condition success cannot mask any old-condition failure. Then replay all48,564 '
                    'views in runtime batches8: score error<=1e-6, both-cut decision changes0, minimum AI shift>=-1e-8.',
        evaluation='Only full TRAIN/runtime pass permits a separately frozen consumed E66 comparison using unchanged '
                   'E83 encoder cache. All20 numeric gates and zero new E43 AI losses per source/condition required; '
                   'also report paired changes against rejected E83. No E49 until DEV passes.',
        hypothesis='One TRAIN-view extension, same neural architecture/seed/optimizer and constrained-head objective. '
                   'No seed, epoch, rank, regularization, margin, weight or cut sweep. This cannot guarantee unseen AI retention.')
    write_once(CONTRACT,c);write_once(EVIDENCE/'e86_fit_contract.json',c|{'contract_sha256':digest(CONTRACT)})
    return {'state':c['state'],'parents':c['parents'],'views':c['views']}


def validate():
    validate_representation();c=read(CONTRACT)
    if digest(CONTRACT)!=read(EVIDENCE/'e86_fit_contract.json')['contract_sha256']:raise ValueError('E86 contract changed')
    for p,s in c['inputs'].items():
        if digest(p)!=s:raise ValueError('E86 input changed: '+p)
    return c


def fit():
    c=validate();torch.set_num_threads(2)
    if CANDIDATE.exists() or REPORT.exists():raise FileExistsError('E86 candidate already fitted')
    start=time.monotonic();deadline=start+c['max_seconds'];resource_check(deadline)
    previous=read(DATA_ROOT/'e71/fit_contract.json');rows,features,head=load_four()
    original=features['dino'].reshape(-1,3072);clip=features['clip'].reshape(-1,1536);dear=features['dear'].reshape(-1,1640)
    with np.load(DATA_ROOT/'e80/correction.npz',allow_pickle=False) as previous_arrays, \
            np.load(DATA_ROOT/'e85/supervised_map.npz',allow_pickle=False) as representation:
        if str(representation['binding'])!=digest(DATA_ROOT/'e85/representation_contract.json') or \
                str(representation['input_map_sha256'])!=digest(DATA_ROOT/'e80/correction.npz'):
            raise ValueError('supervised map bindings differ')
        arrays=model.assemble(previous_arrays,representation)
    labels=np.repeat([r['label'] for r in rows],4)
    if int(np.sum(labels==1))!=c['ai_views']:raise ValueError('incomplete original AI replay')
    with threadpool_limits(limits=2):
        baseline=head.predict_proba(original)[:,1];logits=head.decision_function(original);x=model.project(head,original,clip,dear,arrays)
        with np.load(DATA_ROOT/'e85/supervised_features.npz',allow_pickle=False) as cached:
            if str(cached['binding'])!=digest(DATA_ROOT/'e85/representation_contract.json') or \
                    set(cached['roles'])!={'TRAIN'} or list(cached['parents'])!=[r['parent_id'] for r in rows] or list(cached['conditions'])!=CONDITIONS or \
                    not np.array_equal(x[:,385:449],cached['features'].reshape(-1,64)):
                raise ValueError('complete supervised TRAIN feature replay differs')
        if not np.array_equal(baseline,model.predict(head,original,clip,dear,arrays)):raise ValueError('zero-init replay differs')
        counter=[0]
        def check():
            resource_check(deadline);counter[0]+=1
            if counter[0]%10==0:print(json.dumps({'E86_iteration':counter[0],'seconds':round(time.monotonic()-start)}),flush=True)
        arrays['weights'],solver=fit_real(x,logits,labels,np.repeat([r['source'] for r in rows],4),
            np.repeat([r['parent_id'] for r in rows],4),np.tile(CONDITIONS,len(rows)),check)
        candidate=model.predict(head,original,clip,dear,arrays)
        with CANDIDATE.open('xb') as f:np.savez_compressed(f,**arrays,contract_sha256=np.array(digest(CONTRACT)),
            reference_sha256=np.array(previous['inputs']['reference']['sha256']))
        with np.load(CANDIDATE,allow_pickle=False) as a:
            if not np.array_equal(candidate,model.predict(head,original,clip,dear,a)):raise ValueError('serialized replay differs')
    gate=check_train_retention(rows,baseline.reshape(-1,4),candidate.reshape(-1,4),conditions=CONDITIONS,ai_cut=model.AI_CUT)
    slices=real_slice_gates(rows,candidate.reshape(-1,4),c['old_parents'])
    absolute=full_training_gates(rows,candidate.reshape(-1,4),c['old_parents'])
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
    result={'state':'E86_TRAIN_guard_passed' if passed else 'E86_TRAIN_guard_failed','dev_scoring_permitted':bool(passed),
        'solver':solver,'train_gate':gate,'real_population_gates':slices,'full_TRAIN_metric_gates':absolute,
        'runtime_batch_replay':runtime,
        'reference_real_slices':real_slice_gates(rows,baseline.reshape(-1,4),c['old_parents']),
        'candidate_sha256':digest(CANDIDATE),'contract_sha256':digest(CONTRACT),'seconds':time.monotonic()-start,
        'conditions':CONDITIONS,'dev_features_or_scores_in_fit':0,'e49_rows_read':0,'promotion_allowed':False,'independent_final_passed':False}
    write_report(result)
    return {'state':result['state'],'seconds':result['seconds'],'real_population_gates':slices,'solver_success':solver['success']}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=['freeze','fit'])
    ROOT.mkdir(exist_ok=True)
    with (ROOT/'fit_execution.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        print(json.dumps({'freeze':freeze,'fit':fit}[parser.parse_args().stage](),indent=2))
