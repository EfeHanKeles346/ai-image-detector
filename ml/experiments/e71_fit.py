"""One E71 TRAIN fit with E66 explicitly recorded as consumed development."""
from __future__ import annotations
import argparse
import fcntl
import json
import os
from pathlib import Path
import socket
import time

if __name__=='__main__':
    def denied(*args,**kwargs):raise RuntimeError('E71 fit network disabled')
    socket.socket.connect=denied;socket.socket.connect_ex=denied;socket.create_connection=denied
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HUB_DISABLE_TELEMETRY='1')

import joblib
import numpy as np
from threadpoolctl import threadpool_limits
from experiments import e71_model as model
from experiments.e71_features import ROOT,EVIDENCE,OUTPUT as CLIP,REPORT as FEATURES_REPORT,resource_check,validate as validate_features
from experiments.e65_acquisition import digest,read,write_once
from pixelproof.retention_gate import check_train_retention
from pixelproof.training_weights import balanced_parent_weights
from pixelproof.project_paths import DATA_ROOT,ML_ROOT

CONTRACT=ROOT/'fit_contract.json'
CANDIDATE=ROOT/'correction.npz'
REPORT=ROOT/'fit.json'


def freeze():
    validate_features()
    previous=read(EVIDENCE/'e61_replay_contract.json')
    inputs={k:previous['inputs'][k] for k in ['manifest','features','reference','gate_code']}
    feature=read(FEATURES_REPORT)
    if digest(FEATURES_REPORT)!=read(EVIDENCE/'e71_features.json')['report_sha256'] or feature['legacy_raw_max_error'] > 1e-5 or feature['shape'] != [11630,3,1536] or \
            feature['contract_sha256']!=digest(ROOT/'features_contract.json') or feature['state']!='E71_TRAIN_CLIP_features_complete':
        raise ValueError('unverified CLIP feature extraction')
    inputs['clip']={'path':str(CLIP),'sha256':feature['feature_sha256']}
    files={'features_report':FEATURES_REPORT,'feature_contract':ROOT/'features_contract.json',
           'model_code':Path(model.__file__),'runner_code':Path(__file__),
           'constraint_code':Path(__file__).with_name('e64_constrained.py'),
           'feature_code':Path(__file__).with_name('e71_features.py'),
           'helper_code':Path(__file__).with_name('e65_acquisition.py'),
           'weights_code':ML_ROOT/'src/pixelproof/training_weights.py',
           'basis_artifact':DATA_ROOT/'e67/correction.npz',
           'basis_contract':DATA_ROOT/'e67/fit_contract.json',
           'basis_report':DATA_ROOT/'e67/fit.json',
           'previous_dev_report':DATA_ROOT/'e70/dev_report.json',
           'previous_dev_scores':DATA_ROOT/'e70/dev_scores.json'}
    inputs.update({k:{'path':str(p),'sha256':digest(p)} for k,p in files.items()})
    dev=DATA_ROOT/'e66/development_manifest.json'
    expected=read(EVIDENCE/'e66_development.json')['manifest_sha256']
    if digest(dev)!=expected:raise ValueError('DEV identity changed')
    population=read(dev)
    exposure=read(files['previous_dev_report'])
    if digest(files['previous_dev_report'])!=digest(EVIDENCE/'e70_development.json') or \
            exposure['passes_limited_dev_screen'] or exposure['scores_sha256']!=digest(files['previous_dev_scores']):
        raise ValueError('locked failed E70 consumed DEV exposure required')
    basis_report=read(files['basis_report'])
    if digest(files['basis_report'])!=digest(EVIDENCE/'e67_fit.json') or \
            digest(files['basis_artifact'])!=basis_report['candidate_sha256'] or \
            digest(files['basis_contract'])!=basis_report['contract_sha256']:
        raise ValueError('original TRAIN basis identity changed')
    if population['class_counts']!={'0':160,'1':160}:
        raise ValueError('complete consumed paired-label DEV required before fit')
    # DEV metadata establishes separation; no DEV features, scores or labels enter fitting arrays.
    train_rows=read(inputs['manifest']['path'])['rows'];train_ids={r['parent_id'] for r in train_rows}
    train_sha={r['sha256'] for r in train_rows}
    if any(r['parent_id'] in train_ids or r['sha256'] in train_sha or r['role']!='DEVELOPMENT' for r in population['rows']):
        raise ValueError('TRAIN/DEV role or identity overlap')
    for v in inputs.values():
        if digest(v['path'])!=v['sha256']:raise ValueError('frozen fit input changed')
    c={'state':'E71_original_plus_CLIP_TRAIN_fit_registered','inputs':inputs,
       'dev_manifest':str(dev),'dev_manifest_sha256':expected,'dev_metadata_only_checked':True,
       'representation':'Reuse exact E67 original TRAIN PCA64 coordinates; TRAIN StandardScaler on CLIP1536 '
                        'followed by PCA64(seed71,power3). Whiten both; intercept129 coefficients. '
                        'Discard old response/weights. Zero-initialization exactly E43.',
       'objective':'Exactly E64 class/source/parent-balanced operating-cut BCE, hard REAL views2x, '
                   'class mass.5 each, L2.01 including intercept. Frozen E43 logit + learned linear correction.',
       'constraints':'Exactly E64: every caught TRAIN AI and every correct TRAIN REAL retains its decision '
                     'with min(existing_margin,1e-7) safety margin. Zero-initialization exactly E43.',
       'optimizer':'CPU float64 SLSQP200,ftol1e-9,final iterate only; no rank/L2/seed/threshold sweep.',
       'parents':11630,'ai_parents':4595,'conditions':['clean','assigned_transport','q75'],
       'ai_cut':model.AI_CUT,'real_cut':model.REAL_CUT,'max_seconds':1800,'cpu_threads':2,
       'train_guard':'Solver success,violation<=1e-8,exact serialized replay,E61 zero newly missed AI, '
                     'zero new REAL errors and REAL TRAIN FPR<=10% in ALL3 conditions.',
       'evaluation':'Only if TRAIN guard passes: separately freeze one CONSUMED E66 DEV original/1080pxQ75 comparison '
                    'with same E43 cuts, all20 absolute/selective gates and per-image/per-source AI retention. '
                    'Only if DEV passes may a separate one-shot consumed E49 regression be preregistered; '
                    'E49 never chooses another recipe. No independent final/promotion from E66/E49.',
       'independent_final_allowed':False,'promotion_allowed':False,'e49_read_allowed':False,
       'dev_labels_used_in_fitting':False,'dev_consumed_by':['E70'],
       'dev_fresh_or_independent':False,'downloads':0}
    write_once(CONTRACT,c);write_once(EVIDENCE/'e71_fit_contract.json',c|{'contract_sha256':digest(CONTRACT)})
    return {'state':c['state'],'dev_manifest_sha256':expected}


def validate():
    c=read(CONTRACT)
    if digest(CONTRACT)!=read(EVIDENCE/'e71_fit_contract.json')['contract_sha256']:raise ValueError('E71 fit contract changed')
    for v in c['inputs'].values():
        if digest(v['path'])!=v['sha256']:raise ValueError('E71 fit input changed')
    if digest(c['dev_manifest'])!=c['dev_manifest_sha256']:raise ValueError('frozen DEV manifest changed')
    return c


def fit():
    c=validate()
    if CANDIDATE.exists() or REPORT.exists():raise FileExistsError('E71 candidate already fitted')
    started=time.monotonic();deadline=started+c['max_seconds'];resource_check(deadline)
    p={k:Path(v['path']) for k,v in c['inputs'].items()};rows=read(p['manifest'])['rows']
    if len(rows)!=11630 or any(r['role'].upper()!='TRAIN' for r in rows):raise ValueError('invalid TRAIN population')
    with np.load(p['features'],allow_pickle=False) as a:
        if str(a['binding'])!=digest(p['manifest']) or a['features'].shape!=(11630,3,3072):raise ValueError('original feature binding differs')
        original=a['features'].reshape(-1,3072)
    with np.load(p['clip'],allow_pickle=False) as a:
        if str(a['binding'])!=digest(p['feature_contract']) or a['features'].shape!=(11630,3,1536) or \
            list(a['parents'])!=[r['parent_id'] for r in rows] or set(a['roles'])!={'TRAIN'}:raise ValueError('CLIP alignment/role differs')
        clip=a['features'].reshape(-1,1536)
    with np.load(p['basis_artifact'],allow_pickle=False) as a:
        if str(a['contract_sha256'])!=digest(p['basis_contract']) or \
                str(a['reference_sha256'])!=c['inputs']['reference']['sha256']:
            raise ValueError('original basis artifact binding differs')
        previous_basis={key:a[key] for key in model.BASE_KEYS}
    head=joblib.load(p['reference'])['head'];labels=np.repeat([r['label'] for r in rows],3)
    sources=np.repeat([r['source'] for r in rows],3);parents=np.repeat([r['parent_id'] for r in rows],3)
    if int(np.sum(labels==1))!=c['ai_parents']*3:raise ValueError('incomplete AI replay')
    with threadpool_limits(limits=2):
        baseline=head.predict_proba(original)[:,1];logits=head.decision_function(original)
        print(json.dumps({'stage':'E71_fit_TRAIN_two_bases','views':len(labels)}),flush=True)
        arrays,basis=model.fit_basis(original,clip,previous_basis)
        x=model.project(head,original,clip,arrays)
        if not np.array_equal(baseline,model.predict(head,original,clip,arrays)):raise ValueError('zero-init parity failed')
        weights=balanced_parent_weights(labels,sources,parents)
        weights[(labels==0)&(baseline>=model.AI_CUT)]*=2
        for y in (0,1):weights[labels==y]*=.5/weights[labels==y].sum()
        count=[0]
        def check():
            resource_check(deadline);count[0]+=1
            if count[0]%10==0:print(json.dumps({'stage':'E71_constrained_fit','iteration':count[0],
                                               'seconds':round(time.monotonic()-started)}),flush=True)
        arrays['weights'],solver=model.fit(x,logits,labels,weights,check)
        candidate=model.predict(head,original,clip,arrays)
        with CANDIDATE.open('xb') as out:np.savez_compressed(out,**arrays,contract_sha256=np.array(digest(CONTRACT)),
            reference_sha256=np.array(c['inputs']['reference']['sha256']))
        with np.load(CANDIDATE,allow_pickle=False) as a:
            if not np.array_equal(candidate,model.predict(head,original,clip,a)):raise ValueError('serialized replay differs')
    gate=check_train_retention(rows,baseline.reshape(-1,3),candidate.reshape(-1,3),conditions=c['conditions'],ai_cut=model.AI_CUT)
    real=[gate['comparisons'][k]['real'] for k in c['conditions']]
    passed=(solver['success'] and solver['max_constraint_violation']<=1e-8 and gate['passes_train_retention']
            and all(v['non_ai_to_ai']==0 and v['new_ai_rate']<=.10 for v in real))
    result={'state':'E71_TRAIN_guard_passed' if passed else 'E71_TRAIN_guard_failed','dev_scoring_permitted':bool(passed),
        'train_gate':gate,'solver':solver,'basis':basis,'seconds':time.monotonic()-started,
        'candidate_sha256':digest(CANDIDATE),'contract_sha256':digest(CONTRACT),
        'reference_sha256':c['inputs']['reference']['sha256'],'dev_features_or_scores_used_in_fit':0,'previous_dev_score_body_hashed_only':True,
        'e49_rows_read':0,'independent_final_passed':False,'promotion_allowed':False}
    write_once(REPORT,result);write_once(EVIDENCE/'e71_fit.json',result)
    return {'state':result['state'],'solver_success':solver['success'],'new_ai_miss_views':gate['new_ai_miss_views'],
            'real_rates':{k:gate['comparisons'][k]['real'] for k in c['conditions']},'seconds':result['seconds']}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=('freeze','fit'))
    with (ROOT/'fit_execution.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        print(json.dumps({'freeze':freeze,'fit':fit}[parser.parse_args().stage](),indent=2))
