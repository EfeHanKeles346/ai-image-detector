"""Frozen TRAIN constraint-slack audit; no candidate, fit, cutoff or population changes."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import socket
import time
if __name__=='__main__':
    def denied(*a,**kw):raise RuntimeError('E77 slack diagnostic is offline')
    socket.socket.connect=denied;socket.socket.connect_ex=denied;socket.create_connection=denied
    os.environ.update(OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
import joblib
import numpy as np
import torch
from scipy.special import expit,logit
from threadpoolctl import threadpool_limits
from experiments import e77_model as model
from experiments.e77_fit import validate as validate_fit,ROOT,CANDIDATE,CONDITIONS
from experiments.e76_fit import combine_rows,real_slice_gates
from experiments.e64_constrained import GUARD_MARGIN
from experiments.e65_acquisition import digest,read,write_once
from pixelproof.project_paths import DATA_ROOT,ML_ROOT
EVIDENCE=ML_ROOT.parent/'evidence'
CONTRACT=ROOT/'slack_diagnostic_contract.json';REPORT=ROOT/'slack_diagnostic.json'


def quantiles(values):
    return dict(zip(['min','q05','median','q95','max'],np.quantile(values,[0,.05,.5,.95,1]).tolist())) if len(values) else None


def summarize(labels,source,logits,shift):
    if labels.shape!=source.shape or labels.shape!=logits.shape or logits.shape!=shift.shape or \
            not np.isfinite(logits).all() or not np.isfinite(shift).all():raise ValueError('finite aligned slack inputs required')
    cut=logit(model.AI_CUT);ai=labels==1;real=~ai
    old_false=real&(logits>=cut);remaining=real&(logits+shift>=cut)
    protected=real&(logits<cut);slack=np.maximum(cut-logits-GUARD_MARGIN,0)-shift
    result={
        'ai_views':int(ai.sum()),'ai_shift_quantiles':quantiles(shift[ai]),
        'ai_near_active_lte_1e_7':int(np.sum(ai&(shift<=1e-7))),
        'ai_by_source':{},'protected_real_views':int(protected.sum()),
        'protected_real_near_active_lte_1e_7':int(np.sum(protected&(slack<=1e-7))),
        'remaining_false_ai_views':int(remaining.sum()),
        'remaining_false_ai_shift_quantiles':quantiles(shift[remaining]),
        'remaining_false_ai_margin_quantiles':quantiles((logits+shift-cut)[remaining]),
        'remaining_false_ai_positive_shift_gt_1e_7':int(np.sum(remaining&(shift>1e-7))),
        'baseline_false_ai_rescued':int(np.sum(old_false&~remaining)),
        'new_real_errors':int(np.sum(remaining&~old_false)),
        'real_by_source':{}}
    for s in sorted(set(source[ai])):
        mask=ai&(source==s)
        result['ai_by_source'][s]={'views':int(mask.sum()),'near_active':int(np.sum(mask&(shift<=1e-7))),
                                 'shift_quantiles':quantiles(shift[mask])}
    for s in sorted(set(source[real])):
        mask=real&(source==s);fp=remaining&mask
        result['real_by_source'][s]={'views':int(mask.sum()),'remaining_false_ai':int(fp.sum()),
            'remaining_positive_shift':int(np.sum(fp&(shift>1e-7))),
            'remaining_shift_quantiles':quantiles(shift[fp]),
            'remaining_margin_quantiles':quantiles((logits+shift-cut)[fp])}
    return result


def freeze():
    validate_fit();fitted=read(ROOT/'fit.json')
    if fitted['dev_scoring_permitted'] or digest(CANDIDATE)!=fitted['candidate_sha256'] or \
            digest(ROOT/'fit.json')!=digest(EVIDENCE/'e77_fit.json'):raise ValueError('frozen rejected E77 required')
    files=[Path(__file__),CANDIDATE,ROOT/'fit_contract.json',ROOT/'fit.json',Path(model.__file__),
           Path(__file__).with_name('e64_constrained.py'),Path(__file__).with_name('e76_fit.py')]
    c={'state':'E77_TRAIN_slack_diagnostic_registered','inputs':{str(p):digest(p) for p in files},
       'population':'All frozen12141 TRAIN parents, all3 conditions; replay frozen REAL slice metrics exactly first.',
       'analysis':'Describe AI correction shifts and correct-REAL constraint slack by source/condition. '
                  'Near-active means slack<=1e-7, purely descriptive; no margin or constraint change. '
                  'Describe remaining REAL false-AI margin/shift quantiles and source counts.',
       'limits':'Near-active constraints need not have nonzero KKT multipliers. This cannot establish causality, '
                'prove infeasibility or predict external retention. Views within parents are dependent.',
       'new_fit':False,'threshold_changed':False,'dev_final_pixels_or_scores':0,'e79_features_read':0}
    write_once(CONTRACT,c);write_once(EVIDENCE/'e77_slack_contract.json',c|{'contract_sha256':digest(CONTRACT)})
    return {'state':c['state']}


def run():
    validate_fit();c=read(CONTRACT)
    if digest(CONTRACT)!=read(EVIDENCE/'e77_slack_contract.json')['contract_sha256']:raise ValueError('diagnostic contract changed')
    for p,s in c['inputs'].items():
        if digest(p)!=s:raise ValueError('diagnostic input changed')
    if REPORT.exists():raise FileExistsError('slack diagnostic already complete')
    started=time.monotonic();previous=read(DATA_ROOT/'e71/fit_contract.json');p={k:Path(v['path']) for k,v in previous['inputs'].items()}
    old=read(p['manifest'])['rows'];rows=combine_rows(old,read(DATA_ROOT/'e72/training_manifest.json')['rows'])
    with np.load(p['features'],allow_pickle=False) as a:original=a['features']
    with np.load(p['clip'],allow_pickle=False) as a:clip=a['features']
    with np.load(DATA_ROOT/'e75/midd_features.npz',allow_pickle=False) as a:
        original=np.concatenate([original,a['dino']]).reshape(-1,3072);clip=np.concatenate([clip,a['clip']]).reshape(-1,1536)
    torch.set_num_threads(2);head=joblib.load(p['reference'])['head']
    with np.load(CANDIDATE,allow_pickle=False) as a,threadpool_limits(limits=2):
        logits=head.decision_function(original);shift=model.project(head,original,clip,a)@a['weights'];scores=expit(logits+shift)
    if real_slice_gates(rows,scores.reshape(-1,3),len(old))!=read(ROOT/'fit.json')['real_population_gates']:
        raise ValueError('frozen fitted operating point differs')
    labels=np.repeat([r['label'] for r in rows],3);source=np.repeat([r['source'] for r in rows],3)
    reports={condition:summarize(labels[j::3],source[j::3],logits[j::3],shift[j::3]) for j,condition in enumerate(CONDITIONS)}
    result={'state':'E77_TRAIN_slack_diagnostic_complete','contract_sha256':digest(CONTRACT),'reports':reports,
            'seconds':time.monotonic()-started,'new_fit':False,'threshold_changed':False,
            'image_files_read':0,'dev_final_rows_read':0,'limits':c['limits']}
    write_once(REPORT,result);write_once(EVIDENCE/'e77_slack_diagnostic.json',result)
    return {k:{field:v[field] for field in ['ai_near_active_lte_1e_7','protected_real_near_active_lte_1e_7',
        'remaining_false_ai_views','remaining_false_ai_positive_shift_gt_1e_7','baseline_false_ai_rescued','new_real_errors']} for k,v in reports.items()}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=['freeze','run'])
    print(json.dumps({'freeze':freeze,'run':run}[parser.parse_args().stage](),indent=2))
