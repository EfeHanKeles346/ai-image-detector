"""Read-only component diagnosis of a completed, rejected E83 consumed DEV screen."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import socket
if __name__=='__main__':
    def denied(*a,**kw):raise RuntimeError('E83 diagnostic offline')
    socket.socket.connect=denied;socket.socket.connect_ex=denied;socket.create_connection=denied
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
import joblib
import numpy as np
import torch
from scipy.special import expit
from threadpoolctl import threadpool_limits
from experiments import e83_model as model
from experiments.e83_development import validate as validate_dev,CONTRACT as DEV_CONTRACT,FEATURES,SCORES,REPORT as DEV_REPORT,CANDIDATE,ROOT,EVIDENCE
from experiments.e65_acquisition import digest,read,write_once
CONTRACT=ROOT/'diagnostic_contract.json';REPORT=ROOT/'component_diagnostic.json'
BLOCKS={'original':(0,64),'clip':(64,128),'bilinear':(128,256),'real_residual':(256,320),
        'dear_pca':(320,384),'dear_head':(384,385),'supervised':(385,449),'intercept':(449,450)}


def check_cache(a,rows,binding):
    if str(a['binding'])!=binding or set(a['roles'])!={'DEVELOPMENT'} or len(a['roles'])!=len(rows) or \
            list(a['record_ids'])!=[r['record_id'] for r in rows] or list(a['image_sha256'])!=[r['sha256'] for r in rows]:
        raise ValueError('complete ordered DEVELOPMENT identities required')
    for k,width in [('dino',3072),('clip',1536),('dear',1640)]:
        if a[k].shape!=(len(rows),width) or a[k].dtype!=np.float32 or not np.isfinite(a[k]).all():
            raise ValueError('complete finite frozen encoder features required')


def freeze():
    validate_dev();result=read(DEV_REPORT);scores=read(SCORES)
    if digest(DEV_REPORT)!=digest(EVIDENCE/'e83_development.json') or result['passes_limited_dev_screen'] or \
            digest(SCORES)!=result['scores_sha256'] or digest(CANDIDATE)!=result['candidate_sha256'] or \
            digest(FEATURES)!=scores['features_sha256']:
        raise ValueError('completed verified failed E83 DEV required')
    paths=[Path(__file__),Path(model.__file__),DEV_CONTRACT,DEV_REPORT,SCORES,FEATURES,CANDIDATE]
    c={'state':'E83_read_only_failed_DEV_component_diagnostic_registered','inputs':{str(p):digest(p) for p in paths},
       'views':640,'batch_size':8,'blocks':BLOCKS,'score_replay_max_error':1e-12,
       'analysis':'Reproduce all locked E83 scores from complete saved encoder features in original batches8. '
                  'Require zero decision changes at both cuts before summaries. Report additive logit contributions '
                  'for every source/condition and paired error transition bin, mean and fixed min/25/50/75/max quantiles. '
                  'Do not remove blocks, fit parameters, choose thresholds, select examples or claim causal attribution.',
       'limits':'Consumed E66 observations, dependent SIDD scenes. Coordinate contributions depend on the fitted basis '
                'and correlations; they are descriptive, not a counterfactual ablation or a generalization estimate.',
       'fit_allowed':False,'image_reads':0,'e49_read_allowed':False,'promotion_allowed':False}
    write_once(CONTRACT,c);write_once(EVIDENCE/'e83_diagnostic_contract.json',c|{'contract_sha256':digest(CONTRACT)})
    return {'state':c['state']}


def summary(values):
    return {'count':len(values),'mean':float(np.mean(values)),
            'quantiles_min_25_50_75_max':np.quantile(values,[0,.25,.5,.75,1]).tolist()}


def run():
    previous=validate_dev();c=read(CONTRACT)
    if digest(CONTRACT)!=read(EVIDENCE/'e83_diagnostic_contract.json')['contract_sha256']:raise ValueError('diagnostic changed')
    for p,s in c['inputs'].items():
        if digest(p)!=s:raise ValueError('diagnostic input changed')
    if REPORT.exists():raise FileExistsError('diagnostic already complete')
    rows=read(SCORES)['rows'];expected=np.array([r['score'] for r in rows]);old=np.array([r['reference_score'] for r in rows])
    if len(rows)!=c['views']:raise ValueError('complete consumed DEV required')
    torch.set_num_threads(2);head=joblib.load(previous['reference']['path'])['head']
    actual=[];contributions=[]
    with np.load(FEATURES,allow_pickle=False) as a,np.load(CANDIDATE,allow_pickle=False) as candidate,threadpool_limits(limits=2):
        check_cache(a,rows,digest(DEV_CONTRACT))
        for start in range(0,len(rows),c['batch_size']):
            selected=slice(start,start+c['batch_size'])
            x=model.project(head,a['dino'][selected],a['clip'][selected],a['dear'][selected],candidate)
            weighted=x*candidate['weights'];logits=head.decision_function(a['dino'][selected])
            actual.append(expit(logits+x@candidate['weights']))
            contributions.append(np.column_stack([weighted[:,lo:hi].sum(axis=1) for lo,hi in BLOCKS.values()]))
    actual=np.concatenate(actual);contributions=np.concatenate(contributions)
    error=float(np.max(np.abs(actual-expected)))
    decisions={str(cut):int(np.sum((actual>=cut)!=(expected>=cut))) for cut in [model.REAL_CUT,model.AI_CUT]}
    if error>c['score_replay_max_error'] or any(decisions.values()):raise ValueError('locked runtime scores differ')
    groups={}
    for condition in sorted({r['condition'] for r in rows}):
        groups[condition]={}
        for source in sorted({r['source'] for r in rows if r['condition']==condition}):
            indices=np.array([i for i,r in enumerate(rows) if r['condition']==condition and r['source']==source])
            y={rows[i]['label'] for i in indices}
            if len(y)!=1:raise ValueError('diagnostic source must be class-specific')
            label=y.pop();bins={}
            for before in (False,True):
                for after in (False,True):
                    selected=indices[((old[indices]>=model.AI_CUT)==before)&((expected[indices]>=model.AI_CUT)==after)]
                    if not len(selected):continue
                    bins[f'old_ai_{int(before)}_new_ai_{int(after)}']={'views':len(selected),
                        'total_shift':summary(contributions[selected].sum(axis=1)),
                        'components':{key:summary(contributions[selected,j]) for j,key in enumerate(BLOCKS)}}
            groups[condition][source]={'label':int(label),'views':len(indices),'transitions':bins}
    result={'state':'E83_failed_DEV_component_diagnostic_complete','contract_sha256':digest(CONTRACT),
        'score_replay_max_error':error,'decision_changes_by_cut':decisions,'groups':groups,
        'new_fits':0,'image_reads':0,'e49_rows_read':0,'limits':c['limits']}
    write_once(REPORT,result);write_once(EVIDENCE/'e83_component_diagnostic.json',result)
    return {'state':result['state'],'score_replay_max_error':error}

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=('freeze','run'))
    print(json.dumps({'freeze':freeze,'run':run}[parser.parse_args().stage](),indent=2))
