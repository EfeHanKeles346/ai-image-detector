"""Retrospective descriptive audit of locked E131 scores; no model or cutoff fit."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import socket
import numpy as np
from experiments.e65_acquisition import digest, read, write_once
from pixelproof.project_paths import DATA_ROOT, ML_ROOT

ROOT=DATA_ROOT/'e134';EVIDENCE=ML_ROOT.parent/'evidence';CONTRACT=ROOT/'contract.json'
QUANTILES=[0,.1,.25,.5,.75,.9,1]


def vector(value):
    score=np.asarray(value,dtype=np.float64)
    if score.ndim!=1 or not len(score) or not np.isfinite(score).all() or np.any((score<0)|(score>1)):
        raise ValueError('Nonempty finite score vector in[0,1] required')
    return score


def distribution(value):
    score=vector(value)
    return {'parents':len(score),'mean':float(score.mean()),'quantiles':np.quantile(score,QUANTILES).tolist(),
            'alerts_at_fixed_half':int((score>=.5).sum()),'alert_fraction':float((score>=.5).mean())}


def paired(before,after,label):
    before=vector(before);after=vector(after)
    if before.shape!=after.shape or type(label) is not int or label not in (0,1):
        raise ValueError('Aligned paired scores and exact binary label required')
    old=(before>=.5)!=label;new=(after>=.5)!=label
    delta=after-before
    return {'mean_score_shift':float(delta.mean()),'score_shift_quantiles':np.quantile(delta,QUANTILES).tolist(),
            'new_errors':int((~old&new).sum()),'rescued_errors':int((old&~new).sum()),
            'to_alert':int(((before<.5)&(after>=.5)).sum()),'to_no_alert':int(((before>=.5)&(after<.5)).sum())}


def freeze():
    base=DATA_ROOT/'e131';report=read(base/'report.json');locked=read(base/'locked_scores.json')
    if digest(base/'report.json')!=digest(EVIDENCE/'e131_source_holdout.json') or \
            digest(base/'contract.json')!=read(EVIDENCE/'e131_source_holdout_contract.json')['contract_sha256'] or \
            digest(base/'locked_scores.json')!=report['locked_scores_sha256'] or \
            digest(base/'scores.npz')!=locked['scores_sha256']:
        raise ValueError('Complete locked E131 evidence required')
    files=[Path(__file__),Path(digest.__code__.co_filename),base/'report.json',base/'locked_scores.json',
           base/'scores.npz',base/'contract.json',EVIDENCE/'e131_source_holdout.json',EVIDENCE/'e131_source_holdout_contract.json']
    c={'state':'E134_locked_source_score_audit_registered','inputs':{str(p):digest(p) for p in files},
       'quantiles':QUANTILES,'cut':.5,'parents':12525,'branches':['center_control','full_frame'],
       'scope':'All declared source-component/classes and four matched processing conditions from E131. Raw score quantiles, mean, fixed-cut alert fraction and paired clean-to-processed transitions.',
       'downloads':0,'new_pixels_read':0,'gpu_operations':0,'new_fits':0,'promotion_allowed':False,
       'limits':'Retrospective consumed TRAIN/INTERNAL_HELD_OUT analysis, not a new predictive test or deployed-E92 evaluation. Different folds use different heads. Unknown generator/prompt/scene ancestry remains; no calibration, threshold selection or causal feature explanation.'}
    ROOT.mkdir(exist_ok=True);write_once(CONTRACT,c)
    write_once(EVIDENCE/'e134_source_scores_contract.json',{k:v for k,v in c.items() if k!='inputs'}|{'contract_sha256':digest(CONTRACT)})
    return {'contract_sha256':digest(CONTRACT)}


def validate():
    c=read(CONTRACT)
    if digest(CONTRACT)!=read(EVIDENCE/'e134_source_scores_contract.json')['contract_sha256']:
        raise ValueError('Score-audit contract differs')
    for path,sha in c['inputs'].items():
        if digest(path)!=sha:raise ValueError('Bound score-audit input differs')
    return c


def audit():
    c=validate();prior=read(DATA_ROOT/'e131/contract.json');rows=prior['rows'];conditions=prior['conditions']
    labels=np.asarray([r['label'] for r in rows]);groups=np.asarray([prior['components'][r['parent_id']] for r in rows])
    folds=np.asarray([prior['outer_fold'][r['parent_id']] for r in rows]);result=[]
    with np.load(DATA_ROOT/'e131/scores.npz',allow_pickle=False) as data:
        if str(data['contract_sha256'])!=digest(DATA_ROOT/'e131/contract.json') or \
                list(data['parents'])!=[r['parent_id'] for r in rows] or len(rows)!=c['parents'] or \
                list(data['conditions'])!=conditions or not np.array_equal(data['outer_fold'],folds) or \
                str(data['global_role'])!='TRAIN' or str(data['usage'])!='INTERNAL_HELD_OUT':
            raise ValueError('Complete parent/fold/condition/role identity differs')
        clean=conditions.index('clean')
        for branch in c['branches']:
            values=data[branch]
            if values.shape!=(len(rows),len(conditions)):raise ValueError('Complete score matrix required')
            vector(values.ravel())
            for group in sorted(set(groups.tolist())):
                if len(set(folds[groups==group].tolist()))!=1:raise ValueError('Component crosses held-out folds')
                for label in (0,1):
                    selected=(groups==group)&(labels==label)
                    if not selected.any():continue
                    sources=sorted({r['source'] for r,take in zip(rows,selected,strict=True) if take})
                    for j,condition in enumerate(conditions):
                        result.append({'branch':branch,'component':group,'sources':sources,'label':label,
                            'held_out_fold':int(folds[selected][0]),'condition':condition,
                            'distribution':distribution(values[selected,j]),
                            'change_from_clean':paired(values[selected,clean],values[selected,j],label)})
    report={'state':'E134_locked_source_score_audit_complete','contract_sha256':digest(CONTRACT),
            'quantile_levels':QUANTILES,'rows':result,'parents':len(rows),'component_count':len(set(groups.tolist())),
            'downloads':0,'new_pixels_read':0,'gpu_operations':0,'new_fits':0,'promotion_allowed':False,'limits':c['limits']}
    write_once(ROOT/'report.json',report);write_once(EVIDENCE/'e134_source_scores.json',report)
    note='\n### E134 locked source-score audit complete\n\nAll'+str(len(result))+' component/class/condition/branch summaries are retained in evidence/e134_source_scores.json. No fits, new pixels, threshold changes or serving change. '+c['limits']+'\n'
    for path in (ML_ROOT.parent/'HISTORY.md',ML_ROOT/'EXPERIMENTS.md'):
        with path.open('a') as f:fcntl.flock(f,fcntl.LOCK_EX);f.write(note)
    return {k:v for k,v in report.items() if k!='rows'}


if __name__=='__main__':
    def denied(*args,**kwargs):raise RuntimeError('Locked-score audit is offline')
    socket.socket.connect=denied;socket.socket.connect_ex=denied;socket.create_connection=denied
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=['freeze','audit'])
    ROOT.mkdir(exist_ok=True)
    with (ROOT/'execution.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        print(json.dumps({'freeze':freeze,'audit':audit}[parser.parse_args().stage](),indent=2))
