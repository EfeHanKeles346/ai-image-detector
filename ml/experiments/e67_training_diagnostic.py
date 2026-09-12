"""Describe locked E67 TRAIN source/condition losses without fitting another candidate."""
import json
from pathlib import Path
import joblib
import numpy as np
from scipy.special import logit
from threadpoolctl import threadpool_limits
from experiments.e65_acquisition import digest,read,write_once
from experiments.e67_fit import validate,CANDIDATE,REPORT,ROOT,EVIDENCE
from experiments import e67_model as model
from pixelproof.training_weights import balanced_parent_weights


def run():
    c=validate();result=read(REPORT)
    if digest(REPORT)!=digest(EVIDENCE/'e67_fit.json') or digest(CANDIDATE)!=result['candidate_sha256']:
        raise ValueError('fitted artifact identity differs')
    rows=read(c['inputs']['manifest']['path'])['rows']
    if any(r['role'].upper()!='TRAIN' for r in rows):raise ValueError('TRAIN only')
    with np.load(c['inputs']['features']['path'],allow_pickle=False) as a:original=a['features'].reshape(-1,3072)
    with np.load(c['inputs']['blur']['path'],allow_pickle=False) as a:blurred=a['features'].reshape(-1,3072)
    with np.load(CANDIDATE,allow_pickle=False) as a:arrays={k:a[k] for k in a.files}
    y=np.repeat([r['label'] for r in rows],3);sources=np.repeat([r['source'] for r in rows],3)
    parents=np.repeat([r['parent_id'] for r in rows],3);conditions=np.tile(c['conditions'],len(rows))
    head=joblib.load(c['inputs']['reference']['path'])['head']
    with threadpool_limits(limits=2):
        baseline=head.decision_function(original);x=model.project(head,original,blurred,arrays)
        z=baseline+x@arrays['weights'];old=head.predict_proba(original)[:,1]
        weights=balanced_parent_weights(y,sources,parents);weights[(y==0)&(old>=model.AI_CUT)]*=2
        for label in (0,1):weights[y==label]*=.5/weights[y==label].sum()
        before=baseline-logit(model.AI_CUT);after=z-logit(model.AI_CUT)
        losses0=np.logaddexp(0,before)-y*before;losses1=np.logaddexp(0,after)-y*after
        groups=[]
        for label,source,cond in sorted(set(zip(y.tolist(),sources.tolist(),conditions.tolist()))):
            use=(y==label)&(sources==source)&(conditions==cond)
            groups.append({'label':label,'source':source,'condition':cond,'views':int(use.sum()),
                'old_mean_cut_bce':float(losses0[use].mean()),'new_mean_cut_bce':float(losses1[use].mean()),
                'E67_loss_mass':float(weights[use].sum()),'mean_logit_shift':float((z-baseline)[use].mean()),
                'mean_original_branch_shift':float((x[:,:64]@arrays['weights'][:64])[use].mean()),
                'mean_response_branch_shift':float((x[:,64:128]@arrays['weights'][64:128])[use].mean())})
    report={'state':'E67_TRAIN_only_source_loss_diagnostic','candidate_sha256':digest(CANDIDATE),
        'fit_report_sha256':digest(REPORT),'code_sha256':digest(__file__),'groups':groups,
        'dev_or_test_rows_read':0,'new_fit_count':0,'new_candidates_scored':0,
        'limitation':'TRAIN descriptions diagnose the fitted objective, not causal attribution or external generalization.'}
    write_once(ROOT/'training_diagnostic.json',report);write_once(EVIDENCE/'e67_training_diagnostic.json',report)
    for label in (0,1):
        print(json.dumps({'label':label,'largest_remaining_group_losses':sorted([g for g in groups if g['label']==label],
            key=lambda g:-g['new_mean_cut_bce'])[:4]},indent=2))


if __name__=='__main__':run()
