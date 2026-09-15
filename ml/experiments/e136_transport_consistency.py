"""One prespecified Model1 consistency objective on frozen source-excluding maps."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import socket
import time
import numpy as np
from threadpoolctl import threadpool_limits
from experiments.e65_acquisition import digest,read,write_once
from experiments.e71_features import save_npz
from experiments.e72_acquisition import resource_check
from experiments import e131_source_holdout as previous
from pixelproof import holdout_linear,source_holdout,transport_consistency
from pixelproof.project_paths import DATA_ROOT,ML_ROOT

ROOT=DATA_ROOT/'e136';EVIDENCE=ML_ROOT.parent/'evidence';CONTRACT=ROOT/'contract.json'
BRANCHES=previous.BRANCHES


def freeze():
    prior=previous.validate();report=read(previous.ROOT/'report.json');locked=read(previous.ROOT/'locked_scores.json')
    if digest(previous.ROOT/'report.json')!=digest(EVIDENCE/'e131_source_holdout.json') or \
            digest(previous.ROOT/'locked_scores.json')!=report['locked_scores_sha256'] or \
            digest(previous.ROOT/'scores.npz')!=locked['scores_sha256']:
        raise ValueError('Complete bound E131 comparator required')
    files=[Path(__file__),Path(transport_consistency.__file__),Path(previous.__file__),previous.CONTRACT,
        previous.ROOT/'report.json',previous.ROOT/'locked_scores.json',previous.ROOT/'scores.npz',
        EVIDENCE/'e131_source_holdout.json']
    for fold in range(3):
        files.append(previous.ROOT/f'fold{fold}_shared.npz')
        for branch in BRANCHES:
            path=previous.ROOT/f'fold{fold}_{branch}.npz'
            if digest(path)!=locked['fits'][f'{fold}_{branch}']['artifact_sha256']:
                raise ValueError('Frozen map/head artifact differs')
            files.append(path)
    c={'state':'E136_Model1_paired_consistency_registered',
       'inputs':prior['inputs']|{str(p):digest(p) for p in files},'parents':prior['parents'],'branches':BRANCHES,
       'conditions':prior['conditions'],'folds':3,'max_seconds':3600,
       'change':'Only add0.5*0.1 times weighted within-parent four-view logit variance to E131 BCE+.005||w||². Equal parent-view weights, FIT-only covariance; intercept unpenalized. Fixed pilot coefficient, no search.',
       'fitting':'Reuse exact E131 FIT-only maps and source folds; zero-start heads, same L-BFGS500/ftol1e-12/gtol1e-7 and gradient<=1e-5. Both geometries preserved; six candidate fits. Old held-out score replay<=1e-10 before comparison.',
       'evaluation':'Lock all six held-out score populations before metrics; fixed0.5 cutoff, per-condition/fold/component AUC/AI recall/REAL FPR, worst groups, paired new/rescued errors and four-view score ranges. Zero new AI misses and zero new REAL false alerts reported separately from aggregate gains.',
       'downloads':0,'new_pixels_read':0,'external_DEV_or_gallery_reads':0,'promotion_allowed':False,
       'generator_family_holdout_supported':False,'limits':prior['limits']+' Adaptive development after E134, not an E92 serving-model evaluation. Consistency can reduce useful discrimination or create new errors; no automatic candidate promotion.'}
    ROOT.mkdir(exist_ok=True);write_once(CONTRACT,c)
    write_once(EVIDENCE/'e136_transport_consistency_contract.json',{k:v for k,v in c.items() if k!='inputs'}|{'contract_sha256':digest(CONTRACT)})
    return {'parents':c['parents'],'candidate_fits':6,'contract_sha256':digest(CONTRACT)}


def validate():
    c=read(CONTRACT)
    if digest(CONTRACT)!=read(EVIDENCE/'e136_transport_consistency_contract.json')['contract_sha256']:
        raise ValueError('Consistency contract differs')
    for path,sha in c['inputs'].items():
        if digest(path)!=sha:raise ValueError('Frozen consistency input differs')
    return c


def transitions(labels,before,after):
    labels=np.asarray(labels);before=np.asarray(before);after=np.asarray(after)
    if labels.shape!=before.shape or labels.shape!=after.shape or labels.ndim!=1 or not len(labels) or \
            not np.isin(labels,[0,1]).all() or not np.isfinite(before).all() or not np.isfinite(after).all() or \
            np.any((before<0)|(before>1)|(after<0)|(after>1)):
        raise ValueError('Aligned finite paired scores and labels required')
    old=(before>=.5)!=labels;new=(after>=.5)!=labels
    result={str(label):{'new_errors':int((~old&new&(labels==label)).sum()),
                      'rescued_errors':int((old&~new&(labels==label)).sum())} for label in (0,1)}
    result['zero_new_AI_misses']=result['1']['new_errors']==0
    result['zero_new_REAL_false_alerts']=result['0']['new_errors']==0
    return result


def fit():
    c=validate();start=time.monotonic();deadline=start+c['max_seconds'];resource_check(deadline)
    write_once(ROOT/'started.json',{'contract_sha256':digest(CONTRACT)})
    prior=read(previous.CONTRACT);rows=prior['rows'];features=previous.load_features(rows);n=len(rows)
    labels=np.asarray([r['label'] for r in rows]);folds=np.asarray([prior['outer_fold'][r['parent_id']] for r in rows])
    groups=np.asarray([prior['components'][r['parent_id']] for r in rows]);scores={b:np.full((n,4),np.nan) for b in BRANCHES}
    with np.load(previous.ROOT/'scores.npz',allow_pickle=False) as data:
        if str(data['contract_sha256'])!=digest(previous.CONTRACT) or list(data['parents'])!=[r['parent_id'] for r in rows] or \
                list(data['conditions'])!=c['conditions'] or not np.array_equal(data['outer_fold'],folds) or \
                str(data['global_role'])!='TRAIN' or str(data['usage'])!='INTERNAL_HELD_OUT':
            raise ValueError('Old score parent/fold/condition/role identity differs')
        baseline={b:data[b].copy() for b in BRANCHES}
    fits={}
    with threadpool_limits(limits=2):
        for fold in range(3):
            parent_fit=folds!=fold;view_fit=np.repeat(parent_fit,4)
            selected=[r for r,take in zip(rows,parent_fit,strict=True) if take]
            weights=source_holdout.balanced_weights(selected,{r['parent_id']:prior['components'][r['parent_id']] for r in selected},4)
            common=[]
            with np.load(previous.ROOT/f'fold{fold}_shared.npz',allow_pickle=False) as maps:
                if str(maps['contract_sha256'])!=digest(previous.CONTRACT):raise ValueError('Shared map contract differs')
                for name in ('dino','clip','dear'):
                    resource_check(deadline)
                    a={key.removeprefix(name+'_'):maps[key] for key in maps.files if key.startswith(name+'_')}
                    common.append(holdout_linear.project(features[name],a))
            common=np.column_stack(common)
            for branch in BRANCHES:
                resource_check(deadline)
                with np.load(previous.ROOT/f'fold{fold}_{branch}.npz',allow_pickle=False) as a:
                    if str(a['contract_sha256'])!=digest(previous.CONTRACT):raise ValueError('Context map contract differs')
                    x=np.column_stack([common,holdout_linear.project(features[branch],a)])
                    old_parameters=a['parameters'].copy()
                replay=holdout_linear.predict(x[~view_fit],old_parameters).reshape(-1,4)
                error=float(np.abs(replay-baseline[branch][~parent_fit]).max())
                if error>1e-10 or not np.array_equal(replay>=.5,baseline[branch][~parent_fit]>=.5):
                    raise ValueError('Original E131 held-out scores do not replay')
                parameters,solver=transport_consistency.fit_head(x[view_fit],np.repeat(labels[parent_fit],4),weights,lambda:resource_check(deadline))
                artifact=ROOT/f'fold{fold}_{branch}.npz';save_npz(artifact,parameters=parameters,contract_sha256=digest(CONTRACT))
                prediction=holdout_linear.predict(x[~view_fit],parameters).reshape(-1,4)
                with np.load(artifact,allow_pickle=False) as saved:
                    saved_replay=holdout_linear.predict(x[~view_fit],saved['parameters']).reshape(-1,4)
                saved_error=float(np.abs(prediction-saved_replay).max())
                if saved_error>1e-10 or not np.array_equal(prediction>=.5,saved_replay>=.5):raise ValueError('Candidate artifact replay differs')
                scores[branch][~parent_fit]=prediction
                matrix=transport_consistency.covariance(x[view_fit],weights)
                fits[f'{fold}_{branch}']={'solver':solver,'old_score_max_error':error,'saved_score_max_error':saved_error,
                    'baseline_FIT_weighted_view_logit_variance':float(old_parameters[:-1]@matrix@old_parameters[:-1]),
                    'artifact_sha256':digest(artifact),'FIT_parents':int(parent_fit.sum()),'held_parents':int((~parent_fit).sum())}
                print(json.dumps({'E136_fold':fold,'branch':branch,'seconds':round(time.monotonic()-start)}),flush=True)
                del x
    if any(not np.isfinite(v).all() for v in scores.values()):raise ValueError('Incomplete candidate scores')
    save_npz(ROOT/'scores.npz',**scores,parents=np.asarray([r['parent_id'] for r in rows]),outer_fold=folds,
             conditions=c['conditions'],global_role='TRAIN',usage='INTERNAL_HELD_OUT',contract_sha256=digest(CONTRACT))
    write_once(ROOT/'locked_scores.json',{'scores_sha256':digest(ROOT/'scores.npz'),'contract_sha256':digest(CONTRACT),'fits':fits})
    reports={}
    for branch in BRANCHES:
        reports[branch]={}
        for j,condition in enumerate(c['conditions']):
            component=[]
            for group in sorted(set(groups.tolist())):
                take=groups==group
                component.append(previous.metrics(labels[take],scores[branch][take,j])|{
                    'sources':sorted({r['source'] for r,t in zip(rows,take,strict=True) if t}),
                    'paired':transitions(labels[take],baseline[branch][take,j],scores[branch][take,j])})
            fp=[r['REAL_FPR'] for r in component if r['REAL_FPR'] is not None];recall=[r['AI_recall'] for r in component if r['AI_recall'] is not None]
            reports[branch][condition]={'pooled_different_fold_models':previous.metrics(labels,scores[branch][:,j]),
                'folds':[previous.metrics(labels[folds==f],scores[branch][folds==f,j]) for f in range(3)],
                'components':component,'REAL_component_worst_FPR':max(fp),'AI_component_worst_recall':min(recall),
                'paired':transitions(labels,baseline[branch][:,j],scores[branch][:,j])}
    stability={b:{str(label):{'baseline_mean_four_view_score_range':float(np.ptp(baseline[b][labels==label],axis=1).mean()),
        'candidate_mean_four_view_score_range':float(np.ptp(scores[b][labels==label],axis=1).mean())} for label in (0,1)} for b in BRANCHES}
    nonregression={b:all(r['paired']['zero_new_AI_misses'] and r['paired']['zero_new_REAL_false_alerts'] for r in reports[b].values()) for b in BRANCHES}
    report={'state':'E136_Model1_consistency_complete','contract_sha256':digest(CONTRACT),'reports':reports,
        'four_view_score_stability':stability,'passes_internal_individual_nonregression':nonregression,
        'seconds':time.monotonic()-start,'locked_scores_sha256':digest(ROOT/'locked_scores.json'),'parents':n,
        'downloads':0,'new_pixels_read':0,'promotion_allowed':False,'limits':c['limits']}
    write_once(ROOT/'report.json',report);write_once(EVIDENCE/'e136_transport_consistency.json',report)
    note='\n### E136 Model1 consistency result\n\n'+json.dumps({k:v for k,v in report.items() if k!='reports'},sort_keys=True)+'\n\nFull source/condition and individual-transition results: evidence/e136_transport_consistency.json. E92 remains unchanged.\n'
    for path in (ML_ROOT.parent/'HISTORY.md',ML_ROOT/'EXPERIMENTS.md',ML_ROOT.parent/'DATASETS.md'):
        with path.open('a') as f:fcntl.flock(f,fcntl.LOCK_EX);f.write(note)
    return {k:v for k,v in report.items() if k!='reports'}


if __name__=='__main__':
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    def denied(*args,**kwargs):raise RuntimeError('Model1 consistency experiment is offline')
    socket.socket.connect=denied;socket.socket.connect_ex=denied;socket.create_connection=denied
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=['freeze','fit'])
    ROOT.mkdir(exist_ok=True)
    with (ROOT/'execution.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        print(json.dumps({'freeze':freeze,'fit':fit}[parser.parse_args().stage](),indent=2))
