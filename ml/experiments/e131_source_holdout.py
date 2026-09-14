"""Two fresh convex heads under conservative internal TRAIN source holdout."""
import argparse
from collections import Counter
import fcntl
import json
import os
from pathlib import Path
import socket
import time
import numpy as np
from sklearn.metrics import average_precision_score,roc_auc_score
from threadpoolctl import threadpool_limits
from experiments.e65_acquisition import digest,read,write_once
from experiments.e71_features import save_npz
from experiments.e72_acquisition import resource_check
from experiments.e91_data import load as load_old,CONDITIONS
from pixelproof import source_holdout,holdout_linear
from pixelproof.project_paths import DATA_ROOT,ML_ROOT

ROOT=DATA_ROOT/'e131';EVIDENCE=ML_ROOT.parent/'evidence';CONTRACT=ROOT/'contract.json'
BRANCHES=('center_control','full_frame')


def freeze():
    report=read(DATA_ROOT/'e124/report.json')
    if digest(DATA_ROOT/'e124/report.json')!=digest(EVIDENCE/'e124_fullframe_features.json') or \
            report['state']!='E124_fullframe_TRAIN_cache_complete' or report['parents']!=12525 or \
            digest(DATA_ROOT/'e124/features.npz')!=report['feature_sha256']:
        raise ValueError('Complete bound E124 cache required')
    old_contract=DATA_ROOT/'e53/contract_v1.json'
    if digest(old_contract)!=read(EVIDENCE/'e53_contract.json')['contract_sha256']:
        raise ValueError('Inherited source/near-duplicate components differ')
    rows=read(DATA_ROOT/'e112/contract.json')['rows']
    groups=source_holdout.components(rows,read(old_contract)['components'])
    assignment=source_holdout.outer_folds(rows,groups)
    # E125 records the complete existing raw cache dependency closure. Its learned
    # E103/E43 coefficients are never fed to these fresh heads or PCA transforms.
    previous=read(DATA_ROOT/'e125/contract.json')
    if digest(DATA_ROOT/'e125/contract.json')!=read(EVIDENCE/'e125_context_fit_contract.json')['contract_sha256']:
        raise ValueError('Existing raw-cache dependency closure differs')
    for p,sha in previous['inputs'].items():
        if digest(p)!=sha:raise ValueError('Bound source/cache input differs')
    files=[Path(__file__),Path(source_holdout.__file__),Path(holdout_linear.__file__),old_contract,
           DATA_ROOT/'e124/report.json',DATA_ROOT/'e125/contract.json',EVIDENCE/'e53_contract.json']
    c={'state':'E131_internal_source_holdout_registered','inputs':previous['inputs']|{str(p):digest(p) for p in files},
       'rows':rows,'components':groups,'outer_fold':assignment,'parents':len(rows),'components_count':len(set(groups.values())),
       'fold_counts':[dict(Counter(str(r['label']) for r in rows if assignment[r['parent_id']]==f)) for f in range(3)],
       'branches':BRANCHES,'conditions':CONDITIONS,'max_seconds':7200,
       'grouping':'Preserve E53 known connected components; whole RR topics/classes, shared E36 AI prompt batch, E32 FLUX/Qwen prompts, Nano family, CSAFE, all MIDD sensors and all SID cameras; join declared FLUX/Qwen/OpenAI/Gemini family bridges and known body/pixel/scene identities. Never split a connected component to balance folds.',
       'representation':'Refit every standardizer/PCA only on outer-FIT rows, all4 conditions. Shared DINO3072/pooledCLIP1536/DEAR1640 each64 PCs; center768 versus fullframe768 each128 PCs. Whiten, seed131,power3;320 coordinates plus intercept. No supervised E43/E92/E103 projections or correction coefficients.',
       'objective':'Zero-start convex weighted BCE plus .5*.01*weight_norm_squared, intercept unpenalized. Unit total weight, half/class then equal component/parent/view. L-BFGS-B500,ftol1e-12,gtol1e-7; final gradient<=1e-5 and success required. No search or early held-out selection.',
       'metrics':'Lock all six fold/branch predictions before held-out metrics. Fixed.5 diagnostic cut, per-condition pooled and per-fold AUC/AP/confusion; component macro/worst REAL FPR and AI recall; paired rescued/new errors. No calibration, inherited20-gate or promotion claim.',
       'limits':'Consumed internal TRAIN diagnostic, three AI-bearing components only. Known-family/prompt links are grouped, unknown RR/community-generator or semantic ancestry can remain. Frozen encoder pretraining is not audited by this split. Each held-out fold uses a different fresh head; this is not one deployable candidate or independent final evidence.',
       'downloads':0,'new_pixels_read':0,'external_dev_or_gallery_reads':0,'promotion_allowed':False}
    ROOT.mkdir(exist_ok=True);write_once(CONTRACT,c)
    write_once(EVIDENCE/'e131_source_holdout_contract.json',{k:v for k,v in c.items() if k not in ('inputs','rows','components','outer_fold')}|{'contract_sha256':digest(CONTRACT)})
    return {'parents':len(rows),'components':c['components_count'],'fold_counts':c['fold_counts']}


def validate():
    c=read(CONTRACT)
    if digest(CONTRACT)!=read(EVIDENCE/'e131_source_holdout_contract.json')['contract_sha256']:raise ValueError('Holdout contract differs')
    for p,sha in c['inputs'].items():
        if digest(p)!=sha:raise ValueError('Bound holdout input differs')
    return c


def load_features(rows):
    old,features,_unused_head=load_old()
    new=read(DATA_ROOT/'e100/training_manifest.json')['rows']
    if [r['parent_id'] for r in old+new]!=[r['parent_id'] for r in rows]:raise ValueError('Complete parent order differs')
    with np.load(DATA_ROOT/'e101/midd_features.npz',allow_pickle=False) as a:
        if list(a['parents'])!=[r['parent_id'] for r in new] or str(a['binding'])!=digest(DATA_ROOT/'e101/features_contract.json'):
            raise ValueError('MIDD feature binding differs')
        for key in features:features[key]=np.concatenate([features[key],a[key]]).reshape(len(rows)*4,-1)
    for key,experiment,field in [('center_control','e112','raw'),('full_frame','e124','full')]:
        with np.load(DATA_ROOT/experiment/'features.npz',allow_pickle=False) as a:
            if str(a['binding'])!=digest(DATA_ROOT/experiment/'contract.json') or list(a['parents'])!=[r['parent_id'] for r in rows] or \
                    list(a['roles'])!=['TRAIN']*len(rows) or list(a['conditions'])!=CONDITIONS:raise ValueError('Context identity differs')
            value=a[field]
            if key=='center_control':value=value[:,:,0,:]
            features[key]=value.reshape(len(rows)*4,768).copy()
    if any(v.dtype!=np.float32 or len(v)!=len(rows)*4 or not np.isfinite(v).all() for v in features.values()):
        raise ValueError('Complete finite raw-feature arrays required')
    return features


def metrics(y,score):
    y=np.asarray(y);score=np.asarray(score)
    if y.shape!=score.shape or y.ndim!=1 or not len(y) or not np.isin(y,[0,1]).all() or \
            not np.isfinite(score).all() or np.any((score<0)|(score>1)):raise ValueError('Valid aligned binary scores required')
    pred=score>=.5;positive=y==1;negative=~positive
    tp=int((pred&positive).sum());fp=int((pred&negative).sum());fn=int((~pred&positive).sum());tn=int((~pred&negative).sum())
    both=bool(positive.any() and negative.any())
    return {'parents':len(y),'tp':tp,'fp':fp,'fn':fn,'tn':tn,
            'REAL_FPR':fp/(fp+tn) if fp+tn else None,'AI_recall':tp/(tp+fn) if tp+fn else None,
            'auc':float(roc_auc_score(y,score)) if both else None,'ap':float(average_precision_score(y,score)) if both else None}


def fit():
    c=validate();start=time.monotonic();deadline=start+c['max_seconds'];resource_check(deadline)
    if (ROOT/'started.json').exists():raise FileExistsError('Holdout run already started; review partial evidence before any new execution')
    write_once(ROOT/'started.json',{'contract_sha256':digest(CONTRACT)})
    rows=c['rows'];features=load_features(rows);n=len(rows)
    labels=np.asarray([r['label'] for r in rows]);folds=np.asarray([c['outer_fold'][r['parent_id']] for r in rows])
    scores={b:np.full((n,4),np.nan,dtype=np.float64) for b in BRANCHES};fits={}
    with threadpool_limits(limits=2):
        for fold in range(3):
            parent_fit=folds!=fold;view_fit=np.repeat(parent_fit,4)
            rs=[r for r,take in zip(rows,parent_fit,strict=True) if take]
            groups={r['parent_id']:c['components'][r['parent_id']] for r in rs}
            weights=source_holdout.balanced_weights(rs,groups,4)
            train_labels=np.repeat(labels[parent_fit],4)
            common=[];stored={}
            for name in ('dino','clip','dear'):
                resource_check(deadline)
                transform=holdout_linear.fit_map(features[name][view_fit],64)
                common.append(holdout_linear.project(features[name],transform))
                stored.update({name+'_'+key:value for key,value in transform.items()})
                print(json.dumps({'E131_fold':fold,'PCA_complete':name,'seconds':round(time.monotonic()-start)}),flush=True)
            common=np.column_stack(common)
            save_npz(ROOT/f'fold{fold}_shared.npz',**stored,contract_sha256=digest(CONTRACT))
            for branch in BRANCHES:
                resource_check(deadline)
                transform=holdout_linear.fit_map(features[branch][view_fit],128)
                x=np.column_stack([common,holdout_linear.project(features[branch],transform)])
                parameters,solver=holdout_linear.fit_head(x[view_fit],train_labels,weights,lambda:resource_check(deadline))
                artifact=ROOT/f'fold{fold}_{branch}.npz'
                save_npz(artifact,**transform,parameters=parameters,contract_sha256=digest(CONTRACT))
                prediction=holdout_linear.predict(x[~view_fit],parameters).reshape(-1,4)
                with np.load(artifact,allow_pickle=False) as saved:
                    replay=holdout_linear.predict(np.column_stack([common[~view_fit],holdout_linear.project(features[branch][~view_fit],saved)]),saved['parameters']).reshape(-1,4)
                replay_error=float(np.abs(prediction-replay).max())
                if replay_error>1e-10 or not np.array_equal(prediction>=.5,replay>=.5):
                    raise ValueError('Saved held-out score/cut replay differs')
                scores[branch][~parent_fit]=prediction
                fits[f'{fold}_{branch}']={'solver':solver,'artifact_sha256':digest(artifact),'replay_max_error':replay_error,'fit_parents':int(parent_fit.sum()),'heldout_parents':int((~parent_fit).sum())}
                print(json.dumps({'E131_fold':fold,'branch_fitted':branch,'seconds':round(time.monotonic()-start)}),flush=True)
                del x
    if any(not np.isfinite(value).all() for value in scores.values()):raise ValueError('Incomplete paired OOF predictions')
    save_npz(ROOT/'scores.npz',**scores,parents=np.asarray([r['parent_id'] for r in rows]),global_role='TRAIN',usage='INTERNAL_HELD_OUT',outer_fold=folds,conditions=CONDITIONS,contract_sha256=digest(CONTRACT))
    write_once(ROOT/'locked_scores.json',{'scores_sha256':digest(ROOT/'scores.npz'),'contract_sha256':digest(CONTRACT),'fits':fits})
    reports={};group_names=sorted(set(c['components'].values()));group_array=np.asarray([c['components'][r['parent_id']] for r in rows])
    for branch in BRANCHES:
        reports[branch]={}
        for j,condition in enumerate(CONDITIONS):
            component=[metrics(labels[group_array==g],scores[branch][group_array==g,j])|{'sources':sorted(set(r['source'] for r in rows if c['components'][r['parent_id']]==g))} for g in group_names]
            fp=[r['REAL_FPR'] for r in component if r['REAL_FPR'] is not None];recall=[r['AI_recall'] for r in component if r['AI_recall'] is not None]
            reports[branch][condition]={'pooled_different_fold_models':metrics(labels,scores[branch][:,j]),
                'folds':[metrics(labels[folds==f],scores[branch][folds==f,j]) for f in range(3)],'components':component,
                'REAL_component_macro_FPR':float(np.mean(fp)),'REAL_component_worst_FPR':float(np.max(fp)),
                'AI_component_macro_recall':float(np.mean(recall)),'AI_component_worst_recall':float(np.min(recall))}
    comparison={}
    for j,condition in enumerate(CONDITIONS):
        before=(scores['center_control'][:,j]>=.5)!=labels;after=(scores['full_frame'][:,j]>=.5)!=labels
        comparison[condition]={str(label):{'rescued_errors':int((before&~after&(labels==label)).sum()),'new_errors':int((~before&after&(labels==label)).sum())} for label in (0,1)}
    result={'state':'E131_paired_internal_source_holdout_complete','contract_sha256':digest(CONTRACT),'parents':n,'source_components':c['components_count'],
        'fold_counts':c['fold_counts'],'reports':reports,'paired_changes_fullframe_vs_center':comparison,'seconds':time.monotonic()-start,
        'locked_scores_sha256':digest(ROOT/'locked_scores.json'),'promotion_allowed':False,'new_pixels_read':0,'downloads':0,'limits':c['limits']}
    write_once(ROOT/'report.json',result);write_once(EVIDENCE/'e131_source_holdout.json',result)
    note='\n### E131 internal source-holdout result\n\n'+json.dumps({k:v for k,v in result.items() if k!='reports'},sort_keys=True)+'\n\nFull component/condition metrics: evidence/e131_source_holdout.json. This is consumed TRAIN analysis using separately fitted fold heads, not an independent final or serving candidate.\n'
    for path in (ML_ROOT.parent/'PLAN.md',ML_ROOT.parent/'HISTORY.md',ML_ROOT/'EXPERIMENTS.md',ML_ROOT.parent/'DATASETS.md'):
        with path.open('a') as f:fcntl.flock(f,fcntl.LOCK_EX);f.write(note)
    return {k:v for k,v in result.items() if k!='reports'}


if __name__=='__main__':
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    def denied(*a,**k):raise RuntimeError('Source-holdout experiment is offline')
    socket.socket.connect=denied;socket.socket.connect_ex=denied;socket.create_connection=denied
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=['freeze','fit'])
    ROOT.mkdir(exist_ok=True)
    with (ROOT/'execution.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        print(json.dumps({'freeze':freeze,'fit':fit}[parser.parse_args().stage](),indent=2))
