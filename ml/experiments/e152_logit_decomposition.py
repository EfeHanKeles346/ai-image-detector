"""No-refit algebraic diagnosis of the frozen E151 source-pixel comparison."""
import argparse
import fcntl
import json
from pathlib import Path
import socket
import time
import numpy as np
from scipy.special import expit
from threadpoolctl import threadpool_limits
from experiments.e65_acquisition import digest, read, write_once
from experiments.e71_features import save_npz
from experiments.e72_acquisition import resource_check
from experiments import e151_source_pixel_comparison as candidate
from experiments.e147_weighted_representation import map_fields, check_replay
from pixelproof import holdout_linear
from pixelproof.project_paths import DATA_ROOT, ML_ROOT

ROOT=DATA_ROOT/'e152';CONTRACT=ROOT/'contract.json';EVIDENCE=ML_ROOT.parent/'evidence'
BASE=candidate.previous;BRANCHES=candidate.BRANCHES
FIELDS=('old_logit','new_logit','semantic_coefficient_change','added_residual','intercept_change')


def decompose(x,z,old,new):
    x=np.asarray(x,dtype=np.float64);z=np.asarray(z,dtype=np.float64)
    old=np.asarray(old,dtype=np.float64);new=np.asarray(new,dtype=np.float64)
    if x.ndim!=2 or z.ndim!=2 or len(x)!=len(z) or old.shape!=(x.shape[1]+1,) or \
            new.shape!=(x.shape[1]+z.shape[1]+1,) or any(not np.isfinite(a).all() for a in (x,z,old,new)):
        raise ValueError('Finite aligned old semantic/new residual coefficients required')
    before=x@old[:-1]+old[-1]
    after=np.column_stack([x,z])@new[:-1]+new[-1]
    semantic=x@(new[:x.shape[1]]-old[:-1])
    residual=z@new[x.shape[1]:-1]
    intercept=np.full(len(x),new[-1]-old[-1])
    result=np.column_stack([before,after,semantic,residual,intercept])
    if not np.isfinite(result).all() or np.max(np.abs(after-before-result[:,2:].sum(axis=1)))>1e-10:
        raise ValueError('Logit change does not reconstruct')
    return result


def summary(values,labels):
    values=np.asarray(values,dtype=np.float64);labels=np.asarray(labels)
    if labels.ndim!=1 or values.ndim!=2 or values.shape!=(len(labels),5) or not len(labels) or \
            not np.isin(labels,[0,1]).all() or not np.isfinite(values).all():
        raise ValueError('Complete finite class-aligned decomposition required')
    old=values[:,0]>=0;new=values[:,1]>=0;y=labels.astype(bool)
    cohorts={'all':np.ones(len(y),dtype=bool),'new_errors':(old==y)&(new!=y),'rescued_errors':(old!=y)&(new==y)}
    result={}
    for name,take in cohorts.items():
        count=int(take.sum());part=values[take,2:]
        if not count:
            result[name]={'parents':0};continue
        # Harmful is negative for AI, positive for REAL. This is an algebraic
        # attribution under the fixed centered maps, not image-provenance causality.
        harm=part*np.where(y[take],-1.,1.)[:,None]
        maxima=harm.max(axis=1)
        ties=np.isclose(harm,maxima[:,None],rtol=0,atol=1e-12).sum(axis=1)>1
        unique=(~ties)&(maxima>0)
        largest=np.argmax(harm,axis=1)
        result[name]={'parents':count,
            'terms':{field:{'mean':float(part[:,j].mean()),'q05_median_q95':np.quantile(part[:,j],[.05,.5,.95]).tolist(),
                           'harmful_sign_count':int((harm[:,j]>0).sum())} for j,field in enumerate(FIELDS[2:])},
            'largest_harmful_term':{field:int((unique&(largest==j)).sum()) for j,field in enumerate(FIELDS[2:])},
            'largest_harmful_tie':int((ties&(maxima>0)).sum()),'no_harmful_term':int((maxima<=0).sum())}
    return result


def freeze():
    prior=candidate.validate();report=read(candidate.ROOT/'report.json');locked=read(candidate.ROOT/'locked_scores.json')
    if digest(candidate.ROOT/'report.json')!=digest(EVIDENCE/'e151_source_pixel_comparison.json') or \
            digest(candidate.ROOT/'locked_scores.json')!=report['locked_scores_sha256'] or \
            digest(candidate.ROOT/'scores.npz')!=locked['scores_sha256']:
        raise ValueError('Complete bound E151 result required')
    files=[Path(__file__),Path(candidate.__file__),candidate.CONTRACT,candidate.ROOT/'report.json',
           candidate.ROOT/'locked_scores.json',candidate.ROOT/'scores.npz',EVIDENCE/'e151_source_pixel_comparison.json']
    for name,fit in locked['fits'].items():
        fold,branch=name.split('_',1)
        for path,key in [(candidate.ROOT/f'fold{fold}_{branch}.npz','artifact_sha256'),
                         (candidate.ROOT/f'fold{fold}_residual.npz','residual_map_sha256')]:
            if digest(path)!=fit[key]:raise ValueError('Bound saved candidate differs')
            files.append(path)
    c={'state':'E152_no_refit_logit_decomposition_registered','inputs':prior['inputs']|{str(p):digest(p) for p in files},
       'versions':candidate.versions(),'parents':prior['parents'],'conditions':prior['conditions'],'branches':BRANCHES,
       'fields':FIELDS,'max_seconds':1800,'cpu_threads':2,
       'definition':'E151 minus E131 held-out logit = x*(new_semantic_coefficients-old_coefficients) + z*new_residual_coefficients + (new_intercept-old_intercept). x is frozen E131320-dimensional semantic map, z is frozen E15164-dimensional residual map. No fit, ablation candidate, threshold or feature change.',
       'checks':'Replay both locked score archives at1e-10/identical0.5 decisions, require logit decomposition and paired clean-to-each-transport differences within1e-10. Serialize/replay and lock every parent/view/branch decomposition before summaries.',
       'summaries':'All four conditions by class and every source component/class; signed mean/q05/median/q95 of three logit-change terms; harmful sign and unique largest harmful term counts, ties explicit. Report all parents and clearly post-hoc new/rescued errors. Pair transport-minus-clean changes of the three terms for all parents/class and require zero paired intercept shift. No selected candidate.',
       'limits':prior['limits']+' Post-hoc algebraic diagnosis on consumed folds. Contributions depend on these fixed centered/whitened coordinates and fitted coefficients; neither dominant-term counts nor zeroing a block establish physical causality, authenticity or a deployable improvement. Repeated views are not independent parents.',
       'downloads':0,'new_pixels_read':0,'fits':0,'promotion_allowed':False}
    ROOT.mkdir(exist_ok=True);write_once(CONTRACT,c)
    write_once(EVIDENCE/'e152_logit_decomposition_contract.json',{k:v for k,v in c.items() if k!='inputs'}|{'contract_sha256':digest(CONTRACT)})
    return {'parents':c['parents'],'fits':0,'contract_sha256':digest(CONTRACT)}


def validate():
    c=read(CONTRACT)
    if digest(CONTRACT)!=read(EVIDENCE/'e152_logit_decomposition_contract.json')['contract_sha256'] or c['versions']!=candidate.versions():
        raise ValueError('Decomposition contract/runtime differs')
    for path,sha in c['inputs'].items():
        if digest(path)!=sha:raise ValueError('Frozen decomposition input differs')
    return c


def audit():
    c=validate();binding=digest(CONTRACT);start=time.monotonic();deadline=start+c['max_seconds'];resource_check(deadline)
    write_once(ROOT/'started.json',{'contract_sha256':binding})
    prior=read(BASE.CONTRACT);rows=prior['rows'];parents=[r['parent_id'] for r in rows]
    folds=np.asarray([prior['outer_fold'][p] for p in parents]);groups=np.asarray([prior['components'][p] for p in parents])
    labels=np.asarray([r['label'] for r in rows]);old_binding=digest(BASE.CONTRACT);new_binding=digest(candidate.CONTRACT)
    feature=candidate.previous.load_features(rows)
    with np.load(candidate.pixels.ROOT/'features.npz',allow_pickle=False) as saved:
        residual=candidate.residual_coordinates(saved,rows,digest(candidate.pixels.FULL),c['conditions'])
    scores={}
    for name,folder,contract_binding in [('old',BASE.ROOT,old_binding),('new',candidate.ROOT,new_binding)]:
        with np.load(folder/'scores.npz',allow_pickle=False) as a:
            if list(a['parents'])!=parents or list(a['conditions'])!=c['conditions'] or not np.array_equal(a['outer_fold'],folds) or \
                    str(a['contract_sha256'])!=contract_binding or str(a['global_role'])!='TRAIN' or str(a['usage'])!='INTERNAL_HELD_OUT':
                raise ValueError('Locked score identity differs')
            scores[name]={b:a[b].copy() for b in BRANCHES}
    values={b:np.full((len(rows),4,5),np.nan) for b in BRANCHES};checks={}
    with threadpool_limits(limits=2):
        for fold in range(3):
            resource_check(deadline);take=folds==fold;view=np.repeat(take,4)
            with np.load(BASE.ROOT/f'fold{fold}_shared.npz',allow_pickle=False) as saved:
                if str(saved['contract_sha256'])!=old_binding:raise ValueError('Base map binding differs')
                common=np.column_stack([holdout_linear.project(feature[k][view],map_fields(saved,k+'_')) for k in candidate.COMMON])
            with np.load(candidate.ROOT/f'fold{fold}_residual.npz',allow_pickle=False) as saved:
                if str(saved['contract_sha256'])!=new_binding:raise ValueError('Residual map binding differs')
                z=holdout_linear.project(residual[view],map_fields(saved))
            for branch in BRANCHES:
                with np.load(BASE.ROOT/f'fold{fold}_{branch}.npz',allow_pickle=False) as old, np.load(candidate.ROOT/f'fold{fold}_{branch}.npz',allow_pickle=False) as new:
                    if str(old['contract_sha256'])!=old_binding or str(new['contract_sha256'])!=new_binding:
                        raise ValueError('Head binding differs')
                    x=np.column_stack([common,holdout_linear.project(feature[branch][view],map_fields(old))])
                    part=decompose(x,z,old['parameters'],new['parameters']).reshape(-1,4,5)
                checks[f'{fold}_{branch}']={'old_score_max_error':check_replay(scores['old'][branch][take],expit(part[:,:,0])),
                    'new_score_max_error':check_replay(scores['new'][branch][take],expit(part[:,:,1])),
                    'logit_reconstruction_max_error':float(np.abs(part[:,:,1]-part[:,:,0]-part[:,:,2:].sum(axis=-1)).max())}
                delta=part[:,1:,:]-part[:,:1,:]
                paired_error=float(np.abs(delta[:,:,1]-delta[:,:,0]-delta[:,:,2:].sum(axis=-1)).max())
                if paired_error>1e-10 or np.any(delta[:,:,4]!=0):raise ValueError('Paired transport decomposition differs')
                checks[f'{fold}_{branch}']['paired_reconstruction_max_error']=paired_error
                values[branch][take]=part
            print(json.dumps({'E152_fold':fold,'seconds':round(time.monotonic()-start)}),flush=True)
    if any(not np.isfinite(v).all() for v in values.values()):raise ValueError('Complete derived arrays required')
    save_npz(ROOT/'decomposition.npz',**values,parents=np.asarray(parents),conditions=c['conditions'],fields=FIELDS,
             outer_fold=folds,global_role='TRAIN',usage='POST_HOC_INTERNAL_DIAGNOSTIC',contract_sha256=binding)
    with np.load(ROOT/'decomposition.npz',allow_pickle=False) as a:
        if any(not np.array_equal(a[b],values[b]) for b in BRANCHES):raise ValueError('Derived array replay differs')
    write_once(ROOT/'locked_arrays.json',{'contract_sha256':binding,'array_sha256':digest(ROOT/'decomposition.npz'),'checks':checks})
    reports={}
    for branch in BRANCHES:
        reports[branch]={}
        for j,condition in enumerate(c['conditions']):
            parts=values[branch][:,j]
            by_class={str(label):summary(parts[labels==label],labels[labels==label]) for label in (0,1)}
            components=[]
            for group in sorted(set(groups.tolist())):
                take=groups==group
                components.append({'sources':sorted({r['source'] for r,t in zip(rows,take,strict=True) if t}),
                    'by_class':{str(label):summary(parts[take&(labels==label)],labels[take&(labels==label)]) for label in (0,1) if (take&(labels==label)).any()}})
            paired={}
            if j:
                for label in (0,1):
                    delta=parts[labels==label,2:]-values[branch][labels==label,0,2:]
                    paired[str(label)]={'parents':len(delta),'term_change':{field:{'mean':float(delta[:,k].mean()),
                        'q05_median_q95':np.quantile(delta[:,k],[.05,.5,.95]).tolist()} for k,field in enumerate(FIELDS[2:])}}
            reports[branch][condition]={'by_class':by_class,'components':components,'paired_transport_minus_clean':paired}
    resource_check(deadline)
    report={'state':'E152_no_refit_logit_decomposition_complete','contract_sha256':binding,'parents':len(rows),'reports':reports,
        'checks':checks,'locked_arrays_sha256':digest(ROOT/'locked_arrays.json'),'seconds':time.monotonic()-start,
        'fits':0,'downloads':0,'new_pixels_read':0,'promotion_allowed':False,'limits':c['limits']}
    write_once(ROOT/'report.json',report);write_once(EVIDENCE/'e152_logit_decomposition.json',report)
    return {k:v for k,v in report.items() if k not in ('reports','checks')}


if __name__=='__main__':
    def denied(*args,**kwargs):raise RuntimeError('E152 is offline')
    socket.socket.connect=denied;socket.socket.connect_ex=denied;socket.create_connection=denied
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=['freeze','audit'])
    ROOT.mkdir(exist_ok=True)
    with (ROOT/'execution.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        print(json.dumps({'freeze':freeze,'audit':audit}[parser.parse_args().stage](),indent=2))
