"""Post-result numerical and error audit; no candidate, external test or tuning."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import warnings

import joblib
import numpy as np
from scipy.special import expit
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from experiments.e51_pipeline import FEATURES
from experiments.e53_coverage import load as coverage, OUT as BASELINE
from experiments.e53_expansion import NATIVE_FEATURES
from experiments.e53_offline import EVIDENCE,digest,fixed_write,publisher
from experiments.e55_color import ROOT,CONTRACT as TRAIN_CONTRACT,load,arrays,mixture_weights
from experiments.e55_resume import inspect_safety
from pixelproof.training_weights import balanced_parent_weights

CONTRACT=ROOT/'audit_contract.json'


def objective_gradient(x,y,weights,coefficient,intercept,C):
    x=np.asarray(x,dtype=np.float64);y=np.asarray(y,dtype=np.float64)
    weights=np.asarray(weights,dtype=np.float64);coef=np.asarray(coefficient,dtype=np.float64).reshape(-1)
    if (x.ndim!=2 or x.shape!=(len(y),len(coef)) or weights.shape!=y.shape
            or not np.isfinite(x).all() or not np.isfinite(weights).all()
            or np.any(weights<=0) or C<=0 or not set(y)<= {0.,1.}):
        raise ValueError('invalid logistic objective inputs')
    mass=weights.sum();z=x@coef+float(intercept)
    loss=float(weights@(np.logaddexp(0,z)-y*z)/mass+coef@coef/(2*C*mass))
    residual=weights*(expit(z)-y)/mass
    gradient=np.r_[x.T@residual+coef/(C*mass),residual.sum()]
    return loss,gradient


def frozen_inputs():
    paths=[TRAIN_CONTRACT,ROOT/'result.json',ROOT/'features.json',
           ROOT.parent/'e54/color_descriptors.json',ROOT.parent/'e54/color_result.json',
           NATIVE_FEATURES,Path(__file__).with_name('e53_coverage.py'),
           Path(__file__).with_name('e55_color.py')]
    for arm in ('duplicate_control','grayscale_20'):
        for fold in range(3):
            paths.extend([ROOT/'training'/f'{arm}_fold{fold}.json',ROOT/'training'/f'{arm}_fold{fold}.joblib'])
    return paths


def freeze():
    load()
    value={'state':'E55_diagnostic_frozen','code_sha256':digest(__file__),
           'inputs':{str(p):digest(p) for p in frozen_inputs()},
           'diagnostic_fit':{'C':.01,'tol':1e-8,'max_iter':1000,'dtype':'float64','seed':53},
           'diagnostic_roles':'FIT only; no new CAL/validation predictions, no saved candidate',
           'old_results_remain_rejected':True,'new_external_test_scores':0}
    fixed_write(CONTRACT,value);fixed_write(EVIDENCE/'e55_audit_contract.json',value|{'contract_sha256':digest(CONTRACT)})
    return value


def run():
    data,_=load();config=json.loads(CONTRACT.read_text())
    if config['code_sha256']!=digest(__file__) or digest(CONTRACT)!=json.loads(
            (EVIDENCE/'e55_audit_contract.json').read_text())['contract_sha256']:
        raise ValueError('diagnostic code/contract changed')
    for path,sha in config['inputs'].items():
        if digest(path)!=sha:raise ValueError('diagnostic input changed')
    base,extra,v2=coverage()
    teacher,gray,parents,labels,sources,conditions,order=arrays(data);del gray
    with np.load(FEATURES,allow_pickle=False) as a:
        use=a['roles']=='TRAIN';original={k:a[k][use] for k in ('dino','parents','labels','sources')}
    with np.load(NATIVE_FEATURES,allow_pickle=False) as a:native=a['features']
    folds=[]
    for fold in v2['folds']:
        if inspect_safety():raise RuntimeError('power/storage safety failed')
        fid=fold['fold'];roles=np.repeat(data['folds'][fid]['roles'],3);ids=order[roles[order]=='FIT']
        old_use=np.asarray([fold['roles'][str(p)]=='FIT' for p in original['parents']])
        mass=int(old_use.sum());mapping=defaultdict(set)
        for row in base['rows']:mapping[publisher(row['source'])].add(fold['roles'][row['parent_id']])
        use=np.asarray([mapping[publisher('e32:'+r['source_id'])]=={'FIT'} for r in extra['rows']])
        nr=[r for r,m in zip(extra['rows'],use,strict=True) if m]
        x=np.r_[original['dino'][old_use],native[use].reshape(-1,3072)]
        y=np.r_[original['labels'][old_use],np.repeat([int(r['label']=='ai') for r in nr],3)].astype(int)
        ss=np.r_[original['sources'][old_use],np.repeat(['e32:'+r['source_id'] for r in nr],3)]
        pp=np.r_[original['parents'][old_use],np.repeat(['e32:'+r['record_id'] for r in nr],3)]
        checks={'features_bitwise_equal':np.array_equal(x,teacher[ids]),'labels_equal':np.array_equal(y,labels[ids]),
                'sources_equal':np.array_equal(ss,sources[ids]),'parents_equal':np.array_equal(pp,parents[ids])}
        if not all(checks.values()):raise ValueError('native/E55 FIT reconstruction mismatch')
        weights=balanced_parent_weights(y,ss,pp);weights*=mass/weights.sum()
        mixture=mixture_weights(y,ss,pp,mass)
        collapsed_error=float(np.max(np.abs(weights-(mixture[:len(y)]+mixture[len(y):]))))
        saved={}
        for name,path in (('native',BASELINE/f'full_expanded3_fold{fid}.joblib'),
                          ('duplicate',ROOT/'training'/f'duplicate_control_fold{fid}.joblib')):
            head=joblib.load(path)['head'];scaler,lr=head.steps[0][1],head.steps[1][1]
            standardized=(x.astype(np.float64)-scaler.mean_)/scaler.scale_
            loss,grad=objective_gradient(standardized,y,weights,lr.coef_,lr.intercept_[0],.01)
            saved[name]={'coefficient_dtype':str(lr.coef_.dtype),'iterations':lr.n_iter_.tolist(),
                         'tol':lr.tol,'objective_on_float64_FIT':loss,'max_gradient_on_float64_FIT':float(np.abs(grad).max()),
                         'head':head}
        sa=saved['native']['head'].steps[0][1];sb=saved['duplicate']['head'].steps[0][1]
        differences={'scaler_mean_max_abs':float(np.abs(sa.mean_-sb.mean_).max()),
                     'scaler_scale_max_abs':float(np.abs(sa.scale_-sb.scale_).max()),
                     'coef_max_abs':float(np.abs(saved['native']['head'][-1].coef_-saved['duplicate']['head'][-1].coef_).max())}
        for value in saved.values():del value['head']
        predictions={};numeric={};x64=x.astype(np.float64)
        with warnings.catch_warnings(),threadpool_limits(limits=2):
            warnings.simplefilter('error',ConvergenceWarning)
            for name in ('original64','duplicate64'):
                xx=x64 if name=='original64' else np.r_[x64,x64]
                yy=y if name=='original64' else np.tile(y,2)
                ww=weights if name=='original64' else mixture
                head=make_pipeline(StandardScaler(),LogisticRegression(C=.01,tol=1e-8,max_iter=1000,solver='lbfgs',random_state=53))
                head.fit(xx,yy,standardscaler__sample_weight=ww,logisticregression__sample_weight=ww)
                predictions[name]=head.predict_proba(x64)[:,1]
                loss,grad=objective_gradient(head[0].transform(x64),y,weights,head[-1].coef_,head[-1].intercept_[0],.01)
                numeric[name]={'iterations':head[-1].n_iter_.tolist(),'FIT_objective':loss,'FIT_max_gradient':float(np.abs(grad).max())}
                del xx,yy,ww,head
        numeric['FIT_score_max_abs_difference']=float(np.max(np.abs(predictions['original64']-predictions['duplicate64'])))
        result={'fold':fid,'input_equality':checks,'weight_collapsed_max_abs_error':collapsed_error,
                'saved_fit_diagnostics':saved,'saved_parameter_differences':differences,'fixed_float64_diagnostic':numeric}
        folds.append(result);print(json.dumps(result),flush=True)
        del x64,x,predictions
    facts=json.loads((ROOT.parent/'e54/color_descriptors.json').read_text())
    mono={r['parent_id']:r['near_monochrome_global_crop'] for r in facts['records']};errors={}
    for arm in ('full_expanded3','duplicate_control','grayscale_20'):
        counts=defaultdict(lambda:{'parents':0,'errors':0})
        for fold in range(3):
            path=(BASELINE if arm=='full_expanded3' else ROOT/'training')/f'{arm}_fold{fold}.json'
            for row in json.loads(path.read_text())['observations']:
                key=f"{row['condition']}:{row['label']}:{'monochrome' if mono[row['parent_id']] else 'other'}"
                counts[key]['parents']+=1;counts[key]['errors']+=int(row['predicted_ai']!=bool(row['label']))
        errors[arm]=dict(counts)
    result={'state':'E55_numerical_and_colour_audit_complete','contract_sha256':digest(CONTRACT),
            'folds':folds,'archived_error_groups':errors,'new_external_scores':0,'candidate_saved':False,
            'serving_changed':False,'old_E55_rejection_unchanged':True,
            'limitations':['FIT-only numerical equivalence is not a quality improvement or a validation result.',
                           'Float64 and tighter tolerance changed together; their individual contribution is not isolated.',
                           'Colour error strata remain confounded with source/content; no causal interpretation.']}
    fixed_write(ROOT/'audit.json',result);fixed_write(EVIDENCE/'e55_audit.json',result)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['freeze','run'])
    args=parser.parse_args();print(json.dumps({'freeze':freeze,'run':run}[args.phase](),indent=2))
