"""Coverage-balanced inner calibration; original E53 outer validation stays fixed."""
import argparse
from collections import defaultdict
import itertools
import json
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from experiments.e51_pipeline import FEATURES,real_safe_threshold,metrics
from experiments.e53_offline import ROOT,EVIDENCE,contract,digest,fixed_write,publisher,VARIANT
from experiments.e53_expansion import NATIVE_FEATURES,load as expansion
from experiments.e53_head_controls import feature_map
from experiments.e53_report import paired_interval,preservation_guard
from pixelproof.training_weights import balanced_parent_weights

OUT=ROOT/'coverage_v2';CONTRACT=OUT/'contract.json'
ARMS={f'{r}_{v}':(w,.01,False,v) for r,w in [('full',3072),('mean',1536)] for v in ('old2','e51_3','order3','expanded3')}
ARMS.update({f'{r}_C001':(w,.001,False,'e51_3') for r,w in [('full',3072),('mean',1536)]})
ARMS.update({f'{r}_L2':(w,.01,True,'e51_3') for r,w in [('full',3072),('mean',1536)]})


def balanced_folds(base):
    groups=defaultdict(list)
    for row in base['rows']:groups[base['components'][row['parent_id']]].append(row)
    counts={g:np.bincount([r['label'] for r in rows],minlength=2) for g,rows in groups.items()}
    validation=[{g for g,rs in groups.items() if f['roles'][rs[0]['parent_id']]=='VALIDATION'} for f in base['folds']]
    options=[]
    for val in validation:
        available=sorted(set(groups)-val);total=sum((counts[g] for g in available),np.zeros(2));choices=[]
        for size in (1,2):
            for cal in itertools.combinations(available,size):
                n=sum((counts[g] for g in cal),np.zeros(2))
                if min(n)>=30 and min(total-n)>=100:
                    choices.append((float(np.sum((n/total-.2)**2)),cal))
        if not choices:raise ValueError('no valid CAL component assignment')
        options.append(sorted(choices))
    best=None
    for proposed in itertools.product(*options):
        trained=set().union(*(set(groups)-validation[i]-set(item[1]) for i,item in enumerate(proposed)))
        if trained!=set(groups):continue
        objective=(sum(item[0] for item in proposed),tuple(item[1] for item in proposed))
        if best is None or objective<best:best=objective
    if best is None:raise ValueError('cannot make every component train without leakage')
    return [{'fold':i,'roles':{r['parent_id']:('VALIDATION' if g in validation[i] else 'CAL' if g in best[1][i] else 'FIT')
                             for g,rs in groups.items() for r in rs}} for i in range(3)]


def freeze():
    if CONTRACT.exists():raise FileExistsError('v2 already frozen')
    base,binding=contract();_,extra,_=expansion();folds=balanced_folds(base)
    report={'state':'coverage_v2_frozen_before_scores','base_contract_sha256':binding,
            'expansion_contract_sha256':digest(ROOT/'expansion_contract.json'),'code_sha256':digest(__file__),
            'helper_sha256':{str(p):digest(p) for p in [Path(__file__).with_name('e53_head_controls.py'),
                Path(__file__).with_name('e53_report.py'),EVIDENCE.parent/'ml/src/pixelproof/training_weights.py']},
            'folds':folds,'arms':ARMS,'outer_validation_changed':False,'each_component_trained_at_least_once':True,
            'new_hyperparameters':False,'model_scores_created':0,'serving_changed':False}
    fixed_write(CONTRACT,json.loads(json.dumps(report)))
    fixed_write(EVIDENCE/'e53_coverage_contract.json',{k:v for k,v in report.items() if k!='folds'}|{'contract_sha256':digest(CONTRACT)})
    return {k:v for k,v in report.items() if k!='folds'}


def load():
    base,binding=contract();_,extra,_=expansion();v2=json.loads(CONTRACT.read_text())
    if v2['base_contract_sha256']!=binding or v2['code_sha256']!=digest(__file__):raise ValueError('v2 contract code/base changed')
    if digest(CONTRACT)!=json.loads((EVIDENCE/'e53_coverage_contract.json').read_text())['contract_sha256']:raise ValueError('v2 contract changed')
    if v2['expansion_contract_sha256']!=digest(ROOT/'expansion_contract.json'):raise ValueError('native contract changed')
    for path,expected in v2['helper_sha256'].items():
        if digest(path)!=expected:raise ValueError('v2 helper changed')
    return base,extra,v2


def fit():
    base,extra,v2=load();binding=digest(CONTRACT)
    for path,expected in [(FEATURES,base['features_sha256']),
        (VARIANT,json.loads((EVIDENCE/'e53_order_features.json').read_text())['feature_sha256']),
        (NATIVE_FEATURES,json.loads((EVIDENCE/'e53_native_features.json').read_text())['feature_sha256'])]:
        if digest(path)!=expected:raise ValueError('feature archive changed')
    with np.load(FEATURES,allow_pickle=False) as a:
        m=a['roles']=='TRAIN';data={k:a[k][m] for k in ('dino','labels','parents','sources','conditions')}
    with np.load(VARIANT,allow_pickle=False) as a:ordered={str(p):v for p,v in zip(a['parents'],a['features'],strict=True)}
    with np.load(NATIVE_FEATURES,allow_pickle=False) as a:
        native=a['features']
        if list(a['record_ids'])!=[r['record_id'] for r in extra['rows']]:raise ValueError('native parent order changed')
    for arm,(width,C,l2,views) in ARMS.items():
        evaluated=feature_map(data['dino'],width,l2)
        for fold in v2['folds']:
            path=OUT/f"{arm}_fold{fold['fold']}.json"
            if path.exists():
                if json.loads(path.read_text())['contract_sha256']!=binding:raise ValueError('v2 result mismatch')
                continue
            start=time.monotonic();roles=np.asarray([fold['roles'][str(p)] for p in data['parents']]);train=roles=='FIT'
            if views=='old2':train&=data['conditions']!='q75'
            if views=='order3':
                train&=data['conditions']=='clean';clean=data['dino'][train];pp=data['parents'][train]
                x=np.stack([np.r_[clean[i][None,:],ordered[str(p)]] for i,p in enumerate(pp)]).reshape(-1,3072)
                y=np.repeat(data['labels'][train],3);ss=np.repeat(data['sources'][train],3);pp=np.repeat(pp,3)
            else:x=data['dino'][train];y=data['labels'][train];ss=data['sources'][train];pp=data['parents'][train]
            original_mass=len(y);added=0
            if views=='expanded3':
                mapping=defaultdict(set)
                for row in base['rows']:mapping[publisher(row['source'])].add(fold['roles'][row['parent_id']])
                if any(len(v)!=1 for v in mapping.values()):raise ValueError('publisher split across roles')
                use=np.asarray([mapping[publisher('e32:'+r['source_id'])]=={'FIT'} for r in extra['rows']]);nr=[r for r,m in zip(extra['rows'],use,strict=True) if m];added=len(nr)
                x=np.r_[x,native[use].reshape(-1,3072)]
                y=np.r_[y,np.repeat([int(r['label']=='ai') for r in nr],3)].astype(int)
                ss=np.r_[ss,np.repeat(['e32:'+r['source_id'] for r in nr],3)];pp=np.r_[pp,np.repeat(['e32:'+r['record_id'] for r in nr],3)]
            x=feature_map(x,width,l2);weights=balanced_parent_weights(y,ss,pp)
            if views=='expanded3':weights*=original_mass/weights.sum()
            head=make_pipeline(StandardScaler(),LogisticRegression(C=C,max_iter=1000,solver='lbfgs',random_state=53))
            ev=np.isin(data['conditions'],['clean','q75']);cal=(roles=='CAL')&ev;val=(roles=='VALIDATION')&ev
            with warnings.catch_warnings(),threadpool_limits(limits=2):
                warnings.simplefilter('error',ConvergenceWarning)
                head.fit(x,y,standardscaler__sample_weight=weights,logisticregression__sample_weight=weights)
                cs=head.predict_proba(evaluated[cal])[:,1];cut=real_safe_threshold(data['labels'][cal],cs,data['sources'][cal],data['conditions'][cal])
                scores=head.predict_proba(evaluated[val])[:,1]
            obs=[{'parent_id':str(p),'label':int(y),'source':str(s),'condition':str(c),'score':float(v),'predicted_ai':bool(v>=cut)} for p,y,s,c,v in zip(data['parents'][val],data['labels'][val],data['sources'][val],data['conditions'][val],scores,strict=True)]
            rates={}
            for c in ('clean','q75'):
                m=data['conditions'][val]==c;rates[c]=metrics(data['labels'][val][m],scores[m],data['sources'][val][m],cut,cut)
            artifact=path.with_suffix('.joblib');artifact.parent.mkdir(parents=True,exist_ok=True)
            joblib.dump({'head':head,'threshold':cut,'width':width,'row_l2':l2,'restriction':'TRAIN-only v2 fold; never serve'},artifact)
            result={'arm':arm,'fold':fold['fold'],'contract_sha256':binding,'threshold':cut,'new_native_fit_parents':added,
                    'rates':rates,'observations':obs,'artifact_sha256':digest(artifact),'seconds':time.monotonic()-start}
            fixed_write(path,result);print(json.dumps({k:v for k,v in result.items() if k not in {'rates','observations'}}),flush=True)
    return {'state':'coverage_v2_twelve_arms_complete','serving_changed':False}


def report():
    base,_,v2=load();binding=digest(CONTRACT);parents=sorted(r['parent_id'] for r in base['rows']);known={r['parent_id']:r for r in base['rows']}
    y=np.asarray([known[p]['label'] for p in parents]);s=np.asarray([known[p]['source'] for p in parents]);g=np.asarray([base['components'][p] for p in parents]);predictions={};arms={};files={}
    for arm in ARMS:
        maps={};folds=[]
        for fold in v2['folds']:
            path=OUT/f"{arm}_fold{fold['fold']}.json";r=json.loads(path.read_text());files[str(path)]=digest(path)
            if r['contract_sha256']!=binding:raise ValueError('mixed v2 contracts')
            for o in r['observations']:
                key=(o['parent_id'],o['condition'])
                if key in maps or fold['roles'][o['parent_id']]!='VALIDATION':raise ValueError('OOF mismatch')
                if o['label']!=known[o['parent_id']]['label'] or o['predicted_ai']!=(o['score']>=r['threshold']):raise ValueError('prediction mismatch')
                maps[key]=o['predicted_ai']
            folds.append({k:v for k,v in r.items() if k!='observations'})
        if set(maps)!={(p,c) for p in parents for c in ('clean','q75')}:raise ValueError('incomplete v2 population')
        predicted=np.asarray([[maps[(p,c)] for c in ('clean','q75')] for p in parents]);predictions[arm]=predicted;rates={}
        for j,c in enumerate(('clean','q75')):
            fp=predicted[y==0,j].mean();tp=predicted[y==1,j].mean()
            rates[c]={'real_false_ai':float(fp),'ai_recall':float(tp),'balanced_accuracy':float((1-fp+tp)/2),
                      'mean_fold_auc':float(np.mean([f['rates'][c]['auc'] for f in folds]))}
        arms[arm]={'conditions':rates,'folds':folds,'comparisons':{}}
    for arm in ARMS:
        for reference in ('full_old2','full_e51_3'):
            ci=paired_interval(y,g,predictions[arm],predictions[reference]);arms[arm]['comparisons'][reference]={
                'intervals':ci,**preservation_guard(y,s,predictions[arm],predictions[reference],ci)}
        arms[arm]['research_guard_passed']=all(v['passed'] for v in arms[arm]['comparisons'].values())
    result={'state':'coverage_v2_TRAIN_report_complete','contract_sha256':binding,'parents':len(parents),
            'arms':arms,'result_bindings':files,'research_guard_survivors':[a for a,v in arms.items() if v['research_guard_passed']],
            'serving_changed':False,'independent_final_passed':False,
            'limitations':['Same consumed TRAIN outer validation as v1; not an independent replication.',
                          'All eleven components train at least once, with no within-fold role mixing.',
                          'Compare algorithms within v2; changed inner calibration prevents a direct v1-model improvement claim.',
                          'Conditional 11-component bootstrap/per-pair multiplicity does not certify a selected universal winner.',
                          'Old2/three-view total-mass confound remains; expansion controls its loss mass.']}
    fixed_write(OUT/'summary.json',result);fixed_write(EVIDENCE/'e53_coverage_result.json',result)
    return {k:v for k,v in result.items() if k not in {'arms','result_bindings'}}


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['freeze','fit','report']);args=parser.parse_args()
    print(json.dumps({'freeze':freeze,'fit':fit,'report':report}[args.phase](),indent=2))
