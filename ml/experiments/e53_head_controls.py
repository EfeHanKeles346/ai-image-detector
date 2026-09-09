"""Four fixed regularization/feature-normalization controls on E53 TRAIN only."""
import json
import time
import warnings

import joblib
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler,normalize
from threadpoolctl import threadpool_limits

from experiments.e51_pipeline import FEATURES,real_safe_threshold,metrics
from experiments.e53_offline import ROOT,EVIDENCE,contract,digest,fixed_write
from pixelproof.training_weights import balanced_parent_weights
import pixelproof.training_weights as weight_module

ARMS={'full_C001':(3072,.001,False),'mean_C001':(1536,.001,False),
      'full_L2':(3072,.01,True),'mean_L2':(1536,.01,True)}
CONTRACT=ROOT/'head_controls_contract.json'


def feature_map(values,width,l2):
    if values.ndim!=2 or width>values.shape[1] or not np.isfinite(values).all():raise ValueError('invalid feature matrix')
    return normalize(values[:,:width],norm='l2',axis=1) if l2 else values[:,:width]


def run():
    base,binding=contract()
    protocol={'base_contract_sha256':binding,'code_sha256':digest(__file__),'weight_code_sha256':digest(weight_module.__file__),
              'arms':ARMS,'all_parameters_fixed':True,'scope':'same TRAIN, same folds/CAL/validation, no external scores',
              'sample_weight_mass':'same historical three-view mass; fast helper is bitwise-equivalent',
              'serving_changed':False,'model_scores_created':0}
    # JSON roundtrip normalizes tuples before immutable-resume comparisons.
    fixed_write(CONTRACT,json.loads(json.dumps(protocol)))
    control_sha=digest(CONTRACT)
    fixed_write(EVIDENCE/'e53_head_controls_contract.json',json.loads(json.dumps(protocol))|{'contract_sha256':control_sha})
    if digest(FEATURES)!=base['features_sha256']:raise ValueError('features changed')
    with np.load(FEATURES,allow_pickle=False) as a:
        m=a['roles']=='TRAIN';data={k:a[k][m] for k in ('dino','labels','parents','sources','conditions')}
    for arm,(width,C,l2) in ARMS.items():
        x=feature_map(data['dino'],width,l2)
        for fold in base['folds']:
            path=ROOT/f"results/{arm}_fold{fold['fold']}.json"
            if path.exists():
                if json.loads(path.read_text()).get('head_controls_contract_sha256')!=control_sha:raise ValueError('control result changed')
                continue
            started=time.monotonic();roles=np.asarray([fold['roles'][str(p)] for p in data['parents']])
            train=roles=='FIT';evaluate=np.isin(data['conditions'],['clean','q75']);cal=(roles=='CAL')&evaluate;val=(roles=='VALIDATION')&evaluate
            weights=balanced_parent_weights(data['labels'][train],data['sources'][train],data['parents'][train])
            head=make_pipeline(StandardScaler(),LogisticRegression(C=C,max_iter=1000,solver='lbfgs',random_state=53))
            with warnings.catch_warnings(),threadpool_limits(limits=2):
                warnings.simplefilter('error',ConvergenceWarning)
                head.fit(x[train],data['labels'][train],standardscaler__sample_weight=weights,logisticregression__sample_weight=weights)
                cs=head.predict_proba(x[cal])[:,1];cut=real_safe_threshold(data['labels'][cal],cs,data['sources'][cal],data['conditions'][cal])
                scores=head.predict_proba(x[val])[:,1]
            observations=[{'parent_id':str(p),'label':int(y),'source':str(s),'condition':str(c),'score':float(v),'predicted_ai':bool(v>=cut)} for p,y,s,c,v in zip(data['parents'][val],data['labels'][val],data['sources'][val],data['conditions'][val],scores,strict=True)]
            rates={}
            for c in ('clean','q75'):
                m=data['conditions'][val]==c;rates[c]=metrics(data['labels'][val][m],scores[m],data['sources'][val][m],cut,cut)
            artifact=path.with_suffix('.joblib');artifact.parent.mkdir(parents=True,exist_ok=True)
            joblib.dump({'head':head,'threshold':cut,'width':width,'row_l2':l2,'restriction':'TRAIN-only fold model; do not serve'},artifact)
            report={'arm':arm,'fold':fold['fold'],'contract_sha256':binding,'head_controls_contract_sha256':control_sha,
                    'fit_views':int(train.sum()),'cal_views':int(cal.sum()),'validation_views':int(val.sum()),
                    'threshold':cut,'rates':rates,'observations':observations,'artifact_sha256':digest(artifact),'seconds':time.monotonic()-started}
            fixed_write(path,report);print(json.dumps({k:v for k,v in report.items() if k not in {'observations','rates'}}),flush=True)
    return {'state':'four_head_controls_complete','contract_sha256':control_sha,'serving_changed':False}


if __name__=='__main__':print(json.dumps(run(),indent=2))
