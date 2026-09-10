"""One preregistered REAL-source supplement versus stable native linear controls."""
import argparse
import json
import os
from pathlib import Path
import time
import warnings

import joblib
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from experiments import e42_features as dino
from experiments.e51_pipeline import metrics, real_safe_threshold
from experiments.e53_offline import EVIDENCE, digest, fixed_write, contract as old_contract
from experiments.e53_coverage import OUT as BASELINE
from experiments.e53_report import paired_interval, preservation_guard
from experiments.e53_adaptation_probe import differentiable_aggregate
from experiments.e54_data import load as old_data, CONTRACT as OLD_DATA, TEACHER, crop_views
from experiments.e54_adapt import CropCache
from experiments.e54_replay import compare_scores
from experiments.e57_data_v2 import ROOT, MANIFEST, CONTRACT as DATA_CONTRACT, load as data_config, safe
from pixelproof.training_weights import balanced_parent_weights

CONTRACT=ROOT/'model_contract.json'
FEATURES=ROOT/'fivek_features.npz'
ARMS=('native64','native64_fivek')
REFERENCES=('full_old2','full_e51_3','full_expanded3')


def weights_with_mass(y,s,p,mass):
    if mass<=0:raise ValueError('invalid original FIT mass')
    w=balanced_parent_weights(y,s,p);w*=mass/w.sum()
    return w


def freeze():
    data_config();old_data();safe(time.monotonic()+3600)
    admission=json.loads((EVIDENCE/'e57_admission.json').read_text())
    if digest(MANIFEST)!=admission['manifest_sha256'] or admission['model_scores']!=0:
        raise ValueError('unbound source admission')
    os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
    model,_,_,weight_sha=dino._load_small();del model
    paths=[MANIFEST,DATA_CONTRACT,OLD_DATA,TEACHER,Path(dino.__file__),
           Path(__file__).with_name('e57_data.py'),Path(__file__).with_name('e57_data_v2.py'),Path(__file__).with_name('e57_raw_probe.py'),Path(__file__).with_name('e54_data.py'),
           Path(__file__).with_name('e54_adapt.py'),Path(__file__).with_name('e53_adaptation_probe.py'),
           Path(__file__).with_name('e51_pipeline.py'),Path(__file__).with_name('e53_report.py'),
           Path(__file__).with_name('e54_replay.py'),EVIDENCE.parent/'ml/src/pixelproof/training_weights.py']
    for arm in REFERENCES:
        for fold in range(3):paths.extend([BASELINE/f'{arm}_fold{fold}.json',BASELINE/f'{arm}_fold{fold}.joblib'])
    value={'state':'E57_model_frozen_before_features_fits','code_sha256':digest(__file__),
        'inputs':{str(p):digest(p) for p in paths},'arms':list(ARMS),'weight_sha256':weight_sha,
        'C':.01,'tol':1e-8,'max_iter':1000,'seed':53,'dtype':'float64','cpu_threads':2,
        'policy':'Admitted new REAL FIT-only in each unchanged fold; all AI replay preserved; unchanged CAL/validation.',
        'mass':'50/50 class/source/parent, total original base FIT view count',
        'external_test_opened':False,'serving_changed':False}
    fixed_write(CONTRACT,value);fixed_write(EVIDENCE/'e57_model_contract.json',value|{'contract_sha256':digest(CONTRACT)})
    return value


def load():
    data_config();data=old_data();value=json.loads(CONTRACT.read_text())
    if digest(__file__)!=value['code_sha256'] or digest(CONTRACT)!=json.loads((EVIDENCE/'e57_model_contract.json').read_text())['contract_sha256']:
        raise ValueError('E57 frozen model protocol changed')
    for p,sha in value['inputs'].items():
        if digest(p)!=sha:raise ValueError('E57 bound input changed: '+p)
    new=json.loads(MANIFEST.read_text())['rows']
    if any(r['label']!=0 or r['role']!='FIT_ONLY_RESEARCH' for r in new):raise ValueError('invalid supplement role/class')
    if {r['parent_id'] for r in new}&{r['parent_id'] for r in data['rows']}:raise ValueError('supplement identity overlaps')
    return data,new,value


def extract():
    os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
    import torch
    data,new,config=load();binding=digest(CONTRACT);deadline=time.monotonic()+3600;safe(deadline)
    model,means,stds,sha=dino._load_small()
    if sha!=config['weight_sha256']:raise ValueError('encoder changed')
    device=torch.device('mps' if torch.backends.mps.is_available() else 'cpu');model=model.to(device).eval()
    for p in model.parameters():p.requires_grad_(False)
    mean=torch.tensor(means,device=device).view(1,3,1,1);std=torch.tensor(stds,device=device).view(1,3,1,1)
    def encode(rgb):
        x=torch.from_numpy(rgb).permute(0,3,1,2).to(device,dtype=torch.float32)/255.
        blocks=model.forward_intermediates((x-mean)/std,indices=list(dino.BLOCKS['small']),return_prefix_tokens=True,norm=True,intermediates_only=True)
        return differentiable_aggregate(torch.stack([b[1][:,0,:] for b in blocks],dim=1)).cpu().numpy()
    cache=CropCache(data);parts=[];chunks={}
    try:
        with torch.inference_mode():
            error=float(np.abs(encode(cache.batch([0,1,2]))-cache.teacher[:3]).max())
            if error>5e-5:raise ValueError('pretrained feature parity failed')
            for i,row in enumerate(new):
                safe(deadline);path=ROOT/'feature_chunks'/f'{i:04d}.npz'
                if digest(row['path'])!=row['sha256']:raise ValueError('admitted pixels changed')
                if path.exists():
                    with np.load(path,allow_pickle=False) as a:
                        if str(a['binding'])!=binding or str(a['parent_id'])!=row['parent_id']:raise ValueError('feature chunk mismatch')
                        values=a['features']
                else:
                    crops=crop_views(Path(row['path']).read_bytes(),row['parent_id']).reshape(-1,224,224,3)
                    values=encode(crops)
                    dino._save_npz(path,{'binding':np.asarray(binding),'parent_id':np.asarray(row['parent_id']),'features':values})
                if values.shape!=(3,3072) or not np.isfinite(values).all():raise ValueError('invalid supplement features')
                parts.append(values);chunks[str(path)]=digest(path)
                if (i+1)%25==0:print(json.dumps({'phase':'E57_features','parents':i+1,'total':len(new)}),flush=True)
    finally:cache.pool.shutdown()
    if FEATURES.exists():raise FileExistsError('feature archive already completed; do not overwrite')
    dino._save_npz(FEATURES,{'features':np.stack(parts),'binding':np.asarray(binding),'parents':np.asarray([r['parent_id'] for r in new])})
    value={'state':'E57_features_complete','contract_sha256':binding,'parents':len(new),'views':3*len(new),
        'feature_sha256':digest(FEATURES),'chunks':chunks,'pretrained_parity_error':error,'new_model_scores':0}
    fixed_write(ROOT/'features.json',value);fixed_write(EVIDENCE/'e57_features.json',{k:v for k,v in value.items() if k!='chunks'}|{'receipt_sha256':digest(ROOT/'features.json')})
    return {k:v for k,v in value.items() if k!='chunks'}


def fit():
    data,new,config=load();binding=digest(CONTRACT);deadline=time.monotonic()+3600
    receipt=json.loads((ROOT/'features.json').read_text())
    if digest(ROOT/'features.json')!=json.loads((EVIDENCE/'e57_features.json').read_text())['receipt_sha256'] or digest(FEATURES)!=receipt['feature_sha256'] or receipt['contract_sha256']!=binding:
        raise ValueError('feature receipt changed')
    with np.load(TEACHER,allow_pickle=False) as a:teacher=a['features'].reshape(-1,3072).astype(np.float64)
    with np.load(FEATURES,allow_pickle=False) as a:
        if str(a['binding'])!=binding or list(a['parents'])!=[r['parent_id'] for r in new]:raise ValueError('new feature order/binding changed')
        added=a['features'].reshape(-1,3072).astype(np.float64)
    rows=data['rows'];parents=np.repeat([r['parent_id'] for r in rows],3);labels=np.repeat([r['label'] for r in rows],3)
    sources=np.repeat([r['source'] for r in rows],3);native=np.repeat([r['native'] for r in rows],3)
    conditions=np.tile(['clean','assigned_transport','q75'],len(rows));np_=np.repeat([r['parent_id'] for r in new],3)
    ns=np.repeat([r['source'] for r in new],3)
    for arm in ARMS:
        for fold in data['folds']:
            safe(deadline);path=ROOT/'training'/f"{arm}_fold{fold['fold']}.json"
            if path.exists():
                previous=json.loads(path.read_text())
                if previous['contract_sha256']!=binding or digest(path.with_suffix('.joblib'))!=previous['artifact_sha256']:raise ValueError('completed fit changed')
                continue
            start=time.monotonic();roles=np.repeat(fold['roles'],3);fit_mask=roles=='FIT'
            x=teacher[fit_mask];y=labels[fit_mask];s=sources[fit_mask];p=parents[fit_mask]
            mass=int((fit_mask&~native).sum());original_ai=int((y==1).sum())
            old_weights=weights_with_mass(y,s,p,mass)
            if arm=='native64_fivek':
                x=np.r_[x,added];y=np.r_[y,np.zeros(len(added),dtype=int)];s=np.r_[s,ns];p=np.r_[p,np_]
            weights=weights_with_mass(y,s,p,mass)
            if int((y==1).sum())!=original_ai or not np.allclose(weights[y==1],old_weights[labels[fit_mask]==1],atol=1e-12,rtol=1e-12):raise ValueError('AI replay/loss mass changed')
            cal=(roles=='CAL')&(conditions!='assigned_transport');val=(roles=='VALIDATION')&(conditions!='assigned_transport')
            head=make_pipeline(StandardScaler(),LogisticRegression(C=config['C'],tol=config['tol'],max_iter=config['max_iter'],random_state=config['seed'],solver='lbfgs'))
            with warnings.catch_warnings(),threadpool_limits(limits=2):
                warnings.simplefilter('error',ConvergenceWarning)
                head.fit(x,y,standardscaler__sample_weight=weights,logisticregression__sample_weight=weights)
                if list(head.classes_)!=[0,1]:raise ValueError('class orientation changed')
                cs=head.predict_proba(teacher[cal])[:,1];cut=real_safe_threshold(labels[cal],cs,sources[cal],conditions[cal])
                score=head.predict_proba(teacher[val])[:,1]
            safe(deadline);artifact=path.with_suffix('.joblib');artifact.parent.mkdir(parents=True,exist_ok=True)
            joblib.dump({'head':head,'threshold':cut,'binding':binding,'arm':arm,'fold':fold['fold'],'restriction':'TRAIN fold only; never serve'},artifact)
            with threadpool_limits(limits=2):again=joblib.load(artifact)['head'].predict_proba(teacher[val])[:,1]
            error,flips=compare_scores(again,score,cut)
            observations=[{'parent_id':str(p),'source':str(s),'label':int(y),'condition':str(c),'score':float(v),'predicted_ai':bool(v>=cut)} for p,s,y,c,v in zip(parents[val],sources[val],labels[val],conditions[val],score,strict=True)]
            value={'state':'E57_fold_complete','contract_sha256':binding,'arm':arm,'fold':fold['fold'],'threshold':cut,
                'observations':observations,'artifact_sha256':digest(artifact),'fit_parents':len(y)//3,
                'AI_views_preserved':original_ai,'original_total_weight_mass':mass,'actual_weight_mass':float(weights.sum()),
                'artifact_replay':{'max_score_error':error,'decision_changes':flips,'passed':error==0 and flips==0},
                'rates':{c:metrics(labels[val][conditions[val]==c],score[conditions[val]==c],sources[val][conditions[val]==c],cut,cut) for c in ('clean','q75')},
                'seconds':time.monotonic()-start,'serving_changed':False}
            fixed_write(path,value);print(json.dumps({k:v for k,v in value.items() if k not in {'observations','rates'}}),flush=True)
    return {'state':'E57_six_fixed_fits_complete'}


def report():
    data,_,_=load();binding=digest(CONTRACT);known={r['parent_id']:r for r in data['rows'] if not r['native']};parents=sorted(known)
    y=np.asarray([known[p]['label'] for p in parents]);s=np.asarray([known[p]['source'] for p in parents])
    original,_=old_contract();g=np.asarray([original['components'][p] for p in parents]);arms={};predictions={};bindings={}
    for arm in (*ARMS,*REFERENCES):
        maps={};folds=[]
        for fold in data['folds']:
            path=(ROOT/'training' if arm in ARMS else BASELINE)/f"{arm}_fold{fold['fold']}.json"
            value=json.loads(path.read_text());bindings[str(path)]=digest(path)
            if arm in ARMS and value['contract_sha256']!=binding:raise ValueError('mixed model contracts')
            if digest(path.with_suffix('.joblib'))!=value['artifact_sha256']:raise ValueError('artifact changed')
            roles={r['parent_id']:role for r,role in zip(data['rows'],fold['roles'],strict=True)}
            for row in value['observations']:
                key=(row['parent_id'],row['condition']);truth=known[row['parent_id']]
                if key in maps or roles[row['parent_id']]!='VALIDATION' or any(row[k]!=truth[k] for k in ('source','label')):raise ValueError('OOF mismatch/leakage')
                if row['predicted_ai']!=(row['score']>=value['threshold']):raise ValueError('cut mismatch')
                maps[key]=row['predicted_ai']
            folds.append({k:v for k,v in value.items() if k!='observations'})
        if set(maps)!={(p,c) for p in parents for c in ('clean','q75')}:raise ValueError('incomplete OOF')
        pred=np.asarray([[maps[(p,c)] for c in ('clean','q75')] for p in parents]);predictions[arm]=pred
        arms[arm]={'folds':folds,'conditions':{c:{'ai_recall':float(pred[y==1,j].mean()),'real_false_ai':float(pred[y==0,j].mean()),
            'balanced_accuracy':float((1-pred[y==0,j].mean()+pred[y==1,j].mean())/2),
            'mean_fold_auc':float(np.mean([f['rates'][c]['auc'] for f in folds]))} for j,c in enumerate(('clean','q75'))}}
    for arm in ARMS:
        refs=REFERENCES if arm=='native64' else (*REFERENCES,'native64');comparisons={}
        for ref in refs:
            ci=paired_interval(y,g,predictions[arm],predictions[ref]);comparisons[ref]={'intervals':ci,**preservation_guard(y,s,predictions[arm],predictions[ref],ci)}
        checks={'all_relative_guards':all(v['passed'] for v in comparisons.values()),
            'all_absolute_fold_gates':all(f['rates'][c]['passed'] for f in arms[arm]['folds'] for c in ('clean','q75')),
            'exact_artifact_replay':all(f['artifact_replay']['passed'] for f in arms[arm]['folds'])}
        arms[arm].update(comparisons=comparisons,checks=checks,eligible_for_next_research_stage=all(checks.values()))
    value={'state':'E57_fixed_REAL_supplement_complete','contract_sha256':binding,'arms':arms,'result_bindings':bindings,
        'eligible':[a for a in ARMS if arms[a]['eligible_for_next_research_stage']], 'serving_changed':False,'independent_final_passed':False,
        'limitations':['Reused TRAIN source-held-out exploration, not an independent final.',
                      'New single-publisher REAL data does not reproduce paired REAL/AI semantic alignment.',
                      'Float64 control and source supplement are separately reported; no test-selected threshold.',
                      'Any pass authorizes only separately frozen next research, not serving.']}
    fixed_write(ROOT/'result.json',value);fixed_write(EVIDENCE/'e57_result.json',value)
    return {k:v for k,v in value.items() if k not in {'arms','result_bindings'}}


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['freeze','extract','fit','report']);args=parser.parse_args()
    print(json.dumps(globals()[args.phase](),indent=2))
