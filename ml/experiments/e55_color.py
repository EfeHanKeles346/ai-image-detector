"""Fixed class-symmetric grayscale ablation; admitted TRAIN only, never serving."""
import argparse
import json
import os
from pathlib import Path
import time
import warnings

import joblib
import numpy as np
from PIL import Image
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from experiments import e42_features as dino
from experiments.e51_pipeline import FEATURES, metrics, real_safe_threshold
from experiments.e53_adaptation_probe import differentiable_aggregate
from experiments.e53_coverage import OUT as BASELINE
from experiments.e53_offline import EVIDENCE, contract as original_contract, digest, fixed_write
from experiments.e53_report import paired_interval, preservation_guard
from experiments.e54_adapt import CropCache
from experiments.e54_data import ROOT as INPUT_ROOT, CONTRACT as INPUT_CONTRACT, TEACHER, INDEX, load as input_data
from experiments.e54_replay import compare_scores
from pixelproof.training_weights import balanced_parent_weights

ROOT=INPUT_ROOT.parent/'e55'
CONTRACT=ROOT/'contract.json'
GRAY=ROOT/'grayscale_features.npz'
ARMS=('duplicate_control','grayscale_20')


def grayscale_crops(crops):
    if crops.ndim!=4 or crops.shape[1:]!=(224,224,3) or crops.dtype!=np.uint8:
        raise ValueError('expected exact uint8 RGB model-input crops')
    return np.stack([np.asarray(Image.fromarray(c).convert('L').convert('RGB')) for c in crops])


def mixture_weights(labels,sources,parents,total_mass,fraction=.2):
    if not 0<fraction<1 or total_mass<=0:
        raise ValueError('invalid frozen mixture mass')
    weights=balanced_parent_weights(labels,sources,parents)
    weights*=total_mass/weights.sum()
    return np.r_[weights*(1-fraction),weights*fraction]


def freeze():
    input_data()
    if CONTRACT.exists():raise FileExistsError('E55 is already frozen')
    helpers=[Path(dino.__file__),Path(__file__).with_name('e53_adaptation_probe.py'),
             Path(__file__).with_name('e54_adapt.py'),Path(__file__).with_name('e54_replay.py'),
             Path(__file__).with_name('e51_pipeline.py'),Path(__file__).with_name('e53_report.py'),
             EVIDENCE.parent/'ml/src/pixelproof/training_weights.py']
    inputs=[INPUT_CONTRACT,TEACHER,INDEX,FEATURES,EVIDENCE/'e54_crop_cache.json',
            INPUT_ROOT/'color_result.json',INPUT_ROOT/'result.json',EVIDENCE/'e53_coverage_result.json']
    for arm in ('full_old2','full_e51_3','full_expanded3'):
        for fold in range(3):
            inputs.extend([BASELINE/f'{arm}_fold{fold}.json',BASELINE/f'{arm}_fold{fold}.joblib'])
    value={'state':'E55_frozen_before_extraction_and_fit','code_sha256':digest(__file__),
           'inputs':{str(p):digest(p) for p in inputs+helpers},'arms':list(ARMS),
           'grayscale_mass':.2,'C':.01,'max_iter':1000,'seed':53,'chunk_views':48,
           'transform':'Pillow RGB->L->RGB on each exact cached uint8 crop, all three FIT views.',
           'control_score_tolerance':5e-5,'control_max_decision_changes':0,
           'fit_mass':'original base FIT three-view count; per-class/source/parent balanced',
           'selection_scope':'Consumed TRAIN source-held-out exploratory follow-up only.',
           'independent_final_opened':False,'serving_changed':False,'new_download_bytes':0}
    fixed_write(CONTRACT,value)
    fixed_write(EVIDENCE/'e55_contract.json',value|{'contract_sha256':digest(CONTRACT)})
    return value


def load():
    data=input_data();config=json.loads(CONTRACT.read_text())
    if config['code_sha256']!=digest(__file__) or digest(CONTRACT)!=json.loads(
            (EVIDENCE/'e55_contract.json').read_text())['contract_sha256']:
        raise ValueError('E55 contract or code changed')
    for p,sha in config['inputs'].items():
        if digest(p)!=sha:raise ValueError('E55 bound input changed: '+p)
    return data,config


def extract():
    os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
    import torch
    data,config=load();binding=digest(CONTRACT)
    if GRAY.exists():raise FileExistsError('E55 feature archive already complete')
    cache=CropCache(data)
    device=torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    model,means,stds,_=dino._load_small();model=model.to(device).eval()
    for p in model.parameters():p.requires_grad_(False)
    mean=torch.tensor(means,device=device).view(1,3,1,1)
    std=torch.tensor(stds,device=device).view(1,3,1,1)
    def encode(rgb):
        tensor=torch.from_numpy(rgb).permute(0,3,1,2).to(device,dtype=torch.float32)/255.
        blocks=model.forward_intermediates((tensor-mean)/std,indices=list(dino.BLOCKS['small']),
            return_prefix_tokens=True,norm=True,intermediates_only=True)
        return differentiable_aggregate(torch.stack([b[1][:,0,:] for b in blocks],dim=1)).cpu().numpy()
    # All rows here are admitted TRAIN; this is encoder parity, not detector tuning.
    with torch.inference_mode():
        parity_ids=np.asarray([0,1,2,3,4,5])
        error=float(np.max(np.abs(encode(cache.batch(parity_ids))-cache.teacher[parity_ids])))
        if error>5e-5:raise ValueError('frozen teacher feature parity failed')
        chunks=[];receipts={};start_time=time.monotonic();total=len(data['rows'])*3
        for start in range(0,total,config['chunk_views']):
            ids=np.arange(start,min(start+config['chunk_views'],total))
            path=ROOT/'chunks'/f'{start:06d}.npz'
            if path.exists():
                with np.load(path,allow_pickle=False) as a:
                    if str(a['binding'])!=binding or not np.array_equal(a['view_ids'],ids):
                        raise ValueError('grayscale chunk binding/order changed')
                    feature=a['features']
            else:
                parts=[]
                for off in range(0,len(ids),16):
                    parts.append(encode(grayscale_crops(cache.batch(ids[off:off+16]))))
                feature=np.concatenate(parts)
                dino._save_npz(path,{'binding':np.asarray(binding),'view_ids':ids,'features':feature})
            if feature.shape!=(len(ids),3072) or not np.isfinite(feature).all():
                raise ValueError('invalid grayscale features')
            chunks.append(feature);receipts[str(path)]=digest(path)
            if start%(config['chunk_views']*20)==0:
                print(json.dumps({'phase':'E55_grayscale_features','views':start+len(ids),
                                  'total':total,'seconds':round(time.monotonic()-start_time)}),flush=True)
        dino._save_npz(GRAY,{'binding':np.asarray(binding),'features':np.concatenate(chunks)})
    cache.pool.shutdown()
    value={'state':'E55_grayscale_features_complete','contract_sha256':binding,
           'feature_sha256':digest(GRAY),'parents':len(data['rows']),'views':total,
           'chunks':receipts,'pretrained_parity_error':error,'source_download_bytes':0,
           'new_independent_parents':0}
    fixed_write(ROOT/'features.json',value)
    fixed_write(EVIDENCE/'e55_features.json',{k:v for k,v in value.items() if k!='chunks'}|
                {'receipt_sha256':digest(ROOT/'features.json')})
    return {k:v for k,v in value.items() if k!='chunks'}


def arrays(data):
    with np.load(TEACHER,allow_pickle=False) as a:teacher=a['features'].reshape(-1,3072)
    receipt=json.loads((ROOT/'features.json').read_text())
    if digest(ROOT/'features.json')!=json.loads((EVIDENCE/'e55_features.json').read_text())['receipt_sha256']:
        raise ValueError('feature receipt changed')
    if receipt['contract_sha256']!=digest(CONTRACT) or digest(GRAY)!=receipt['feature_sha256']:
        raise ValueError('grayscale feature archive changed')
    with np.load(GRAY,allow_pickle=False) as a:
        if str(a['binding'])!=digest(CONTRACT):raise ValueError('grayscale feature binding changed')
        gray=a['features']
    if gray.shape!=teacher.shape:raise ValueError('incomplete feature population')
    rows=data['rows'];parents=np.repeat([r['parent_id'] for r in rows],3)
    labels=np.repeat([r['label'] for r in rows],3);sources=np.repeat([r['source'] for r in rows],3)
    conditions=np.tile(['clean','assigned_transport','q75'],len(rows))
    # Preserve the archived native baseline's base-array order, then native row order.
    lookup={r['parent_id']:i for i,r in enumerate(rows)}
    with np.load(FEATURES,allow_pickle=False) as a:
        use=a['roles']=='TRAIN';pp=a['parents'][use];cc=a['conditions'][use]
    order=[]
    for p,c in zip(pp,cc,strict=True):
        slot=0 if c=='clean' else 2 if c=='q75' else 1
        if slot==1 and str(c)!=dino.assigned_transport(str(p)):raise ValueError('transport mismatch')
        order.append(lookup[str(p)]*3+slot)
    order.extend(i*3+j for i,r in enumerate(rows) if r['native'] for j in range(3))
    if sorted(order)!=list(range(len(teacher))):raise ValueError('teacher population order mismatch')
    return teacher,gray,parents,labels,sources,conditions,np.asarray(order)


def fit():
    data,config=load();binding=digest(CONTRACT)
    teacher,gray,parents,labels,sources,conditions,order=arrays(data)
    native=np.repeat([r['native'] for r in data['rows']],3)
    for arm in ARMS:
        for fold in data['folds']:
            path=ROOT/'training'/f"{arm}_fold{fold['fold']}.json"
            if path.exists():
                if json.loads(path.read_text())['contract_sha256']!=binding:raise ValueError('result binding changed')
                continue
            started=time.monotonic();roles=np.repeat(fold['roles'],3)
            ids=order[roles[order]=='FIT'];base_mass=int(((roles=='FIT')&~native).sum())
            x=np.r_[teacher[ids],teacher[ids] if arm=='duplicate_control' else gray[ids]]
            y=np.tile(labels[ids],2)
            weights=mixture_weights(labels[ids],sources[ids],parents[ids],base_mass,config['grayscale_mass'])
            head=make_pipeline(StandardScaler(),LogisticRegression(C=config['C'],solver='lbfgs',
                max_iter=config['max_iter'],random_state=config['seed']))
            cal=(roles=='CAL')&(conditions!='assigned_transport')
            val=(roles=='VALIDATION')&(conditions!='assigned_transport')
            with warnings.catch_warnings(),threadpool_limits(limits=2):
                warnings.simplefilter('error',ConvergenceWarning)
                head.fit(x,y,standardscaler__sample_weight=weights,logisticregression__sample_weight=weights)
                if list(head.classes_)!=[0,1]:raise ValueError('class orientation changed')
                cs=head.predict_proba(teacher[cal])[:,1]
                cut=real_safe_threshold(labels[cal],cs,sources[cal],conditions[cal])
                scores=head.predict_proba(teacher[val])[:,1]
            observations=[{'parent_id':str(p),'source':str(s),'label':int(y),'condition':str(c),
                           'score':float(v),'predicted_ai':bool(v>=cut)} for p,s,y,c,v in
                          zip(parents[val],sources[val],labels[val],conditions[val],scores,strict=True)]
            artifact=path.with_suffix('.joblib');artifact.parent.mkdir(parents=True,exist_ok=True)
            joblib.dump({'head':head,'threshold':cut,'binding':binding,'arm':arm,'fold':fold['fold'],
                         'restriction':'Consumed TRAIN-only fold; never serve'},artifact)
            with threadpool_limits(limits=2):
                restored=joblib.load(artifact);again=restored['head'].predict_proba(teacher[val])[:,1]
            replay_error,replay_flips=compare_scores(again,scores,cut)
            control=None
            if arm=='duplicate_control':
                ref=json.loads((BASELINE/f"full_expanded3_fold{fold['fold']}.json").read_text())
                by_key={(o['parent_id'],o['condition']):o for o in ref['observations']}
                expected=np.asarray([by_key[(o['parent_id'],o['condition'])]['score'] for o in observations])
                error=float(np.max(np.abs(scores-expected)))
                flips=sum(o['predicted_ai']!=by_key[(o['parent_id'],o['condition'])]['predicted_ai'] for o in observations)
                control={'max_score_error':error,'decision_changes':flips,
                         'passed':bool(error<=config['control_score_tolerance'] and flips==0)}
            result={'state':'E55_fold_complete','contract_sha256':binding,'arm':arm,'fold':fold['fold'],
                    'threshold':cut,'observations':observations,'artifact_sha256':digest(artifact),
                    'fit_parents':len(ids)//3,'original_total_weight_mass':base_mass,'actual_weight_mass':float(weights.sum()),
                    'control_parity':control,'artifact_replay':{'max_score_error':replay_error,'decision_changes':replay_flips,
                        'passed':bool(replay_error==0 and replay_flips==0)},
                    'rates':{c:metrics(labels[val][conditions[val]==c],scores[conditions[val]==c],
                        sources[val][conditions[val]==c],cut,cut) for c in ('clean','q75')},
                    'seconds':time.monotonic()-started,'serving_changed':False}
            fixed_write(path,result)
            print(json.dumps({k:v for k,v in result.items() if k not in {'observations','rates'}}),flush=True)
    return {'state':'E55_six_fixed_fits_complete'}


def report():
    data,config=load();binding=digest(CONTRACT);base=[r for r in data['rows'] if not r['native']]
    known={r['parent_id']:r for r in base};parents=sorted(known)
    y=np.asarray([known[p]['label'] for p in parents]);s=np.asarray([known[p]['source'] for p in parents])
    original,_=original_contract();g=np.asarray([original['components'][p] for p in parents])
    arms={};predictions={};bindings={}
    for arm in (*ARMS,'full_old2','full_e51_3','full_expanded3'):
        maps={};folds=[]
        for fold in data['folds']:
            path=(ROOT/'training' if arm in ARMS else BASELINE)/f"{arm}_fold{fold['fold']}.json"
            value=json.loads(path.read_text());bindings[str(path)]=digest(path)
            if arm in ARMS and value['contract_sha256']!=binding:raise ValueError('mixed E55 contracts')
            if digest(path.with_suffix('.joblib'))!=value['artifact_sha256']:raise ValueError('saved artifact changed')
            roles={r['parent_id']:v for r,v in zip(data['rows'],fold['roles'],strict=True)}
            for r in value['observations']:
                key=(r['parent_id'],r['condition']);truth=known[r['parent_id']]
                if key in maps or roles[r['parent_id']]!='VALIDATION':raise ValueError('OOF overlap/leakage')
                if r['label']!=truth['label'] or r['source']!=truth['source']:raise ValueError('label/source mismatch')
                if r['predicted_ai']!=(r['score']>=value['threshold']):raise ValueError('prediction/cut mismatch')
                maps[key]=r['predicted_ai']
            folds.append({k:v for k,v in value.items() if k!='observations'})
        if set(maps)!={(p,c) for p in parents for c in ('clean','q75')}:raise ValueError('incomplete OOF')
        pred=np.asarray([[maps[(p,c)] for c in ('clean','q75')] for p in parents]);predictions[arm]=pred
        rates={c:{'ai_recall':float(pred[y==1,j].mean()),'real_false_ai':float(pred[y==0,j].mean()),
                  'balanced_accuracy':float((1-pred[y==0,j].mean()+pred[y==1,j].mean())/2),
                  'mean_fold_auc':float(np.mean([f['rates'][c]['auc'] for f in folds]))} for j,c in enumerate(('clean','q75'))}
        arms[arm]={'folds':folds,'conditions':rates}
    control_ok=all(f['control_parity']['passed'] for f in arms['duplicate_control']['folds'])
    for arm in ARMS:
        comparisons={}
        for ref in ('full_old2','full_e51_3','full_expanded3'):
            ci=paired_interval(y,g,predictions[arm],predictions[ref])
            comparisons[ref]={'intervals':ci,**preservation_guard(y,s,predictions[arm],predictions[ref],ci)}
        checks={'all_relative_guards':all(v['passed'] for v in comparisons.values()),
                'all_absolute_fold_gates':all(f['rates'][c]['passed'] for f in arms[arm]['folds'] for c in ('clean','q75')),
                'duplicate_control_parity':control_ok,
                'all_artifacts_replayed':all(f['artifact_replay']['passed'] for f in arms[arm]['folds'])}
        arms[arm].update(comparisons=comparisons,checks=checks,eligible_for_next_research_stage=all(checks.values()))
    ci=paired_interval(y,g,predictions['grayscale_20'],predictions['duplicate_control'])
    arms['grayscale_20']['control_comparison']={'intervals':ci,**preservation_guard(
        y,s,predictions['grayscale_20'],predictions['duplicate_control'],ci)}
    value={'state':'E55_fixed_colour_ablation_complete','contract_sha256':binding,'arms':arms,
           'result_bindings':bindings,'eligible':[a for a in ARMS if arms[a]['eligible_for_next_research_stage']],
           'serving_changed':False,'independent_final_passed':False,
           'limits':['Exploratory follow-up on reused TRAIN validation; not a new final.',
                     'Changed FIT colour processing is class-identical; source/content confounds still exist.',
                     'No hyperparameter search or threshold tuning on outer validation.',
                     'Passing only permits separately frozen full-data/CAL and DEV research, not direct serving.']}
    fixed_write(ROOT/'result.json',value);fixed_write(EVIDENCE/'e55_result.json',value)
    return {k:v for k,v in value.items() if k not in {'arms','result_bindings'}}


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['freeze','extract','fit','report'])
    args=parser.parse_args();print(json.dumps(globals()[args.phase](),indent=2))
