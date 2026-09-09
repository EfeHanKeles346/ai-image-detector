"""Bounded offline E53 TRAIN-only transport/representation ablation.

No model downloads, test scoring, serving writes, or historical head warm starts.
Contracts and outputs are immutable; interrupted feature chunks resume by input/code binding.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
from io import BytesIO
import itertools
import json
from pathlib import Path
import sqlite3
import time
import warnings

import joblib
import numpy as np
from PIL import Image, ImageOps
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from experiments import e42_features as dino
from experiments.e43_train import parent_source_weights
from experiments.e51_pipeline import admission, FEATURES, real_safe_threshold, metrics
from experiments.e51_prefit_audit import VERSION, cross_role_matches
from experiments.e51_protected_pixels import DATABASE
from experiments.e51_train_cal_realize import _write_atomic
from pixelproof.project_paths import DATA_ROOT, ML_ROOT

ROOT = DATA_ROOT / 'e53'
CONTRACT = ROOT / 'contract_v1.json'
EVIDENCE = ML_ROOT.parent / 'evidence'
ARMS = {'full_old2': ('full', 'old2'), 'full_e51_3': ('full', 'e51_3'),
        'mean_old2': ('mean', 'old2'), 'mean_e51_3': ('mean', 'e51_3'),
        'full_order3': ('full', 'order3'), 'mean_order3': ('mean', 'order3')}
VARIANT = ROOT / 'features/order_v1.npz'


def digest(path):
    return dino._sha256_file(Path(path))


def fixed_write(path, value):
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError(f'immutable output differs: {path}')
        return
    _write_atomic(path, value)


def publisher(source):
    if source.startswith('rr:'): return 'rr_publisher_all_topics'
    if source.startswith('e36:'):
        return 'e36_real_publisher' if source.startswith('e36:device_') else 'e36_ai_shared_prompt_batch'
    if source.startswith('e32:csafe-'): return 'csafe_publisher'
    if source in {'e32:flux2-klein-9b', 'e32:qwen-image-2512'}: return 'e32_flux_qwen_shared_prompts'
    if source.startswith('e32:nano-banana'): return 'e32_nano_family'
    return source


def assign_folds(rows, matches):
    roots = {r['parent_id']: r['parent_id'] for r in rows}
    def find(x):
        while roots[x] != x:
            roots[x] = roots[roots[x]]; x = roots[x]
        return x
    def union(a, b):
        a, b = find(a), find(b)
        if a != b: roots[max(a,b)] = min(a,b)
    sources = {}
    for row in rows:
        name = publisher(row['source']); parent = row['parent_id']
        if name in sources: union(parent, sources[name])
        sources[name] = parent
    for match in matches: union(match['train_parent'], match['cal_parent'])
    groups = defaultdict(list)
    for row in rows: groups[find(row['parent_id'])].append(row)
    total = np.bincount([r['label'] for r in rows], minlength=2)
    bins = np.zeros((3,2)); folds = {}
    counts = {g: np.bincount([r['label'] for r in rs], minlength=2) for g,rs in groups.items()}
    for group in sorted(groups, key=lambda g: (-float(np.max(counts[g]/total)), g)):
        def cost(fold):
            candidate = bins.copy(); candidate[fold] += counts[group]
            return float(np.sum((candidate/total)**2)), fold
        fold = min(range(3), key=cost); bins[fold] += counts[group]; folds[group] = fold
    if np.any(bins == 0): raise ValueError('source components cannot populate three outer folds')
    result = []
    for fold in range(3):
        remaining = sorted(g for g in groups if folds[g] != fold)
        options = []
        remainder_total = sum((counts[g] for g in remaining), np.zeros(2))
        for size in (1,2):
            for selected in itertools.combinations(remaining,size):
                cal = sum((counts[g] for g in selected), np.zeros(2))
                fit = remainder_total-cal
                if min(cal) < 30 or min(fit) < 100: continue
                options.append((float(np.sum((cal/remainder_total-.2)**2)), selected))
        if not options: raise ValueError('no disjoint inner CAL with both labels')
        selected = set(min(options)[1])
        roles = {r['parent_id']: ('VALIDATION' if folds[g]==fold else 'CAL' if g in selected else 'FIT')
                 for g,rs in groups.items() for r in rs}
        result.append({'fold':fold, 'roles':roles,
                       'counts':dict(Counter(f"{roles[r['parent_id']]}:{r['label']}" for r in rows))})
    return {r['parent_id']:find(r['parent_id']) for r in rows}, result


def freeze():
    if CONTRACT.exists(): raise FileExistsError('E53 contract already frozen')
    admitted, sha = admission(); rows = admitted['train']
    feature_receipt = json.loads((EVIDENCE/'e51_features.json').read_text())
    if feature_receipt['admission_sha256'] != sha or digest(FEATURES) != feature_receipt['feature_sha256']:
        raise ValueError('E51 feature/admission mismatch')
    with sqlite3.connect(f'file:{DATABASE}?mode=ro', uri=True) as db:
        cache = {s:json.loads(v) for s,v in db.execute('SELECT sha,facts FROM fingerprints WHERE version=?',(VERSION,))}
    def checked(row):
        if digest(row['path']) != row['sha256']: raise ValueError('TRAIN bytes changed')
        return {**row, **cache[row['sha256']], 'condition':'original'}
    with ThreadPoolExecutor(max_workers=4) as pool:
        train = list(pool.map(checked, rows))
    cal = [{**r, **cache[r['sha256']]} for r in admitted['cal']]
    dev_path = DATA_ROOT/'e51/development/manifest_unscored.json'
    dev_sha = json.loads((EVIDENCE/'e51_development_manifest.json').read_text())['manifest_sha256']
    if digest(dev_path) != dev_sha: raise ValueError('latest DEV manifest changed')
    dev = json.loads(dev_path.read_text())['rows']
    if cross_role_matches(train,cal) or cross_role_matches(train,dev):
        raise ValueError('E53 TRAIN overlaps protected E51 CAL/DEV')
    matches = [m for m in cross_role_matches(train,train) if m['train_parent']<m['cal_parent']]
    components, folds = assign_folds(train, matches)
    report = {'schema_version':1, 'state':'E53_frozen_before_scores', 'model_scores_created':0,
              'admission_sha256':sha, 'features_sha256':feature_receipt['feature_sha256'],
              'dev_manifest_sha256':dev_sha, 'code_sha256':digest(__file__),
              'dependencies_sha256':{str(p):digest(p) for p in [Path(dino.__file__),
                  ML_ROOT/'experiments/e51_pipeline.py',ML_ROOT/'experiments/e43_train.py']},
              'train_parents':len(train), 'rows':rows, 'components':components,'folds':folds,
              'internal_pair_count':len(matches), 'internal_matches':matches,
              'arms':ARMS,'C':.01,'seed':53,'label_mapping':{'REAL':0,'AI':1},
              'protected_cross_role_matches':0,'policy':'TRAIN-only research, no serving promotion; all arms reported',
              'limits':['Inherited E51 consumed replay is TRAIN, not fresh historical-CAL evidence.',
                        'Publisher grouping is conservative; unknown semantic dependence can remain.',
                        'No genuinely independent final or universal no-loss guarantee.']}
    _write_atomic(CONTRACT, report)
    summary = {k:v for k,v in report.items() if k not in {'rows','components','folds','internal_matches'}}
    summary['contract_sha256'] = digest(CONTRACT)
    summary['fold_counts'] = [f['counts'] for f in folds]
    fixed_write(EVIDENCE/'e53_contract.json',summary)
    return summary


def contract():
    receipt = json.loads((EVIDENCE/'e53_contract.json').read_text())
    if digest(CONTRACT) != receipt['contract_sha256']: raise ValueError('contract changed')
    value = json.loads(CONTRACT.read_text())
    if digest(__file__) != value['code_sha256']: raise ValueError('study code changed after freeze')
    for path, sha in value['dependencies_sha256'].items():
        if digest(path) != sha: raise ValueError('feature/fit dependency changed')
    return value, receipt['contract_sha256']


def ordered_views(image, parent):
    bits = hashlib.sha256(f'E53_TRANSPORT_V1|{parent}'.encode()).digest()
    scale = (.5,.75,1.)[bits[0]%3]; quality = (75,90)[bits[1]%2]
    size = tuple(max(1,round(x*scale)) for x in image.size)
    def jpeg(im):
        output = BytesIO(); im.save(output,format='JPEG',quality=quality,subsampling=2,optimize=False)
        with Image.open(BytesIO(output.getvalue())) as decoded: return decoded.convert('RGB')
    return [jpeg(image.resize(size,Image.Resampling.LANCZOS)),
            jpeg(image).resize(size,Image.Resampling.LANCZOS)]


def extract(batch_parents=8):
    import os
    os.environ['HF_HUB_OFFLINE']='1'; os.environ['TRANSFORMERS_OFFLINE']='1'
    import torch
    value, binding = contract()
    if VARIANT.exists(): raise FileExistsError('variant already extracted')
    model, means, stds, weight_sha = dino._load_small()
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    model = model.to(device).eval()
    mean = torch.tensor(means,device=device).view(1,3,1,1)
    std = torch.tensor(stds,device=device).view(1,3,1,1)
    rows = value['rows']; chunks=[]; start_time=time.monotonic()
    def prepare(row):
        raw = Path(row['path']).read_bytes()
        if hashlib.sha256(raw).hexdigest()!=row['sha256']:raise ValueError('changed TRAIN payload')
        with Image.open(BytesIO(raw)) as im: rgb=ImageOps.exif_transpose(im).convert('RGB')
        return [dino.texture_crops(im) for im in ordered_views(rgb,row['parent_id'])]
    with torch.inference_mode(), ThreadPoolExecutor(max_workers=4) as pool:
        for start in range(0,len(rows),batch_parents):
            selected = rows[start:start+batch_parents]
            path=ROOT/f'features/chunks/{start:06d}.npz'
            ids=np.asarray([r['parent_id'] for r in selected])
            if path.exists():
                if any(digest(r['path'])!=r['sha256'] for r in selected):raise ValueError('changed resumed TRAIN payload')
                with np.load(path,allow_pickle=False) as a:
                    if str(a['binding'])!=binding or not np.array_equal(ids,a['parents']):raise ValueError('chunk binding mismatch')
                    features=a['features']
            else:
                prepared=list(pool.map(prepare,selected))
                arrays=np.stack([crop for parent in prepared for view in parent for crop in view])
                batch=torch.from_numpy(arrays).permute(0,3,1,2).to(device=device,dtype=torch.float32)/255.
                blocks=model.forward_intermediates((batch-mean)/std,indices=list(dino.BLOCKS['small']),
                    return_prefix_tokens=True,norm=True,intermediates_only=True)
                tokens=torch.stack([item[1][:,0,:] for item in blocks],dim=1)
                features=dino.aggregate_tokens(tokens.float().cpu().numpy(),len(selected)*2).reshape(len(selected),2,3072)
                dino._save_npz(path,{'features':features,'parents':ids,'binding':np.asarray(binding)})
            chunks.append(features)
            if start%80==0:print(json.dumps({'phase':'order_features','parents':start+len(selected),'total':len(rows),'seconds':round(time.monotonic()-start_time)}),flush=True)
    dino._save_npz(VARIANT,{'features':np.concatenate(chunks),'parents':np.asarray([r['parent_id'] for r in rows]),
                           'binding':np.asarray(binding)})
    result={'state':'E53_order_features_complete','feature_sha256':digest(VARIANT),
            'contract_sha256':binding,'weight_sha256':weight_sha,'parents':len(rows),'views':len(rows)*2}
    fixed_write(EVIDENCE/'e53_order_features.json',result)
    return result


def study(include_order=False):
    value,binding=contract()
    if digest(FEATURES)!=value['features_sha256']:raise ValueError('base features changed')
    with np.load(FEATURES,allow_pickle=False) as a:
        mask=a['roles']=='TRAIN'; data={k:a[k][mask] for k in ('dino','labels','parents','sources','conditions')}
    known={r['parent_id']:r for r in value['rows']}
    if set(data['parents'])!=set(known):raise ValueError('non-TRAIN or missing feature parents')
    for p,y,s in zip(data['parents'],data['labels'],data['sources'],strict=True):
        if known[str(p)]['label']!=int(y) or known[str(p)]['source']!=str(s):raise ValueError('feature metadata mismatch')
    order=None
    if include_order:
        receipt=json.loads((EVIDENCE/'e53_order_features.json').read_text())
        if receipt['contract_sha256']!=binding or digest(VARIANT)!=receipt['feature_sha256']:raise ValueError('order features changed')
        with np.load(VARIANT,allow_pickle=False) as a:
            order={str(p):x for p,x in zip(a['parents'],a['features'],strict=True)}
    eval_mask=np.isin(data['conditions'],['clean','q75'])
    results=[]
    for name,(representation,views) in ARMS.items():
        if views=='order3' and not include_order:continue
        if views!='order3' and include_order:continue
        for fold in value['folds']:
            output=ROOT/f"results/{name}_fold{fold['fold']}.json"
            if output.exists():
                prior=json.loads(output.read_text())
                if prior['contract_sha256']!=binding:raise ValueError('result binding changed')
                results.append(prior);continue
            started=time.monotonic(); roles=np.asarray([fold['roles'][str(p)] for p in data['parents']])
            train=(roles=='FIT') & ((data['conditions']!='q75') if views=='old2' else np.ones(len(roles),dtype=bool))
            if views=='order3':
                train=(roles=='FIT')&(data['conditions']=='clean')
                parents=data['parents'][train]; clean_x=data['dino'][train]
                x=np.stack([np.concatenate([clean_x[i][None,:],order[str(p)]]) for i,p in enumerate(parents)]).reshape(-1,3072)
                y=np.repeat(data['labels'][train],3);sources=np.repeat(data['sources'][train],3);parents=np.repeat(parents,3)
            else:x=data['dino'][train];y=data['labels'][train];sources=data['sources'][train];parents=data['parents'][train]
            width=3072 if representation=='full' else 1536
            weights=parent_source_weights(y,sources,parents)
            head=make_pipeline(StandardScaler(),LogisticRegression(C=.01,max_iter=1000,solver='lbfgs',random_state=53))
            with warnings.catch_warnings(),threadpool_limits(limits=2):
                warnings.simplefilter('error',ConvergenceWarning)
                head.fit(x[:,:width],y,standardscaler__sample_weight=weights,logisticregression__sample_weight=weights)
                cal=(roles=='CAL')&eval_mask; val=(roles=='VALIDATION')&eval_mask
                cal_scores=head.predict_proba(data['dino'][cal,:width])[:,1]
                cut=real_safe_threshold(data['labels'][cal],cal_scores,data['sources'][cal],data['conditions'][cal])
                scores=head.predict_proba(data['dino'][val,:width])[:,1]
            observations=[{'parent_id':str(p),'label':int(y),'source':str(s),'condition':str(c),
                           'score':float(score),'predicted_ai':bool(score>=cut)} for p,y,s,c,score in zip(
                data['parents'][val],data['labels'][val],data['sources'][val],data['conditions'][val],scores,strict=True)]
            rates={}
            for c in ('clean','q75'):
                m=data['conditions'][val]==c
                rates[c]=metrics(data['labels'][val][m],scores[m],data['sources'][val][m],cut,cut)
            modelpath=output.with_suffix('.joblib'); modelpath.parent.mkdir(parents=True,exist_ok=True)
            joblib.dump({'head':head,'threshold':cut,'fold':fold['fold'],'arm':name,'contract_sha256':binding,
                         'restriction':'TRAIN outer-fold diagnostic; never serve this head'},modelpath)
            report={'arm':name,'fold':fold['fold'],'contract_sha256':binding,'threshold':cut,'fit_views':len(y),
                    'cal_views':int(cal.sum()),'validation_views':int(val.sum()),'rates':rates,'observations':observations,
                    'artifact_sha256':digest(modelpath),'seconds':time.monotonic()-started}
            fixed_write(output,report);results.append(report)
            print(json.dumps({k:v for k,v in report.items() if k not in {'observations','rates'}}),flush=True)
    return {'completed_folds':len(results),'order_only':include_order,'contract_sha256':binding}


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['freeze','extract','base','order'])
    args=parser.parse_args()
    result={'freeze':freeze,'extract':extract,'base':study,'order':lambda:study(True)}[args.phase]()
    print(json.dumps(result,indent=2))
