"""Controlled native-TRAIN replay expansion; CAL/validation from E53 remain unchanged."""
import argparse
from collections import defaultdict,Counter
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
import hashlib
import json
import time
import warnings

import joblib
import numpy as np
from PIL import Image,ImageOps
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from experiments.e32_r0_input import source_path,PARQUET_SOURCES,_parquet_raw
from experiments.e43_train import parent_source_weights
from experiments.e51_pipeline import FEATURES,real_safe_threshold,metrics
from experiments.e53_offline import ROOT,EVIDENCE,contract,digest,fixed_write,publisher
from experiments import e42_features as dino
from experiments.e53_latest_reserves import close as close_latest_reserves
from pixelproof.project_paths import DATA_ROOT

CONTRACT=ROOT/'expansion_contract.json'
NATIVE_FEATURES=ROOT/'features/native_expansion.npz'
ARMS={'full_expanded3':3072,'mean_expanded3':1536}


def select_rows(records,allowed,internal_pairs):
    excluded={p for pair in internal_pairs for p in (pair['train_parent'],pair['cal_parent'])}
    groups=defaultdict(lambda:defaultdict(list))
    for row in records:
        if row['record_id'] in allowed and 'e32:'+row['record_id'] not in excluded:
            if row['role']!='TRAIN':raise ValueError('non-TRAIN expansion row')
            groups[row['source_id']][row['role_group']].append(row)
    result=[]
    for source,buckets in sorted(groups.items()):
        maximum=1000 if next(iter(buckets.values()))[0]['label']=='real' else 500
        queues={g:sorted(rs,key=lambda r:hashlib.sha256(f"E53_NATIVE|{r['record_id']}".encode()).hexdigest()) for g,rs in buckets.items()}
        taken=0
        for rank in range(max(map(len,queues.values()))):
            for group in sorted(queues,key=lambda g:hashlib.sha256(g.encode()).hexdigest()):
                if rank<len(queues[group]) and taken<maximum:
                    result.append(queues[group][rank]);taken+=1
        if taken<50:raise ValueError('native source depleted below fixed 50-parent floor')
    return result


def freeze():
    if CONTRACT.exists():raise FileExistsError('expansion already frozen')
    base,binding=contract();audit_path=ROOT/'native_inventory_result.json'
    audit=json.loads(audit_path.read_text());receipt=json.loads((EVIDENCE/'e53_native_inventory.json').read_text())
    if digest(audit_path)!=receipt['report_sha256']:raise ValueError('native audit changed')
    c3path=DATA_ROOT/'e32/c3_role_manifest.json'
    if digest(c3path)!='0b6656a25762dbb097d18634d4193282e7c03d2e6bbe257f31471434f67b91eb':raise ValueError('C3 changed')
    records=json.loads(c3path.read_text())['records']
    allowed=close_latest_reserves(records,set(audit['remaining_record_ids']))
    rows=select_rows(records,allowed,audit['internal_pairs'])
    if {'e32:'+r['record_id'] for r in rows}&{r['parent_id'] for r in base['rows']}:
        raise ValueError('native expansion duplicates an existing TRAIN parent')
    if len(rows)>6000 or len(rows)+base['train_parents']>16000:raise ValueError('parent budget exceeded')
    # Original source keys/receipts document these nine research sources; no unknown repacks.
    if len({r['source_id'] for r in rows})!=9:raise ValueError('missing source or unknown expansion schema')
    roles=[]
    for fold in base['folds']:
        mapping=defaultdict(set)
        for old in base['rows']:mapping[publisher(old['source'])].add(fold['roles'][old['parent_id']])
        role={}
        for row in rows:
            oldroles=mapping[publisher('e32:'+row['source_id'])]
            if len(oldroles)!=1:raise ValueError('unknown/ambiguous source component')
            role[row['record_id']]=next(iter(oldroles))
        roles.append(role)
    report={'schema_version':1,'state':'native_expansion_frozen_before_features_and_scores',
            'base_contract_sha256':binding,'native_audit_sha256':digest(audit_path),'code_sha256':digest(__file__),
            'latest_reserve_report_sha256':digest(ROOT/'latest_reserves.json'),
            'rows':rows,'fold_roles':roles,'arms':ARMS,'C':.01,'new_parents':len(rows),
            'source_counts':dict(Counter(r['source_id'] for r in rows)),
            'sample_weight_mass':'Match the original E51 three-view FIT observation count in each fold, even after adding native rows.',
            'model_scores_created':0,'serving_changed':False,
            'policy':'Add only matching-source FIT rows; unchanged E53 CAL/VALIDATION. Not final proof.',
            'licence_boundary':'Existing C3 research terms; CF non-commercial plus per-model terms. No redistribution of images or models.'}
    fixed_write(CONTRACT,report)
    fixed_write(EVIDENCE/'e53_expansion_contract.json',{k:v for k,v in report.items() if k not in {'rows','fold_roles'}}|{'contract_sha256':digest(CONTRACT)})
    return {k:v for k,v in report.items() if k not in {'rows','fold_roles'}}


def load():
    base,binding=contract();value=json.loads(CONTRACT.read_text())
    if value['base_contract_sha256']!=binding or value['code_sha256']!=digest(__file__):raise ValueError('expansion binding changed')
    if digest(CONTRACT)!=json.loads((EVIDENCE/'e53_expansion_contract.json').read_text())['contract_sha256']:raise ValueError('expansion contract changed')
    if digest(ROOT/'native_inventory_result.json')!=value['native_audit_sha256']:raise ValueError('native audit binding changed')
    if digest(ROOT/'latest_reserves.json')!=value['latest_reserve_report_sha256']:raise ValueError('latest reserve audit changed')
    return base,value,binding


def extract():
    import os
    os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
    import torch
    _,value,_=load();binding=digest(CONTRACT)
    if NATIVE_FEATURES.exists():raise FileExistsError('native features already complete')
    model,means,stds,_=dino._load_small();device=torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    model=model.to(device).eval();mean=torch.tensor(means,device=device).view(1,3,1,1);std=torch.tensor(stds,device=device).view(1,3,1,1)
    outputs={};rows=value['rows'];chunks=ROOT/'features/native_chunks';started=time.monotonic()
    def prepare(pair):
        row,raw=pair
        if hashlib.sha256(raw).hexdigest()!=row['sha256']:raise ValueError('native source bytes changed')
        with Image.open(BytesIO(raw)) as im:rgb=ImageOps.exif_transpose(im).convert('RGB')
        buffer=BytesIO();rgb.save(buffer,format='JPEG',quality=75,subsampling=2,optimize=False)
        with Image.open(BytesIO(buffer.getvalue())) as im:q75=im.convert('RGB')
        parent='e32:'+row['record_id']
        views=[dino.transport_image(rgb,'clean'),dino.transport_image(rgb,dino.assigned_transport(parent)),dino.transport_image(q75,'clean')]
        return [crop for im in views for crop in dino.texture_crops(im)]
    with torch.inference_mode(),ThreadPoolExecutor(max_workers=4) as pool:
        def process(batch):
            pending=[]
            for row,raw in batch:
                if hashlib.sha256(raw).hexdigest()!=row['sha256']:raise ValueError('changed native bytes')
                path=chunks/(row['record_id']+'.npz')
                if path.exists():
                    with np.load(path,allow_pickle=False) as a:
                        if str(a['binding'])!=binding:raise ValueError('native chunk binding changed')
                        outputs[row['record_id']]=a['features']
                else:pending.append((row,raw))
            if pending:
                arrays=np.stack([crop for prepared in pool.map(prepare,pending) for crop in prepared])
                batch=torch.from_numpy(arrays).permute(0,3,1,2).to(device=device,dtype=torch.float32)/255.
                blocks=model.forward_intermediates((batch-mean)/std,indices=list(dino.BLOCKS['small']),return_prefix_tokens=True,norm=True,intermediates_only=True)
                tokens=torch.stack([b[1][:,0,:] for b in blocks],dim=1).float().cpu().numpy()
                features=dino.aggregate_tokens(tokens,len(pending)*3).reshape(len(pending),3,3072)
                for (row,_),f in zip(pending,features,strict=True):
                    dino._save_npz(chunks/(row['record_id']+'.npz'),{'features':f,'binding':np.asarray(binding)})
                    outputs[row['record_id']]=f
            if len(outputs)%80<8:print(json.dumps({'phase':'native_features','parents':len(outputs),'total':len(rows),'seconds':round(time.monotonic()-started)}),flush=True)
        loose=[r for r in rows if r['source_id'] not in PARQUET_SOURCES]
        for start in range(0,len(loose),8):process([(r,source_path(r['source_id'],r['source_key']).read_bytes()) for r in loose[start:start+8]])
        for source,(folder,column) in PARQUET_SOURCES.items():
            pending=[]
            for pair in _parquet_raw([r for r in rows if r['source_id']==source],folder,column):
                pending.append(pair)
                if len(pending)==8:process(pending);pending=[]
            if pending:process(pending)
    features=np.stack([outputs[r['record_id']] for r in rows])
    if not np.isfinite(features).all():raise ValueError('non-finite native feature')
    dino._save_npz(NATIVE_FEATURES,{'features':features,'record_ids':np.asarray([r['record_id'] for r in rows]),'binding':np.asarray(binding)})
    result={'contract_sha256':binding,'feature_sha256':digest(NATIVE_FEATURES),'parents':len(rows),'views':len(rows)*3,'state':'native_features_complete'}
    fixed_write(EVIDENCE/'e53_native_features.json',result);return result


def fit():
    base,value,binding=load();expansion_sha=digest(CONTRACT)
    if digest(FEATURES)!=base['features_sha256']:raise ValueError('base features changed')
    receipt=json.loads((EVIDENCE/'e53_native_features.json').read_text())
    if receipt['contract_sha256']!=expansion_sha or digest(NATIVE_FEATURES)!=receipt['feature_sha256']:raise ValueError('native features changed')
    with np.load(FEATURES,allow_pickle=False) as a:
        m=a['roles']=='TRAIN';data={k:a[k][m] for k in ('dino','labels','parents','sources','conditions')}
    with np.load(NATIVE_FEATURES,allow_pickle=False) as a:
        native=a['features'];ids=a['record_ids']
    if list(ids)!=[r['record_id'] for r in value['rows']]:raise ValueError('native order changed')
    for arm,width in ARMS.items():
        for fold in base['folds']:
            path=ROOT/f"results/{arm}_fold{fold['fold']}.json"
            if path.exists():continue
            started=time.monotonic();roles=np.asarray([fold['roles'][str(p)] for p in data['parents']])
            train=roles=='FIT';nmask=np.asarray([value['fold_roles'][fold['fold']][str(p)]=='FIT' for p in ids]);nr=[r for r,m in zip(value['rows'],nmask,strict=True) if m]
            x=np.concatenate([data['dino'][train],native[nmask].reshape(-1,3072)])[:,:width]
            y=np.r_[data['labels'][train],np.repeat([int(r['label']=='ai') for r in nr],3)].astype(int)
            sources=np.r_[data['sources'][train],np.repeat(['e32:'+r['source_id'] for r in nr],3)]
            parents=np.r_[data['parents'][train],np.repeat(['e32:'+r['record_id'] for r in nr],3)]
            weights=parent_source_weights(y,sources,parents)
            # Keep effective regularization fixed relative to full_e51_3/mean_e51_3.
            weights*=int(train.sum())/weights.sum()
            head=make_pipeline(StandardScaler(),LogisticRegression(C=.01,max_iter=1000,solver='lbfgs',random_state=53))
            evaluate=np.isin(data['conditions'],['clean','q75']);cal=(roles=='CAL')&evaluate;val=(roles=='VALIDATION')&evaluate
            with warnings.catch_warnings(),threadpool_limits(limits=2):
                warnings.simplefilter('error',ConvergenceWarning)
                head.fit(x,y,standardscaler__sample_weight=weights,logisticregression__sample_weight=weights)
                cs=head.predict_proba(data['dino'][cal,:width])[:,1];cut=real_safe_threshold(data['labels'][cal],cs,data['sources'][cal],data['conditions'][cal])
                scores=head.predict_proba(data['dino'][val,:width])[:,1]
            observations=[{'parent_id':str(p),'label':int(y),'source':str(s),'condition':str(c),'score':float(v),'predicted_ai':bool(v>=cut)} for p,y,s,c,v in zip(data['parents'][val],data['labels'][val],data['sources'][val],data['conditions'][val],scores,strict=True)]
            rates={}
            for c in ('clean','q75'):
                m=data['conditions'][val]==c;rates[c]=metrics(data['labels'][val][m],scores[m],data['sources'][val][m],cut,cut)
            artifact=path.with_suffix('.joblib');artifact.parent.mkdir(parents=True,exist_ok=True)
            joblib.dump({'head':head,'threshold':cut,'restriction':'TRAIN-only fold model; do not serve'},artifact)
            result={'arm':arm,'fold':fold['fold'],'contract_sha256':binding,'expansion_contract_sha256':expansion_sha,
                    'fit_views':len(y),'new_native_fit_parents':len(nr),'cal_views':int(cal.sum()),'validation_views':int(val.sum()),
                    'threshold':cut,'rates':rates,'observations':observations,'artifact_sha256':digest(artifact),'seconds':time.monotonic()-started}
            fixed_write(path,result);print(json.dumps({k:v for k,v in result.items() if k not in {'observations','rates'}}),flush=True)
    return {'state':'expanded_folds_complete','arms':list(ARMS),'serving_changed':False}


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['freeze','extract','fit']);args=parser.parse_args()
    print(json.dumps({'freeze':freeze,'extract':extract,'fit':fit}[args.phase](),indent=2))
