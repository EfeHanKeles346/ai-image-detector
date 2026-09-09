"""Fixed E54 head-only versus restricted-backbone adaptation on admitted TRAIN folds."""
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
import hashlib
import json
import os
from pathlib import Path
import time

import joblib
import numpy as np

from experiments import e42_features as dino
from experiments.e51_pipeline import real_safe_threshold, metrics
from experiments.e53_adaptation_probe import differentiable_aggregate
from experiments.e53_coverage import OUT as BASELINE
from experiments.e53_offline import EVIDENCE, digest, fixed_write
from experiments.e53_report import paired_interval, preservation_guard
from experiments.e54_data import ROOT, CONTRACT as DATA_CONTRACT, TEACHER, INDEX, load as data_contract
from pixelproof.training_weights import balanced_parent_weights

CONTRACT = ROOT/'training_contract.json'
ARMS = ('head_only', 'last2_anchor')


def freeze():
    if CONTRACT.exists():
        raise FileExistsError('E54 training already frozen')
    data = data_contract()
    helpers = [Path(dino.__file__), Path(__file__).with_name('e53_adaptation_probe.py'),
               Path(__file__).with_name('e51_pipeline.py'),
               EVIDENCE.parent/'ml/src/pixelproof/training_weights.py']
    baselines = {}
    for fold in data['folds']:
        path = BASELINE/f"full_expanded3_fold{fold['fold']}.json"
        result = json.loads(path.read_text())
        artifact = path.with_suffix('.joblib')
        if digest(artifact) != result['artifact_sha256']:
            raise ValueError('FIT-only initializer changed')
        baselines[str(fold['fold'])] = {'path': str(artifact), 'sha256': digest(artifact),
                                      'result_sha256': digest(path)}
    value = {'state':'E54_training_frozen_before_steps','data_contract_sha256':digest(DATA_CONTRACT),
        'code_sha256':digest(__file__),'helper_sha256':{str(p):digest(p) for p in helpers},
        'initializers':baselines,'arms':list(ARMS),'epochs':2,'seed':54,'batch_views':8,
        'backbone_lr':1e-6,'head_lr':1e-5,'weight_decay':.01,'gradient_clip':1.,
        'cosine_anchor_weight':.1,'checkpoint_every_steps':200,'final_epoch_only':True,
        'serving_changed':False,'independent_final_opened':False,
        'policy':'Same source-held-out TRAIN folds; no validation-driven stopping or hyperparameter changes.'}
    fixed_write(CONTRACT,value)
    fixed_write(EVIDENCE/'e54_training_contract.json',value|{'contract_sha256':digest(CONTRACT)})
    return value


def load():
    data = data_contract(); value = json.loads(CONTRACT.read_text())
    if value['data_contract_sha256']!=digest(DATA_CONTRACT) or value['code_sha256']!=digest(__file__):
        raise ValueError('E54 training code/data changed')
    if digest(CONTRACT)!=json.loads((EVIDENCE/'e54_training_contract.json').read_text())['contract_sha256']:
        raise ValueError('E54 training contract changed')
    for path,expected in value['helper_sha256'].items():
        if digest(path)!=expected:
            raise ValueError('E54 helper changed')
    return data,value


def deterministic_order(view_ids, fold, epoch, seed=54):
    return np.random.default_rng(seed+100*fold+epoch).permutation(view_ids)


def torch_head(pipeline, device):
    import torch
    scaler,classifier = pipeline.steps[0][1],pipeline.steps[1][1]
    if list(classifier.classes_) != [0,1]:
        raise ValueError('classifier class orientation changed')
    head=torch.nn.Linear(3072,1).to(device)
    coefficient=classifier.coef_/scaler.scale_
    intercept=classifier.intercept_-coefficient@scaler.mean_
    with torch.no_grad():
        head.weight.copy_(torch.tensor(coefficient,device=device,dtype=torch.float32))
        head.bias.copy_(torch.tensor(intercept,device=device,dtype=torch.float32))
    return head


def atomic_checkpoint(path, value):
    import torch
    def cpu(item):
        if isinstance(item,torch.Tensor):
            return item.detach().cpu()
        if isinstance(item,dict):
            return {k:cpu(v) for k,v in item.items()}
        if isinstance(item,(tuple,list)):
            return type(item)(cpu(v) for v in item)
        return item
    path.parent.mkdir(parents=True,exist_ok=True)
    part=path.with_suffix('.pt.part')
    with part.open('wb') as stream:
        torch.save(cpu(value),stream);stream.flush();os.fsync(stream.fileno())
    part.replace(path)


class CropCache:
    def __init__(self, data):
        self.rows=data['rows'];self.binding=digest(DATA_CONTRACT)
        receipt=json.loads((EVIDENCE/'e54_crop_cache.json').read_text())
        if digest(INDEX)!=receipt['index_sha256']:
            raise ValueError('crop index changed')
        index=json.loads(INDEX.read_text())
        if index['contract_sha256']!=self.binding or digest(TEACHER)!=index['teacher_sha256']:
            raise ValueError('crop/teacher binding changed')
        self.records=index['records']
        with np.load(TEACHER,allow_pickle=False) as a:
            if str(a['binding'])!=self.binding:
                raise ValueError('teacher role mismatch')
            self.teacher=a['features'].reshape(-1,3072)
        self.pool=ThreadPoolExecutor(max_workers=4)

    def one(self, view):
        parent,slot=divmod(int(view),3)
        record=self.records[self.rows[parent]['parent_id']]
        raw=Path(record['path']).read_bytes()
        if hashlib.sha256(raw).hexdigest()!=record['sha256']:
            raise ValueError('crop bytes changed')
        with np.load(BytesIO(raw),allow_pickle=False) as a:
            if str(a['binding'])!=self.binding:
                raise ValueError('crop role mismatch')
            return a['crops'][slot]

    def batch(self, ids):
        return np.concatenate(list(self.pool.map(self.one,ids)),axis=0)


def run(arm, fold_id):
    os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
    import torch
    data,config=load();binding=digest(CONTRACT)
    output=ROOT/'training'/f'{arm}_fold{fold_id}.json'
    if output.exists():
        raise FileExistsError('completed E54 fold cannot be overwritten')
    torch.manual_seed(config['seed'])
    cache=CropCache(data);rows=data['rows'];roles=np.asarray(data['folds'][fold_id]['roles'])
    fit=np.flatnonzero(roles=='FIT');view_ids=(fit[:,None]*3+np.arange(3)).reshape(-1)
    labels=np.repeat([r['label'] for r in rows],3)
    sources=np.repeat([r['source'] for r in rows],3)
    parents=np.repeat([r['parent_id'] for r in rows],3)
    weights=np.zeros(len(labels),dtype=np.float32)
    weights[view_ids]=balanced_parent_weights(labels[view_ids],sources[view_ids],parents[view_ids])
    initializer=config['initializers'][str(fold_id)]
    if digest(initializer['path'])!=initializer['sha256']:
        raise ValueError('FIT-only initializer drift')
    reference=joblib.load(initializer['path'])['head']
    device=torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    model,means,stds,_=dino._load_small();model=model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    if arm=='last2_anchor':
        for block in model.blocks[-2:]:
            for p in block.parameters():p.requires_grad_(True)
    head=torch_head(reference,device)
    mean=torch.tensor(means,device=device).view(1,3,1,1)
    std=torch.tensor(stds,device=device).view(1,3,1,1)
    def encode(ids):
        arrays=cache.batch(ids)
        tensor=torch.from_numpy(arrays).permute(0,3,1,2).to(device,dtype=torch.float32)/255.
        blocks=model.forward_intermediates((tensor-mean)/std,indices=list(dino.BLOCKS['small']),
            return_prefix_tokens=True,norm=True,intermediates_only=True)
        return differentiable_aggregate(torch.stack([b[1][:,0,:] for b in blocks],dim=1))
    def frozen_sha():
        h=hashlib.sha256()
        for name,p in model.named_parameters():
            if not p.requires_grad:
                h.update(name.encode());h.update(p.detach().cpu().numpy().tobytes())
        return h.hexdigest()
    frozen=frozen_sha()
    # One clean FIT parent per source, fixed by hash, checks preprocessing/teacher parity.
    probe=[]
    for source in sorted(set(sources[view_ids])):
        chosen=min((i for i in fit if rows[i]['source']==source),
                   key=lambda i:hashlib.sha256(rows[i]['parent_id'].encode()).hexdigest())
        probe.extend(chosen*3+np.arange(3))
    error=0.;score_error=0.
    with torch.no_grad():
        for start in range(0,len(probe),8):
            ids=probe[start:start+8];features=encode(ids)
            error=max(error,float(np.max(np.abs(features.cpu().numpy()-cache.teacher[ids]))))
            actual=torch.sigmoid(head(features).flatten()).cpu().numpy()
            score_error=max(score_error,float(np.max(np.abs(actual-reference.predict_proba(cache.teacher[ids])[:,1]))))
    if max(error,score_error)>5e-5:
        raise ValueError(f'pre-fit parity failed: {error}, {score_error}')
    groups=[{'params':list(head.parameters()),'lr':config['head_lr']}]
    backbone=[p for p in model.parameters() if p.requires_grad]
    if backbone:groups.append({'params':backbone,'lr':config['backbone_lr']})
    optimizer=torch.optim.AdamW(groups,weight_decay=config['weight_decay'])
    trainable=list(head.parameters())+backbone
    checkpoint=output.with_suffix('.pt');epoch0=0;offset0=0;step=0;history=[]
    if checkpoint.exists():
        saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
        if saved['binding']!=binding or saved['arm']!=arm or saved['fold']!=fold_id:
            raise ValueError('checkpoint role mismatch')
        for i,state in saved['blocks'].items():model.blocks[int(i)].load_state_dict(state)
        head.load_state_dict(saved['head']);optimizer.load_state_dict(saved['optimizer'])
        epoch0=saved['epoch'];offset0=saved['offset'];step=saved['step'];history=saved['history']
    def save(epoch,offset):
        atomic_checkpoint(checkpoint,{'binding':binding,'arm':arm,'fold':fold_id,'epoch':epoch,'offset':offset,
            'step':step,'history':history,'head':head.state_dict(),'optimizer':optimizer.state_dict(),
            'blocks':{i:model.blocks[i].state_dict() for i in (10,11)} if arm=='last2_anchor' else {},
            'frozen_sha256':frozen})
    started=time.monotonic()
    for epoch in range(epoch0,config['epochs']):
        order=deterministic_order(view_ids,fold_id,epoch);total=0.;batches=0
        for offset in range(offset0 if epoch==epoch0 else 0,len(order),config['batch_views']):
            ids=order[offset:offset+config['batch_views']]
            teacher=torch.tensor(cache.teacher[ids],device=device)
            features=encode(ids) if arm=='last2_anchor' else teacher
            target=torch.tensor(labels[ids],device=device,dtype=torch.float32)
            w=torch.tensor(weights[ids],device=device)
            optimizer.zero_grad(set_to_none=True)
            losses=torch.nn.functional.binary_cross_entropy_with_logits(head(features).flatten(),target,reduction='none')
            if arm=='last2_anchor':
                losses=losses+config['cosine_anchor_weight']*(1-torch.nn.functional.cosine_similarity(features,teacher,dim=1))
            loss=(losses*w).mean()
            if not torch.isfinite(loss):raise ValueError('non-finite TRAIN loss')
            loss.backward()
            norm=torch.nn.utils.clip_grad_norm_(trainable,config['gradient_clip'],error_if_nonfinite=True)
            optimizer.step();step+=1;total+=float(loss.detach().cpu());batches+=1
            if step%config['checkpoint_every_steps']==0:
                save(epoch,offset+len(ids))
                print(json.dumps({'phase':'E54_train','arm':arm,'fold':fold_id,'epoch':epoch+1,'step':step,
                    'views_this_epoch':offset+len(ids),'total_views':len(order),'seconds':round(time.monotonic()-started),
                    'train_loss_recent':total/max(1,batches)}),flush=True)
        history.append({'epoch':epoch+1,'mean_batch_loss_this_process_segment':total/max(1,batches)})
        save(epoch+1,0)
    if frozen_sha()!=frozen:raise ValueError('frozen backbone changed')
    # Final epoch is fixed before opening inner CAL/outer validation predictions.
    scoring={};scores_by_role={}
    with torch.inference_mode():
        for role in ('CAL','VALIDATION'):
            parent_ids=np.flatnonzero(roles==role);ids=(parent_ids[:,None]*3+np.array([0,2])).reshape(-1)
            predictions=[]
            for start in range(0,len(ids),16):
                part=ids[start:start+16]
                features=encode(part) if arm=='last2_anchor' else torch.tensor(cache.teacher[part],device=device)
                predictions.extend(torch.sigmoid(head(features).flatten()).cpu().numpy().tolist())
                if start%800==0:print(json.dumps({'phase':'E54_score','arm':arm,'fold':fold_id,'role':role,'views':start+len(part),'total':len(ids)}),flush=True)
            scores_by_role[role]=(ids,np.asarray(predictions))
    cal_ids,cal_scores=scores_by_role['CAL']
    conditions=np.asarray(['clean' if i%3==0 else 'q75' for i in cal_ids])
    cut=real_safe_threshold(labels[cal_ids],cal_scores,sources[cal_ids],conditions)
    val_ids,val_scores=scores_by_role['VALIDATION'];conditions=np.asarray(['clean' if i%3==0 else 'q75' for i in val_ids])
    obs=[{'parent_id':str(parents[i]),'label':int(labels[i]),'source':str(sources[i]),'condition':str(c),
          'score':float(s),'predicted_ai':bool(s>=cut)} for i,c,s in zip(val_ids,conditions,val_scores,strict=True)]
    result={'state':'E54_fold_complete','arm':arm,'fold':fold_id,'contract_sha256':binding,
        'threshold':cut,'observations':obs,'artifact_sha256':digest(checkpoint),'fit_parents':len(fit),
        'fit_views':len(view_ids),'steps':step,'epochs':config['epochs'],'training_history':history,
        'pre_fit_feature_error':error,'pre_fit_score_error':score_error,'frozen_parameters_unchanged':True,
        'rates':{c:metrics(labels[val_ids][conditions==c],val_scores[conditions==c],sources[val_ids][conditions==c],cut,cut) for c in ('clean','q75')},
        'serving_changed':False,'seconds_this_process':time.monotonic()-started}
    fixed_write(output,result);cache.pool.shutdown()
    return {k:v for k,v in result.items() if k!='observations'}


def report():
    data,config=load();binding=digest(CONTRACT)
    baseline_receipt=json.loads((EVIDENCE/'e53_coverage_result.json').read_text())
    base=[r for r in data['rows'] if not r['native']];parents=sorted(r['parent_id'] for r in base)
    known={r['parent_id']:r for r in base};y=np.asarray([known[p]['label'] for p in parents]);s=np.asarray([known[p]['source'] for p in parents])
    # Conservative publisher grouping inherits E53 canonical component ids exactly.
    from experiments.e53_offline import contract as original_contract
    original,_=original_contract();g=np.asarray([original['components'][p] for p in parents])
    predictions={};results={};bindings={}
    for arm in (*ARMS,'full_old2','full_e51_3','full_expanded3'):
        maps={};folds=[]
        for fold in range(3):
            path=(ROOT/'training' if arm in ARMS else BASELINE)/f'{arm}_fold{fold}.json'
            result=json.loads(path.read_text());bindings[str(path)]=digest(path)
            if arm in ARMS and result['contract_sha256']!=binding:raise ValueError('mixed training contract')
            if arm not in ARMS and digest(path)!=baseline_receipt['result_bindings'][str(path)]:
                raise ValueError('frozen comparison result changed')
            roles={r['parent_id']:role for r,role in zip(data['rows'],data['folds'][fold]['roles'],strict=True)}
            for row in result['observations']:
                key=(row['parent_id'],row['condition'])
                if key in maps or row['label']!=known[row['parent_id']]['label'] or row['source']!=known[row['parent_id']]['source']:
                    raise ValueError('prediction alignment changed')
                if roles[row['parent_id']]!='VALIDATION':raise ValueError('non-held-out prediction')
                if row['predicted_ai']!=(row['score']>=result['threshold']):raise ValueError('cut mismatch')
                maps[key]=row['predicted_ai']
            folds.append({k:v for k,v in result.items() if k!='observations'})
        if set(maps)!={(p,c) for p in parents for c in ('clean','q75')}:raise ValueError('incomplete OOF')
        pred=np.asarray([[maps[(p,c)] for c in ('clean','q75')] for p in parents]);predictions[arm]=pred
        rates={}
        for j,c in enumerate(('clean','q75')):
            fp=pred[y==0,j].mean();tp=pred[y==1,j].mean()
            rates[c]={'real_false_ai':float(fp),'ai_recall':float(tp),'balanced_accuracy':float((1-fp+tp)/2),
                      'mean_fold_auc':float(np.mean([f['rates'][c]['auc'] for f in folds]))}
        results[arm]={'conditions':rates,'folds':folds}
    for arm in ARMS:
        comparisons={}
        for ref in ('full_old2','full_e51_3','full_expanded3'):
            ci=paired_interval(y,g,predictions[arm],predictions[ref])
            comparisons[ref]={'intervals':ci,**preservation_guard(y,s,predictions[arm],predictions[ref],ci)}
        results[arm]['comparisons']=comparisons
        results[arm]['research_guard_passed']=all(r['passed'] for r in comparisons.values())
    ci=paired_interval(y,g,predictions['last2_anchor'],predictions['head_only'])
    results['last2_anchor']['head_only_comparison']={'intervals':ci,**preservation_guard(
        y,s,predictions['last2_anchor'],predictions['head_only'],ci)}
    result={'state':'E54_controlled_adaptation_complete','contract_sha256':binding,'arms':results,'result_bindings':bindings,
            'survivors':[a for a in ARMS if results[a]['research_guard_passed']],
            'serving_changed':False,'independent_final_passed':False,
            'limits':['Consumed TRAIN source-held-out study, not E52 or independent replication.',
                      'Comparisons use refitted fold models, not the full E43 94.3% benchmark.',
                      'Cosine anchoring and limited updates are not a reproduction of Effort.',
                      'Eleven-source conditional intervals cannot certify universal transfer.']}
    fixed_write(EVIDENCE/'e54_result.json',result);fixed_write(ROOT/'result.json',result)
    return {k:v for k,v in result.items() if k not in {'arms','result_bindings'}}


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['freeze','run','report'])
    parser.add_argument('--arm',choices=ARMS);parser.add_argument('--fold',type=int,choices=range(3));args=parser.parse_args()
    if args.phase=='run' and (args.arm is None or args.fold is None):parser.error('run requires arm/fold')
    result=freeze() if args.phase=='freeze' else report() if args.phase=='report' else run(args.arm,args.fold)
    print(json.dumps(result,indent=2))
