"""Adaptive internal two-placement patch learning with unchanged source-held-out folds."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import socket
import time
import numpy as np
from PIL import Image
from threadpoolctl import threadpool_limits
from experiments.e65_acquisition import digest, read, write_once
from experiments.e71_features import save_npz
from experiments.e72_acquisition import resource_check
from experiments import e130_patch_drift_audit as encoder, e132_patch_learning as old
from experiments import e133_mask_location as location
from pixelproof import patch_learning, holdout_linear, spatial_evaluation, patch_drift
from pixelproof.project_paths import DATA_ROOT, ML_ROOT

ROOT=DATA_ROOT/'e135';EVIDENCE=ML_ROOT.parent/'evidence';CONTRACT=ROOT/'contract.json'
PLACEMENTS=('original','corner');CONDITIONS=old.CONDITIONS;VARIANTS=old.VARIANTS


def expanded_rows(original,corner,folds):
    if len(original)!=16 or len(corner)!=16 or len(folds)!=16:
        raise ValueError('Exact paired16-parent population required')
    rows=[];expanded=[]
    for i,(a,b,fold) in enumerate(zip(original,corner,folds,strict=True)):
        if a['index']!=i or b['index']!=i or any(a.get(k)!=b.get(k) for k in
            ('parent_id','source','scene_group','source_body_sha256','role','seed')) or a['role']!='TRAIN_RESEARCH_PILOT':
            raise ValueError('Paired placement ancestry differs')
        if a['files']['original']!=b['files']['original']:
            raise ValueError('Authentic image must be identical across placements')
        for placement,source in enumerate((a,b)):
            rows.append(source|{'index':2*i+placement,'parent_index':i,'placement':PLACEMENTS[placement]})
            expanded.append(fold)
    return rows,expanded


def journal(title,value):
    note='\n### E135 '+title+'\n\n'+json.dumps(value,sort_keys=True)+'\n'
    for path in (ML_ROOT.parent/'HISTORY.md',ML_ROOT/'EXPERIMENTS.md',ML_ROOT.parent/'DATASETS.md'):
        with path.open('a') as f:fcntl.flock(f,fcntl.LOCK_EX);f.write(note)


def freeze():
    prior=old.validate();location_prior=location.validate();corner=location.prepared_rows()
    if location.verify_cases()!=list(range(16)) or not read(EVIDENCE/'e133_location_generation.json')['full16_engineering_gate_passed']:
        raise ValueError('All16 corner generations must pass before scoped learning')
    if digest(location.ROOT/'report.json')!=digest(EVIDENCE/'e133_mask_location.json'):
        raise ValueError('Complete E133 comparison required')
    rows,folds=expanded_rows(prior['rows'],corner,prior['folds'])
    files=[Path(__file__),Path(patch_learning.__file__),Path(holdout_linear.__file__),Path(spatial_evaluation.__file__),
        Path(patch_drift.__file__),location.CONTRACT,location.ROOT/'prepared.json',location.ROOT/'generation.json',
        location.ROOT/'report.json',location.ROOT/'measurements.json',EVIDENCE/'e133_mask_location.json',
        EVIDENCE/'e133_location_preparation.json',EVIDENCE/'e133_location_generation.json',old.ROOT/'report.json',
        old.ROOT/'measurements.json',old.CONTRACT,DATA_ROOT/'e130/scoring.json',DATA_ROOT/'e130/tokens.npz']
    for base in (old.ROOT,location.ROOT):
        if digest(base/'measurements.json')!=read(base/'report.json')['measurements_sha256']:
            raise ValueError('Previous matched measurements differ')
    for row in corner:
        for value in row['files'].values():files.append(Path(value['path']))
        folder=location.ROOT/'fp16_sdpa'/f"{row['index']:03d}"
        files.extend(folder/name for name in ('raw.png','composite.png','case.json'))
    c={'state':'E135_two_placement_patch_learning_registered',
       'inputs':prior['inputs']|location_prior['inputs']|{str(p):digest(p) for p in files},
       'rows':rows,'folds':folds,'unique_parents':16,'placement_records':32,'fold_count':prior['fold_count'],
       'placements':PLACEMENTS,'conditions':CONDITIONS,'cache_seconds':900,'fit_seconds':3600,
       'representation':'Exact frozen E130384-D original tokens. Reuse original-placement tokens; encode new classical/composite corner views; authentic replay<=1e-5 then reuse old authentic token. No position/mask/reference difference at inference.',
       'training':'Same E132 objective and guards: zero-start weighted BCE+.005||w||²; FIT-only weighted mean/std, no PCA;500 L-BFGS iterations,ftol1e-12,gtol1e-7, final gradient<=1e-5. Half positive mass, one-sixth each authentic/classical/background negative, equal parent/placement/condition. Authentic duplicate bookkeeping has unchanged total weight.',
       'split':'Exact original E132 source/known-ancestry folds. Both placements and all views of a parent stay together. No final full-data head.',
       'targets':'Pure16px cells inside fixed8px eroded mask positive; pure exterior negative; mixed composite cells omitted from fitting. Authentic/classical always negative; score every cell.',
       'cut':.5,'metrics':'Lock all192 maps before evaluation. Original/corner and original/JPEG75 separately: per-parent full/interior-background AUC, IoU and negative flagged area; paired changes and counts of higher/lower parent AUC versus E132/E133.',
       'scope_admission':'Only the already accepted16 E129 and16 E133 composites with matched controls enter E135 TRAIN research. Global roles unchanged.',
       'downloads':0,'promotion_allowed':False,
       'limits':'Adaptive internal development after observing E133; sixteen previously inspected MIDD parents, one SD1.5 editor, two masks per parent. Source exclusion prevents declared parent/scene leakage, not unknown ancestry or adaptive validation bias. More views are not more independent parents. Raw scores uncalibrated; no deployment or universal/editor-transfer proof.'}
    ROOT.mkdir(exist_ok=True);write_once(CONTRACT,c)
    write_once(EVIDENCE/'e135_location_learning_contract.json',{k:v for k,v in c.items() if k not in ('inputs','rows')}|{'contract_sha256':digest(CONTRACT)})
    return {'contract_sha256':digest(CONTRACT),'unique_parents':16,'placement_records':32}


def validate():
    c=read(CONTRACT)
    if digest(CONTRACT)!=read(EVIDENCE/'e135_location_learning_contract.json')['contract_sha256']:
        raise ValueError('Two-placement contract differs')
    for path,sha in c['inputs'].items():
        if digest(path)!=sha:raise ValueError('Bound two-placement input differs')
    return c


def cache():
    c=validate();start=time.monotonic();deadline=start+c['cache_seconds'];resource_check(deadline)
    write_once(ROOT/'cache_started.json',{'contract_sha256':digest(CONTRACT)})
    import torch,timm
    from safetensors.torch import load_file
    from timm.models.vision_transformer import checkpoint_filter_fn
    torch.set_num_threads(2);device=torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    model=timm.create_model(encoder.DINO_MODEL_ID,pretrained=False,num_classes=0,img_size=448)
    model.load_state_dict(checkpoint_filter_fn(load_file(read(encoder.CONTRACT)['backbone']),model),strict=True)
    config=timm.data.resolve_data_config({},model=model);model=model.to(device).eval()
    for p in model.parameters():p.requires_grad_(False)
    mean=torch.tensor(config['mean'],device=device).view(1,3,1,1);std=torch.tensor(config['std'],device=device).view(1,3,1,1)
    tokens={};parity=0.;peak=0
    with np.load(DATA_ROOT/'e130/tokens.npz',allow_pickle=False) as previous:
        if str(previous['contract_sha256'])!=digest(encoder.CONTRACT):raise ValueError('Original token binding differs')
        for row in c['rows']:
            for variant in VARIANTS:
                for condition in CONDITIONS:
                    resource_check(deadline)
                    key=f"{row['index']:03d}_{variant}_{condition}";oldkey=f"{row['parent_index']:03d}_{variant}_{condition}"
                    if row['placement']=='original':tokens[key]=previous[oldkey];continue
                    if variant=='ai_composite':path=location.ROOT/'fp16_sdpa'/f"{row['parent_index']:03d}"/'composite.png'
                    else:path=Path(row['files']['original' if variant=='authentic' else 'classic']['path'])
                    image=encoder.decode(path)
                    if condition!='original':image=encoder.social_view(image)
                    _,value,_,_=encoder.encoded_maps(model,torch,image,device,mean,std)
                    if variant=='authentic':
                        parity=max(parity,float(np.abs(value-previous[oldkey]).max()))
                        if parity>1e-5:raise ValueError('Authentic replay differs')
                        value=previous[oldkey]
                    tokens[key]=value
                    if device.type=='mps':
                        peak=max(peak,torch.mps.driver_allocated_memory())
                        if peak>6*1024**3:raise MemoryError('Token-cache MPS budget exceeded')
            print(json.dumps({'E135_cached_placement':row['index']+1,'placements':32}),flush=True)
    if len(tokens)!=192:raise ValueError('Complete192 token views required')
    save_npz(ROOT/'tokens.npz',**tokens,contract_sha256=digest(CONTRACT))
    report={'state':'E135_two_placement_tokens_complete','token_sha256':digest(ROOT/'tokens.npz'),
            'contract_sha256':digest(CONTRACT),'views':192,'unique_parents':16,'max_authentic_error':parity,
            'seconds':time.monotonic()-start,'peak_mps_bytes':peak,'downloads':0}
    write_once(ROOT/'cache.json',report);write_once(EVIDENCE/'e135_location_tokens.json',report);journal('cache complete',report)
    return report


def summarize(results,references):
    summaries={}
    for placement in PLACEMENTS:
        summaries[placement]={}
        for condition in CONDITIONS:
            selected=[r for r in results if r['placement']==placement and r['condition']==condition]
            before={r['index']:r for r in references[placement] if r['condition']==condition}
            entry={'unique_parents':len(selected)}
            for variant in VARIANTS:
                values=[r['metrics']['maps'][variant] for r in selected]
                prev=[before[r['parent_index']]['metrics']['maps'][variant] for r in selected]
                flags=[r['flagged_area_fraction'] for r in values];oldflags=[r['flagged_area_fraction'] for r in prev]
                entry[variant]={'mean_flagged_area':float(np.mean(flags)),
                    'mean_flagged_area_change':float(np.mean(np.asarray(flags)-oldflags)),
                    'mean_iou':float(np.mean([r['iou'] for r in values if r['iou'] is not None])) if any(r['iou'] is not None for r in values) else None}
                if variant=='ai_composite':
                    auc=np.asarray([r['all_pixel_ranking']['auc'] for r in values]);oldauc=np.asarray([r['all_pixel_ranking']['auc'] for r in prev])
                    entry[variant].update(mean_pixel_auc=float(auc.mean()),mean_auc_change=float((auc-oldauc).mean()),
                        mean_interior_background_auc=float(np.mean([r['interior_background_ranking']['auc'] for r in values])),
                        parents_higher_auc=int((auc>oldauc).sum()),parents_lower_auc=int((auc<oldauc).sum()))
            summaries[placement][condition]=entry
    return summaries


def fit():
    c=validate();start=time.monotonic();deadline=start+c['fit_seconds'];resource_check(deadline)
    if digest(ROOT/'cache.json')!=digest(EVIDENCE/'e135_location_tokens.json') or \
            digest(ROOT/'tokens.npz')!=read(ROOT/'cache.json')['token_sha256']:raise ValueError('Complete bound cache required')
    write_once(ROOT/'fit_started.json',{'contract_sha256':digest(CONTRACT)})
    rows=c['rows'];folds=np.asarray(c['folds']);masks={}
    with np.load(ROOT/'tokens.npz',allow_pickle=False) as data:
        if str(data['contract_sha256'])!=digest(CONTRACT):raise ValueError('Token contract differs')
        tokens={k:data[k] for k in data.files if k!='contract_sha256'}
    for row in rows:
        with Image.open(row['files']['mask']['path']) as im:masks[row['index']]=np.asarray(im)==255
    maps={};fits={}
    with threadpool_limits(limits=2):
        for fold in sorted(set(folds.tolist())):
            resource_check(deadline)
            train=[r for r,f in zip(rows,folds,strict=True) if f!=fold];held=[r for r,f in zip(rows,folds,strict=True) if f==fold]
            if {r['parent_id'] for r in train}&{r['parent_id'] for r in held}:raise ValueError('Parent crossed folds')
            x,y,weights=patch_learning.training_arrays(train,tokens,masks);center,scale=patch_learning.fit_normalizer(x,weights)
            x=(x-center)/scale
            parameters,solver=holdout_linear.fit_head(x,y,weights,lambda:resource_check(deadline))
            artifact=ROOT/f'fold{fold}.npz';save_npz(artifact,center=center,scale=scale,parameters=parameters,contract_sha256=digest(CONTRACT))
            parity=0.
            with np.load(artifact,allow_pickle=False) as saved:
                for row in held:
                    for variant in VARIANTS:
                        for condition in CONDITIONS:
                            key=f"{row['index']:03d}_{variant}_{condition}";value=tokens[key].reshape(-1,384)
                            prediction=holdout_linear.predict((value-center)/scale,parameters)
                            replay=holdout_linear.predict((value-saved['center'])/saved['scale'],saved['parameters'])
                            parity=max(parity,float(np.abs(prediction-replay).max()))
                            if parity>1e-10 or not np.array_equal(prediction>=.5,replay>=.5):raise ValueError('Saved-head score replay differs')
                            maps[key]=prediction.reshape(32,32)
            fits[str(fold)]={'solver':solver,'artifact_sha256':digest(artifact),'max_score_error':parity,
                'train_parent_indices':sorted({r['parent_index'] for r in train}),'held_parent_indices':sorted({r['parent_index'] for r in held})}
            print(json.dumps({'E135_fold_complete':fold,'seconds':round(time.monotonic()-start)}),flush=True)
    if len(maps)!=192:raise ValueError('All192 held-out maps required')
    save_npz(ROOT/'scores.npz',**maps,contract_sha256=digest(CONTRACT))
    write_once(ROOT/'locked_scores.json',{'scores_sha256':digest(ROOT/'scores.npz'),'contract_sha256':digest(CONTRACT),'fits':fits})
    results=[]
    for row in rows:
        for condition in CONDITIONS:
            dense=[patch_drift.expand_grid(maps[f"{row['index']:03d}_{v}_{condition}"],(512,512)) for v in VARIANTS]
            metrics=spatial_evaluation.evaluate_triplet(*dense,masks[row['index']],threshold=.5,boundary_width=8)
            results.append({'parent_index':row['parent_index'],'placement':row['placement'],'condition':condition,'metrics':metrics})
    write_once(ROOT/'measurements.json',{'results':results,'contract_sha256':digest(CONTRACT)})
    refs={'original':read(old.ROOT/'measurements.json')['results'],'corner':read(location.ROOT/'measurements.json')['results']}
    report={'state':'E135_two_placement_internal_learning_complete','contract_sha256':digest(CONTRACT),
        'unique_parents':16,'placement_records':32,'folds':c['fold_count'],'summary':summarize(results,refs),
        'seconds':time.monotonic()-start,'locked_scores_sha256':digest(ROOT/'locked_scores.json'),
        'measurements_sha256':digest(ROOT/'measurements.json'),'downloads':0,'promotion_allowed':False,'limits':c['limits']}
    write_once(ROOT/'report.json',report);write_once(EVIDENCE/'e135_location_learning.json',report);journal('learning result',report)
    return report


if __name__=='__main__':
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    def denied(*args,**kwargs):raise RuntimeError('Two-placement learning is offline')
    socket.socket.connect=denied;socket.socket.connect_ex=denied;socket.create_connection=denied
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=['freeze','cache','fit'])
    ROOT.mkdir(exist_ok=True)
    with (ROOT/'execution.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        print(json.dumps({'freeze':freeze,'cache':cache,'fit':fit}[parser.parse_args().stage](),indent=2))
