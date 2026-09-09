"""All admitted TRAIN clean/Q75 crop-dedup parity against pinned E51 features.

Only numerical equivalence is measured, never in-sample classification accuracy.
"""
from concurrent.futures import ThreadPoolExecutor
import json
import time

import joblib
import numpy as np

from experiments import e42_features as dino
from experiments.e51_pipeline import FEATURES,prepare
from experiments.e53_offline import ROOT,EVIDENCE,contract,digest,fixed_write
from experiments.e53_crop_dedup import unique_crops
import experiments.e53_crop_dedup as helper
from pixelproof.project_paths import DATA_ROOT


def parity(reference,optimized,head,threshold):
    if reference.shape!=optimized.shape or not np.isfinite(optimized).all():raise ValueError('invalid optimized feature batch')
    a,b=head.predict_proba(reference)[:,1],head.predict_proba(optimized)[:,1]
    return {'max_feature_error':float(np.max(np.abs(reference-optimized))),
            'max_score_error':float(np.max(np.abs(a-b))),
            'decision_flips':int(np.sum((a>=threshold)!=(b>=threshold)))}


def run():
    import os
    os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
    import torch
    base,binding=contract()
    if (EVIDENCE/'e53_crop_dedup_full.json').exists():raise FileExistsError('full TRAIN parity already complete')
    if digest(FEATURES)!=base['features_sha256']:raise ValueError('reference archive changed')
    artifact_path=DATA_ROOT/'e51/training/e51_A.joblib'
    artifact_sha=json.loads((EVIDENCE/'e51_cal_result.json').read_text())['candidates']['A']['artifact_sha256']
    if digest(artifact_path)!=artifact_sha:raise ValueError('reference head changed')
    artifact=joblib.load(artifact_path)
    rows=[{**r,'role':'TRAIN','condition':c} for r in base['rows'] for c in ('clean','q75')]
    protocol={'base_contract_sha256':binding,'code_sha256':digest(__file__),'helper_sha256':digest(helper.__file__),
              'feature_sha256':base['features_sha256'],'head_sha256':artifact_sha,'views':len(rows),
              'parent_count':len(base['rows']),'conditions':['clean','q75'],'feature_tolerance':.00005,
              'score_tolerance':.00005,'allowed_decision_flips':0,'batch_views':16,
              'purpose':'Numerical regression only; no training accuracy or final claim.'}
    fixed_write(ROOT/'crop_dedup_full_contract.json',protocol)
    run_binding=digest(ROOT/'crop_dedup_full_contract.json')
    with np.load(FEATURES,allow_pickle=False) as saved:
        mask=(saved['roles']=='TRAIN')&np.isin(saved['conditions'],['clean','q75'])
        keys=list(zip(saved['parents'][mask].tolist(),saved['conditions'][mask].tolist(),strict=True))
        reference={key:row for key,row in zip(keys,saved['dino'][mask],strict=True)}
    if set(reference)!={(r['parent_id'],r['condition']) for r in rows}:raise ValueError('reference population mismatch')
    model,means,stds,_=dino._load_small();device=torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    model=model.to(device).eval();mean=torch.tensor(means,device=device).view(1,3,1,1);std=torch.tensor(stds,device=device).view(1,3,1,1)
    results=[];started=time.monotonic()
    with torch.inference_mode(),ThreadPoolExecutor(max_workers=4) as pool:
        for start in range(0,len(rows),16):
            selected=rows[start:start+16];path=ROOT/f'crop_dedup_full_chunks/{start:06d}.json'
            if path.exists():
                chunk=json.loads(path.read_text())
                if chunk['binding']!=run_binding:raise ValueError('parity checkpoint changed')
            else:
                arrays=np.stack([crop for prepared in pool.map(prepare,selected) for crop in prepared[0]])
                unique,inverse=unique_crops(arrays)
                batch=torch.from_numpy(unique).permute(0,3,1,2).to(device=device,dtype=torch.float32)/255.
                blocks=model.forward_intermediates((batch-mean)/std,indices=list(dino.BLOCKS['small']),return_prefix_tokens=True,norm=True,intermediates_only=True)
                tokens=torch.stack([item[1][:,0,:] for item in blocks],dim=1).float().cpu().numpy()[inverse]
                optimized=dino.aggregate_tokens(tokens,len(selected))
                expected=np.stack([reference[(r['parent_id'],r['condition'])] for r in selected])
                chunk={'binding':run_binding,'views':len(selected),'input_crops':len(arrays),'unique_crops':len(unique),
                       **parity(expected,optimized,artifact['head'],artifact['threshold'])}
                fixed_write(path,chunk)
            results.append(chunk)
            if start%160==0:print(json.dumps({'phase':'full_TRAIN_parity','views':start+len(selected),'total':len(rows),'seconds':round(time.monotonic()-started)}),flush=True)
    result={'state':'all_TRAIN_crop_dedup_parity_complete','contract_sha256':run_binding,
            'parents':len(base['rows']),'views':sum(r['views'] for r in results),
            'input_crops':sum(r['input_crops'] for r in results),'unique_crops':sum(r['unique_crops'] for r in results),
            'max_feature_error':max(r['max_feature_error'] for r in results),'max_score_error':max(r['max_score_error'] for r in results),
            'decision_flips':sum(r['decision_flips'] for r in results),'serving_changed':False,'accuracy_gain_claimed':False,
            'limits':['All current TRAIN original/Q75 views, not all future uploads or a new final.',
                      'Reference features are hash-pinned historical E51 outputs; no new baseline inference timing is claimed.']}
    result['numerical_guard_passed']=(result['max_feature_error']<=.00005 and result['max_score_error']<=.00005 and result['decision_flips']==0)
    fixed_write(EVIDENCE/'e53_crop_dedup_full.json',result);return result


if __name__=='__main__':print(json.dumps(run(),indent=2))
