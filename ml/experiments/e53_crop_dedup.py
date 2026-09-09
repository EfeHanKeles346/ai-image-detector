"""Optional exact-crop deduplication benchmark, not enabled in serving or frozen studies."""
import hashlib
import json
import time

import joblib
import numpy as np

from experiments import e42_features as dino
from experiments.e51_pipeline import prepare
from experiments.e53_offline import ROOT,EVIDENCE,contract,digest,fixed_write
from pixelproof.project_paths import DATA_ROOT


def unique_crops(arrays):
    """Stable exact-byte deduplication; confirm equality even within a digest bucket."""
    if arrays.ndim!=4 or arrays.shape[-1]!=3 or arrays.dtype!=np.uint8 or len(arrays)==0:
        raise ValueError('nonempty NHWC uint8 RGB crops required')
    buckets={};unique=[];inverse=[]
    for array in arrays:
        key=hashlib.sha256(array.tobytes()).digest();found=None
        for index in buckets.get(key,[]):
            if np.array_equal(unique[index],array):found=index;break
        if found is None:
            found=len(unique);unique.append(array);buckets.setdefault(key,[]).append(found)
        inverse.append(found)
    return np.stack(unique),np.asarray(inverse,dtype=np.int64)


def benchmark():
    import os
    os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
    import torch
    value,binding=contract()
    if (EVIDENCE/'e53_crop_dedup_benchmark.json').exists():raise FileExistsError('benchmark already complete')
    # Two hash-ranked TRAIN parents per source; no metric-selected examples or external test access.
    sources=sorted({r['source'] for r in value['rows']})
    selected=[]
    for source in sources:
        rows=sorted((r for r in value['rows'] if r['source']==source),
                    key=lambda r:hashlib.sha256(f"E53_SPEED|{r['parent_id']}".encode()).hexdigest())
        selected.extend(rows[:2])
    protocol={'contract_sha256':binding,'code_sha256':digest(__file__),
              'parent_ids':[r['parent_id'] for r in selected],'conditions':['clean','q75'],
              'warmup_passes':1,'timed_passes':3,'batch_views':16,'tolerance_features':.00005,
              'tolerance_scores':.00005,'allowed_decision_flips':0,
              'serving_changed':False,'purpose':'numerical equivalence and runtime, not accuracy evaluation'}
    fixed_write(ROOT/'crop_dedup_contract.json',protocol)
    model,means,stds,_=dino._load_small();device=torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    model=model.to(device).eval();mean=torch.tensor(means,device=device).view(1,3,1,1);std=torch.tensor(stds,device=device).view(1,3,1,1)
    arrays=np.stack([crop for row in selected for condition in ('clean','q75')
                     for crop in prepare({**row,'role':'TRAIN','condition':condition})[0]])
    artifact_path=DATA_ROOT/'e51/training/e51_A.joblib'
    expected=json.loads((EVIDENCE/'e51_cal_result.json').read_text())['candidates']['A']['artifact_sha256']
    if digest(artifact_path)!=expected:raise ValueError('comparison head changed')
    artifact=joblib.load(artifact_path)
    def sync():
        if device.type=='mps':torch.mps.synchronize()
    def forward(batch,dedup):
        sync();started=time.perf_counter()
        inputs,inverse=unique_crops(batch) if dedup else (batch,np.arange(len(batch)))
        tensor=torch.from_numpy(inputs).permute(0,3,1,2).to(device=device,dtype=torch.float32)/255.
        blocks=model.forward_intermediates((tensor-mean)/std,indices=list(dino.BLOCKS['small']),
            return_prefix_tokens=True,norm=True,intermediates_only=True)
        tokens=torch.stack([item[1][:,0,:] for item in blocks],dim=1).float().cpu().numpy()[inverse]
        result=dino.aggregate_tokens(tokens,len(batch)//3);sync()
        return result,time.perf_counter()-started,len(inputs)
    durations={False:[],True:[]};feature_error=0.;score_error=0.;flips=0;unique_count=0
    with torch.inference_mode():
        for repetition in range(4):
            totals={False:0.,True:0.}
            for start in range(0,len(arrays),48):
                batch=arrays[start:start+48];values={}
                for dedup in ((False,True) if repetition%2==0 else (True,False)):
                    features,elapsed,count=forward(batch,dedup);totals[dedup]+=elapsed;values[dedup]=features
                    if dedup and repetition==1:unique_count+=count
                feature_error=max(feature_error,float(np.max(np.abs(values[False]-values[True]))))
                scores=[artifact['head'].predict_proba(values[d])[:,1] for d in (False,True)]
                score_error=max(score_error,float(np.max(np.abs(scores[0]-scores[1]))))
                flips+=int(np.sum((scores[0]>=artifact['threshold'])!=(scores[1]>=artifact['threshold'])))
            if repetition>0:
                for dedup in (False,True):durations[dedup].append(totals[dedup])
    baseline=float(np.median(durations[False]));optimized=float(np.median(durations[True]))
    result={'state':'exact_crop_dedup_benchmark_complete','contract_sha256':digest(ROOT/'crop_dedup_contract.json'),
            'parents':len(selected),'views':len(selected)*2,'input_crops':len(arrays),'unique_crops_per_pass':unique_count,
            'max_feature_error':feature_error,'max_score_error':score_error,'decision_flips_across_four_passes':flips,
            'plain_seconds':durations[False],'dedup_seconds':durations[True],'median_speedup':baseline/optimized,
            'numerical_guard_passed':feature_error<=.00005 and score_error<=.00005 and flips==0,
            'serving_changed':False,'accuracy_gain_claimed':False,
            'limits':['Small fixed TRAIN sample; device/runtime-specific speed, not whole-population decision equivalence.',
                      'Current production and frozen E53 extraction code were not changed.']}
    fixed_write(EVIDENCE/'e53_crop_dedup_benchmark.json',result)
    return result


if __name__=='__main__':print(json.dumps(benchmark(),indent=2))
