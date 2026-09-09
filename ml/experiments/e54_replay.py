"""Reconstruct E54 artifacts and replay already-consumed OOF predictions only."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np

from experiments import e42_features as dino
from experiments.e53_adaptation_probe import differentiable_aggregate
from experiments.e53_offline import EVIDENCE, digest, fixed_write
from experiments.e54_adapt import ARMS, CONTRACT as TRAIN_CONTRACT, CropCache, load
from experiments.e54_data import ROOT

CONTRACT = ROOT/'replay_contract.json'


def compare_scores(actual,expected,cut):
    # Archived probabilities are Python/float64 values. Preserve that comparison
    # precision when CAL chose nextafter(score,+inf); numpy float32 promotion
    # could otherwise round the threshold back down and invent a decision flip.
    actual=np.asarray(actual,dtype=np.float64);expected=np.asarray(expected,dtype=np.float64)
    if actual.shape!=expected.shape or not np.isfinite(actual).all():
        raise ValueError('invalid reconstructed scores')
    return float(np.max(np.abs(actual-expected))),int(np.sum((actual>=cut)!=(expected>=cut)))


def freeze():
    load()
    report = json.loads((ROOT/'result.json').read_text())
    paths = [ROOT/'result.json',TRAIN_CONTRACT,Path(__file__).with_name('e54_adapt.py'),
             Path(dino.__file__),Path(__file__).with_name('e53_adaptation_probe.py')]
    for arm in ARMS:
        for fold in range(3):
            path = ROOT/'training'/f'{arm}_fold{fold}.json'
            if digest(path) != report['result_bindings'][str(path)]:
                raise ValueError('result binding changed')
            artifact = path.with_suffix('.pt')
            if digest(artifact) != json.loads(path.read_text())['artifact_sha256']:
                raise ValueError('checkpoint changed')
            paths.extend([path,artifact])
    value = {'state':'E54_replay_frozen','code_sha256':digest(__file__),
             'inputs':{str(p):digest(p) for p in paths},'score_tolerance':5e-5,
             'max_decision_changes':0,'independent_test_opened':False,'serving_changed':False}
    fixed_write(CONTRACT,value)
    fixed_write(EVIDENCE/'e54_replay_contract.json',value|{'contract_sha256':digest(CONTRACT)})
    return value


def run():
    os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
    import torch
    config = json.loads(CONTRACT.read_text())
    if config['code_sha256'] != digest(__file__) or digest(CONTRACT) != json.loads(
            (EVIDENCE/'e54_replay_contract.json').read_text())['contract_sha256']:
        raise ValueError('replay contract/code changed')
    for path,sha in config['inputs'].items():
        if digest(path) != sha:
            raise ValueError('replay input changed')
    data,training = load();cache=CropCache(data);results=[]
    device=torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    for arm in ARMS:
        for fold in range(3):
            path=ROOT/'training'/f'{arm}_fold{fold}.json'
            result=json.loads(path.read_text())
            saved=torch.load(path.with_suffix('.pt'),map_location='cpu',weights_only=True)
            roles=np.asarray(data['folds'][fold]['roles'])
            expected_steps=training['epochs']*math.ceil(3*int((roles=='FIT').sum())/training['batch_views'])
            if (saved['binding']!=digest(TRAIN_CONTRACT) or saved['epoch']!=training['epochs']
                    or saved['offset']!=0 or saved['step']!=expected_steps
                    or saved['arm']!=arm or saved['fold']!=fold):
                raise ValueError('checkpoint does not represent declared final epoch')
            if set(saved['blocks'])!=({10,11} if arm=='last2_anchor' else set()):
                raise ValueError('unexpected trainable block set')
            model,means,stds,_=dino._load_small();model=model.to(device).eval()
            for parameter in model.parameters():parameter.requires_grad_(False)
            if arm=='last2_anchor':
                for i in (10,11):
                    model.blocks[i].load_state_dict(saved['blocks'][i])
                    for parameter in model.blocks[i].parameters():parameter.requires_grad_(True)
            h=hashlib.sha256()
            for name,parameter in model.named_parameters():
                if not parameter.requires_grad:
                    h.update(name.encode());h.update(parameter.detach().cpu().numpy().tobytes())
            if h.hexdigest()!=saved['frozen_sha256']:
                raise ValueError('frozen backbone reconstruction changed')
            head=torch.nn.Linear(3072,1).to(device);head.load_state_dict(saved['head'])
            mean=torch.tensor(means,device=device).view(1,3,1,1)
            std=torch.tensor(stds,device=device).view(1,3,1,1)
            def encode(ids):
                tensor=torch.from_numpy(cache.batch(ids)).permute(0,3,1,2).to(device,dtype=torch.float32)/255.
                blocks=model.forward_intermediates((tensor-mean)/std,indices=list(dino.BLOCKS['small']),
                    return_prefix_tokens=True,norm=True,intermediates_only=True)
                return differentiable_aggregate(torch.stack([b[1][:,0,:] for b in blocks],dim=1))
            parents=np.flatnonzero(roles=='VALIDATION')
            ids=(parents[:,None]*3+np.array([0,2])).reshape(-1)
            by_key={(r['parent_id'],r['condition']):r for r in result['observations']}
            if len(by_key)!=len(ids):raise ValueError('incomplete/duplicate archived predictions')
            max_error=0.;flips=0
            with torch.inference_mode():
                for start in range(0,len(ids),16):
                    part=ids[start:start+16]
                    features=encode(part) if arm=='last2_anchor' else torch.tensor(cache.teacher[part],device=device)
                    actual=torch.sigmoid(head(features).flatten()).cpu().numpy()
                    expected=np.asarray([by_key[(data['rows'][int(i)//3]['parent_id'],
                        'clean' if i%3==0 else 'q75')]['score'] for i in part])
                    error,changed=compare_scores(actual,expected,result['threshold'])
                    max_error=max(max_error,error);flips+=changed
                    if start%1600==0:
                        print(json.dumps({'phase':'E54_artifact_replay','arm':arm,'fold':fold,
                                          'views':start+len(part),'total':len(ids)}),flush=True)
            results.append({'arm':arm,'fold':fold,'observations':len(ids),'max_score_error':max_error,
                            'decision_changes':flips,'frozen_parameters_unchanged':True,
                            'passed':bool(max_error<=config['score_tolerance'] and flips==0)})
            del model,head
    cache.pool.shutdown()
    value={'state':'E54_artifacts_replayed','contract_sha256':digest(CONTRACT),'results':results,
           'passed':all(r['passed'] for r in results),'independent_test_opened':False,'serving_changed':False}
    fixed_write(ROOT/'replay.json',value);fixed_write(EVIDENCE/'e54_replay.json',value)
    return value


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['freeze','run'])
    args=parser.parse_args();print(json.dumps({'freeze':freeze,'run':run}[args.phase](),indent=2))
