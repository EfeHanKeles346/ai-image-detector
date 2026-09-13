"""Synthetic-only DEAR-r source parity and bounded runtime probe; no image files."""
from __future__ import annotations
import argparse
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import socket
import time
if __name__=='__main__':
    def denied(*a,**kw):raise RuntimeError('E78 synthetic probe is offline')
    socket.socket.connect=denied;socket.socket.connect_ex=denied;socket.create_connection=denied
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
import numpy as np
import torch
import torchvision
from experiments import e78_model as model
from experiments.e78_acquisition import validate as validate_acquisition,ROOT,WEIGHT,RESEARCH,EVIDENCE
from experiments.e65_acquisition import digest,read,write_once
from experiments.e72_acquisition import resource_check
CONTRACT=ROOT/'probe_contract.json';REPORT=ROOT/'probe.json'
SOURCE=RESEARCH/'code/dear/nn_classifier/resnet.py'


def freeze():
    validate_acquisition();receipt=read(ROOT/'acquisition.json')
    if digest(ROOT/'acquisition.json')!=digest(EVIDENCE/'e78_acquisition.json') or receipt['weight_sha256']!=digest(WEIGHT):
        raise ValueError('verified acquisition required')
    paths=[Path(__file__),Path(model.__file__),SOURCE,WEIGHT,ROOT/'acquisition.json',Path(__file__).with_name('e72_acquisition.py')]
    c={'state':'E78_synthetic_parity_resource_registered','inputs':{str(p):digest(p) for p in paths},
       'synthetic':'NumPy default_rng78 uint8 uniform RGB:1x96x96,1x113x157,3x224x224,1x383x511.',
       'reference':'Reviewed hash-pinned author ResNet50(stride0=1).change_output(1), strict weights_only state, '
                   'author binary gate applied before spatial pooling, original ChannelLinear head.',
       'candidate':'Installed torchvision ResNet50 stride0=1 equivalent, exact gate, pooled Linear head.',
       'cpu_parity':'All pooled2048 channels maxabs<=1e-4; logit maxabs<=1e-5; zero sign decisions changed.',
       'mps_parity':'CPU vs MPS on3x224x224 only: pooled feature maxabs<=5e-4, logit<=1e-4, zero sign changes.',
       'runtime':'After2 warmups,10 fixed batch3x224 forwards on MPS or CPU, synchronize each. '
                 'Report seconds/crop and projected12141parent*9crop extraction; no image scores.',
       'resource_gate':'Parity passes, finite outputs, gate820/2048, projected extraction<=7200s, '
                       'MPS driver allocation<=6GiB (or CPU), AC+20GiB reserve, max600s execution.',
       'max_seconds':600,'seed':78,'torch_version':str(torch.__version__),'torchvision_version':torchvision.__version__,
       'image_rows_read':0,'classifier_fit':False,'promotion_allowed':False,
       'adaptation_limits':'Future E54 global+2texture crops would be our crop adaptation, not official full-native image inference. '
                           'Synthetic parity validates implementation, not detector performance.'}
    write_once(CONTRACT,c);write_once(EVIDENCE/'e78_probe_contract.json',c|{'contract_sha256':digest(CONTRACT)})
    return {'state':c['state']}


def validate():
    validate_acquisition();c=read(CONTRACT)
    if digest(CONTRACT)!=read(EVIDENCE/'e78_probe_contract.json')['contract_sha256'] or \
            c['torch_version']!=str(torch.__version__) or c['torchvision_version']!=torchvision.__version__:
        raise ValueError('probe contract/runtime changed')
    for p,s in c['inputs'].items():
        if digest(p)!=s:raise ValueError('probe input changed: '+p)
    return c


def probe():
    c=validate()
    if REPORT.exists():raise FileExistsError('synthetic probe already complete')
    start=time.monotonic();deadline=start+c['max_seconds'];resource_check(deadline);torch.set_num_threads(2)
    spec=importlib.util.spec_from_file_location('e78_reviewed_author_resnet',SOURCE);upstream=importlib.util.module_from_spec(spec);spec.loader.exec_module(upstream)
    reference=upstream.resnet50(pretrained=False,stride0=1).change_output(1).eval()
    checkpoint=torch.load(WEIGHT,map_location='cpu',weights_only=True)['model']
    reference.load_state_dict({k.removeprefix('backbone.'):v for k,v in checkpoint.items() if k.startswith('backbone.')},strict=True)
    candidate=model.load(WEIGHT);rng=np.random.default_rng(78);checks=[];batch=None
    with torch.inference_mode():
        for n,h,w in [(1,96,96),(1,113,157),(3,224,224),(1,383,511)]:
            resource_check(deadline);rgb=rng.integers(0,256,size=(n,h,w,3),dtype=np.uint8);x=model.normalize(rgb)
            raw=reference.forward_features(x)*checkpoint['gate.gate'].view(1,-1,1,1)
            expected=reference.avgpool(raw).flatten(1);logits=reference.forward_head(raw)
            actual=candidate.pooled(x);out=candidate.backbone.fc(actual)
            checks.append({'shape':list(rgb.shape),'feature_max_abs':float(torch.max(torch.abs(expected-actual))),
                'logit_max_abs':float(torch.max(torch.abs(logits-out))),
                'decision_changes':int(((logits>0)!=(out>0)).sum())})
            if n==3:batch=rgb;cpu_features=actual.numpy().copy();cpu_logits=out.numpy().copy()
    device='mps' if torch.backends.mps.is_available() else 'cpu';candidate=candidate.to(device);x=model.normalize(batch).to(device)
    def sync():
        if device=='mps':torch.mps.synchronize()
    with torch.inference_mode():
        features=candidate.pooled(x);logits=candidate.backbone.fc(features);sync()
        device_error={'feature_max_abs':float(np.max(np.abs(features.cpu().numpy()-cpu_features))),
                      'logit_max_abs':float(np.max(np.abs(logits.cpu().numpy()-cpu_logits))),
                      'decision_changes':int(np.sum((logits.cpu().numpy()>0)!=(cpu_logits>0)))}
        for _ in range(2):candidate(x);sync()
        timings=[]
        for _ in range(10):
            resource_check(deadline);t=time.monotonic();candidate(x);sync();timings.append(time.monotonic()-t)
    per_crop=sum(timings)/30;projected=per_crop*12141*9
    allocated=torch.mps.driver_allocated_memory() if device=='mps' else None
    passed=all(v['feature_max_abs']<=1e-4 and v['logit_max_abs']<=1e-5 and not v['decision_changes'] for v in checks) and \
        device_error['feature_max_abs']<=5e-4 and device_error['logit_max_abs']<=1e-4 and not device_error['decision_changes'] and \
        projected<=7200 and (allocated is None or allocated<=6*1024**3)
    result={'state':'E78_synthetic_probe_passed' if passed else 'E78_synthetic_probe_failed',
        'contract_sha256':digest(CONTRACT),'source_parity':checks,'device':device,'device_parity':device_error,
        'seconds_per_crop':per_crop,'projected_extraction_seconds':projected,'mps_driver_bytes':allocated,
        'seconds':time.monotonic()-start,'image_rows_read':0,'classifier_fit':False,
        'feature_extraction_registration_permitted':bool(passed),'promotion_allowed':False}
    write_once(REPORT,result);write_once(EVIDENCE/'e78_probe.json',result);return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=['freeze','probe'])
    with (ROOT/'probe_execution.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        print(json.dumps({'freeze':freeze,'probe':probe}[parser.parse_args().stage](),indent=2))
