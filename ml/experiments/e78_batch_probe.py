"""One synthetic batch9 throughput repair; unchanged model, pixels, precision and gates."""
from __future__ import annotations
import argparse
import fcntl
import json
import os
from pathlib import Path
import socket
import time
if __name__=='__main__':
    def denied(*a,**kw):raise RuntimeError('E78 batch probe is offline')
    socket.socket.connect=denied;socket.socket.connect_ex=denied;socket.create_connection=denied
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
import numpy as np
import torch
from experiments import e78_model as model
from experiments.e78_probe import validate as validate_previous
from experiments.e78_acquisition import ROOT,WEIGHT,EVIDENCE
from experiments.e65_acquisition import digest,read,write_once
from experiments.e72_acquisition import resource_check
CONTRACT=ROOT/'batch_probe_contract.json';REPORT=ROOT/'batch_probe.json'


def freeze():
    validate_previous();previous=read(ROOT/'probe.json')
    if digest(ROOT/'probe.json')!=digest(EVIDENCE/'e78_probe.json') or previous['state']!='E78_synthetic_probe_failed' or \
            previous['projected_extraction_seconds']<=7200 or previous['mps_driver_bytes']>6*1024**3 or \
            any(v['feature_max_abs']>1e-4 or v['logit_max_abs']>1e-5 or v['decision_changes'] for v in previous['source_parity']) or \
            previous['device_parity']['feature_max_abs']>5e-4 or previous['device_parity']['logit_max_abs']>1e-4 or \
            previous['device_parity']['decision_changes']:
        raise ValueError('only throughput failed; numeric/resource identity required')
    files=[Path(__file__),Path(model.__file__),ROOT/'probe.json',ROOT/'probe_contract.json']
    c={'state':'E78_batch9_throughput_repair_registered','inputs':{str(p):digest(p) for p in files},
       'change':'Only inference batch3->batch9. Same frozen model/uint8 normalization/float32/gate and nine crops per parent.',
       'synthetic':'default_rng78 uint8 uniform9x224x224 RGB. CPU batch3 vs MPS batch9 (CPU fallback).',
       'parity':'Feature maxabs<=5e-4, logit<=1e-4, zero sign changes. Batch3 same-device features/logits same tolerances.',
       'runtime':'Two warmups and10 synchronized batch9 encode calls, including normalization/device copy/output copy. '
                 'Projected12141*9crop time<=7200s; MPS driver<=6GiB. No changed acceptance limits.',
       'max_seconds':600,'image_rows_read':0,'promotion_allowed':False}
    write_once(CONTRACT,c);write_once(EVIDENCE/'e78_batch_probe_contract.json',c|{'contract_sha256':digest(CONTRACT)})
    return {'state':c['state']}


def validate():
    validate_previous();c=read(CONTRACT)
    if digest(CONTRACT)!=read(EVIDENCE/'e78_batch_probe_contract.json')['contract_sha256']:raise ValueError('batch contract changed')
    for p,s in c['inputs'].items():
        if digest(p)!=s:raise ValueError('batch input changed: '+p)
    return c


def probe():
    c=validate()
    if REPORT.exists():raise FileExistsError('batch probe already complete')
    started=time.monotonic();deadline=started+c['max_seconds'];resource_check(deadline);torch.set_num_threads(2)
    rgb=np.random.default_rng(78).integers(0,256,size=(9,224,224,3),dtype=np.uint8)
    net=model.load(WEIGHT);cpu=[model.encode(net,rgb[i:i+3]) for i in range(0,9,3)]
    device='mps' if torch.backends.mps.is_available() else 'cpu';net=net.to(device)
    small=[model.encode(net,rgb[i:i+3]) for i in range(0,9,3)];large=model.encode(net,rgb)
    errors={}
    for name,old in [('cpu',cpu),('same_device_batch3',small)]:
        features=np.concatenate([v[0] for v in old]);logits=np.concatenate([v[1] for v in old])
        errors[name]={'feature_max_abs':float(np.max(np.abs(features-large[0]))),
            'logit_max_abs':float(np.max(np.abs(logits-large[1]))),
            'decision_changes':int(np.sum((logits>0)!=(large[1]>0)))}
    for _ in range(2):model.encode(net,rgb)
    timings=[]
    for _ in range(10):
        resource_check(deadline);start=time.monotonic();model.encode(net,rgb)
        if device=='mps':torch.mps.synchronize()
        timings.append(time.monotonic()-start)
    per_crop=sum(timings)/90;projected=per_crop*12141*9
    allocated=torch.mps.driver_allocated_memory() if device=='mps' else None
    passed=all(v['feature_max_abs']<=5e-4 and v['logit_max_abs']<=1e-4 and not v['decision_changes'] for v in errors.values()) and \
        projected<=7200 and (allocated is None or allocated<=6*1024**3)
    result={'state':'E78_batch9_probe_passed' if passed else 'E78_batch9_probe_failed','contract_sha256':digest(CONTRACT),
        'device':device,'parity':errors,'seconds_per_crop':per_crop,'projected_extraction_seconds':projected,
        'mps_driver_bytes':allocated,'seconds':time.monotonic()-started,'image_rows_read':0,
        'feature_extraction_registration_permitted':bool(passed),'promotion_allowed':False}
    write_once(REPORT,result);write_once(EVIDENCE/'e78_batch_probe.json',result);return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=['freeze','probe'])
    with (ROOT/'batch_probe_execution.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        print(json.dumps({'freeze':freeze,'probe':probe}[parser.parse_args().stage](),indent=2))
