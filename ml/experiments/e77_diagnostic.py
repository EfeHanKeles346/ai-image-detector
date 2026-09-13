"""Read-only full operating-point audit of the frozen failed E77 TRAIN candidate."""
from __future__ import annotations
import json
import os
from pathlib import Path
import socket
import time
if __name__=='__main__':
    def denied(*a,**kw):raise RuntimeError('E77 TRAIN diagnostic is offline')
    socket.socket.connect=denied;socket.socket.connect_ex=denied;socket.create_connection=denied
    os.environ.update(OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
import joblib
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from experiments import e77_model as model
from experiments.e77_fit import validate,ROOT,CANDIDATE,CONTRACT as FIT_CONTRACT
from experiments.e76_fit import combine_rows,real_slice_gates
from experiments.e49_evaluation import evaluate_condition
from experiments.e65_acquisition import digest,read,write_once
from pixelproof.project_paths import DATA_ROOT,ML_ROOT
EVIDENCE=ML_ROOT.parent/'evidence';REPORT=ROOT/'train_operating_point.json'


def run():
    validate()
    if REPORT.exists():raise FileExistsError('frozen TRAIN diagnostic already complete')
    fitted=read(ROOT/'fit.json')
    if digest(ROOT/'fit.json')!=digest(EVIDENCE/'e77_fit.json') or digest(CANDIDATE)!=fitted['candidate_sha256']:
        raise ValueError('failed candidate/report changed')
    previous=read(DATA_ROOT/'e71/fit_contract.json');p={k:Path(v['path']) for k,v in previous['inputs'].items()}
    old=read(p['manifest'])['rows'];rows=combine_rows(old,read(DATA_ROOT/'e72/training_manifest.json')['rows'])
    with np.load(p['features'],allow_pickle=False) as a:original=a['features']
    with np.load(p['clip'],allow_pickle=False) as a:clip=a['features']
    with np.load(DATA_ROOT/'e75/midd_features.npz',allow_pickle=False) as a:
        original=np.concatenate([original,a['dino']]).reshape(-1,3072);clip=np.concatenate([clip,a['clip']]).reshape(-1,1536)
    torch.set_num_threads(2);head=joblib.load(p['reference'])['head'];started=time.monotonic()
    with threadpool_limits(limits=2),np.load(CANDIDATE,allow_pickle=False) as a:
        scores=model.predict(head,original,clip,a).reshape(-1,3)
    if real_slice_gates(rows,scores,len(old))!=fitted['real_population_gates']:
        raise ValueError('frozen fitted operating point replay differs')
    reports={}
    for j,condition in enumerate(['clean','assigned_transport','q75']):
        views=[{'parent_id':r['parent_id'],'label':r['label'],'source':r['source'],'score':float(scores[i,j]),'role':'TRAIN'} for i,r in enumerate(rows)]
        reports[condition]={'old_TRAIN':evaluate_condition(views[:len(old)]),'expanded_TRAIN':evaluate_condition(views)}
    files=[Path(__file__),Path(model.__file__),Path(__file__).with_name('e49_evaluation.py'),
           ML_ROOT/'src/pixelproof/benchmark_metrics.py',CANDIDATE,FIT_CONTRACT,ROOT/'fit.json']
    result={'state':'E77_frozen_TRAIN_operating_point_diagnostic','inputs':{str(p):digest(p) for p in files},
        'reports':reports,'seconds':time.monotonic()-started,'new_fit':False,'threshold_changed':False,
        'e49_rows_or_scores_read':0,'dev_rows_read':0,'image_files_read':0,'promotion_allowed':False,
        'interpretation':'The generic frozen E49 metric function is applied to TRAIN only. '
                         'No E49 population validation/bootstrap, no external quality or independence claim.'}
    write_once(REPORT,result);write_once(EVIDENCE/'e77_train_operating_point.json',result)
    return {k:{s:{'gate':v['gate'],'selective':v['selective']} for s,v in values.items()} for k,values in reports.items()}


if __name__=='__main__':print(json.dumps(run(),indent=2))
