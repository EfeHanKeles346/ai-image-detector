"""Add one exact social1080/JPEG75 view to every admitted TRAIN parent, offline and resumable."""
from __future__ import annotations
import argparse
import fcntl
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
import socket
import time
if __name__=='__main__':
    def denied(*a,**kw):raise RuntimeError('E84B feature extraction offline')
    socket.socket.connect=denied;socket.socket.connect_ex=denied;socket.create_connection=denied
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
import joblib
import numpy as np
from PIL import Image
import torch
from threadpoolctl import threadpool_limits
from experiments import e42_features as dino,e78_model as dear_model
from experiments.e80_development import dear_features
from experiments.e71_development import clip_aggregate
from experiments.e71_features import load_encoder,load_crops,array_sha,save_npz
from experiments.e65_diagnostic import social_q75_bytes
from experiments.e65_acquisition import digest,read,write_once
from experiments.e72_acquisition import resource_check
from experiments.e76_fit import combine_rows
from experiments.e75_features import crop_views
from experiments.e83_fit import validate as validate_previous
from experiments.e64_constrained import AI_CUT,REAL_CUT
from pixelproof.project_paths import DATA_ROOT,ML_ROOT
from experiments.e84_features import Encoders,windows,check_chunk,create_chunk,parity
from experiments.e84_sources import materialized_rows,INDEX as SOURCE_INDEX,REPORT as SOURCE_REPORT,CONTRACT as SOURCE_CONTRACT

ROOT=DATA_ROOT/'e84b';EVIDENCE=ML_ROOT.parent/'evidence'
CONTRACT=ROOT/'features_contract.json';PROBE=ROOT/'probe.json';REPORT=ROOT/'features.json';OUTPUT=ROOT/'social_features.npz'
MANIFEST=DATA_ROOT/'e54/data_contract_v2.json';NEW_MANIFEST=DATA_ROOT/'e72/training_manifest.json'
CROP_INDEX=DATA_ROOT/'e54/crop_index.json';WIDTHS={'dino':3072,'clip':1536,'dear':1640}


def population():return materialized_rows()


def freeze():
    ROOT.mkdir(exist_ok=True);validate_previous();rows=population()
    report=read(DATA_ROOT/'e83/dev_report.json');diagnostic=read(DATA_ROOT/'e83/component_diagnostic.json')
    if digest(DATA_ROOT/'e83/dev_report.json')!=digest(EVIDENCE/'e83_development.json') or report['passes_limited_dev_screen'] or \
            digest(DATA_ROOT/'e83/component_diagnostic.json')!=digest(EVIDENCE/'e83_component_diagnostic.json') or \
            diagnostic['score_replay_max_error']!=0:
        raise ValueError('verified rejected E83 and read-only diagnosis required')
    paths=[Path(__file__),Path(__file__).with_name('e84_features.py'),Path(__file__).with_name('e84_sources.py'),
        SOURCE_INDEX,SOURCE_REPORT,SOURCE_CONTRACT,EVIDENCE/'e84_probe_attempt1.json',MANIFEST,NEW_MANIFEST,CROP_INDEX,DATA_ROOT/'e83/dev_report.json',
        DATA_ROOT/'e83/component_diagnostic.json',DATA_ROOT/'e83/fit_contract.json',DATA_ROOT/'e83/fit.json',
        Path(dino.__file__),Path(dear_model.__file__),Path(__file__).with_name('e80_development.py'),
        Path(__file__).with_name('e71_development.py'),Path(__file__).with_name('e71_features.py'),
        Path(__file__).with_name('e65_diagnostic.py'),Path(__file__).with_name('e75_features.py'),
        Path(__file__).with_name('e76_fit.py'),Path(__file__).with_name('e72_acquisition.py')]
    count=len(windows(rows));probe=read(DATA_ROOT/'e84/features_contract.json')['probe_windows']
    if len(probe)!=16 or probe!=sorted(set(probe)) or min(probe)<0 or max(probe)>=count:raise ValueError('original probe plan differs')
    c={'state':'E84B_all_TRAIN_social_transport_features_registered','inputs':{str(p):digest(p) for p in paths},
       'engineering_revision':'E84B only resolves original source-key records through the verified complete source index. '
                              'Same12,141 parents/transform/encoders/probe windows/numeric/resource guards as E84. '
                              'Prior64 completed E84 views are re-encoded and must reproduce exactly; old contract/chunks retained.',
       'parents':12141,'old_parents':11630,'new_parents':511,'real_parents':7546,'ai_parents':4595,
       'condition':'social_q75','batch_views':8,'probe_windows':probe,'probe_max_seconds':900,'max_seconds':10800,
       'mps_driver_limit_bytes':6*1024**3,'cpu_threads':2,
       'population':'All exact E76 admitted TRAIN parents, both labels. Keep old three-condition caches immutable; '
                    'new source-body hashes verified before exact1080px LANCZOS cap then JPEG75/subsampling2. '
                    'Global plus two texture224 crops exactly as E83 DEV, using frozen E65 transport helper. '
                    'No source, score, error, label or content-based selection; no new admission.',
       'encoding':'DINOv2S blocks2/5/8/11 CLS mean/std3072; official frozen CLIP ViT-L/14 batch3 per view '
                  'mean/std1536; DEAR-r820 channels batch9 mean/std1640 using E80 helper. float32. '
                  'Fixed windows8; pad last incomplete window with its last view to8 and discard outputs. '
                  'No classifier head for new social views, no PCA or feature learning.',
       'parity_parents':[min(r['parent_id'] for r in rows if r['source']==source) for source in sorted({r['source'] for r in rows})],
       'parity':'Before full extraction replay34 fixed source representatives on old clean crops. E43 score error<=5e-5 '
                'and zero changes at both cuts; CLIP/DEAR feature max error<=1e-5. Repeat first new probe window '
                'with every feature max error<=1e-5. Old TRAIN reference scores only; never new-view classifier scores.',
       'resource':'AC and20GiB reserve. First16 hash-selected8-parent windows form a bounded cost probe; retain their '
                  'immutable chunks. Project observed encode/source-read/chunk-write time to whole population; '
                  'must be<=10800s and all numeric/memory guards pass before full extraction. Model setup/parity '
                  'reported separately. Runtime deadline10800s per execution, no quality guard change on resume.',
       'resume':'Immutable numeric window chunks, contract/ordered parent/source/derived-body/feature hashes and TRAIN roles. '
                'No partial window acceptance. Full extraction resumes missing windows and verifies existing chunks. '
                'An interrupted probe replays all16 fixed windows for timing, requiring exact existing feature/body equality; '
                'never overwrite prior chunks. Probe timing is per execution, with resumed window count reported.',
       'next':'Only complete bound features permit a separately frozen four-condition supervised representation '
              'with unchanged E82 architecture/optimizer/initialization, then a constrained head. No automatic fit.',
       'limits':'E83 consumed DEV gap motivated transport coverage; no E66 features, pixels or labels in TRAIN. '
                'Existing RR lineage and MIDD/DEAR research-use limits persist. This is not independent validation.',
       'downloads':0,'new_view_classifier_scores':0,'dev_final_image_reads':0,'training_fit_allowed':False,
       'e49_read_allowed':False,'promotion_allowed':False}
    write_once(CONTRACT,c);write_once(EVIDENCE/'e84b_features_contract.json',c|{'contract_sha256':digest(CONTRACT)})
    return {'state':c['state'],'parents':len(rows),'probe_windows':probe}


def validate():
    validate_previous();c=read(CONTRACT)
    if digest(CONTRACT)!=read(EVIDENCE/'e84b_features_contract.json')['contract_sha256']:raise ValueError('E84B contract changed')
    for p,s in c['inputs'].items():
        if digest(p)!=s:raise ValueError('E84B input changed: '+p)
    return c


def probe():
    c=validate()
    if PROBE.exists():raise FileExistsError('E84B probe already complete')
    started=time.monotonic();deadline=started+c['probe_max_seconds'];resource_check(deadline)
    rows=population();batches=windows(rows);binding=digest(CONTRACT);(ROOT/'chunks').mkdir(exist_ok=True)
    enc=Encoders();setup=time.monotonic()-started;check=parity(enc,rows,c,deadline);parity_seconds=time.monotonic()-started-setup
    if not check['passed']:
        result={'state':'E84B_probe_failed','full_extraction_permitted':False,'parity':check,'setup_seconds':setup,'parity_seconds':parity_seconds}
    else:
        elapsed=0.;count=0;repeat=0.;peak=enc.memory();resumed=0;legacy_replays=0
        for j,index in enumerate(c['probe_windows']):
            resource_check(deadline);batch=batches[index];path=ROOT/'chunks'/f'{index:05d}.npz'
            resumed+=int(path.exists())
            before=time.monotonic();crops,features=create_chunk(enc,batch,path,binding);elapsed+=time.monotonic()-before;count+=len(batch)
            legacy_path=DATA_ROOT/'e84/chunks'/f'{index:05d}.npz'
            if legacy_path.exists():
                with np.load(legacy_path,allow_pickle=False) as a:
                    previous=check_chunk(a,digest(DATA_ROOT/'e84/features_contract.json'),batch)
                    if any(not np.array_equal(previous[k],features[k]) for k in WIDTHS):
                        raise ValueError('E84 engineering source revision changed features')
                legacy_replays+=1
            if j==0:
                again=enc.encode(crops);repeat=max(float(np.max(np.abs(features[k]-again[k]))) for k in WIDTHS)
            peak=max(peak,enc.memory())
            if peak>c['mps_driver_limit_bytes']:raise RuntimeError('E84B memory budget exceeded')
            print(json.dumps({'E84B_probe_windows':j+1,'parents':count,'seconds':round(elapsed)}),flush=True)
        projected=elapsed/count*len(rows)
        passed=repeat<=1e-5 and projected<=c['max_seconds'] and peak<=c['mps_driver_limit_bytes']
        result={'state':'E84B_probe_passed' if passed else 'E84B_probe_failed','full_extraction_permitted':passed,
                'parity':check,'setup_seconds':setup,'parity_seconds':parity_seconds,'probe_parents':count,
                'predecessor_windows_replayed_exactly':legacy_replays,
                'probe_encoding_seconds':elapsed,'resumed_probe_windows':resumed,'projected_encoding_seconds':projected,'repeat_max_feature_error':repeat,
                'mps_driver_peak_bytes':peak}
    result.update(contract_sha256=binding,total_seconds=time.monotonic()-started,new_view_classifier_scores=0,dev_final_image_reads=0)
    write_once(PROBE,result);write_once(EVIDENCE/'e84b_probe.json',result)
    return result


def extract():
    c=validate();probe_result=read(PROBE)
    if digest(PROBE)!=digest(EVIDENCE/'e84b_probe.json') or not probe_result['full_extraction_permitted']:raise ValueError('passed E84B probe required')
    if REPORT.exists():raise FileExistsError('E84B features already complete')
    started=time.monotonic();deadline=started+c['max_seconds'];resource_check(deadline)
    rows=population();binding=digest(CONTRACT);enc=Encoders();parts={k:[] for k in WIDTHS};chunks={};created=0;peak=enc.memory()
    for index,batch in enumerate(windows(rows)):
        resource_check(deadline);path=ROOT/'chunks'/f'{index:05d}.npz'
        if not path.exists():create_chunk(enc,batch,path,binding);created+=len(batch)
        with np.load(path,allow_pickle=False) as a:features=check_chunk(a,binding,batch)
        for k,v in features.items():parts[k].append(v)
        chunks[str(path)]=digest(path);peak=max(peak,enc.memory())
        if peak>c['mps_driver_limit_bytes']:raise RuntimeError('E84B MPS memory budget exceeded')
        if index%16==0:print(json.dumps({'E84B_parents':min((index+1)*8,len(rows)),'total':len(rows),
            'created_this_run':created,'seconds':round(time.monotonic()-started),'mps_driver_peak_bytes':peak}),flush=True)
    arrays={k:np.concatenate(v) for k,v in parts.items()};parents=np.array([r['parent_id'] for r in rows])
    if any(v.shape!=(len(rows),WIDTHS[k]) for k,v in arrays.items()):raise ValueError('complete social TRAIN features required')
    if OUTPUT.exists():
        with np.load(OUTPUT,allow_pickle=False) as a:
            if str(a['binding'])!=binding or not np.array_equal(a['parents'],parents) or set(a['roles'])!={'TRAIN'} or \
                    any(not np.array_equal(a[k],v) for k,v in arrays.items()):raise ValueError('unreceipted full archive differs')
    else:save_npz(OUTPUT,**arrays,parents=parents,roles=np.repeat('TRAIN',len(rows)),binding=np.array(binding),condition=np.array('social_q75'))
    result={'state':'E84B_social_TRAIN_features_complete','contract_sha256':binding,'parents':len(rows),
        'shapes':{k:list(v.shape) for k,v in arrays.items()},'feature_sha256':digest(OUTPUT),
        'created_parents_this_run':created,'seconds_this_run':time.monotonic()-started,'mps_driver_peak_bytes':peak,
        'chunks':chunks,'new_view_classifier_scores':0,'dev_final_image_reads':0,'downloads':0,
        'source_index_sha256':digest(SOURCE_INDEX)}
    write_once(REPORT,result);write_once(EVIDENCE/'e84b_features.json',{k:v for k,v in result.items() if k!='chunks'}|{'report_sha256':digest(REPORT)})
    return {k:v for k,v in result.items() if k!='chunks'}

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=('freeze','probe','extract'))
    ROOT.mkdir(exist_ok=True)
    with (ROOT/'execution.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        print(json.dumps({'freeze':freeze,'probe':probe,'extract':extract}[parser.parse_args().stage](),indent=2))
