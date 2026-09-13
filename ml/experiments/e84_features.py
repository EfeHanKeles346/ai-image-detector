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
    def denied(*a,**kw):raise RuntimeError('E84 feature extraction offline')
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
ROOT=DATA_ROOT/'e84';EVIDENCE=ML_ROOT.parent/'evidence'
CONTRACT=ROOT/'features_contract.json';PROBE=ROOT/'probe.json';REPORT=ROOT/'features.json';OUTPUT=ROOT/'social_features.npz'
MANIFEST=DATA_ROOT/'e54/data_contract_v2.json';NEW_MANIFEST=DATA_ROOT/'e72/training_manifest.json'
CROP_INDEX=DATA_ROOT/'e54/crop_index.json';WIDTHS={'dino':3072,'clip':1536,'dear':1640}


def population():return combine_rows(read(MANIFEST)['rows'],read(NEW_MANIFEST)['rows'])


def social_crops(body):
    q75=social_q75_bytes(BytesIO(body))
    with Image.open(BytesIO(q75)) as image:crops=np.stack(dino.texture_crops(dino.transport_image(image,'clean')))
    return crops,hashlib.sha256(q75).hexdigest()


def windows(rows,size=8):return [rows[i:i+size] for i in range(0,len(rows),size)]


def check_chunk(a,binding,rows):
    if str(a['binding'])!=binding or list(a['parents'])!=[r['parent_id'] for r in rows] or \
            list(a['source_sha256'])!=[r['sha256'] for r in rows] or len(a['roles'])!=len(rows) or set(a['roles'])!={'TRAIN'} or \
            a['transport_sha256'].shape!=(len(rows),) or any(len(s)!=64 for s in a['transport_sha256']):
        raise ValueError('complete ordered TRAIN chunk identities required')
    features={}
    for key,width in WIDTHS.items():
        v=a[key]
        if v.shape!=(len(rows),width) or v.dtype!=np.float32 or not np.isfinite(v).all() or str(a[key+'_sha256'])!=array_sha(v):
            raise ValueError('finite feature body and dimensions required')
        features[key]=v
    return features


class Encoders:
    def __init__(self):
        torch.set_num_threads(2);self.device=torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        net,means,stds,_=dino._load_small();self.dino=net.to(self.device).eval()
        self.mean=torch.tensor(means,device=self.device).view(1,3,1,1)
        self.std=torch.tensor(stds,device=self.device).view(1,3,1,1)
        self.clip,self.preprocess,self.clip_device=load_encoder()
        self.dear=dear_model.load(DATA_ROOT/'e78/dear_r.pth',self.device)
    def encode(self,crops):
        if crops.ndim!=5 or crops.shape[1:]!=(3,224,224,3) or crops.dtype!=np.uint8 or not 1<=len(crops)<=8:
            raise ValueError('one to eight complete RGB view crops required')
        count=len(crops)
        if count<8:crops=np.concatenate([crops,np.repeat(crops[-1:],8-count,axis=0)])
        clip=[]
        with torch.inference_mode(),threadpool_limits(limits=2):
            for pack in crops:
                tensor=torch.stack([self.preprocess(Image.fromarray(crop)) for crop in pack]).to(self.clip_device)
                clip.append(clip_aggregate(self.clip.encode_image(tensor).float().cpu().numpy()))
            flat=crops.reshape(-1,224,224,3)
            tensor=torch.from_numpy(flat).to(self.device).permute(0,3,1,2).float().div_(255)
            blocks=self.dino.forward_intermediates((tensor-self.mean)/self.std,indices=list(dino.BLOCKS['small']),
                return_prefix_tokens=True,norm=True,intermediates_only=True)
            tokens=torch.stack([b[1][:,0,:] for b in blocks],dim=1).float().cpu().numpy()
            result={'dino':dino.aggregate_tokens(tokens,8),'clip':np.stack(clip),'dear':dear_features(self.dear,crops)}
        return {k:v[:count] for k,v in result.items()}
    def memory(self):return torch.mps.driver_allocated_memory() if self.device.type=='mps' else 0


def freeze():
    ROOT.mkdir(exist_ok=True);validate_previous();rows=population()
    report=read(DATA_ROOT/'e83/dev_report.json');diagnostic=read(DATA_ROOT/'e83/component_diagnostic.json')
    if digest(DATA_ROOT/'e83/dev_report.json')!=digest(EVIDENCE/'e83_development.json') or report['passes_limited_dev_screen'] or \
            digest(DATA_ROOT/'e83/component_diagnostic.json')!=digest(EVIDENCE/'e83_component_diagnostic.json') or \
            diagnostic['score_replay_max_error']!=0:
        raise ValueError('verified rejected E83 and read-only diagnosis required')
    paths=[Path(__file__),MANIFEST,NEW_MANIFEST,CROP_INDEX,DATA_ROOT/'e83/dev_report.json',
        DATA_ROOT/'e83/component_diagnostic.json',DATA_ROOT/'e83/fit_contract.json',DATA_ROOT/'e83/fit.json',
        Path(dino.__file__),Path(dear_model.__file__),Path(__file__).with_name('e80_development.py'),
        Path(__file__).with_name('e71_development.py'),Path(__file__).with_name('e71_features.py'),
        Path(__file__).with_name('e65_diagnostic.py'),Path(__file__).with_name('e75_features.py'),
        Path(__file__).with_name('e76_fit.py'),Path(__file__).with_name('e72_acquisition.py')]
    count=len(windows(rows));probe=sorted(sorted(range(count),key=lambda i:hashlib.sha256(f'E84_probe|{i}'.encode()).digest())[:16])
    c={'state':'E84_all_TRAIN_social_transport_features_registered','inputs':{str(p):digest(p) for p in paths},
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
    write_once(CONTRACT,c);write_once(EVIDENCE/'e84_features_contract.json',c|{'contract_sha256':digest(CONTRACT)})
    return {'state':c['state'],'parents':len(rows),'probe_windows':probe}


def validate():
    validate_previous();c=read(CONTRACT)
    if digest(CONTRACT)!=read(EVIDENCE/'e84_features_contract.json')['contract_sha256']:raise ValueError('E84 contract changed')
    for p,s in c['inputs'].items():
        if digest(p)!=s:raise ValueError('E84 input changed: '+p)
    return c


def create_chunk(enc,rows,path,binding):
    crops=[];transport=[]
    for row in rows:
        body=Path(row['path']).read_bytes()
        if hashlib.sha256(body).hexdigest()!=row['sha256']:raise ValueError('TRAIN source body changed')
        crop,qsha=social_crops(body);crops.append(crop);transport.append(qsha)
    features=enc.encode(np.stack(crops))
    if path.exists():
        with np.load(path,allow_pickle=False) as a:
            old=check_chunk(a,binding,rows)
            if list(a['transport_sha256'])!=transport or any(not np.array_equal(old[k],v) for k,v in features.items()):
                raise ValueError('resumed probe source/feature replay differs')
    else:
        save_npz(path,**features,**{k+'_sha256':np.array(array_sha(v)) for k,v in features.items()},
            parents=np.array([r['parent_id'] for r in rows]),source_sha256=np.array([r['sha256'] for r in rows]),
            transport_sha256=np.array(transport),roles=np.repeat('TRAIN',len(rows)),binding=np.array(binding))
    with np.load(path,allow_pickle=False) as a:check_chunk(a,binding,rows)
    return np.stack(crops),features


def parity(enc,rows,c,deadline):
    indices={r['parent_id']:i for i,r in enumerate(rows)};ids=c['parity_parents'];positions=[indices[p] for p in ids]
    previous=read(DATA_ROOT/'e71/fit_contract.json');expected={}
    for key,path,key_in in [('dino',DATA_ROOT/'e54/teacher.npz','features'),
                            ('clip',DATA_ROOT/'e71/clip_features.npz','features'),
                            ('dear',DATA_ROOT/'e79/dear_features.npz','features')]:
        resource_check(deadline)
        with np.load(path,allow_pickle=False) as a:
            full=a[key_in];expected[key]=np.stack([full[i,0] for i in positions if i<11630 or key=='dear'])
        del full
        if key!='dear':
            with np.load(DATA_ROOT/'e75/midd_features.npz',allow_pickle=False) as a:
                new=a[key];old_iter=iter(expected[key]);expected[key]=np.stack([next(old_iter) if i<11630 else new[i-11630,0] for i in positions])
    index=read(CROP_INDEX)['records'];packs=[];data_binding=digest(MANIFEST)
    for i in positions:
        resource_check(deadline);row=rows[i]
        if i<11630:packs.append(load_crops(index[row['parent_id']],data_binding)[0])
        else:
            body=Path(row['path']).read_bytes()
            if hashlib.sha256(body).hexdigest()!=row['sha256']:raise ValueError('new parity source changed')
            packs.append(crop_views(body,row['parent_id'])[0])
    actual={k:[] for k in WIDTHS}
    for start in range(0,len(packs),8):
        resource_check(deadline);part=enc.encode(np.stack(packs[start:start+8]))
        for k,v in part.items():actual[k].append(v)
    actual={k:np.concatenate(v) for k,v in actual.items()}
    head=joblib.load(previous['inputs']['reference']['path'])['head']
    with threadpool_limits(limits=2):old=head.predict_proba(expected['dino'])[:,1];new=head.predict_proba(actual['dino'])[:,1]
    error=float(np.max(np.abs(old-new)));changes={str(cut):int(np.sum((old>=cut)!=(new>=cut))) for cut in [REAL_CUT,AI_CUT]}
    features={k:float(np.max(np.abs(actual[k]-expected[k]))) for k in ['clip','dear']}
    return {'parents':len(ids),'reference_max_score_error':error,'decision_changes_by_cut':changes,
            'feature_max_errors':features,'passed':error<=5e-5 and not any(changes.values()) and max(features.values())<=1e-5}


def probe():
    c=validate()
    if PROBE.exists():raise FileExistsError('E84 probe already complete')
    started=time.monotonic();deadline=started+c['probe_max_seconds'];resource_check(deadline)
    rows=population();batches=windows(rows);binding=digest(CONTRACT);(ROOT/'chunks').mkdir(exist_ok=True)
    enc=Encoders();setup=time.monotonic()-started;check=parity(enc,rows,c,deadline);parity_seconds=time.monotonic()-started-setup
    if not check['passed']:
        result={'state':'E84_probe_failed','full_extraction_permitted':False,'parity':check,'setup_seconds':setup,'parity_seconds':parity_seconds}
    else:
        elapsed=0.;count=0;repeat=0.;peak=enc.memory();resumed=0
        for j,index in enumerate(c['probe_windows']):
            resource_check(deadline);batch=batches[index];path=ROOT/'chunks'/f'{index:05d}.npz'
            resumed+=int(path.exists())
            before=time.monotonic();crops,features=create_chunk(enc,batch,path,binding);elapsed+=time.monotonic()-before;count+=len(batch)
            if j==0:
                again=enc.encode(crops);repeat=max(float(np.max(np.abs(features[k]-again[k]))) for k in WIDTHS)
            peak=max(peak,enc.memory())
            if peak>c['mps_driver_limit_bytes']:raise RuntimeError('E84 memory budget exceeded')
            print(json.dumps({'E84_probe_windows':j+1,'parents':count,'seconds':round(elapsed)}),flush=True)
        projected=elapsed/count*len(rows)
        passed=repeat<=1e-5 and projected<=c['max_seconds'] and peak<=c['mps_driver_limit_bytes']
        result={'state':'E84_probe_passed' if passed else 'E84_probe_failed','full_extraction_permitted':passed,
                'parity':check,'setup_seconds':setup,'parity_seconds':parity_seconds,'probe_parents':count,
                'probe_encoding_seconds':elapsed,'resumed_probe_windows':resumed,'projected_encoding_seconds':projected,'repeat_max_feature_error':repeat,
                'mps_driver_peak_bytes':peak}
    result.update(contract_sha256=binding,total_seconds=time.monotonic()-started,new_view_classifier_scores=0,dev_final_image_reads=0)
    write_once(PROBE,result);write_once(EVIDENCE/'e84_probe.json',result)
    return result


def extract():
    c=validate();probe_result=read(PROBE)
    if digest(PROBE)!=digest(EVIDENCE/'e84_probe.json') or not probe_result['full_extraction_permitted']:raise ValueError('passed E84 probe required')
    if REPORT.exists():raise FileExistsError('E84 features already complete')
    started=time.monotonic();deadline=started+c['max_seconds'];resource_check(deadline)
    rows=population();binding=digest(CONTRACT);enc=Encoders();parts={k:[] for k in WIDTHS};chunks={};created=0;peak=enc.memory()
    for index,batch in enumerate(windows(rows)):
        resource_check(deadline);path=ROOT/'chunks'/f'{index:05d}.npz'
        if not path.exists():create_chunk(enc,batch,path,binding);created+=len(batch)
        with np.load(path,allow_pickle=False) as a:features=check_chunk(a,binding,batch)
        for k,v in features.items():parts[k].append(v)
        chunks[str(path)]=digest(path);peak=max(peak,enc.memory())
        if peak>c['mps_driver_limit_bytes']:raise RuntimeError('E84 MPS memory budget exceeded')
        if index%16==0:print(json.dumps({'E84_parents':min((index+1)*8,len(rows)),'total':len(rows),
            'created_this_run':created,'seconds':round(time.monotonic()-started),'mps_driver_peak_bytes':peak}),flush=True)
    arrays={k:np.concatenate(v) for k,v in parts.items()};parents=np.array([r['parent_id'] for r in rows])
    if any(v.shape!=(len(rows),WIDTHS[k]) for k,v in arrays.items()):raise ValueError('complete social TRAIN features required')
    if OUTPUT.exists():
        with np.load(OUTPUT,allow_pickle=False) as a:
            if str(a['binding'])!=binding or not np.array_equal(a['parents'],parents) or set(a['roles'])!={'TRAIN'} or \
                    any(not np.array_equal(a[k],v) for k,v in arrays.items()):raise ValueError('unreceipted full archive differs')
    else:save_npz(OUTPUT,**arrays,parents=parents,roles=np.repeat('TRAIN',len(rows)),binding=np.array(binding),condition=np.array('social_q75'))
    result={'state':'E84_social_TRAIN_features_complete','contract_sha256':binding,'parents':len(rows),
        'shapes':{k:list(v.shape) for k,v in arrays.items()},'feature_sha256':digest(OUTPUT),
        'created_parents_this_run':created,'seconds_this_run':time.monotonic()-started,'mps_driver_peak_bytes':peak,
        'chunks':chunks,'new_view_classifier_scores':0,'dev_final_image_reads':0,'downloads':0}
    write_once(REPORT,result);write_once(EVIDENCE/'e84_features.json',{k:v for k,v in result.items() if k!='chunks'}|{'report_sha256':digest(REPORT)})
    return {k:v for k,v in result.items() if k!='chunks'}

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=('freeze','probe','extract'))
    ROOT.mkdir(exist_ok=True)
    with (ROOT/'execution.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        print(json.dumps({'freeze':freeze,'probe':probe,'extract':extract}[parser.parse_args().stage](),indent=2))
