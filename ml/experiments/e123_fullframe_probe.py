"""TRAIN-only uncropped available-frame CLIP feasibility; preserve old center-vector parity."""
import argparse
import fcntl
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
import socket
import time
import numpy as np
from PIL import Image,ImageOps
from experiments import e42_features as views
from experiments.e65_acquisition import digest,read,write_once
from experiments.e65_diagnostic import social_q75_bytes
from experiments.e71_features import load_encoder,save_npz
from experiments.e72_acquisition import resource_check
from experiments.e75_features import crop_views
from experiments.e84_features import social_crops
from experiments.e112_context_features import validate as validate_old
from pixelproof.e32_candidate import standardized_array
from pixelproof.project_paths import DATA_ROOT,ML_ROOT

ROOT=DATA_ROOT/'e123';EVIDENCE=ML_ROOT.parent/'evidence';CONTRACT=ROOT/'contract.json'


def fullframe_array(image):
    # Fixed warp instead of source-aspect-ratio padding bars; no area is cropped.
    image=ImageOps.exif_transpose(image).convert('RGB').resize((224,224),Image.Resampling.LANCZOS)
    stream=BytesIO();image.save(stream,format='JPEG',quality=90,subsampling=0,optimize=False,progressive=False)
    with Image.open(BytesIO(stream.getvalue())) as decoded:return np.asarray(decoded.convert('RGB')).copy()


def transported_images(body,parent):
    with Image.open(BytesIO(body)) as im:image=ImageOps.exif_transpose(im).convert('RGB')
    stream=BytesIO();image.save(stream,format='JPEG',quality=75,subsampling=2,optimize=False)
    with Image.open(BytesIO(stream.getvalue())) as im:q75=im.convert('RGB')
    with Image.open(BytesIO(social_q75_bytes(BytesIO(body)))) as im:social=im.convert('RGB')
    return [views.transport_image(image,'clean'),views.transport_image(image,views.assigned_transport(parent)),
        views.transport_image(q75,'clean'),views.transport_image(social,'clean')]


def selected(rows):
    if any(str(r['role']).upper()!='TRAIN' for r in rows):raise ValueError('TRAIN only')
    groups=sorted({(r['label'],r['source']) for r in rows})
    return [min((i for i,r in enumerate(rows) if (r['label'],r['source'])==g),
        key=lambda i:hashlib.sha256(('E123|'+rows[i]['parent_id']).encode()).hexdigest()) for g in groups]


def freeze():
    old=validate_old();receipt=DATA_ROOT/'e112/report.json'
    if digest(receipt)!=digest(EVIDENCE/'e112_context_features.json') or \
            digest(DATA_ROOT/'e112/features.npz')!=read(receipt)['feature_sha256']:
        raise ValueError('Complete raw center reference required')
    files=[Path(__file__),ML_ROOT/'experiments/e112_context_features.py',Path(standardized_array.__code__.co_filename),
        Path(social_q75_bytes.__code__.co_filename),Path(views.__file__),ML_ROOT/'experiments/e75_features.py',
        ML_ROOT/'experiments/e84_features.py',DATA_ROOT/'e112/contract.json',receipt,DATA_ROOT/'e112/features.npz',
        EVIDENCE/'e121b_context_transfer.json']
    c={'state':'E123_uncropped_frame_TRAIN_probe_registered','inputs':old['inputs']|{str(p):digest(p) for p in files},
        'selected_indices':selected(old['rows']),'conditions':['clean','assigned_transport','q75','social_q75'],
        'selection':'One hash-selected admitted TRAIN original per complete class/source group, all four conditions; no DEV/gallery/final use or image/score selection.',
        'new_view':'Full available transported frame warped to224x224 with LANCZOS, then identical JPEG90/4:4:4 center-branch encoding. No padding bars or crop; aspect distortion is a limitation. Resized publisher inputs remain resized inputs, not recovered native originals.',
        'encoder':'Unchanged pinned CLIP. Batch3=[old center,new full frame,new full frame] for exact center-reference and same-batch repeated-new-vector checks. No classification/PCA fitting.',
        'acceptance':'All selected sources/conditions; old center pixels match native E75/E84 pipeline exactly; old center CLIP error<=1e-5; repeated new CLIP error<=1e-5; all finite. Record full/center distances and actual cost without claiming improved accuracy.',
        'max_seconds':1800,'mps_limit_bytes':6*1024**3,'parity_max_abs':1e-5,'downloads':0,'model_scores':0,
        'next':'Only a successful probe permits separately registering full-population extraction and paired full-frame versus center-only capacity controls. Keep E103 base and all AI/correct-REAL retention guards. No final/gallery access.',
        'limits':'Source aspect ratios are class-confounded; full-frame warp can create new biases. This feasibility test cannot establish benefit or explain the known missed DEV example.'}
    ROOT.mkdir(exist_ok=True);write_once(CONTRACT,c)
    write_once(EVIDENCE/'e123_fullframe_contract.json',c|{'contract_sha256':digest(CONTRACT)})
    return {'parents':len(c['selected_indices']),'views':4*len(c['selected_indices'])}


def probe():
    import torch
    c=read(CONTRACT)
    if digest(CONTRACT)!=read(EVIDENCE/'e123_fullframe_contract.json')['contract_sha256']:raise ValueError('Full-frame contract differs')
    for p,s in c['inputs'].items():
        if digest(p)!=s:raise ValueError('Full-frame bound input differs')
    if (ROOT/'report.json').exists():raise FileExistsError('Feasibility already completed')
    start=time.monotonic();deadline=start+c['max_seconds'];resource_check(deadline);torch.set_num_threads(2)
    all_rows=read(DATA_ROOT/'e112/contract.json')['rows']
    with np.load(DATA_ROOT/'e112/features.npz',allow_pickle=False) as a:
        if list(a['parents'])!=[r['parent_id'] for r in all_rows]:raise ValueError('Reference parent order differs')
        reference=a['raw'][c['selected_indices'],:,0].copy()
    model,preprocess,device=load_encoder();new=[];facts=[];max_old=0.;max_repeat=0.;peak=0;loop_start=time.monotonic()
    with torch.inference_mode():
        for j,i in enumerate(c['selected_indices']):
            row=all_rows[i];resource_check(deadline)
            if digest(row['path'])!=row['sha256']:raise ValueError('TRAIN source changed')
            body=Path(row['path']).read_bytes();images=transported_images(body,row['parent_id'])
            expected_pixels=np.concatenate([crop_views(body,row['parent_id']),social_crops(body)[0][None]])[:,0]
            parent=[]
            for k,image in enumerate(images):
                center=standardized_array(image);full=fullframe_array(image)
                if not np.array_equal(center,expected_pixels[k]):raise ValueError('Native center transform differs')
                raw=model.encode_image(torch.stack([preprocess(Image.fromarray(x)) for x in (center,full,full)]).to(device)).float().cpu().numpy()
                if not np.isfinite(raw).all():raise ValueError('Nonfinite full-frame vectors')
                old=float(np.abs(raw[0]-reference[j,k]).max());repeat=float(np.abs(raw[1]-raw[2]).max())
                max_old=max(max_old,old);max_repeat=max(max_repeat,repeat)
                if max(old,repeat)>c['parity_max_abs']:raise ValueError('Center/repeated full-frame vector parity failed')
                parent.append(raw[1]);facts.append({'label':row['label'],'source':row['source'],'condition':c['conditions'][k],
                    'full_minus_center_l2':float(np.linalg.norm(raw[1]-raw[0]))})
                peak=max(peak,torch.mps.driver_allocated_memory() if device.type=='mps' else 0)
                if peak>c['mps_limit_bytes']:raise RuntimeError('Full-frame MPS limit')
            new.append(parent);print(json.dumps({'E123_completed_parents':j+1,'total':len(c['selected_indices']),'max_old_error':max_old}),flush=True)
    loop_seconds=time.monotonic()-loop_start
    save_npz(ROOT/'features.npz',full=np.array(new,dtype=np.float32),reference_center=reference,
        parents=np.array([all_rows[i]['parent_id'] for i in c['selected_indices']]),roles=np.repeat('TRAIN',len(new)),
        conditions=np.array(c['conditions']),binding=digest(CONTRACT))
    result={'state':'E123_fullframe_feasibility_passed','contract_sha256':digest(CONTRACT),'parents':len(new),'views':len(facts),
        'features_sha256':digest(ROOT/'features.npz'),'max_center_vector_error':max_old,'max_repeat_error':max_repeat,
        'peak_mps_bytes':peak,'seconds':time.monotonic()-start,'loop_seconds':loop_seconds,'source_view_distances':facts,
        'model_scores':0,'fit_allowed':False,'limits':c['limits'],'next':c['next']}
    write_once(ROOT/'report.json',result);write_once(EVIDENCE/'e123_fullframe_probe.json',result)
    return {k:v for k,v in result.items() if k!='source_view_distances'}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['freeze','probe']);args=p.parse_args()
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    def denied(*a,**kw):raise RuntimeError('Full-frame probe is offline')
    socket.socket.connect=denied;socket.socket.connect_ex=denied;socket.create_connection=denied
    ROOT.mkdir(exist_ok=True)
    with (ROOT/'execution.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        print(json.dumps({'freeze':freeze,'probe':probe}[args.stage](),indent=2))
