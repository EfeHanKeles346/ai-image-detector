"""Fixed TRAIN-research patch-drift localization audit; no mask input or training."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import socket
import time
import numpy as np
from PIL import Image
from sklearn.metrics import roc_auc_score
from experiments.e65_acquisition import digest, read, write_once
from experiments.e71_features import save_npz
from experiments.e72_acquisition import resource_check
from pixelproof import patch_drift, spatial_evaluation
from pixelproof.e92_demo import social_view
from pixelproof.e32_candidate import DINO_MODEL_ID, DINO_REPO_ID, DINO_WEIGHT_SHA256
from pixelproof.project_paths import DATA_ROOT, ML_ROOT

ROOT = DATA_ROOT/'e130'; EVIDENCE = ML_ROOT.parent/'evidence'; CONTRACT = ROOT/'contract.json'
CONDITIONS = ('original', 'jpeg75')
THRESHOLD = .0025980165154915157 / 2


def population(prepared, report):
    rows, cases = prepared['rows'], report['rows']
    if report.get('state') != 'E129_context_generation_complete' or len(rows) != 16 or len(cases) != 16 or \
            [r['index'] for r in rows] != list(range(16)) or [r['index'] for r in cases] != list(range(16)):
        raise ValueError('Complete fixed E128/E129 16-attempt population required')
    for row, case in zip(rows, cases, strict=True):
        if row['role'] != 'TRAIN_RESEARCH_PILOT' or row['seed'] != case['seed'] or \
                row['rendering'] != 'center_square_resized_512' or case['branch'] != 'fp16_sdpa' or \
                set(case['checks']) != {'background_exact','masked_pixels_changed','nonblank_output','safety_check_negative'} or \
                any(type(x) is not bool for x in case['checks'].values()) or \
                type(case['passed']) is not bool or case['passed'] != all(case['checks'].values()):
            raise ValueError('Role, rendering, seed or acceptance identity differs')
    return rows, cases


def freeze():
    prepared = DATA_ROOT/'e128/prepared.json'; report = DATA_ROOT/'e129/report.json'
    if digest(prepared) != read(EVIDENCE/'e128_context_inputs.json')['prepared_sha256'] or \
            digest(report) != read(EVIDENCE/'e129_context_replay.json')['report_sha256']:
        raise ValueError('Bound preparation/generation receipts differ')
    rows, cases = population(read(prepared), read(report))
    from huggingface_hub import snapshot_download
    backbone = Path(snapshot_download(DINO_REPO_ID, local_files_only=True))/'model.safetensors'
    if digest(backbone) != DINO_WEIGHT_SHA256: raise ValueError('Cached DINO identity differs')
    files = [Path(__file__), Path(patch_drift.__file__), Path(spatial_evaluation.__file__),
        Path(social_view.__code__.co_filename), prepared, report, backbone,
        DATA_ROOT/'e128/contract.json', DATA_ROOT/'e129/contract.json',
        EVIDENCE/'e128_context_inputs.json', EVIDENCE/'e129_context_replay.json',
        EVIDENCE/'trail_reference_receipt.json', ML_ROOT/'third_party/TRAIL_LICENSE.txt']
    for row, case in zip(rows, cases, strict=True):
        for item in row['files'].values():
            path = Path(item['path'])
            if digest(path) != item['sha256']: raise ValueError('Prepared file differs')
            files.append(path)
        folder = DATA_ROOT/'e129/fp16_sdpa'/f"{case['index']:03d}"
        if read(folder/'case.json') != case: raise ValueError('Case receipt differs')
        files.append(folder/'case.json')
        for name in ('raw', 'composite'):
            path = folder/(name+'.png')
            if digest(path) != case[name+'_sha256']: raise ValueError('Generated body differs')
            files.append(path)
    c = {'state': 'E130_mask_blind_patch_drift_audit_registered', 'inputs': {str(p):digest(p) for p in files},
        'backbone': str(backbone), 'parents':16, 'accepted_generations':sum(x['passed'] for x in cases),
        'geometry':'All512px square inputs resized448 LANCZOS. Existing frozen timm DINOv2-S/14, block11 one-based (index10), final LayerNorm, CLS excluded. 32x32 patch maps repeated16x to original512 geometry.',
        'method':'Global two-level Haar attenuation alpha.2, one-minus-cosine divided by2, reflect3x3 median. Both labels/controls and original/JPEG75 conditions identical. No per-image normalization, threshold search or oracle mask use.',
        'threshold':THRESHOLD, 'threshold_status':'Half of external TRAIL DINOv3-7B CocoGlide cut; NOT calibrated or assumed transferable to our smaller DINOv2 model. Fixed diagnostic only.',
        'boundary_width_pixels':8, 'conditions':CONDITIONS,
        'population':'All16 authentic and classical controls. Composite scoring only for existing E129 accepted generations, with every rejected attempt retained in counts. No refill or training admission.',
        'guards':'Finite384-D patch grids; repeated original in batch[original,perturbed,original] parity<=1e-5; exact local composite mask/background provenance before metrics. All maps locked before masks enter metrics.',
        'metrics':'Per-parent regional flags and full/interior-background ranking; negatives always AI-negative. Paired original/edited spatial-ranking contrast plus matched Haar pixel-RMS, radial-center and constant ranking baselines; no pooled-pixel independent intervals. Original and JPEG75 separately. Pixel-RMS has no transferred classification threshold.',
        'cache':'Retain original patch tokens for each scored view locally, with image/role/condition and code hashes; no automatic later training authorization.',
        'max_seconds':2400, 'mps_limit_bytes':6*1024**3, 'downloads':0, 'training_admission':False,
        'promotion_allowed':False, 'limits':'Small correlated TRAIN research, one old editor, simple masks, conditional accepted-generation subset. This is an adapted edit-response baseline, not an exact TRAIL reproduction, calibrated AI probability, semantic annotation or independent detector evidence.'}
    ROOT.mkdir(exist_ok=True); write_once(CONTRACT,c)
    write_once(EVIDENCE/'e130_patch_drift_contract.json',{k:v for k,v in c.items() if k not in ('inputs','backbone')}|{'contract_sha256':digest(CONTRACT)})
    return {'parents':16,'accepted_generations':c['accepted_generations'],'training_admission':False}


def validate():
    c=read(CONTRACT)
    if digest(CONTRACT)!=read(EVIDENCE/'e130_patch_drift_contract.json')['contract_sha256']:
        raise ValueError('Audit contract differs')
    for path,sha in c['inputs'].items():
        if digest(path)!=sha: raise ValueError('Audit input differs')
    return c


def decode(path):
    with Image.open(path) as image:
        if image.mode!='RGB' or image.size!=(512,512): raise ValueError('Exact prepared512 RGB required')
        return image.copy()


def encoded_maps(model, torch, image, device, mean, std):
    value=np.asarray(image.resize((448,448),Image.Resampling.LANCZOS))
    plain=value.astype(np.float32)/255.
    perturbed=patch_drift.haar_attenuate(value)
    tensor=torch.from_numpy(np.stack([plain,perturbed,plain])).to(device).permute(0,3,1,2)
    with torch.inference_mode():
        block=model.forward_intermediates((tensor-mean)/std,indices=[10],norm=True,
            return_prefix_tokens=False,intermediates_only=True)[0]
    token=block.float().cpu().numpy().transpose(0,2,3,1)
    if token.shape!=(3,32,32,384) or not np.isfinite(token).all(): raise ValueError('Invalid DINO patch grid')
    error=float(np.abs(token[0]-token[2]).max())
    if error>1e-5: raise ValueError('Repeated original patch tokens differ')
    grid=patch_drift.median3(patch_drift.half_cosine_drift(token[0],token[1]))
    pixel=patch_drift.median3(patch_drift.pixel_response(value,perturbed))
    return grid,token[0].copy(),error,pixel


def audit():
    c=validate(); start=time.monotonic(); deadline=start+c['max_seconds']; resource_check(deadline)
    if (ROOT/'report.json').exists() or (ROOT/'scores.npz').exists(): raise FileExistsError('Audit already scored')
    import torch,timm
    from safetensors.torch import load_file
    from timm.models.vision_transformer import checkpoint_filter_fn
    torch.set_num_threads(2)
    device=torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    model=timm.create_model(DINO_MODEL_ID,pretrained=False,num_classes=0,img_size=448)
    model.load_state_dict(checkpoint_filter_fn(load_file(c['backbone']),model),strict=True)
    config=timm.data.resolve_data_config({},model=model)
    model=model.to(device).eval()
    for p in model.parameters(): p.requires_grad_(False)
    mean=torch.tensor(config['mean'],device=device).view(1,3,1,1)
    std=torch.tensor(config['std'],device=device).view(1,3,1,1)
    rows,cases=population(read(DATA_ROOT/'e128/prepared.json'),read(DATA_ROOT/'e129/report.json'))
    maps={}; pixel_maps={}; tokens={}; identities=[]; parity=0.; peak=0
    for row,case in zip(rows,cases,strict=True):
        paths={'authentic':Path(row['files']['original']['path']), 'classical_edit':Path(row['files']['classic']['path'])}
        if case['passed']: paths['ai_composite']=DATA_ROOT/'e129/fp16_sdpa'/f"{row['index']:03d}"/'composite.png'
        for variant,path in paths.items():
            source=decode(path)
            for condition in CONDITIONS:
                resource_check(deadline)
                image=source if condition=='original' else social_view(source)
                key=f"{row['index']:03d}_{variant}_{condition}"
                grid,token,error,pixel=encoded_maps(model,torch,image,device,mean,std)
                maps[key]=grid; pixel_maps[key]=pixel; tokens[key]=token; parity=max(parity,error)
                identities.append({'key':key,'index':row['index'],'parent_id':row['parent_id'],'role':row['role'],
                    'condition':condition,'variant':variant,'source_body_sha256':digest(path)})
                if device.type=='mps':
                    peak=max(peak,torch.mps.driver_allocated_memory())
                    if peak>c['mps_limit_bytes']: raise MemoryError('Patch-drift MPS budget exceeded')
        print(json.dumps({'E130_scored_parent':row['index']+1,'parents':16,'seconds':round(time.monotonic()-start)}),flush=True)
    save_npz(ROOT/'scores.npz',**maps,**{'pixel__'+key:value for key,value in pixel_maps.items()},contract_sha256=digest(CONTRACT))
    save_npz(ROOT/'tokens.npz',**tokens,contract_sha256=digest(CONTRACT))
    write_once(ROOT/'scoring.json',{'contract_sha256':digest(CONTRACT),'identities':identities,
        'scores_sha256':digest(ROOT/'scores.npz'),'tokens_sha256':digest(ROOT/'tokens.npz')})
    # Scores are fixed above. Only now do evaluation masks enter the computation.
    results=[]
    for row,case in zip(rows,cases,strict=True):
        with Image.open(row['files']['mask']['path']) as image: raw_mask=np.asarray(image)
        if raw_mask.shape!=(512,512) or not np.isin(raw_mask,[0,255]).all(): raise ValueError('Exact binary mask required')
        mask=raw_mask==255
        if case['passed']:
            raw=np.asarray(decode(DATA_ROOT/'e129/fp16_sdpa'/f"{row['index']:03d}"/'raw.png'))
            composite=np.asarray(decode(DATA_ROOT/'e129/fp16_sdpa'/f"{row['index']:03d}"/'composite.png'))
            original=np.asarray(decode(row['files']['original']['path']))
            if not np.array_equal(composite,np.where(mask[:,:,None],raw,original)):
                raise ValueError('Exact hard-composite provenance differs')
        for condition in CONDITIONS:
            def dense(name):
                key=f"{row['index']:03d}_{name}_{condition}"
                return patch_drift.expand_grid(maps[key],(512,512)) if key in maps else None
            metrics=spatial_evaluation.evaluate_triplet(dense('authentic'),dense('classical_edit'),dense('ai_composite'),
                mask,threshold=c['threshold'],boundary_width=c['boundary_width_pixels'])
            # The authentic mask here denotes an aligned location, never an AI label.
            metrics['aligned_region_auc_diagnostic']={name:float(roc_auc_score(mask.ravel(),dense(name).ravel()))
                for name in metrics['maps']}
            metrics['pixel_response_aligned_auc_diagnostic']={name:float(roc_auc_score(mask.ravel(),
                patch_drift.expand_grid(pixel_maps[f"{row['index']:03d}_{name}_{condition}"],(512,512)).ravel())) for name in metrics['maps']}
            yy,xx=np.mgrid[:512,:512]
            center=np.clip(1-np.sqrt(((xx+.5-256)/256)**2+((yy+.5-256)/256)**2),0,1)
            metrics['radial_center_auc']=float(roc_auc_score(mask.ravel(),center.ravel()))
            metrics['constant_auc']=.5
            results.append({'index':row['index'],'condition':condition,'metrics':metrics})
    write_once(ROOT/'measurements.json',{'contract_sha256':digest(CONTRACT),'results':results})
    summary={}
    for condition in CONDITIONS:
        group=[r['metrics'] for r in results if r['condition']==condition]
        summary[condition]={}
        for variant in ('authentic','classical_edit','ai_composite'):
            selected=[r['maps'][variant] for r in group if variant in r['maps']]
            fields={'flagged_area_fraction':[r['flagged_area_fraction'] for r in selected],
                'interior_flagged_fraction':[r['regions']['interior']['flagged_fraction'] for r in selected],
                'background_flagged_fraction':[r['regions']['background']['flagged_fraction'] for r in selected],
                'pixel_auc':[r['all_pixel_ranking']['auc'] for r in selected],
                'interior_background_auc':[r['interior_background_ranking']['auc'] for r in selected]}
            summary[condition][variant]={'parents':len(selected),'metrics':{key:{'available_parents':len(valid),
                'mean':float(np.mean(valid)) if valid else None} for key,values in fields.items() for valid in [[v for v in values if v is not None]]}}
        paired=[r['aligned_region_auc_diagnostic'] for r in group if r['composite_available']]
        summary[condition]['accepted_parent_spatial_contrast']={'parents':len(paired),
            'AI_minus_authentic_auc':float(np.mean([r['ai_composite']-r['authentic'] for r in paired])) if paired else None,
            'AI_minus_classical_edit_auc':float(np.mean([r['ai_composite']-r['classical_edit'] for r in paired])) if paired else None,
            'interpretation':'Within-image aligned-region ranking differences, not REAL-versus-AI classification accuracy.'}
        accepted=[r for r in group if r['composite_available']]
        summary[condition]['accepted_parent_ranking_baselines']={'parents':len(accepted),
            'pixel_response_auc':float(np.mean([r['pixel_response_aligned_auc_diagnostic']['ai_composite'] for r in accepted])) if accepted else None,
            'radial_center_auc':float(np.mean([r['radial_center_auc'] for r in accepted])) if accepted else None,
            'constant_auc':.5 if accepted else None,
            'interpretation':'Same accepted parents/masks. Pixel-change and location controls are rankings, not calibrated AI detectors.'}
    result={'state':'E130_patch_drift_audit_complete','contract_sha256':digest(CONTRACT),'summary':summary,
        'parents':16,'accepted_composites':sum(x['passed'] for x in cases),'excluded_generations':sum(not x['passed'] for x in cases),
        'max_duplicate_token_error':parity,'peak_mps_bytes':peak,'seconds':time.monotonic()-start,
        'scoring_sha256':digest(ROOT/'scoring.json'),'measurements_sha256':digest(ROOT/'measurements.json'),
        'training_admission':False,'promotion_allowed':False,'limits':c['limits']}
    write_once(ROOT/'report.json',result);write_once(EVIDENCE/'e130_patch_drift_audit.json',result)
    message='\n### E130 patch-drift diagnostic completed\n\n'+json.dumps(result,sort_keys=True)+'\n\nNo trained localizer, calibrated probability, independent proof or training admission is claimed.\n'
    for path in (ML_ROOT.parent/'PLAN.md',ML_ROOT.parent/'HISTORY.md',ML_ROOT/'EXPERIMENTS.md',ML_ROOT.parent/'DATASETS.md'):
        with path.open('a') as f:fcntl.flock(f,fcntl.LOCK_EX);f.write(message)
    return result


if __name__=='__main__':
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    def denied(*a,**k): raise RuntimeError('Patch-drift audit is offline')
    socket.socket.connect=denied;socket.socket.connect_ex=denied;socket.create_connection=denied
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=['freeze','audit'])
    ROOT.mkdir(exist_ok=True)
    with (ROOT/'execution.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        print(json.dumps({'freeze':freeze,'audit':audit}[parser.parse_args().stage](),indent=2))
