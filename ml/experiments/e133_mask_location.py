"""One fixed off-center generation per parent, evaluated by frozen E132 fold heads."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import socket
import time
import numpy as np
from PIL import Image, ImageFilter
from sklearn.metrics import roc_auc_score
from experiments.e65_acquisition import digest, read, write_once
from experiments.e71_features import save_npz
from experiments.e72_acquisition import resource_check
from experiments import e118_inpaint_assets as assets, e119_inpainting_pilot as pilot
from experiments import e120_inpaint_numerics as numerics, e130_patch_drift_audit as encoder
from experiments import e132_patch_learning as learned
from pixelproof import mask_location, holdout_linear, patch_drift, spatial_evaluation
from pixelproof.project_paths import DATA_ROOT, ML_ROOT

ROOT = DATA_ROOT/'e133'; EVIDENCE = ML_ROOT.parent/'evidence'; CONTRACT = ROOT/'contract.json'
CONDITIONS = ('original', 'jpeg75')
VARIANTS = ('authentic', 'classical_edit', 'ai_composite')


def journal(title, value):
    note = '\n### E133 '+title+'\n\n'+json.dumps(value, sort_keys=True)+'\n'
    for path in (ML_ROOT.parent/'HISTORY.md', ML_ROOT/'EXPERIMENTS.md', ML_ROOT.parent/'DATASETS.md'):
        with path.open('a') as f:
            fcntl.flock(f, fcntl.LOCK_EX); f.write(note)


def freeze():
    prior = learned.validate(); assets.validate(); numerical = numerics.validate()
    if digest(learned.ROOT/'report.json') != digest(EVIDENCE/'e132_patch_learning.json'):
        raise ValueError('Complete bound E132 report required')
    report = read(learned.ROOT/'report.json'); locked = read(learned.ROOT/'locked_scores.json')
    if digest(learned.ROOT/'locked_scores.json') != report['locked_scores_sha256'] or \
            digest(learned.ROOT/'scores.npz') != locked['scores_sha256']:
        raise ValueError('Frozen E132 score receipt differs')
    rows = prior['rows']; geometry = []; files = [Path(__file__), Path(mask_location.__file__),
        Path(numerics.__file__), assets.CONTRACT, assets.ROOT/'download.json',
        learned.ROOT/'report.json', learned.ROOT/'locked_scores.json', learned.ROOT/'scores.npz',
        learned.ROOT/'measurements.json', EVIDENCE/'e132_patch_learning.json', learned.CONTRACT]
    if digest(learned.ROOT/'measurements.json') != report['measurements_sha256']:
        raise ValueError('Old matched measurements differ')
    for row, fold in zip(rows, prior['folds'], strict=True):
        with Image.open(row['files']['mask']['path']) as im:
            _, position = mask_location.corner_mask(np.asarray(im), row['index'])
        geometry.append(position)
        fit = locked['fits'][str(fold)]
        if row['index'] not in fit['held_parents'] or row['index'] in fit['train_parents']:
            raise ValueError('Parent was not excluded from frozen head')
        path = learned.ROOT/f'fold{fold}.npz'
        if digest(path) != fit['artifact_sha256']:
            raise ValueError('Frozen head differs')
        files.append(path)
    c = {'state': 'E133_frozen_head_location_challenge_registered',
        'inputs': prior['inputs'] | numerical['inputs'] | {str(p): digest(p) for p in files},
        'rows': rows, 'folds': prior['folds'], 'geometry': geometry, 'parents': 16,
        'intervention': 'Translate unchanged tight binary masks to index%4 corners with16px margin. Preserve exact area/shape, image bodies, seeds, prompt and generation parameters. Four parents per corner, one placement per parent.',
        'generation': read(DATA_ROOT/'e129/contract.json')['generation'],
        'generation_seconds': 2400, 'score_seconds': 900, 'mps_limit_bytes': 10*1024**3,
        'scoring': 'Use the existing E132 source-excluding head per parent and exact E130 encoder. No fitting, threshold changes or position/mask inference input. Original/JPEG75; authentic-token parity<=1e-5 and authentic-score parity<=1e-6. Fixed0.5 diagnostic cut.',
        'accounting': 'All16 authentic/classical negatives; only previously accepted generated composites. Retain every failed attempt, no reroll/refill. Lock all maps before new-mask metrics. All16-pass generation is reported, not a requirement to hide failed-parent negatives.',
        'metrics': 'Per-parent full/interior-background AUC, IoU and flagged area; AI-authentic/classical aligned contrast; same-mask radial and constant rankings; paired E132 original-placement metrics restricted to accepted parents. No pixel-IID intervals.',
        'downloads': 0, 'training_admission': False, 'promotion_allowed': False,
        'limits': 'Consumed sixteen-parent MIDD research with one old editor and known source-excluding heads. New placement changes edited semantic content and diffusion output; not pure causal position attribution or a fully crossed corner experiment. Intended masks are not semantic-change annotations. No fresh-source/editor proof, calibration or serving promotion.'}
    ROOT.mkdir(exist_ok=True); write_once(CONTRACT, c)
    public = {k:v for k,v in c.items() if k not in ('inputs','rows')} | {'contract_sha256':digest(CONTRACT)}
    write_once(EVIDENCE/'e133_mask_location_contract.json', public)
    return {'parents': 16, 'corners': 4, 'contract_sha256': digest(CONTRACT)}


def validate():
    c = read(CONTRACT)
    if digest(CONTRACT) != read(EVIDENCE/'e133_mask_location_contract.json')['contract_sha256']:
        raise ValueError('Location contract differs')
    for path, sha in c['inputs'].items():
        if digest(path) != sha:
            raise ValueError('Location input differs')
    return c


def prepare():
    c = validate(); resource_check(time.monotonic()+300)
    write_once(ROOT/'preparation_started.json', {'contract_sha256': digest(CONTRACT)})
    rows = []
    for old, geometry in zip(c['rows'], c['geometry'], strict=True):
        with Image.open(old['files']['mask']['path']) as im:
            mask, observed = mask_location.corner_mask(np.asarray(im), old['index'])
        if observed != geometry:
            raise ValueError('Prespecified geometry differs')
        authentic = encoder.decode(old['files']['original']['path']); mask_image = Image.fromarray(mask)
        classic = pilot.composite(authentic, authentic.filter(ImageFilter.GaussianBlur(6)), mask_image)
        if not np.array_equal(np.asarray(classic)[mask==0], np.asarray(authentic)[mask==0]):
            raise ValueError('Classical background differs')
        folder = ROOT/'parents'/f"{old['index']:03d}"; folder.mkdir(parents=True)
        files = {key:old['files'][key] for key in ('original','AI_negative_target')}
        for name, image in (('mask',mask_image),('classic',classic)):
            path = folder/(name+'.png'); image.save(path)
            files[name] = {'path':str(path), 'sha256':digest(path)}
        rows.append(old | {'files':files, 'location_intervention':geometry})
    write_once(ROOT/'prepared.json', {'rows':rows, 'contract_sha256':digest(CONTRACT)})
    result = {'parents':16, 'prepared_sha256':digest(ROOT/'prepared.json'), 'downloads':0,
              'contract_sha256':digest(CONTRACT), 'training_admission':False}
    write_once(EVIDENCE/'e133_location_preparation.json', result); journal('preparation complete',result)
    return result


def prepared_rows():
    if digest(ROOT/'prepared.json') != read(EVIDENCE/'e133_location_preparation.json')['prepared_sha256']:
        raise ValueError('Location preparation differs')
    rows = read(ROOT/'prepared.json')['rows']
    for row in rows:
        for f in row['files'].values():
            if digest(f['path']) != f['sha256']:
                raise ValueError('Prepared pixels differ')
    return rows


def generate():
    c = validate(); rows = prepared_rows(); start = time.monotonic()
    for row in read(assets.CONTRACT)['files']:
        assets.verify(assets.ROOT/'model'/row['rfilename'], row)
    write_once(ROOT/'generation_started.json', {'contract_sha256':digest(CONTRACT)})
    cases, peak = numerics.run_cases('fp16_sdpa', rows, ROOT, start+c['generation_seconds'], c['mps_limit_bytes'])
    accepted = mask_location.accepted_indices(cases)
    result = {'state':'E133_location_generation_complete', 'rows':cases, 'accepted':accepted,
              'parents':16, 'passed':len(accepted), 'full16_engineering_gate_passed':len(accepted)==16,
              'seconds':time.monotonic()-start, 'peak_mps_bytes':peak, 'contract_sha256':digest(CONTRACT),
              'downloads':0, 'training_admission':False, 'promotion_allowed':False}
    write_once(ROOT/'generation.json', result)
    public = {k:v for k,v in result.items() if k not in ('rows','accepted')} | {'generation_sha256':digest(ROOT/'generation.json')}
    write_once(EVIDENCE/'e133_location_generation.json', public); journal('generation complete',public)
    return public


def verify_cases():
    if digest(ROOT/'generation.json') != read(EVIDENCE/'e133_location_generation.json')['generation_sha256']:
        raise ValueError('Generation receipt differs')
    cases = read(ROOT/'generation.json')['rows']; accepted = mask_location.accepted_indices(cases)
    for case in cases:
        folder = ROOT/'fp16_sdpa'/f"{case['index']:03d}"
        if read(folder/'case.json') != case:
            raise ValueError('Case receipt differs')
        for name in ('raw','composite'):
            if name+'_sha256' in case and digest(folder/(name+'.png')) != case[name+'_sha256']:
                raise ValueError('Generated body differs')
    return accepted


def summarize(results, old, accepted):
    output = {}
    for condition in CONDITIONS:
        group = [r for r in results if r['condition']==condition]; valid = [r for r in group if r['index'] in accepted]
        entry = {'all_parents':len(group), 'accepted_parents':len(valid)}
        for variant in VARIANTS:
            selected = [r['metrics']['maps'][variant] for r in group if variant in r['metrics']['maps']]
            def mean(values):
                values = [v for v in values if v is not None]
                return float(np.mean(values)) if values else None
            entry[variant] = {'parents':len(selected),
                'mean_flagged_area':mean([r['flagged_area_fraction'] for r in selected]),
                'mean_pixel_auc':mean([r['all_pixel_ranking']['auc'] for r in selected]),
                'mean_interior_background_auc':mean([r['interior_background_ranking']['auc'] for r in selected]),
                'mean_iou':mean([r['iou'] for r in selected])}
        previous = [r for r in old if r['condition']==condition and r['index'] in accepted]
        entry['matched_accepted_comparison'] = {
            'parents':len(valid),
            'old_E132_mean_pixel_auc':mean([r['metrics']['maps']['ai_composite']['all_pixel_ranking']['auc'] for r in previous]),
            'new_radial_center_auc':mean([r['radial_center_auc'] for r in valid]),
            'constant_auc':.5 if valid else None,
            'AI_minus_authentic_aligned_auc':mean([r['aligned_auc']['ai_composite']-r['aligned_auc']['authentic'] for r in valid]),
            'AI_minus_classical_aligned_auc':mean([r['aligned_auc']['ai_composite']-r['aligned_auc']['classical_edit'] for r in valid])}
        output[condition] = entry
    return output


def score():
    c = validate(); rows = prepared_rows(); accepted = verify_cases(); start = time.monotonic()
    deadline = start+c['score_seconds']; resource_check(deadline)
    write_once(ROOT/'scoring_started.json', {'contract_sha256':digest(CONTRACT), 'generation_sha256':digest(ROOT/'generation.json')})
    import torch, timm
    from safetensors.torch import load_file
    from timm.models.vision_transformer import checkpoint_filter_fn
    torch.set_num_threads(2); device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    model = timm.create_model(encoder.DINO_MODEL_ID, pretrained=False, num_classes=0, img_size=448)
    model.load_state_dict(checkpoint_filter_fn(load_file(read(encoder.CONTRACT)['backbone']), model), strict=True)
    config = timm.data.resolve_data_config({},model=model); model = model.to(device).eval()
    for p in model.parameters():p.requires_grad_(False)
    mean = torch.tensor(config['mean'],device=device).view(1,3,1,1)
    std = torch.tensor(config['std'],device=device).view(1,3,1,1)
    maps = {}; token_error = 0.; score_error = 0.; peak = 0
    with np.load(DATA_ROOT/'e130/tokens.npz',allow_pickle=False) as previous_tokens, \
            np.load(learned.ROOT/'scores.npz',allow_pickle=False) as previous_scores:
        for row, fold in zip(rows,c['folds'],strict=True):
            paths = {'authentic':Path(row['files']['original']['path']), 'classical_edit':Path(row['files']['classic']['path'])}
            if row['index'] in accepted:
                paths['ai_composite'] = ROOT/'fp16_sdpa'/f"{row['index']:03d}"/'composite.png'
            with np.load(learned.ROOT/f'fold{fold}.npz',allow_pickle=False) as head:
                if str(head['contract_sha256']) != digest(learned.CONTRACT):raise ValueError('Head contract differs')
                for variant,path in paths.items():
                    source = encoder.decode(path)
                    for condition in CONDITIONS:
                        resource_check(deadline); key = f"{row['index']:03d}_{variant}_{condition}"
                        image = source if condition=='original' else encoder.social_view(source)
                        _,tokens,_,_ = encoder.encoded_maps(model,torch,image,device,mean,std)
                        pred = holdout_linear.predict((tokens.reshape(-1,384)-head['center'])/head['scale'],head['parameters']).reshape(32,32)
                        if variant=='authentic':
                            token_error = max(token_error,float(np.abs(tokens-previous_tokens[key]).max()))
                            score_error = max(score_error,float(np.abs(pred-previous_scores[key]).max()))
                            if token_error>1e-5 or score_error>1e-6:raise ValueError('Authentic replay differs')
                        maps[key] = pred
                        if device.type=='mps':
                            peak = max(peak,torch.mps.driver_allocated_memory())
                            if peak>6*1024**3:raise MemoryError('Location-score MPS budget exceeded')
            print(json.dumps({'E133_scored_parent':row['index']+1,'parents':16}),flush=True)
    save_npz(ROOT/'scores.npz',**maps,contract_sha256=digest(CONTRACT))
    write_once(ROOT/'locked_scores.json', {'contract_sha256':digest(CONTRACT),'scores_sha256':digest(ROOT/'scores.npz'),
        'generation_sha256':digest(ROOT/'generation.json'),'accepted_indices':accepted})
    results = []; yy,xx = np.mgrid[:512,:512]
    radial = np.clip(1-np.sqrt(((xx+.5-256)/256)**2+((yy+.5-256)/256)**2),0,1)
    for row in rows:
        with Image.open(row['files']['mask']['path']) as im:mask = np.asarray(im)==255
        if row['index'] in accepted:
            folder = ROOT/'fp16_sdpa'/f"{row['index']:03d}"
            raw = np.asarray(encoder.decode(folder/'raw.png')); composite = np.asarray(encoder.decode(folder/'composite.png'))
            authentic = np.asarray(encoder.decode(row['files']['original']['path']))
            if not np.array_equal(composite,np.where(mask[:,:,None],raw,authentic)):
                raise ValueError('Hard composite provenance differs')
        for condition in CONDITIONS:
            dense = {v:patch_drift.expand_grid(maps[f"{row['index']:03d}_{v}_{condition}"],(512,512))
                     for v in VARIANTS if f"{row['index']:03d}_{v}_{condition}" in maps}
            metrics = spatial_evaluation.evaluate_triplet(dense['authentic'],dense['classical_edit'],dense.get('ai_composite'),
                mask,threshold=.5,boundary_width=8)
            results.append({'index':row['index'],'condition':condition,'metrics':metrics,
                'aligned_auc':{v:float(roc_auc_score(mask.ravel(),value.ravel())) for v,value in dense.items()},
                'radial_center_auc':float(roc_auc_score(mask.ravel(),radial.ravel()))})
    write_once(ROOT/'measurements.json', {'results':results,'contract_sha256':digest(CONTRACT)})
    report = {'state':'E133_frozen_head_location_challenge_complete','contract_sha256':digest(CONTRACT),
        'parents':16,'accepted_composites':len(accepted),'rejected_attempts':16-len(accepted),
        'summary':summarize(results,read(learned.ROOT/'measurements.json')['results'],accepted),
        'max_authentic_token_error':token_error,'max_authentic_score_error':score_error,'peak_mps_bytes':peak,
        'seconds':time.monotonic()-start,'locked_scores_sha256':digest(ROOT/'locked_scores.json'),
        'measurements_sha256':digest(ROOT/'measurements.json'),'downloads':0,'training_admission':False,
        'promotion_allowed':False,'limits':c['limits']}
    write_once(ROOT/'report.json',report);write_once(EVIDENCE/'e133_mask_location.json',report)
    journal('frozen-head location result',report);return report


if __name__=='__main__':
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    def denied(*args,**kwargs):raise RuntimeError('Location challenge is offline')
    socket.socket.connect=denied;socket.socket.connect_ex=denied;socket.create_connection=denied
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=['freeze','prepare','generate','score'])
    ROOT.mkdir(exist_ok=True)
    with (ROOT/'execution.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        print(json.dumps({'freeze':freeze,'prepare':prepare,'generate':generate,'score':score}[parser.parse_args().stage](),indent=2))
