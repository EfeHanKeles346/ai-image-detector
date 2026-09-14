"""Traceable, score-blind TRAIN-only inpainting engineering pilot; never a detector test."""
import argparse
from collections import Counter
import fcntl
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import random
import socket
import time
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageOps
from experiments.e65_acquisition import digest, read, write_once
from experiments.e72_acquisition import resource_check
from experiments import e118_inpaint_assets as assets
from pixelproof.project_paths import DATA_ROOT, ML_ROOT, WORK_ROOT

ROOT = DATA_ROOT/'e119'; EVIDENCE = ML_ROOT.parent/'evidence'; CONTRACT = ROOT/'contract.json'
SENSORS = ('Hynix_SL846', 'ISOCELL_3P9', 'Sony_IMX258', 'OmniVision_OV32A',
    'ISOCELL_GN1', 'ISOCELL_HM3', 'OmniVision_OV64B', 'Sony_IMX766')
WHEEL = WORK_ROOT/'model2-runtime/wheels/diffusers-0.40.0-py3-none-any.whl'
WHEEL_SHA = '5b5da7c3ddb62152fa4afc577f02e050af688c797375c34fb2d01006da3f3541'


def select(rows):
    chosen = []; seen = set()
    for sensor in SENSORS:
        candidates = [r for r in rows if r['source'] == 'MIDD:'+sensor]
        candidates.sort(key=lambda r: hashlib.sha256(('E119|'+r['parent_id']).encode()).hexdigest())
        count = 0
        for row in candidates:
            if row.get('role') != 'TRAIN' or row.get('label') != 0 or row.get('training_allowed') is not True or min(row['width'], row['height']) < 512:
                raise ValueError('Only admitted native REAL TRAIN originals allowed')
            if row['scene_group'] in seen: continue
            seen.add(row['scene_group']); chosen.append(row); count += 1
            if count == 2: break
        if count != 2: raise ValueError('Two existing TRAIN scene groups per sensor required')
    if len({r['parent_id'] for r in chosen}) != 16 or len({r['sha256'] for r in chosen}) != 16:
        raise ValueError('Duplicate pilot original identity')
    return chosen


def make_mask(parent, shape):
    if shape not in ('rectangle', 'ellipse'): raise ValueError('Unregistered mask shape')
    seed = int(hashlib.sha256(('E119-mask|'+parent).encode()).hexdigest()[:16], 16)
    rng = random.Random(seed); w = rng.randrange(160, 241); h = rng.randrange(160, 241)
    x = rng.randrange(16, 512-w-16); y = rng.randrange(16, 512-h-16)
    mask = Image.new('L', (512, 512)); draw = ImageDraw.Draw(mask)
    getattr(draw, shape)((x, y, x+w-1, y+h-1), fill=255)
    return mask


def composite(original, generated, mask):
    a = np.asarray(original); b = np.asarray(generated); m = np.asarray(mask)
    if a.shape != (512, 512, 3) or b.shape != a.shape or a.dtype != np.uint8 or b.dtype != np.uint8 or \
            m.shape != a.shape[:2] or m.dtype != np.uint8 or set(np.unique(m)) != {0, 255}:
        raise ValueError('Exact RGB geometry and binary partial mask required')
    return Image.fromarray(np.where(m[..., None] > 0, b, a).astype(np.uint8))


def freeze():
    assets.validate()
    source = DATA_ROOT/'e112/contract.json'
    if digest(source) != read(EVIDENCE/'e112_context_contract.json')['contract_sha256']:
        raise ValueError('Admitted original manifest changed')
    rows = select(read(source)['rows'])
    if digest(WHEEL) != WHEEL_SHA or version('diffusers') != '0.40.0':
        raise ValueError('Pinned isolated Diffusers wheel required')
    runtime = WORK_ROOT/'model2-runtime/site-packages/diffusers'
    files = [Path(__file__), source, assets.CONTRACT, WHEEL, Path(assets.__file__),
        ML_ROOT/'experiments/e65_acquisition.py', ML_ROOT/'experiments/e72_acquisition.py'] + sorted(runtime.rglob('*.py'))
    if len(files) < 100: raise ValueError('Installed isolated runtime missing')
    c = {'state': 'E119_TRACEABLE_TRAIN_PILOT_registered', 'inputs': {str(p): digest(p) for p in files},
        'versions': {p: version(p) for p in ('diffusers', 'torch', 'transformers', 'safetensors', 'numpy', 'Pillow')},
        'rows': rows, 'parents': 16, 'parent_role': 'TRAIN', 'all_descendants_role': 'TRAIN_RESEARCH_PILOT',
        'selection': 'First two distinct existing scene_group IDs per eight admitted MIDD sensors, ranked by SHA256(E119|parent_id). No images/scores used; no replacements for generation/quality failure. Groups are not proven independent scenes.',
        'image_geometry': 'EXIF transpose to RGB then exact native center512 crop, no resize. All authentic/classic/AI paired outputs use this same geometry and lossless PNG encoding.',
        'mask': 'Alternating rectangle/ellipse; deterministic parent-seeded position and160..240 side lengths with16px margins. Store intended mask separately from zero AI targets of authentic/classic controls.',
        'generation': {'repository': assets.REPO, 'revision': assets.REVISION, 'dtype': 'float16', 'device': 'mps',
            'steps': 30, 'guidance_scale': 7.5, 'strength': 1.0, 'batch_size': 1,
            'prompt': 'a natural photograph, consistent lighting and texture', 'seed_base': 119000,
            'scheduler': 'pinned repository scheduler configuration', 'attention_slicing': True, 'safety_checker': True},
        'controls': 'Unmodified authentic input plus GaussianBlur(radius6) inside the same mask, composed by the identical hard mask. Both are AI-negative; they are not equivalent to an unedited authenticity label. This isolates some compositing/mask shortcuts.',
        'raw_output': 'Retain full raw inpainting output and measure out-of-mask pixel drift. Only the separately labelled hard composite has guaranteed unchanged background. Do not treat raw output as localized ground truth.',
        'acceptance': 'All16 attempted, unchanged source/contract/runtime, finite nonblank512 RGB output, safety check negative, >=95% masked pixels changed, composite background exactly original. No detector scores; generator feasibility is not semantic quality or detector performance.',
        'max_seconds': 2400, 'mps_limit_bytes': 8*1024**3, 'downloads_during_generation': 0,
        'licence': 'MIDD CC BY-NC-SA4.0 research originals; preserve attribution and share-alike terms for derivatives. Generator CreativeML OpenRAIL-M retained by E118. No image/weight publication.',
        'limits': '16 already-used TRAIN parents, one old editor, correlated sensor scenes and fixed prompt. This is an engineering pilot. No calibration, independent evaluation, universal claim or automatic training/serving admission.'}
    ROOT.mkdir(exist_ok=True); write_once(CONTRACT, c)
    write_once(EVIDENCE/'e119_inpainting_pilot_contract.json', {k:v for k,v in c.items() if k not in ('inputs', 'rows')} |
        {'contract_sha256': digest(CONTRACT), 'source_manifest_sha256': digest(source), 'wheel_sha256': WHEEL_SHA})
    return {'parents': 16, 'role': c['all_descendants_role'], 'generator': assets.REPO}


def validate():
    c = read(CONTRACT)
    if digest(CONTRACT) != read(EVIDENCE/'e119_inpainting_pilot_contract.json')['contract_sha256']:
        raise ValueError('Pilot contract differs')
    for path, sha in c['inputs'].items():
        if digest(path) != sha: raise ValueError('Frozen pilot input differs')
    if any(version(p) != v for p,v in c['versions'].items()): raise ValueError('Pilot runtime version differs')
    return c


def prepare():
    c = validate(); rows = []
    for i, row in enumerate(c['rows']):
        if digest(row['path']) != row['sha256']: raise ValueError('Original source body differs')
        with Image.open(row['path']) as image:
            image = ImageOps.exif_transpose(image).convert('RGB'); w,h = image.size
            original = image.crop(((w-512)//2, (h-512)//2, (w-512)//2+512, (h-512)//2+512))
        mask = make_mask(row['parent_id'], 'rectangle' if i%2 == 0 else 'ellipse')
        classic = composite(original, original.filter(ImageFilter.GaussianBlur(6)), mask)
        dest = ROOT/'parents'/f'{i:03d}'; dest.mkdir(parents=True, exist_ok=True); files = {}
        for name, image in [('original', original), ('mask', mask), ('classic', classic), ('AI_negative_target', Image.new('L', (512,512)))]:
            p = dest/(name+'.png')
            if p.exists(): raise FileExistsError('Pilot source preparation already exists; inspect receipt')
            image.save(p, format='PNG'); files[name] = {'path': str(p), 'sha256': digest(p)}
        rows.append({'index': i, 'parent_id': row['parent_id'], 'scene_group': row['scene_group'], 'source': row['source'],
            'original_sha256': row['sha256'], 'role': c['all_descendants_role'], 'files': files,
            'mask_fraction': float((np.asarray(mask)>0).mean()), 'seed': c['generation']['seed_base']+i})
    result = {'state': 'E119_paired_inputs_prepared', 'contract_sha256': digest(CONTRACT), 'rows': rows,
        'generated_images': 0, 'detector_scores': 0}
    write_once(ROOT/'prepared.json', result)
    write_once(EVIDENCE/'e119_inpainting_preparation.json', {k:v for k,v in result.items() if k != 'rows'} |
        {'prepared_sha256': digest(ROOT/'prepared.json'), 'parents': len(rows), 'source_counts': dict(Counter(r['source'] for r in rows)),
         'mask_fraction_range': [min(r['mask_fraction'] for r in rows), max(r['mask_fraction'] for r in rows)]})
    return {'parents_prepared': len(rows), 'generated_images': 0}


def generate():
    import torch
    from diffusers import StableDiffusionInpaintPipeline
    c = validate(); assets.validate()
    if digest(assets.ROOT/'download.json') != digest(EVIDENCE/'e118_inpaint_assets.json'):
        raise ValueError('Asset completion receipt differs')
    for row in read(assets.CONTRACT)['files']: assets.verify(assets.ROOT/'model'/row['rfilename'], row)
    if digest(ROOT/'prepared.json') != read(EVIDENCE/'e119_inpainting_preparation.json')['prepared_sha256']:
        raise ValueError('Prepared parent receipt differs')
    if (ROOT/'result.json').exists(): raise FileExistsError('Pilot completed; no hidden re-generation')
    start = time.monotonic(); deadline = start+c['max_seconds']; resource_check(deadline)
    torch.set_num_threads(1)
    if not torch.backends.mps.is_available(): raise RuntimeError('Registered MPS backend unavailable')
    pipe = StableDiffusionInpaintPipeline.from_pretrained(str(assets.ROOT/'model'), local_files_only=True,
        torch_dtype=torch.float16, variant='fp16', use_safetensors=True, low_cpu_mem_usage=False).to('mps')
    if pipe.safety_checker is None: raise ValueError('Registered safety checker missing')
    pipe.enable_attention_slicing(); pipe.set_progress_bar_config(disable=True); peak = 0; results = []
    def step(pipeline, index, timestep, callback_kwargs):
        nonlocal peak
        resource_check(deadline); peak = max(peak, torch.mps.driver_allocated_memory())
        if peak > c['mps_limit_bytes']: raise RuntimeError('Pilot MPS memory bound')
        return callback_kwargs
    for row in read(ROOT/'prepared.json')['rows']:
        resource_check(deadline)
        dest = ROOT/'parents'/f'{row["index"]:03d}'
        if (dest/'generation.json').exists(): raise FileExistsError('Partial generation requires explicit review, not automatic reroll')
        for info in row['files'].values():
            if digest(info['path']) != info['sha256']: raise ValueError('Prepared pixels differ')
        with Image.open(row['files']['original']['path']) as im: original = im.convert('RGB')
        with Image.open(row['files']['mask']['path']) as im: mask = im.convert('L')
        with torch.inference_mode():
            output = pipe(prompt=c['generation']['prompt'], image=original, mask_image=mask, height=512, width=512,
                num_inference_steps=30, guidance_scale=7.5, strength=1.0, num_images_per_prompt=1,
                generator=torch.Generator(device='cpu').manual_seed(row['seed']), callback_on_step_end=step)
        generated = output.images[0].convert('RGB'); generated.save(dest/'raw.png')
        edited = composite(original, generated, mask); edited.save(dest/'composite.png')
        a = np.asarray(original); b = np.asarray(generated); m = np.asarray(mask)>0; diff = np.any(a != b, axis=2)
        flags = output.nsfw_content_detected
        checks = {'safety_check_negative': flags is not None and len(flags)==1 and not bool(flags[0]),
            'nonblank_output': bool(b.std() > 1), 'masked_pixels_changed': bool(diff[m].mean() >= .95),
            'background_exact': bool(np.array_equal(np.asarray(edited)[~m], a[~m]))}
        record = {'index': row['index'], 'parent_id': row['parent_id'], 'role': row['role'], 'seed': row['seed'],
            'raw_sha256': digest(dest/'raw.png'), 'composite_sha256': digest(dest/'composite.png'),
            'raw_changed_fraction_inside_mask': float(diff[m].mean()), 'raw_changed_fraction_outside_mask': float(diff[~m].mean()),
            'checks': checks, 'passed': all(checks.values()), 'contract_sha256': digest(CONTRACT)}
        write_once(dest/'generation.json', record); results.append(record)
        print(json.dumps({'E119_generated': len(results), 'passed': record['passed']}), flush=True)
    result = {'state': 'E119_inpainting_engineering_pilot_complete', 'contract_sha256': digest(CONTRACT),
        'rows': results, 'passed': len(results)==16 and all(r['passed'] for r in results),
        'seconds': time.monotonic()-start, 'peak_mps_bytes': peak, 'detector_scores': 0,
        'training_admission': False, 'promotion_allowed': False, 'limits': c['limits']}
    write_once(ROOT/'result.json', result)
    public = {k:v for k,v in result.items() if k != 'rows'} | {'result_sha256': digest(ROOT/'result.json'),
        'parents': len(results), 'passed_parents': sum(r['passed'] for r in results),
        'mean_raw_background_changed_fraction': float(np.mean([r['raw_changed_fraction_outside_mask'] for r in results]))}
    write_once(EVIDENCE/'e119_inpainting_pilot.json', public)
    text = '\n### E119 automatic engineering pilot result\n\n'+json.dumps(public, sort_keys=True)+'\n\nThis measures generation feasibility only. Spatial-head training requires a new protocol; no independent detector evidence or serving change.\n'
    for path in (ML_ROOT.parent/'PLAN.md', ML_ROOT.parent/'HISTORY.md', ML_ROOT/'EXPERIMENTS.md', ML_ROOT.parent/'DATASETS.md'):
        with path.open('a') as f: fcntl.flock(f, fcntl.LOCK_EX); f.write(text)
    return public


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('stage', choices=['freeze', 'prepare', 'generate'])
    args = parser.parse_args()
    if args.stage == 'generate':
        import os
        os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
        def denied(*a, **kw): raise RuntimeError('Local generation is offline')
        socket.socket.connect = denied; socket.socket.connect_ex = denied; socket.create_connection = denied
    ROOT.mkdir(exist_ok=True)
    with (ROOT/'execution.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        print(json.dumps({'freeze': freeze, 'prepare': prepare, 'generate': generate}[args.stage](), indent=2))
