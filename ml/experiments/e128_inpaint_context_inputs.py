"""Same-parent resized-context preparation for a separately registered inpainting ablation."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import socket
import time
import numpy as np
from PIL import Image, ImageFilter, ImageOps
from experiments import e119_inpainting_pilot as original
from experiments.e65_acquisition import digest, read, write_once
from experiments.e72_acquisition import resource_check
from pixelproof.project_paths import DATA_ROOT, ML_ROOT

ROOT = DATA_ROOT/'e128'; EVIDENCE = ML_ROOT.parent/'evidence'; CONTRACT = ROOT/'contract.json'


def context_image(image):
    image = ImageOps.exif_transpose(image).convert('RGB')
    width, height = image.size; side = min(width, height)
    if side < 512: raise ValueError('Original side below registered512px minimum')
    left, top = (width-side)//2, (height-side)//2
    box = (left, top, left+side, top+side)
    result = image.crop(box).resize((512, 512), Image.Resampling.LANCZOS)
    return result, {'oriented_size': [width, height], 'native_center_square': list(box),
        'native_square_side': side, 'linear_scale': 512/side,
        'source_area_fraction': side*side/(width*height), 'resized': side != 512}


def texture_proxy(image):
    a = np.asarray(image, dtype=np.float64)/255.
    if a.shape != (512, 512, 3): raise ValueError('Matched512 RGB proxy required')
    return float((np.abs(a[1:]-a[:-1]).mean()+np.abs(a[:, 1:]-a[:, :-1]).mean())/2)


def validate_population(rows, prepared):
    if len(rows) != 16 or len(prepared) != 16 or [r['index'] for r in prepared] != list(range(16)):
        raise ValueError('Complete original16-parent order required')
    for row, old in zip(rows, prepared, strict=True):
        if row['parent_id'] != old['parent_id'] or row['sha256'] != old['original_sha256'] or \
                row['role'] != 'TRAIN' or row['label'] != 0 or row.get('training_allowed') is not True or \
                old['role'] != 'TRAIN_RESEARCH_PILOT' or old['seed'] != 119000+old['index']:
            raise ValueError('Original TRAIN ancestry/seed/role differs')


def freeze():
    parent = original.validate()
    preparation = original.ROOT/'prepared.json'; replay = DATA_ROOT/'e122/report.json'
    if digest(preparation) != read(EVIDENCE/'e119_inpainting_preparation.json')['prepared_sha256'] or \
            digest(replay) != read(EVIDENCE/'e122_inpaint_replay.json')['report_sha256']:
        raise ValueError('Prior preparation or complete replay identity differs')
    prepared = read(preparation)['rows']; validate_population(parent['rows'], prepared)
    files = [Path(__file__), Path(original.__file__), original.CONTRACT, preparation, replay,
        EVIDENCE/'e127_pixel_audit.json', ML_ROOT/'experiments/e72_acquisition.py', ML_ROOT/'experiments/e65_acquisition.py']
    for row, old in zip(parent['rows'], prepared, strict=True):
        if digest(row['path']) != row['sha256']: raise ValueError('Native parent body differs')
        files.append(Path(row['path']))
        for entry in old['files'].values():
            if digest(entry['path']) != entry['sha256']: raise ValueError('Old prepared body differs')
            files.append(Path(entry['path']))
    c = {'state': 'E128_same_parent_context_preparation_registered',
        'inputs': {str(p): digest(p) for p in files}, 'parents': 16,
        'source_contract_sha256': digest(original.CONTRACT),
        'geometry': 'EXIF-oriented RGB; native center square side min(width,height), then LANCZOS512. No aspect distortion. Exact same16 source parents and order as E119/E122, no refill.',
        'controls': 'Retain exact previous512 masks and zero AI targets. Construct resized authentic input and GaussianBlur6 same-mask classical edit with identical hard compositing and lossless PNG encoding.',
        'roles': 'TRAIN_RESEARCH_PILOT descendants of unchanged TRAIN originals. Resized derivatives, not native sensor evidence or new independent parents.',
        'proxy': 'Mean of vertical/horizontal absolute RGB neighbour differences on normalized512 images, recorded for old native crop and new context. Texture proxy, not a noise estimator, quality gate, selection criterion or detector score.',
        'max_seconds': 900, 'downloads': 0, 'gpu_operations': 0, 'detector_scores': 0,
        'training_admission': False, 'promotion_allowed': False,
        'limits': 'Input preparation changes field of view and resampling together, not an isolated denoising effect. No explanation of safety-filter decisions or proof of semantic quality. All16 attempts required in later generation.',
        'next': 'Only complete verified preparation permits a separate E129 fixed fp16+SDPA replay, preserving original parameters/seeds/checker and full16/16 engineering gate.'}
    ROOT.mkdir(exist_ok=True); write_once(CONTRACT, c)
    write_once(EVIDENCE/'e128_context_input_contract.json', {k:v for k,v in c.items() if k != 'inputs'} |
        {'contract_sha256': digest(CONTRACT)})
    return {'parents': 16, 'generation_started': False}


def validate():
    c = read(CONTRACT)
    if digest(CONTRACT) != read(EVIDENCE/'e128_context_input_contract.json')['contract_sha256']:
        raise ValueError('Context input contract differs')
    for path, sha in c['inputs'].items():
        if digest(path) != sha: raise ValueError('Frozen context input differs')
    return c


def prepare():
    c = validate(); start = time.monotonic(); deadline = start+c['max_seconds']; resource_check(deadline)
    if (ROOT/'prepared.json').exists(): raise FileExistsError('Context preparation already complete')
    parents = read(original.CONTRACT)['rows']; previous = read(original.ROOT/'prepared.json')['rows']
    validate_population(parents, previous); rows = []; proxies = []
    for source, old in zip(parents, previous, strict=True):
        resource_check(deadline)
        with Image.open(source['path']) as image: authentic, geometry = context_image(image)
        with Image.open(old['files']['mask']['path']) as image: mask = image.convert('L')
        with Image.open(old['files']['original']['path']) as image: old_proxy = texture_proxy(image)
        classic = original.composite(authentic, authentic.filter(ImageFilter.GaussianBlur(6)), mask)
        dest = ROOT/'parents'/f"{old['index']:03d}"; dest.mkdir(parents=True, exist_ok=True)
        files = {key: old['files'][key] for key in ('mask', 'AI_negative_target')}
        for name, image in [('original', authentic), ('classic', classic)]:
            path = dest/(name+'.png')
            if path.exists(): raise FileExistsError('Partial preparation exists; explicit recovery required')
            image.save(path, format='PNG'); files[name] = {'path': str(path), 'sha256': digest(path)}
        a = np.asarray(authentic); b = np.asarray(classic); intended = np.asarray(mask)>0
        with Image.open(files['AI_negative_target']['path']) as negative:
            if np.asarray(negative).any(): raise ValueError('Inherited non-AI target differs')
        if not np.array_equal(a[~intended], b[~intended]): raise ValueError('Classical control background changed')
        rows.append(old | {'files': files, 'geometry': geometry, 'rendering': 'center_square_resized_512',
            'source_body_sha256': source['sha256'], 'preparation_contract_sha256': digest(CONTRACT)})
        proxies.append({'index': old['index'], 'source': old['source'], 'old_native_crop': old_proxy,
            'resized_context': texture_proxy(authentic), 'geometry': geometry})
    result = {'state': 'E128_same_parent_context_inputs_complete', 'contract_sha256': digest(CONTRACT),
        'rows': rows, 'texture_proxies': proxies, 'parents': len(rows), 'seconds': time.monotonic()-start,
        'training_admission': False, 'generation_started': False}
    write_once(ROOT/'prepared.json', result)
    public = {k:v for k,v in result.items() if k not in ('rows', 'texture_proxies')} | {
        'prepared_sha256': digest(ROOT/'prepared.json'), 'all_masks_seeds_roles_preserved': True,
        'mean_old_native_crop_proxy': float(np.mean([x['old_native_crop'] for x in proxies])),
        'mean_resized_context_proxy': float(np.mean([x['resized_context'] for x in proxies])),
        'source_area_fraction_range': [min(x['geometry']['source_area_fraction'] for x in proxies),
            max(x['geometry']['source_area_fraction'] for x in proxies)], 'limits': c['limits']}
    write_once(EVIDENCE/'e128_context_inputs.json', public)
    return public


if __name__ == '__main__':
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    def denied(*a, **kw): raise RuntimeError('Context preparation is offline')
    socket.socket.connect = denied; socket.socket.connect_ex = denied; socket.create_connection = denied
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('stage', choices=['freeze', 'prepare'])
    ROOT.mkdir(exist_ok=True)
    with (ROOT/'execution.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        print(json.dumps({'freeze': freeze, 'prepare': prepare}[parser.parse_args().stage](), indent=2))
