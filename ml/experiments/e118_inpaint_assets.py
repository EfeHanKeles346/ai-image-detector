"""Pin public SD1.5 inpainting assets for a small, TRAIN-only Model2 feasibility pilot."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import shutil
import time
import requests
from experiments.e65_acquisition import digest, read, write_once
from pixelproof.project_paths import DATA_ROOT, ML_ROOT

ROOT = DATA_ROOT/'e118'; EVIDENCE = ML_ROOT.parent/'evidence'; CONTRACT = ROOT/'contract.json'
REPO = 'stable-diffusion-v1-5/stable-diffusion-inpainting'
REVISION = '8a4288a76071f7280aedbdb3253bdb9e9d5d84bb'
LICENSE_REVISION = '14d42d09bffd871b1666a084fc954a50cff72ac0'
LICENSE_BLOB = '0e609df0d8cd3b5d11a1ea962a56b604b70846a5'
FILES = ('README.md', 'model_index.json', 'feature_extractor/preprocessor_config.json',
    'safety_checker/config.json', 'safety_checker/model.fp16.safetensors',
    'scheduler/scheduler_config.json', 'text_encoder/config.json', 'text_encoder/model.fp16.safetensors',
    'tokenizer/merges.txt', 'tokenizer/special_tokens_map.json', 'tokenizer/tokenizer_config.json',
    'tokenizer/vocab.json', 'unet/config.json', 'unet/diffusion_pytorch_model.fp16.safetensors',
    'vae/config.json', 'vae/diffusion_pytorch_model.fp16.safetensors')


def git_blob(body):
    return hashlib.sha1(b'blob '+str(len(body)).encode()+b'\0'+body).hexdigest()


def verify(path, row):
    if path.stat().st_size != row['size']: raise ValueError('Asset byte count differs')
    sha = digest(path)
    if 'lfs' in row:
        if sha != row['lfs']['sha256']: raise ValueError('Publisher asset SHA256 differs')
    elif git_blob(path.read_bytes()) != row['blobId']:
        raise ValueError('Publisher Git blob identity differs')
    return sha


def freeze():
    url = f'https://huggingface.co/api/models/{REPO}/revision/{REVISION}?blobs=true'
    r = requests.get(url, timeout=(10, 30)); r.raise_for_status(); metadata = r.json()
    if metadata['sha'] != REVISION or metadata.get('gated'):
        raise ValueError('Pinned public model revision required')
    lookup = {r['rfilename']: r for r in metadata['siblings']}
    rows = [lookup[name] for name in FILES]
    if sum(r['size'] for r in rows) > 3*1024**3 or any('lfs' not in r for r in rows if r['rfilename'].endswith('.safetensors')):
        raise ValueError('Bound safetensors inventory differs')
    ROOT.mkdir(exist_ok=True)
    license_url = f'https://huggingface.co/spaces/CompVis/stable-diffusion-license/resolve/{LICENSE_REVISION}/license.txt'
    r = requests.get(license_url, timeout=(10, 30)); r.raise_for_status()
    if len(r.content) != 14385 or git_blob(r.content) != LICENSE_BLOB: raise ValueError('Licence identity differs')
    path = ROOT/'CreativeML-OpenRAIL-M.txt'
    if path.exists() and path.read_bytes() != r.content: raise ValueError('Local licence changed')
    if not path.exists(): path.write_bytes(r.content)
    c = {'state': 'E118_public_inpainting_assets_registered', 'repository': REPO, 'revision': REVISION,
        'inputs': {str(Path(__file__)): digest(__file__), str(path): digest(path)}, 'files': rows,
        'expected_asset_bytes': sum(r['size'] for r in rows), 'max_seconds': 1800,
        'licence': 'CreativeML OpenRAIL-M; pinned author licence retained locally. Underlying image terms remain separate.',
        'licence_url': license_url, 'licence_bytes': len(r.content),
        'provenance': 'Community mirror of the deprecated Runway repository, explicitly unaffiliated with Runway; not a new 2026 generator. Pin all four safetensors LFS SHA256 values and small Git blobs. Do not load pickle or remote code.',
        'purpose': 'Create a reproducible 512px local inpainting feasibility pilot from already admitted TRAIN-only originals, after active Model1 work. Recover every original/mask/prompt/seed relationship ourselves because compiled Model2 ancestry is inadequate.',
        'limits': 'Asset acquisition only, no generation or detector training. Single old generator cannot establish modern/editor-held-out generalization. Generator pretraining overlaps are not fully auditable; no independent test claim.',
        'model_scores': 0, 'generation_allowed_by_this_contract': False, 'promotion_allowed': False}
    write_once(CONTRACT, c); write_once(EVIDENCE/'e118_inpaint_asset_contract.json', c | {'contract_sha256': digest(CONTRACT)})
    return {'expected_asset_bytes': c['expected_asset_bytes'], 'files': len(rows)}


def validate():
    c = read(CONTRACT)
    if digest(CONTRACT) != read(EVIDENCE/'e118_inpaint_asset_contract.json')['contract_sha256']:
        raise ValueError('Asset contract differs')
    for path, sha in c['inputs'].items():
        if digest(path) != sha: raise ValueError('Frozen asset input differs')
    return c


def download():
    c = validate(); start = time.monotonic(); deadline = start+c['max_seconds']; transferred = 0; done = []
    if (ROOT/'download.json').exists(): raise FileExistsError('Asset receipt already sealed')
    for row in c['files']:
        path = ROOT/'model'/row['rfilename']; path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            part = path.with_name(path.name+'.part')
            for attempt in range(3):
                if time.monotonic() >= deadline or shutil.disk_usage(ROOT).free < 20*1024**3:
                    raise RuntimeError('Bounded acquisition resource limit')
                offset = part.stat().st_size if part.exists() else 0
                if offset == row['size']: break
                if offset > row['size']: raise ValueError('Partial asset is oversized')
                headers = {'Accept-Encoding': 'identity'}
                if offset: headers['Range'] = f'bytes={offset}-'
                url = f'https://huggingface.co/{REPO}/resolve/{REVISION}/{row["rfilename"]}'
                try:
                    with requests.get(url, headers=headers, stream=True, timeout=(10, 40)) as response:
                        if response.status_code not in (200, 206):
                            raise RuntimeError('Asset HTTP status '+str(response.status_code))
                        if offset and (response.status_code != 206 or response.headers.get('Content-Range') != f'bytes {offset}-{row["size"]-1}/{row["size"]}'):
                            raise ValueError('Resume range differs')
                        with part.open('ab' if offset else 'wb') as output:
                            for chunk in response.iter_content(1024**2):
                                offset += len(chunk); transferred += len(chunk)
                                if offset > row['size'] or time.monotonic() >= deadline:
                                    raise RuntimeError('Asset transfer limit')
                                output.write(chunk)
                    if offset != row['size']: raise requests.exceptions.ChunkedEncodingError('Incomplete asset')
                    break
                except requests.RequestException:
                    if attempt == 2: raise RuntimeError('Asset transport failed after three attempts') from None
                    time.sleep(2)
            verify(part, row); part.replace(path)
        sha = verify(path, row)
        done.append({'file': row['rfilename'], 'bytes': row['size'], 'sha256': sha})
        print(json.dumps({'E118_asset': row['rfilename'], 'verified_bytes': row['size']}), flush=True)
    result = {'state': 'E118_pinned_inpainting_assets_ready', 'contract_sha256': digest(CONTRACT),
        'repository': REPO, 'revision': REVISION, 'files': done, 'asset_bytes': sum(x['bytes'] for x in done),
        'transferred_bytes_this_execution': transferred, 'seconds': time.monotonic()-start,
        'generated_images': 0, 'detector_scores': 0, 'training_allowed': False, 'limits': c['limits']}
    write_once(ROOT/'download.json', result); write_once(EVIDENCE/'e118_inpaint_assets.json', result)
    return {k:v for k,v in result.items() if k != 'files'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('stage', choices=['freeze', 'download'])
    ROOT.mkdir(exist_ok=True)
    with (ROOT/'execution.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        print(json.dumps({'freeze': freeze, 'download': download}[parser.parse_args().stage](), indent=2))
