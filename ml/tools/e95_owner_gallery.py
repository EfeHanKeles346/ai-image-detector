"""Offline E92 consumed owner-gallery diagnostic. Private rows never enter Git."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import socket
import time

os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_HUB_DISABLE_TELEMETRY='1')
def denied(*args, **kwargs):
    raise RuntimeError('E95 outgoing network disabled')
socket.socket.connect = denied
socket.socket.connect_ex = denied
socket.create_connection = denied

from PIL import Image, ImageOps
from pixelproof.project_paths import ML_ROOT, WORK_ROOT
from pixelproof.e32_candidate import image_paths
from pixelproof.e92_demo import E92Engine, AI_CUT, digest, social_view, MANIFEST
from pixelproof.demo_policy import display_result
from pixelproof.research_serve import decode_demo
from pixelproof.image_input import DEFAULT_LIMITS, ImagePolicyError
from experiments.e42_data import OWNER_DEFAULT, OWNER_IDENTITY, RESERVE

ROOT = WORK_ROOT / 'e95_owner_gallery'
EVIDENCE = ML_ROOT.parent / 'evidence'
CONTRACT = ROOT / 'contract.json'
SCORES = ROOT / 'scores.json'


def read(p):
    return json.loads(p.read_text())


def write(p, value):
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('x') as f:
        json.dump(value, f, indent=2, sort_keys=True); f.write('\n')


def freeze():
    paths = image_paths([str(OWNER_DEFAULT)])
    reserved = [p for p in paths if p.name == RESERVE['name'] and digest(p) == RESERVE['sha256']]
    selected = [p for p in paths if p not in reserved]
    identity = [{'name': p.name, 'bytes': p.stat().st_size, 'sha256': digest(p)} for p in selected]
    binding = hashlib.sha256((json.dumps(identity, indent=2, sort_keys=True) + '\n').encode()).hexdigest()
    if binding != OWNER_IDENTITY or len(selected) != 210 or len({r['sha256'] for r in identity}) != 206:
        raise ValueError('Historical owner-gallery identity changed; do not infer labels for new files')
    files = [Path(__file__), MANIFEST, ML_ROOT/'src/pixelproof/e92_demo.py',
             ML_ROOT/'src/pixelproof/demo_policy.py', ML_ROOT/'src/pixelproof/research_serve.py',
             ML_ROOT/'src/pixelproof/image_input.py']
    c = {'state': 'E95_owner_gallery_registered_before_scores', 'identity_sha256': binding,
         'files': [r | {'path': str(p)} for r, p in zip(identity, selected)],
         'inputs': {str(p): digest(p) for p in files}, 'label': 0, 'role': 'CONSUMED_DEVELOPMENT',
         'conditions': ['original', 'social_q75'], 'reserved_stills_excluded': len(reserved),
         'unsupported_nonimages': sum(p.is_file() and p.suffix.lower() == '.mov' for p in OWNER_DEFAULT.iterdir()),
         'policy': 'Keep E92 cuts/weights unchanged; all210 files and206 unique byte parents. Separate UI admission and raw native predictions. No fit/threshold selection on gallery. No fresh-final claim. Save individual results privately, only aggregate receipts to Git.',
         'max_seconds': 1800, 'training_allowed': False, 'e49_read_allowed': False, 'downloads_allowed': False}
    write(CONTRACT, c)
    write(EVIDENCE/'e95_gallery_contract.json', {k: v for k,v in c.items() if k not in ('files', 'inputs')} |
          {'contract_sha256': digest(CONTRACT), 'code_sha256': digest(Path(__file__)), 'file_count': 210, 'unique_parents':206})
    return {'state': c['state'], 'files':210, 'unique_parents':206, 'reserved_excluded':len(reserved)}


def verify():
    c = read(CONTRACT)
    if digest(CONTRACT) != read(EVIDENCE/'e95_gallery_contract.json')['contract_sha256']:
        raise ValueError('E95 contract changed')
    for p, h in c['inputs'].items():
        if digest(Path(p)) != h:
            raise ValueError('E95 input changed')
    return c


def run():
    c = verify()
    if SCORES.exists():
        raise FileExistsError('E95 already scored')
    import subprocess
    if 'AC Power' not in subprocess.check_output(['pmset','-g','batt'],text=True):
        raise RuntimeError('AC power required')
    started = time.monotonic(); engine = E92Engine(); done=[]; cached={}
    for index, r in enumerate(c['files']):
        if time.monotonic()-started > c['max_seconds']:
            raise TimeoutError('E95 deadline')
        path=Path(r['path'])
        if digest(path) != r['sha256']:
            raise ValueError('Gallery file identity changed')
        if r['sha256'] in cached:
            done.append(cached[r['sha256']] | {'name':r['name'],'path':r['path']})
            continue
        raw=path.read_bytes()
        with Image.open(path) as opened:
            image = ImageOps.exif_transpose(opened).convert('RGB')
            exif=opened.getexif()
            camera=str(exif.get(272,'unknown')).strip()
            format_name=opened.format
        eligibility='accepted'
        if len(raw)>DEFAULT_LIMITS.max_upload_bytes:
            eligibility='over_12_MiB'
        else:
            try:
                decoded=decode_demo(raw)
                if min(decoded.size)<224:eligibility='too_small_abstention'
            except ImagePolicyError as exc:
                eligibility=f'input_policy_{exc.status_code}'
        scores,reference=engine.score_views([image,social_view(image)])
        row=r|{'width':image.width,'height':image.height,'camera':camera,'format':format_name,
               'original_score':float(scores[0]),'social_q75_score':float(scores[1]),
               'reference_original':float(reference[0]),'reference_q75':float(reference[1]),
               'display':display_result(scores,reference),'eligibility':eligibility,'label':0}
        cached[r['sha256']]=row;done.append(row)
        if (index+1)%20==0:print(json.dumps({'files':index+1,'total':210}),flush=True)
    verify()
    write(SCORES,{'state':'E95_scores_locked_before_metrics','rows':done,'seconds':time.monotonic()-started})
    write(EVIDENCE/'e95_gallery_scores_receipt.json',{'scores_sha256':digest(SCORES),'contract_sha256':digest(CONTRACT),'files':len(done),'native_views':2*len(cached)})
    return {'state':'scores_locked','files':len(done)}


def counts(rows):
    return {'files':len(rows), 'original_false_ai':sum(r['original_score']>=AI_CUT for r in rows),
            'q75_false_ai':sum(r['social_q75_score']>=AI_CUT for r in rows),
            'e43_original_false_ai':sum(r['reference_original']>=AI_CUT for r in rows),
            'e43_q75_false_ai':sum(r['reference_q75']>=AI_CUT for r in rows),
            'display':dict(Counter(r['display']['outcome'] for r in rows)),
            'positive_with_warning':sum(r['display']['outcome']=='ai_signal' and r['display']['review_required'] for r in rows),
            'eligibility':dict(Counter(r['eligibility'] for r in rows))}


def report():
    verify()
    if digest(SCORES)!=read(EVIDENCE/'e95_gallery_scores_receipt.json')['scores_sha256']:raise ValueError('Scores changed')
    rows=read(SCORES)['rows'];unique=list({r['sha256']:r for r in rows}.values())
    result={'state':'E95_consumed_owner_gallery_complete','contract_sha256':digest(CONTRACT),
            'scores_sha256':digest(SCORES),'all_files':counts(rows),'unique_byte_parents':counts(unique),
            'api_accepted':counts([r for r in rows if r['eligibility']=='accepted']),
            'camera_groups':{k:counts([r for r in unique if r['camera']==k]) for k in sorted({r['camera'] for r in unique})},
            'formats':{k:counts([r for r in unique if r['format']==k]) for k in sorted({r['format'] for r in unique})},
            'limits':'Owner-declared REAL, historically consumed DEVELOPMENT. Duplicate files reported separately; bursts/scenes are not independent. No AI-recall estimate, label/source-specific exception, fit, threshold tuning or E49 access. Private images/filenames/per-image scores stay in ignored work directory.'}
    write(EVIDENCE/'e95_gallery_report.json',result)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['freeze','run','report'])
    print(json.dumps({'freeze':freeze,'run':run,'report':report}[p.parse_args().stage](),indent=2))
