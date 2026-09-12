"""Frozen E43 scoring on E65 REAL diagnostics, never training or benchmark tuning."""
from __future__ import annotations
import argparse
from collections import defaultdict
import fcntl
from io import BytesIO
import json
import os
from pathlib import Path
import socket
import time

if __name__ == '__main__':
    def denied(*args, **kwargs): raise RuntimeError('E65 diagnostic network disabled')
    socket.socket.connect = denied
    socket.socket.connect_ex = denied
    socket.create_connection = denied
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_HUB_DISABLE_TELEMETRY='1')

import joblib
import numpy as np
from PIL import Image, ImageOps
from threadpoolctl import threadpool_limits
from experiments.e65_acquisition import ROOT, EVIDENCE, digest, read, write_once
from experiments.e65_audit import REPORT as AUDIT, verified
from pixelproof.project_paths import DATA_ROOT, ML_ROOT

REFERENCE = DATA_ROOT / 'e43/e43_small_predev.joblib'
REFERENCE_SHA = 'a3aec445926bcc8707b3775f01d2cdd9491ba8495ad8a8ec306840556ca47390'
AI_CUT = 0.07940196245908739
REAL_CUT = 0.011505939625203613
CONTRACT = ROOT / 'diagnostic_contract.json'
SCORES = ROOT / 'diagnostic_scores.json'
REPORT = ROOT / 'diagnostic_report.json'


def social_q75_bytes(path):
    # Exact E49 transport convention, copied to avoid historical acquisition imports.
    with Image.open(path) as opened:
        image = ImageOps.exif_transpose(opened).convert('RGB')
        if max(image.size) > 1080:
            scale = 1080 / max(image.size)
            image = image.resize((max(1, round(image.width * scale)), max(1, round(image.height * scale))),
                                 Image.Resampling.LANCZOS)
        output = BytesIO()
        image.save(output, format='JPEG', quality=75, subsampling=2, optimize=False)
        return output.getvalue()


def freeze():
    verified(AUDIT, read(EVIDENCE / 'e65_audit.json')['report_sha256'])
    verified(REFERENCE, REFERENCE_SHA)
    audit = read(AUDIT); rows = audit['records']
    if not rows or audit['model_scores_created'] != 0 or audit['training_allowed']:
        raise ValueError('diagnostic requires nonempty unscored no-training audit')
    if len({r['parent_id'] for r in rows}) != len(rows) or any(
            r['role'] != 'DIAGNOSTIC_DEV_ONLY' or r['label'] != 0 or r['training_allowed'] for r in rows):
        raise ValueError('diagnostic population role/identity changed')
    files = [Path(__file__), Path(__file__).with_name('e65_acquisition.py'),
        Path(__file__).with_name('e65_audit.py'), Path(__file__).with_name('e42_features.py'),
        ML_ROOT / 'src/pixelproof/e32_candidate.py', AUDIT, REFERENCE]
    c = {'state': 'E65_frozen_E43_REAL_diagnostic_registered', 'inputs': {str(p): digest(p) for p in files},
         'observations': len(rows), 'views': 2*len(rows), 'conditions': ['original','social_q75'],
         'ai_cut': AI_CUT, 'real_cut': REAL_CUT, 'batch_views': 8, 'max_seconds': 1200,
         'report': 'All admitted views; per source/camera/CFA/capture condition and WIFD camera+ISO counts. '
                   'RawNIND linked low/high scene differences and transport transitions, descriptive only; '
                   'no independent-view intervals, no noise-causality or AI-retention claim.',
         'training_allowed': False, 'new_threshold_allowed': False, 'promotion_allowed': False,
         'independent_final_claim_allowed': False, 'e49_read_allowed': False}
    write_once(CONTRACT, c)
    write_once(EVIDENCE / 'e65_diagnostic_contract.json', c | {'contract_sha256': digest(CONTRACT)})
    return {'state': c['state'], 'views': c['views']}


def validate():
    verified(CONTRACT, read(EVIDENCE / 'e65_diagnostic_contract.json')['contract_sha256'])
    c = read(CONTRACT)
    for p, sha in c['inputs'].items(): verified(p, sha)
    return c


def score():
    c = validate()
    if SCORES.exists(): raise FileExistsError('diagnostic scores already locked')
    from experiments import e42_features as dino
    import torch
    torch.set_num_threads(2)
    start_time = time.monotonic()
    head = joblib.load(REFERENCE)['head']
    model, means, stds, backbone_sha = dino._load_small()
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    model = model.to(device).eval()
    mean = torch.tensor(means,device=device).view(1,3,1,1)
    std = torch.tensor(stds,device=device).view(1,3,1,1)
    rows = []
    (ROOT / 'q75').mkdir(exist_ok=True)
    for r in read(AUDIT)['records']:
        verified(r['path'],r['sha256'])
        q = ROOT / 'q75' / (r['parent_id'].split(':')[1]+'.jpg')
        body = social_q75_bytes(r['path'])
        if q.exists():
            if q.read_bytes() != body: raise ValueError('Q75 derivative replay differs')
        else:
            with q.open('xb') as f: f.write(body)
        rows.extend([r | {'capture_condition': r['condition'], 'condition': condition,
                         'path': str(p), 'sha256': digest(p), 'record_id': r['parent_id']+'|'+condition}
                     for condition,p in [('original',Path(r['path'])),('social_q75',q)]])
    result=[]; chunks=[]
    with torch.inference_mode(), threadpool_limits(limits=2):
        for start in range(0,len(rows),c['batch_views']):
            if time.monotonic()-start_time > c['max_seconds']: raise TimeoutError('diagnostic budget reached')
            batch=rows[start:start+c['batch_views']]; arrays=[]
            for r in batch:
                verified(r['path'],r['sha256'])
                with Image.open(r['path']) as im:
                    arrays.extend(dino.texture_crops(dino.transport_image(im,'clean')))
            tensor=torch.from_numpy(np.stack(arrays)).to(device).permute(0,3,1,2).float().div_(255)
            blocks=model.forward_intermediates((tensor-mean)/std,indices=list(dino.BLOCKS['small']),
                return_prefix_tokens=True,norm=True,intermediates_only=True)
            tokens=torch.stack([b[1][:,0,:] for b in blocks],dim=1).float().cpu().numpy()
            features=dino.aggregate_tokens(tokens,len(batch)); chunks.append(features)
            probabilities=head.predict_proba(features)[:,1]
            if not np.isfinite(probabilities).all() or np.any((probabilities<0)|(probabilities>1)):
                raise ValueError('invalid diagnostic score')
            result.extend(r | {'score':float(s)} for r,s in zip(batch,probabilities,strict=True))
            print(json.dumps({'scored_views':len(result),'total':len(rows)}),flush=True)
    if len(result) != c['views']: raise ValueError('incomplete coverage')
    feature_path=ROOT/'diagnostic_features.npz'
    with feature_path.open('xb') as f:
        np.savez_compressed(f,features=np.concatenate(chunks),record_ids=np.array([r['record_id'] for r in result]),
                            role=np.array('DIAGNOSTIC_DEV_ONLY'),contract_sha256=np.array(digest(CONTRACT)))
    value={'state':'E65_scores_locked_before_metrics','rows':result,'contract_sha256':digest(CONTRACT),
           'feature_sha256':digest(feature_path),'reference_sha256':REFERENCE_SHA,
           'backbone_sha256':backbone_sha,'seconds':time.monotonic()-start_time}
    write_once(SCORES,value)
    summary={k:v for k,v in value.items() if k!='rows'} | {'views':len(result),'scores_sha256':digest(SCORES),'metrics_opened':False}
    write_once(EVIDENCE/'e65_diagnostic_scores.json',summary)
    return summary


def rates(rows):
    scores=np.array([r['score'] for r in rows])
    if not len(rows): raise ValueError('empty metric group')
    return {'views':len(rows),'false_ai':int(np.sum(scores>=AI_CUT)),
        'real_fpr':float(np.mean(scores>=AI_CUT)), 'automatic_real':int(np.sum(scores<REAL_CUT)),
        'uncertain':int(np.sum((scores>=REAL_CUT)&(scores<AI_CUT))),
        'median_score':float(np.median(scores))}


def paired_report(rows):
    pairs=defaultdict(dict)
    for r in rows:
        key=(r['parent_id'],r['condition'])
        if r['condition'] in pairs[r['parent_id']]: raise ValueError('duplicate transport view')
        pairs[r['parent_id']][r['condition']]=r
    if any(set(p)!= {'original','social_q75'} for p in pairs.values()):
        raise ValueError('incomplete transport pair')
    by_source={}
    for source in sorted({r['source'] for r in rows}):
        p=[v for v in pairs.values() if v['original']['source']==source]
        by_source[source]={'pairs':len(p),
            'q75_new_false_ai':sum(v['original']['score']<AI_CUT<=v['social_q75']['score'] for v in p),
            'q75_rescued_real':sum(v['social_q75']['score']<AI_CUT<=v['original']['score'] for v in p),
            'median_q75_minus_original_score':float(np.median([v['social_q75']['score']-v['original']['score'] for v in p]))}
    return by_source


def report():
    c=validate(); verified(SCORES,read(EVIDENCE/'e65_diagnostic_scores.json')['scores_sha256'])
    rows=read(SCORES)['rows']; groups=defaultdict(list)
    for r in rows:
        groups[f"{r['source']}|{r['condition']}"] .append(r)
        identity=r['camera'] if r['source']=='WIFD' else r['cfa']
        groups[f"{r['source']}|{identity}|{r['condition']}"] .append(r)
        capture=str(r['iso']) if r['source']=='WIFD' else r['capture_condition']
        groups[f"{r['source']}|{identity}|{capture}|{r['condition']}"] .append(r)
    raw_pairs=defaultdict(dict)
    for r in rows:
        if r['source']=='RawNIND': raw_pairs[r['scene_group'],r['condition']][r['capture_condition']]=r
    raw=[]
    for (scene,transport),p in sorted(raw_pairs.items()):
        if set(p)!= {'low_iso_raw','high_iso_raw'}: raise ValueError('incomplete RAW scene pair')
        low,high=p['low_iso_raw'],p['high_iso_raw']
        raw.append({'scene':scene,'transport':transport,'cfa':low['cfa'],
                    'low_score':low['score'],'high_score':high['score'],
                    'high_minus_low':high['score']-low['score'],
                    'low_false_ai':low['score']>=AI_CUT,'high_false_ai':high['score']>=AI_CUT})
    result={'state':'E65_REAL_diagnostic_complete_no_candidate', 'contract_sha256':digest(CONTRACT),
            'scores_sha256':digest(SCORES),'reference_sha256':REFERENCE_SHA,
            'views':len(rows),'groups':{k:rates(v) for k,v in sorted(groups.items())},
            'transport_pairs':paired_report(rows),'raw_scene_pairs':raw,
            'ai_recall_measured':False,'serving_changed':False,'candidate_created':False,
            'independent_final_passed':False,'training_allowed':False,
            'limitations':['REAL-only diagnostic cannot establish AI retention or target success.',
                'WIFD scenes unknown, internal near-duplicates/repeated views are dependent.',
                'Eight selected RAW scenes at most; different CFA/cameras/scenes/brightness and RAW development '
                'confound causal interpretations of ISO/noise. File counts are not independent scene counts.',
                'Scores are AI signals, not calibrated authenticity probabilities; no confidence intervals on repeated views.']}
    write_once(REPORT,result);write_once(EVIDENCE/'e65_diagnostic.json',result)
    return result


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('freeze','score','report'))
    with (ROOT/'execution.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        print(json.dumps({'freeze':freeze,'score':score,'report':report}[parser.parse_args().stage](),indent=2))
