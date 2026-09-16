"""Score-blind local residual/processing feasibility probe; no classifier or promotion."""
import argparse
from collections import defaultdict
import fcntl
import hashlib
import json
from pathlib import Path
import platform
import resource
import socket
import sys
import time
import numpy as np
import PIL
from experiments.e65_acquisition import digest, read, write_once
from experiments.e71_features import save_npz
from experiments.e72_acquisition import resource_check
from pixelproof import source_pixel_residual as features
from pixelproof.project_paths import DATA_ROOT, ML_ROOT

ROOT = DATA_ROOT/'e149'
CONTRACT = ROOT/'contract.json'
EVIDENCE = ML_ROOT.parent/'evidence'
INVENTORY = DATA_ROOT/'e148'
TRAIN = DATA_ROOT/'e131/contract.json'


def select(records):
    if not records or len({r['parent_id'] for r in records}) != len(records) or any('header_error' in r for r in records):
        raise ValueError('Complete unique header-verified inventory required')
    rank = lambda r: hashlib.sha256(('E149|'+r['parent_id']).encode()).hexdigest()
    strata = defaultdict(list)
    for row in records:
        strata[(row['source'], row['label'], row['processing_history'])].append(row)
    selected = {min(part, key=rank)['parent_id'] for part in strata.values()}
    for mode in {r['mode'] for r in records}:
        selected.add(min((r for r in records if r['mode']==mode), key=rank)['parent_id'])
    selected.add(min(records, key=lambda r: (min(r['width'],r['height']),rank(r)))['parent_id'])
    selected.add(min(records, key=lambda r: (-r['width']*r['height'],rank(r)))['parent_id'])
    return sorted(selected), len(strata)


def versions():
    return dict(python=platform.python_version(), numpy=np.__version__, pillow=PIL.__version__)


def freeze():
    old = read(INVENTORY/'report.json')
    if digest(INVENTORY/'report.json') != digest(EVIDENCE/'e148_processing_inventory.json') or \
            digest(INVENTORY/'private_records.json') != old['private_records_sha256'] or \
            digest(TRAIN) != read(EVIDENCE/'e131_source_holdout_contract.json')['contract_sha256']:
        raise ValueError('Complete bound source inventory required')
    if old['total']['parents'] != 12525 or any(old['total'][k] for k in ('header_errors','geometry_mismatches','processing_geometry_conflicts')):
        raise ValueError('Successful full processing inventory required')
    records = read(INVENTORY/'private_records.json')['rows']
    rows = read(TRAIN)['rows']
    if [r['parent_id'] for r in records] != [r['parent_id'] for r in rows] or any(r['role'].upper()!='TRAIN' for r in rows):
        raise ValueError('Exact TRAIN inventory order required')
    selected, strata = select(records)
    inputs = [Path(__file__), Path(features.__file__), TRAIN, INVENTORY/'report.json',
              INVENTORY/'private_records.json', EVIDENCE/'e148_processing_inventory.json',
              ML_ROOT/'experiments/e65_acquisition.py', ML_ROOT/'experiments/e71_features.py',
              ML_ROOT/'experiments/e72_acquisition.py']
    c = dict(state='E149_source_pixel_probe_registered', inputs={str(p):digest(p) for p in inputs},
        selected=selected, parents=len(selected), strata=strata, versions=versions(),
        max_seconds=900, max_rss_MiB=1024, max_body_bytes=100*1024**2,
        selection='One SHA256(E149|parent)-first parent per source/class/recorded-processing stratum, one per RGB/L/RGBA mode, global smallest-short-side and largest-pixel-area stress cases; deduplicate, no replacements or scores.',
        geometry='Existing source bytes only; EXIF orient, center128 crop at source-pixel scale, then RGB conversion (alpha discarded as in existing code). No resampling, texture selection or padding on source branch. Control downsamples the same RGB patch to64 then upsamples to128 using Pillow LANCZOS. Not reproduction of whole-image2048 capping or earlier processing history.',
        representation='RGB, derivative orders1/2, axes y/x; int16 differences, nearest-even quantization at step4, clip[-2,2]. Joint histogram of adjacent quantized residuals along the derivative axis,25 normalized bins. Channel/order/axis order,300float32 coordinates per view; two views. No rank/step/filter/crop sweep.',
        acceptance='Every selected body verifies and completes; exact feature and crop replay on a second decode; exact saved numeric array replay, finite nonnegative histograms each summing to1 within1e-6, peak RSS<=1024MiB, total<=900s. Report all responses including zeros; changed features are not a quality gate or detection gain.',
        reporting='Aggregate timing and mean-absolute paired feature response by class/source/processing/mode; private parent features/patch hashes remain on disk, no patch image written. Timing is a small stratified warm-cache pilot, not guaranteed full extraction cost.',
        downloads=0, fits=0, model_scores=0, promotion_allowed=False,
        limits='Consumed TRAIN engineering pilot; no class separability, calibration, AI recall, false-positive, independent-source or universality claim. RGB conversion and prior source processing remain. No DEV/gallery/reserves, learned encoders or classifiers loaded. Later four-condition extraction requires a separate full-population geometry/protocol freeze.')
    ROOT.mkdir(exist_ok=True)
    write_once(CONTRACT,c)
    write_once(EVIDENCE/'e149_source_pixel_probe_contract.json',
               {k:v for k,v in c.items() if k not in ('inputs','selected')} | {'contract_sha256':digest(CONTRACT)})
    return {'parents':len(selected),'strata':strata,'contract_sha256':digest(CONTRACT)}


def run():
    c = read(CONTRACT)
    if digest(CONTRACT) != read(EVIDENCE/'e149_source_pixel_probe_contract.json')['contract_sha256'] or c['versions']!=versions():
        raise ValueError('Probe contract or numeric runtime differs')
    for path, sha in c['inputs'].items():
        if digest(path)!=sha:
            raise ValueError('Frozen probe input differs')
    rows = {r['parent_id']:r for r in read(TRAIN)['rows']}
    inventory = {r['parent_id']:r for r in read(INVENTORY/'private_records.json')['rows']}
    start = time.monotonic(); deadline = start+c['max_seconds']
    resource_check(deadline)
    write_once(ROOT/'started.json',{'contract_sha256':digest(CONTRACT)})
    arrays, records = [], []
    for parent in c['selected']:
        resource_check(deadline)
        row = rows[parent]; meta = inventory[parent]; before = time.monotonic()
        if row['role'].upper()!='TRAIN' or not 0<Path(row['path']).stat().st_size<=c['max_body_bytes']:
            raise ValueError('Invalid TRAIN source')
        raw = Path(row['path']).read_bytes()
        if hashlib.sha256(raw).hexdigest()!=row['sha256']:
            raise ValueError('Source body differs')
        patch, result = features.extract(raw)
        first_seconds = time.monotonic()-before
        again, replay = features.extract(raw)
        if not np.array_equal(patch,again) or not np.array_equal(result,replay) or \
                not np.isfinite(result).all() or (result<0).any() or not np.allclose(result.reshape(2,12,25).sum(axis=2),1,rtol=0,atol=1e-6):
            raise ValueError('Probe replay or histogram validity failed')
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1024**2 if sys.platform=='darwin' else 1024)
        if peak>c['max_rss_MiB']:
            raise RuntimeError('Probe memory ceiling exceeded')
        arrays.append(result)
        records.append(dict(parent_id=parent, source=row['source'],label=row['label'],
            processing_history=meta['processing_history'],mode=meta['mode'],first_seconds=first_seconds,
            paired_mean_absolute_response=float(np.abs(result[1]-result[0]).mean()),
            patch_sha256=hashlib.sha256(patch.tobytes()).hexdigest(),source_bytes=len(raw)))
        print(json.dumps({'E149_parents':len(records),'seconds':round(time.monotonic()-start)}),flush=True)
    resource_check(deadline)
    values = np.stack(arrays)
    save_npz(ROOT/'features.npz',features=values,parents=np.asarray(c['selected']),contract_sha256=digest(CONTRACT))
    with np.load(ROOT/'features.npz',allow_pickle=False) as saved:
        if not np.array_equal(saved['features'],values) or list(saved['parents'])!=c['selected'] or str(saved['contract_sha256'])!=digest(CONTRACT):
            raise ValueError('Saved artifact replay differs')
    write_once(ROOT/'private_records.json',{'rows':records,'contract_sha256':digest(CONTRACT)})
    groups = []
    for key in sorted({(r['source'],r['label'],r['processing_history'],r['mode']) for r in records}):
        part=[r for r in records if (r['source'],r['label'],r['processing_history'],r['mode'])==key]
        groups.append(dict(source=key[0],label=key[1],processing_history=key[2],mode=key[3],parents=len(part),
            mean_first_seconds=float(np.mean([r['first_seconds'] for r in part])),
            mean_absolute_paired_feature_response=float(np.mean([r['paired_mean_absolute_response'] for r in part]))))
    report=dict(state='E149_source_pixel_probe_complete',contract_sha256=digest(CONTRACT),parents=len(records),
        views=len(records)*2,pixel_decodes=len(records)*2,feature_dimensions=300,all_engineering_gates_passed=True,
        exact_repeat_and_serialization_replay=True,peak_RSS_MiB=peak,seconds=time.monotonic()-start,
        first_pass_seconds=sum(r['first_seconds'] for r in records),source_bytes_read=sum(r['source_bytes'] for r in records),
        parents_with_nonzero_paired_feature_response=sum(r['paired_mean_absolute_response']>0 for r in records),
        by_stratum=groups,features_sha256=digest(ROOT/'features.npz'),private_records_sha256=digest(ROOT/'private_records.json'),
        downloads=0,fits=0,model_scores=0,promotion_allowed=False,limits=c['limits'])
    write_once(ROOT/'report.json',report);write_once(EVIDENCE/'e149_source_pixel_probe.json',report)
    return {k:v for k,v in report.items() if k!='by_stratum'}


if __name__=='__main__':
    def denied(*args,**kwargs):
        raise RuntimeError('Source-pixel probe is offline')
    socket.socket.connect=denied;socket.socket.connect_ex=denied;socket.create_connection=denied
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=['freeze','run'])
    ROOT.mkdir(exist_ok=True)
    with (ROOT/'execution.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        print(json.dumps({'freeze':freeze,'run':run}[parser.parse_args().stage](),indent=2))
