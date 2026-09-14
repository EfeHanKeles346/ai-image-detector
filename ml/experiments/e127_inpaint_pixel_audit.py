"""Read-only, CPU-only audit of E122 pixel drift and mask-edge shortcuts; no admission."""
import argparse
from collections import Counter
import fcntl
import json
import os
from pathlib import Path
import socket
import time
import numpy as np
from PIL import Image
from experiments.e65_acquisition import digest, read, write_once
from pixelproof.project_paths import DATA_ROOT, ML_ROOT

ROOT = DATA_ROOT/'e127'
EVIDENCE = ML_ROOT.parent/'evidence'
CONTRACT = ROOT/'contract.json'
PREPARED = DATA_ROOT/'e119/prepared.json'
REPLAY = DATA_ROOT/'e122/report.json'


def mask_array(value, shape):
    value = np.asarray(value)
    if value.shape != shape or not np.isin(value, [0, 255]).all():
        raise ValueError('Same-geometry binary intended mask required')
    mask = value == 255
    if not mask.any() or mask.all(): raise ValueError('Partial mask required')
    return mask


def image_metrics(original, changed, mask):
    original = np.asarray(original); changed = np.asarray(changed)
    if original.shape != changed.shape or original.ndim != 3 or original.shape[2] != 3 or \
            original.dtype != np.uint8 or changed.dtype != np.uint8:
        raise ValueError('Matched uint8 RGB images required')
    mask = mask_array(mask, original.shape[:2])
    x = original.astype(np.float64)/255.; y = changed.astype(np.float64)/255.
    error = np.abs(y-x); modified = np.any(original != changed, axis=2)
    # Undirected four-neighbour edges that cross exactly one intended-mask boundary.
    horizontal = mask[:, 1:] != mask[:, :-1]
    vertical = mask[1:, :] != mask[:-1, :]
    def edges(a):
        return np.concatenate([np.abs(a[:, 1:]-a[:, :-1]).mean(axis=2)[horizontal],
            np.abs(a[1:]-a[:-1]).mean(axis=2)[vertical]])
    before = edges(x); after = edges(y)
    return {'inside_mae': float(error[mask].mean()), 'outside_mae': float(error[~mask].mean()),
        'outside_channel_abs_error_p95': float(np.quantile(error[~mask], .95)),
        'outside_rmse': float(np.sqrt(np.square(y-x)[~mask].mean())),
        'inside_changed_pixel_fraction': float(modified[mask].mean()),
        'outside_changed_pixel_fraction': float(modified[~mask].mean()),
        'boundary_edges': len(before), 'boundary_gradient': float(after.mean()),
        'boundary_gradient_original': float(before.mean()),
        'boundary_gradient_delta': float((after-before).mean()),
        'boundary_gradient_absolute_change': float(np.abs(after-before).mean())}


def summarize(records):
    if not records: raise ValueError('No accepted outputs to summarize')
    return {key: {'mean': float(np.mean(values)), 'median': float(np.median(values)),
        'min': float(np.min(values)), 'max': float(np.max(values))}
        for key in records[0] for values in [[r[key] for r in records]]}


def freeze():
    publication = read(EVIDENCE/'e122_inpaint_replay.json')
    if digest(REPLAY) != publication['report_sha256'] or \
            digest(DATA_ROOT/'e122/contract.json') != publication['contract_sha256']:
        raise ValueError('Frozen complete replay identity differs')
    if digest(PREPARED) != read(EVIDENCE/'e119_inpainting_preparation.json')['prepared_sha256']:
        raise ValueError('Prepared original/control population differs')
    prepared = read(PREPARED); report = read(REPLAY)
    if digest(DATA_ROOT/'e119/contract.json') != prepared['contract_sha256']:
        raise ValueError('Original preparation contract differs')
    rows = prepared['rows']; cases = report['rows']
    if len(rows) != 16 or [r['index'] for r in rows] != list(range(16)) or \
            [r['index'] for r in cases] != list(range(16)) or \
            any(r['role'] != 'TRAIN_RESEARCH_PILOT' for r in rows):
        raise ValueError('Unchanged complete TRAIN research pilot required')
    files = [Path(__file__), PREPARED, REPLAY, DATA_ROOT/'e119/contract.json', DATA_ROOT/'e122/contract.json',
        EVIDENCE/'e122_inpaint_replay.json', EVIDENCE/'e119_inpainting_preparation.json']
    for row, case in zip(rows, cases, strict=True):
        if row['seed'] != case['seed'] or case.get('branch') != report['branch'] or \
                type(case.get('passed')) is not bool or case['passed'] != all(case['checks'].values()):
            raise ValueError('Replay case identity or acceptance differs')
        for item in row['files'].values():
            path = Path(item['path'])
            if digest(path) != item['sha256']: raise ValueError('Prepared image/mask body differs')
            files.append(path)
        folder = DATA_ROOT/'e122'/report['branch']/f"{case['index']:03d}"
        if read(folder/'case.json') != case: raise ValueError('Replay case receipt differs')
        files.append(folder/'case.json')
        for name in ('raw', 'composite'):
            path = folder/(name+'.png')
            if digest(path) != case[name+'_sha256']: raise ValueError('Replay output body differs')
            files.append(path)
    c = {'state': 'E127_readonly_pixel_audit_registered', 'inputs': {str(p): digest(p) for p in files},
        'parents': 16, 'branch': report['branch'], 'max_seconds': 300,
        'population': 'Account for all16 attempts. Check original/mask/classic/zero-target invariants for all16. Quantify raw/composite quality only for cases passing the unchanged preexisting E122 checks; report excluded attempts explicitly. No refilling or selection by pixel results.',
        'metrics': 'Normalize channels to0..1. Inside/outside MAE, outside RMSE and95th percentile absolute channel error, changed-pixel fractions. For every four-neighbour edge crossing the intended mask, mean absolute RGB gradient, its signed difference and absolute change from authentic. Compare raw/composite/traditional controls on the same11 accepted parent masks; also summarize all16 traditional controls.',
        'interpretation': 'Pixel-change fraction is not semantic drift. Boundary gradients are a shortcut-risk diagnostic, not proof that a detector learned seams. Intended masks are not semantic annotations. Correlated TRAIN parents, single old generator, simple shapes and hard composites cannot establish generalization.',
        'downloads': 0, 'detector_scores': 0, 'gpu_operations': 0,
        'training_admission': False, 'promotion_allowed': False,
        'next': 'Retain failed E122 gate and quarantine. A future independently registered recipe needs paired non-AI edit controls, matched compositing/encoding and interior/boundary evaluation before spatial training; no safety-filter bypass.'}
    ROOT.mkdir(exist_ok=True); write_once(CONTRACT, c)
    write_once(EVIDENCE/'e127_pixel_audit_contract.json', {k:v for k,v in c.items() if k != 'inputs'} |
        {'contract_sha256': digest(CONTRACT)})
    return {'parents': 16, 'accepted_output_subset': sum(r['passed'] for r in cases), 'training_admission': False}


def audit():
    start = time.monotonic(); c = read(CONTRACT)
    if digest(CONTRACT) != read(EVIDENCE/'e127_pixel_audit_contract.json')['contract_sha256']:
        raise ValueError('Pixel audit contract differs')
    for path, sha in c['inputs'].items():
        if digest(path) != sha: raise ValueError('Bound pixel audit input differs')
    if (ROOT/'report.json').exists(): raise FileExistsError('Pixel audit already complete')
    def pixels(path, mode):
        with Image.open(path) as image:
            if image.size != (512, 512) or image.mode != mode: raise ValueError('Pilot PNG geometry/mode differs')
            return np.asarray(image).copy()
    prepared = read(PREPARED)['rows']; cases = read(REPLAY)['rows']
    records = []; all_classic = []; failures = Counter()
    for row, case in zip(prepared, cases, strict=True):
        if time.monotonic()-start >= c['max_seconds']: raise TimeoutError('Pixel audit time budget exceeded')
        original = pixels(row['files']['original']['path'], 'RGB')
        mask = pixels(row['files']['mask']['path'], 'L')
        negative = pixels(row['files']['AI_negative_target']['path'], 'L')
        if negative.any(): raise ValueError('Traditional/authentic target is not AI-negative')
        classic = pixels(row['files']['classic']['path'], 'RGB')
        control = image_metrics(original, classic, mask); all_classic.append(control)
        if control['outside_changed_pixel_fraction'] != 0: raise ValueError('Classic background changed')
        if not case['passed']:
            failures.update(k for k,v in case['checks'].items() if not v)
            continue  # Never interpret safety-filter placeholders as generated content.
        folder = DATA_ROOT/'e122'/c['branch']/f"{row['index']:03d}"
        raw = pixels(folder/'raw.png', 'RGB'); composite = pixels(folder/'composite.png', 'RGB')
        intended = mask_array(mask, original.shape[:2])
        if not np.array_equal(composite, np.where(intended[:, :, None], raw, original)):
            raise ValueError('Recorded hard composite differs from exact provenance')
        raw_metrics = image_metrics(original, raw, mask)
        if abs(raw_metrics['outside_changed_pixel_fraction']-case['raw_background_changed_fraction']) > 1e-12:
            raise ValueError('E122 background measurement replay differs')
        records.append({'index': row['index'], 'classic': control, 'raw': raw_metrics,
            'composite': image_metrics(original, composite, mask)})
    if len(records) != read(REPLAY)['passed_parents']: raise ValueError('Accepted subset count differs')
    result = {'state': 'E127_pixel_audit_complete', 'contract_sha256': digest(CONTRACT),
        'parents_accounted': len(prepared), 'accepted_output_pairs': len(records),
        'excluded_attempts': len(prepared)-len(records), 'failed_checks_counts': dict(failures),
        'all_classic_controls': summarize(all_classic),
        'matched_accepted_subset': {name: summarize([r[name] for r in records]) for name in ('classic', 'raw', 'composite')},
        'composite_boundary_greater_than_classic_pairs': sum(r['composite']['boundary_gradient'] > r['classic']['boundary_gradient'] for r in records),
        'all_control_backgrounds_exact': True, 'all_negative_targets_zero': True,
        'all_accepted_composites_exact': True, 'seconds': time.monotonic()-start,
        'downloads': 0, 'detector_scores': 0, 'gpu_operations': 0, 'training_admission': False,
        'promotion_allowed': False, 'limits': c['interpretation'], 'next': c['next']}
    write_once(ROOT/'measurements.json', {'contract_sha256': digest(CONTRACT), 'rows': records})
    result['measurements_sha256'] = digest(ROOT/'measurements.json')
    write_once(ROOT/'report.json', result); write_once(EVIDENCE/'e127_pixel_audit.json', result)
    return result


if __name__ == '__main__':
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    def denied(*a, **kw): raise RuntimeError('Pixel audit is offline')
    socket.socket.connect = denied; socket.socket.connect_ex = denied; socket.create_connection = denied
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('stage', choices=['freeze', 'audit'])
    ROOT.mkdir(exist_ok=True)
    with (ROOT/'execution.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        print(json.dumps({'freeze': freeze, 'audit': audit}[parser.parse_args().stage](), indent=2))
