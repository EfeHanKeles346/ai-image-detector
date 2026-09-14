"""Score-blind TRAIN audit of crop-role information lost by mean/std pooling."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from experiments.e65_acquisition import digest, read, write_once
from experiments.e71_features import aggregate, read_chunk, check_population
from pixelproof.project_paths import DATA_ROOT, ML_ROOT

ROOT = DATA_ROOT/'e104'
EVIDENCE = ML_ROOT.parent/'evidence'
CONTRACT = ROOT/'audit_contract.json'
REPORT = ROOT/'audit.json'
MANIFEST = DATA_ROOT/'e54/data_contract_v2.json'
OLD_CONTRACT = DATA_ROOT/'e71/features_contract.json'
OLD_REPORT = DATA_ROOT/'e71/features.json'
INVENTORY = DATA_ROOT/'e71/legacy_inventory.json'


def context_coordinates(raw):
    """Preserve center-context identity separately from the two texture crops.

Crop0 is the historical resized CENTER crop, not an uncropped full frame.
No fitting, classification, normalization, label or threshold is involved.
"""
    raw = np.asarray(raw)
    if raw.ndim != 3 or raw.shape[1] != 3 or not raw.shape[0] or not raw.shape[2] or \
            raw.dtype != np.float32 or not np.isfinite(raw).all():
        raise ValueError('finite float32 [view, three ordered crops, width] required')
    center = raw[:, 0].astype(np.float64)
    local = raw[:, 1:].astype(np.float64).mean(axis=1)
    return np.concatenate([center, center-local], axis=1)


def selected_indices(rows):
    if not rows or any(str(r['role']).upper() != 'TRAIN' for r in rows) or \
            len({r['parent_id'] for r in rows}) != len(rows):
        raise ValueError('unique TRAIN parents only')
    return sorted(min((i for i, r in enumerate(rows) if (r['label'], r['source']) == group),
        key=lambda i: hashlib.sha256(('E104|'+rows[i]['parent_id']).encode()).digest())
        for group in sorted({(r['label'], r['source']) for r in rows}))


def freeze():
    old = read(OLD_CONTRACT); receipt = read(EVIDENCE/'e71_features.json')
    if digest(OLD_REPORT) != receipt['report_sha256'] or digest(OLD_CONTRACT) != receipt['contract_sha256']:
        raise ValueError('E71 receipt binding differs')
    for p in (MANIFEST, INVENTORY, Path(aggregate.__code__.co_filename)):
        if digest(p) != old['inputs'][str(p)]:
            raise ValueError('E71 bound TRAIN/code input differs')
    rows = read(MANIFEST)['rows']; check_population(rows)
    cached = read(INVENTORY)['cached']; chunks = read(OLD_REPORT)['chunks']; selected = []
    for i in selected_indices(rows):
        if str(i) in cached:
            path = Path(cached[str(i)]['path']); binding = old['legacy_contract_sha256']
        else:
            path = DATA_ROOT/'e71/chunks'/f'{i:05d}.npz'; binding = digest(OLD_CONTRACT)
        expected = chunks[str(path)]
        if digest(path) != expected:
            raise ValueError('Selected TRAIN raw chunk differs')
        selected.append({'index': i, 'parent_id': rows[i]['parent_id'], 'label': rows[i]['label'],
            'source': rows[i]['source'], 'path': str(path), 'sha256': expected, 'binding': binding})
    inputs = [Path(__file__), MANIFEST, OLD_CONTRACT, OLD_REPORT, INVENTORY,
              Path(aggregate.__code__.co_filename), ML_ROOT/'src/pixelproof/e32_candidate.py',
              ML_ROOT/'experiments/e42_features.py']
    c = {'state': 'E104_score_blind_crop_role_audit_registered',
        'inputs': {str(p): digest(p) for p in inputs}, 'selected': selected,
        'selection': 'One SHA256(E104|parent_id)-first TRAIN parent per label/source; all three old conditions.',
        'metrics': 'Exact raw->saved mean/std replay; center/local swap pooled roundoff and signed role '
            'coordinate change. Descriptive center-minus-local L2 only; no classifier score or error selection.',
        'conditions': ['clean', 'assigned_transport', 'q75'],
        'limits': 'A lossy representation property, not a demonstrated cause of any error or a trained '
            'reviewer. Historical global crop is resized-center JPEG90, not an uncropped full frame. '
            'Only legacy TRAIN raw cache is audited; social/new-cohort raw coverage is not established.',
        'new_image_reads': 0, 'dev_final_rows_read': 0, 'downloads': 0, 'fit_allowed': False}
    ROOT.mkdir(parents=True, exist_ok=True)
    write_once(CONTRACT, c)
    write_once(EVIDENCE/'e104_context_contract.json', {k:v for k,v in c.items() if k != 'selected'} |
        {'contract_sha256': digest(CONTRACT), 'selected_parents': len(selected)})
    return {'state': c['state'], 'selected_parents': len(selected)}


def audit():
    c = read(CONTRACT)
    if digest(CONTRACT) != read(EVIDENCE/'e104_context_contract.json')['contract_sha256']:
        raise ValueError('E104 contract changed')
    for p, sha in c['inputs'].items():
        if digest(p) != sha:
            raise ValueError('E104 input changed')
    records = []
    for item in c['selected']:
        raw, saved = read_chunk(Path(item['path']), item['binding'], item['parent_id'], item['sha256'])
        if not np.array_equal(aggregate(raw), saved):
            raise ValueError('Historical raw aggregation changed')
        swapped = raw[:, [1, 0, 2], :]
        change = float(np.abs(aggregate(swapped)-saved).max())
        # Pooling is permutation invariant algebraically; retain float32 roundoff as measured.
        coordinates = context_coordinates(raw); other = context_coordinates(swapped)
        records.append({'label': item['label'], 'source': item['source'],
            'pooled_swap_max_abs_error': change,
            'role_coordinate_swap_l2': np.linalg.norm(coordinates-other, axis=1).tolist(),
            'center_minus_local_l2': np.linalg.norm(coordinates[:, 768:], axis=1).tolist()})
    differences = np.array([r['center_minus_local_l2'] for r in records])
    result = {'state': 'E104_score_blind_crop_role_audit_complete',
        'contract_sha256': digest(CONTRACT), 'TRAIN_parents': len(records), 'TRAIN_views': len(records)*3,
        'saved_raw_aggregate_exact': True,
        'max_pooled_swap_float32_roundoff': max(r['pooled_swap_max_abs_error'] for r in records),
        'min_role_coordinate_swap_l2': min(min(r['role_coordinate_swap_l2']) for r in records),
        'center_minus_local_l2_quantiles': {q: float(v) for q,v in zip(
            ['min', 'median', 'max'], np.quantile(differences, [0, .5, 1]))},
        'by_source': records, 'new_image_reads': 0, 'classifier_scores': 0,
        'dev_final_rows_read': 0, 'downloads': 0, 'fit_allowed': False, 'limitations': c['limits']}
    write_once(REPORT, result); write_once(EVIDENCE/'e104_context_audit.json', result)
    return {k:v for k,v in result.items() if k != 'by_source'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['freeze', 'audit'])
    print(json.dumps({'freeze': freeze, 'audit': audit}[parser.parse_args().stage](), indent=2))
