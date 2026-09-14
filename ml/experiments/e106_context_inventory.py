"""Verify every legacy TRAIN raw CLIP chunk before extending crop-role features."""
import argparse
import json
from collections import Counter
from pathlib import Path
import numpy as np
from experiments.e65_acquisition import digest, read, write_once
from experiments.e71_features import aggregate, read_chunk, check_population
from experiments.e104_context_audit import context_coordinates
from pixelproof.project_paths import DATA_ROOT, ML_ROOT

ROOT = DATA_ROOT/'e106'
EVIDENCE = ML_ROOT.parent/'evidence'
CONTRACT = ROOT/'contract.json'


def freeze():
    old = DATA_ROOT/'e71/features_contract.json'
    report = DATA_ROOT/'e71/features.json'
    receipt = read(EVIDENCE/'e71_features.json')
    if digest(old) != receipt['contract_sha256'] or digest(report) != receipt['report_sha256']:
        raise ValueError('E71 receipt differs')
    manifest = DATA_ROOT/'e54/data_contract_v2.json'
    inventory = DATA_ROOT/'e71/legacy_inventory.json'
    for p in (manifest, inventory, Path(aggregate.__code__.co_filename)):
        if digest(p) != read(old)['inputs'][str(p)]:
            raise ValueError('E71 upstream input differs')
    rows = read(manifest)['rows']; check_population(rows)
    files = [Path(__file__), old, report, manifest, inventory,
             Path(context_coordinates.__code__.co_filename), Path(aggregate.__code__.co_filename),
             ML_ROOT/'experiments/e42_features.py', ML_ROOT/'src/pixelproof/e32_candidate.py']
    c = {'state': 'E106_full_legacy_TRAIN_raw_inventory_registered',
         'inputs': {str(p): digest(p) for p in files}, 'parents': len(rows),
         'conditions': ['clean', 'assigned_transport', 'q75'],
         'selection': 'Every E54 TRAIN parent in frozen order; no score-based selection.',
         'missing_views_in_full_E103_population': 15210,
         'purpose': 'Body and array hash verification, exact aggregate replay, finite crop-role coordinates. No classifier, image decode or feature fitting.'}
    ROOT.mkdir(parents=True, exist_ok=True); write_once(CONTRACT, c)
    write_once(EVIDENCE/'e106_context_contract.json', c | {'contract_sha256': digest(CONTRACT)})
    return {'registered_parents': len(rows)}


def audit():
    c = read(CONTRACT)
    if digest(CONTRACT) != read(EVIDENCE/'e106_context_contract.json')['contract_sha256']:
        raise ValueError('E106 contract differs')
    for p, sha in c['inputs'].items():
        if digest(p) != sha:
            raise ValueError('E106 input differs')
    old = read(DATA_ROOT/'e71/features_contract.json')
    rows = read(DATA_ROOT/'e54/data_contract_v2.json')['rows']
    cached = read(DATA_ROOT/'e71/legacy_inventory.json')['cached']
    chunks = read(DATA_ROOT/'e71/features.json')['chunks']
    counts = Counter(); total = 0
    for i, row in enumerate(rows):
        if str(i) in cached:
            path = Path(cached[str(i)]['path']); binding = old['legacy_contract_sha256']
        else:
            path = DATA_ROOT/'e71/chunks'/f'{i:05d}.npz'
            binding = digest(DATA_ROOT/'e71/features_contract.json')
        raw, saved = read_chunk(path, binding, row['parent_id'], chunks[str(path)])
        if raw.shape != (3, 3, 768) or not np.array_equal(aggregate(raw), saved):
            raise ValueError('Unexpected raw shape or aggregation')
        if context_coordinates(raw).shape != (3, 1536):
            raise ValueError('Unexpected role coordinates')
        total += path.stat().st_size; counts[f"{row['label']}:{row['source']}"] += 1
        if (i+1) % 1000 == 0:
            print(json.dumps({'E106_verified_parents': i+1}), flush=True)
    r = {'state': 'E106_full_legacy_TRAIN_raw_inventory_complete',
         'contract_sha256': digest(CONTRACT), 'parents_verified': len(rows),
         'views_verified': 3*len(rows), 'raw_chunk_bytes_verified': total,
         'by_label_source': dict(counts), 'all_body_array_hashes_and_aggregates_exact': True,
         'missing_raw_views_for_full_E103_extension': c['missing_views_in_full_E103_population'],
         'new_image_reads': 0, 'dev_final_reads': 0, 'classifier_calls': 0,
         'promotion_allowed': False,
         'next': 'Register extraction of all missing cohorts/conditions before a full-population paired representation fit.'}
    write_once(ROOT/'audit.json', r); write_once(EVIDENCE/'e106_context_inventory.json', r)
    return r


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['freeze', 'audit'])
    print(json.dumps({'freeze': freeze, 'audit': audit}[parser.parse_args().stage](), indent=2))
