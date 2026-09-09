"""Model-blind overlap screening of the evaluation-only MNW reserve."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
from io import BytesIO
import json
from pathlib import Path
import sqlite3

from PIL import Image, ImageOps

from experiments.e51_prefit_audit import VERSION, fingerprint, cross_role_matches
from experiments.e51_protected_pixels import DATABASE
from experiments.e51_train_cal_realize import _q75
from experiments.e53_offline import EVIDENCE, digest, fixed_write
from experiments.e54_data import CONTRACT as TRAIN_CONTRACT
from experiments.e54_mnw_reserve import ROOT, MANIFEST
from pixelproof.project_paths import DATA_ROOT

CONTRACT = ROOT/'overlap_contract.json'
RESULT = ROOT/'overlap.json'
NATIVE_DB = DATA_ROOT/'e53/native_fingerprints.sqlite3'
ROUTE = DATA_ROOT/'e51/route/contract_untransferred.json'
DEV = DATA_ROOT/'e51/development/manifest_unscored.json'
CLOSURE = DATA_ROOT/'e51/audit/reserve_closure_v1.json'


def rejected_parents(cross, internal):
    # Both endpoints of a possible within-reserve duplicate are withheld, without
    # choosing by detector score or pretending perceptual similarity proves lineage.
    return ({m['train_parent'] for m in cross}
            | {m[k] for m in internal for k in ('train_parent', 'cal_parent')})


def freeze():
    paths = [MANIFEST, ROOT/'download.json', TRAIN_CONTRACT, DATABASE, NATIVE_DB,
             ROUTE, DEV, CLOSURE, Path(__file__).with_name('e51_prefit_audit.py'),
             Path(__file__).with_name('e51_train_cal_realize.py')]
    download = json.loads((ROOT/'download.json').read_text())
    receipt = json.loads((EVIDENCE/'e54_mnw_reserve_download.json').read_text())
    if digest(ROOT/'download.json') != receipt['report_sha256'] or download['parents'] != 300:
        raise ValueError('unbound or incomplete MNW download')
    value = {'state': 'MNW_overlap_screen_frozen_before_matching',
             'code_sha256': digest(__file__), 'inputs': {str(p): digest(p) for p in paths},
             'canonical_version': VERSION, 'training_allowed': False,
             'model_scores_created': 0, 'balanced_final_admitted': False,
             'policy': 'Withhold cross-protected matches and both internal-pair endpoints. No replacement or score-based filtering.'}
    fixed_write(CONTRACT, value)
    fixed_write(EVIDENCE/'e54_mnw_overlap_contract.json', value | {'contract_sha256': digest(CONTRACT)})
    return value


def run():
    if RESULT.exists():
        raise FileExistsError('MNW overlap report already frozen')
    config = json.loads(CONTRACT.read_text())
    if config['code_sha256'] != digest(__file__):
        raise ValueError('overlap code changed')
    receipt = json.loads((EVIDENCE/'e54_mnw_overlap_contract.json').read_text())
    if digest(CONTRACT) != receipt['contract_sha256']:
        raise ValueError('overlap contract changed')
    for path, expected in config['inputs'].items():
        if digest(path) != expected:
            raise ValueError(f'overlap input changed: {path}')
    refs = {}
    def retain(fact):
        if fact['version'] != VERSION:
            raise ValueError('mixed fingerprint convention')
        sha = fact['sha256']
        if sha in refs and any(fact[k] != refs[sha][k] for k in
                              ('pixel_sha256', 'canonical_dhash', 'phash63')):
            raise ValueError('same bytes have conflicting fingerprints')
        refs[sha] = {**fact, 'parent_id': 'protected:' + sha, 'label': -1, 'condition': 'protected'}
    for path, sql, args in ((DATABASE, 'SELECT facts FROM fingerprints WHERE version=?', (VERSION,)),
                            (NATIVE_DB, 'SELECT facts FROM fingerprints', ())):
        with sqlite3.connect(f'file:{path}?mode=ro', uri=True) as db:
            for (text,) in db.execute(sql, args):
                retain(json.loads(text))
    for fact in json.loads(CLOSURE.read_text())['additional_fingerprints']:
        retain(fact)
    selected = json.loads(DEV.read_text())['rows']
    for fact in selected:
        retain(fact)
    route = json.loads(ROUTE.read_text())['roles']['development_ai']['rows']
    chosen = {r['parent_id'] for r in selected if r['label'] == 1}
    remaining = [r for r in route if r['identity'] not in chosen]
    if (len(route), len(chosen), len(remaining)) != (920, 800, 120):
        raise ValueError('latest AI reserve coverage changed')
    def verify_reserve(row):
        path = DATA_ROOT/'e51/development/ai_reserve'/(row['expected_sha256']+'.image')
        raw = path.read_bytes()
        if len(raw) != row['expected_bytes'] or hashlib.sha256(raw).hexdigest() != row['expected_sha256']:
            raise ValueError('old reserve bytes changed')
        with Image.open(BytesIO(raw)) as im:
            q75 = _q75(ImageOps.exif_transpose(im).convert('RGB'))
        if q75 != (DATA_ROOT/'e51/development/q75'/(row['expected_sha256']+'.jpg')).read_bytes():
            raise ValueError('old reserve Q75 changed')
        return [fingerprint(raw), fingerprint(q75)]
    with ThreadPoolExecutor(max_workers=4) as pool:
        for pair in pool.map(verify_reserve, remaining):
            for fact in pair:
                retain(fact)
    training = json.loads(TRAIN_CONTRACT.read_text())['rows']
    if any(r['sha256'] not in refs for r in training):
        raise ValueError('current TRAIN originals not fully covered')
    queries = json.loads((ROOT/'download.json').read_text())['records']
    for row in queries:
        if digest(row['path']) != row['sha256'] or row['label'] != 1:
            raise ValueError('MNW payload/label changed')
        if row['version'] != VERSION:
            raise ValueError('MNW fingerprint convention changed')
    reference = [refs[s] for s in sorted(refs)]
    cross = cross_role_matches(queries, reference)
    internal = [m for m in cross_role_matches(queries, queries) if m['train_parent'] < m['cal_parent']]
    rejected = rejected_parents(cross, internal)
    value = {'state': 'MNW_identity_screen_complete_still_unscored_reserve',
             'contract_sha256': digest(CONTRACT), 'reference_unique_bodies': len(refs),
             'current_train_parents_covered': len(training), 'latest_ai_reserve_parents_covered': len(route),
             'reference_facts_sha256': hashlib.sha256(json.dumps(reference, sort_keys=True).encode()).hexdigest(),
             'query_parents': len(queries), 'cross_matches': cross, 'internal_pairs': internal,
             'withheld_parents': sorted(rejected),
             'eligible_parent_ids': [r['parent_id'] for r in queries if r['parent_id'] not in rejected],
             'eligible_by_family': dict(Counter(r['family'] for r in queries if r['parent_id'] not in rejected)),
             'model_scores_created': 0, 'training_allowed': False, 'balanced_final_admitted': False,
             'limits': ['Exact RGB/byte plus radius-4 dHash/pHash screening is not exhaustive semantic deduplication.',
                        'Unknown shared prompts: reserve entire publisher, not 300 independent prompt groups.',
                        'Unavailable historical metadata-only bodies cannot receive pixel comparisons.',
                        'AI-only reserve cannot measure REAL safety or pass balanced E52.',
                        'Future TRAIN admission must protect this manifest and overlap report as well as old registries.']}
    fixed_write(RESULT, value)
    summary = {k: v for k, v in value.items() if k not in {'eligible_parent_ids', 'cross_matches', 'internal_pairs'}}
    fixed_write(EVIDENCE/'e54_mnw_overlap.json', summary | {'report_sha256': digest(RESULT),
        'cross_match_observations': len(cross), 'internal_pair_observations': len(internal)})
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('phase', choices=['freeze', 'run'])
    args = parser.parse_args()
    print(json.dumps({'freeze': freeze, 'run': run}[args.phase](), indent=2))
