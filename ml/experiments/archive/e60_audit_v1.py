"""Offline E43 exposure audit. No acquisition, fitting, or evaluation scoring."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np

from experiments.e42_train import _fit_view_mask
from experiments.e43_train import E42_FEATURES, E42_FEATURES_SHA256, RR_FEATURES, RR_FEATURES_SHA256
from experiments.e54_data import load as load_data, CONTRACT as DATA_CONTRACT, TEACHER, INDEX
from pixelproof.project_paths import DATA_ROOT, ML_ROOT

ROOT = DATA_ROOT / 'e60'
EVIDENCE = ML_ROOT.parent / 'evidence'
REFERENCE = DATA_ROOT / 'e43/e43_small_predev.joblib'
REFERENCE_SHA = 'a3aec445926bcc8707b3775f01d2cdd9491ba8495ad8a8ec306840556ca47390'
CONTRACT = ROOT / 'audit_contract.json'
RESULT = ROOT / 'audit.json'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    if path.exists():
        raise FileExistsError(f'immutable E60 output already exists: {path}')
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.part')
    temp.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    temp.replace(path)


def verified(path, expected):
    if digest(path) != expected:
        raise ValueError(f'bound input changed: {path}')
    return Path(path)


def metadata(path):
    with np.load(path, allow_pickle=False) as a:
        return {k: a[k] for k in a.files if k != 'features'}


def reconstruct_fit(e42, rr):
    result = {}
    view_counts = Counter()
    for origin, a, mask in [('e42', e42, _fit_view_mask(e42)),
                            ('rr', rr, rr['roles'] == 'train')]:
        for p, y, s in zip(a['parent_ids'][mask], a['labels'][mask], a['sources'][mask], strict=True):
            p = str(p)
            item = {'parent_id': p, 'label': int(y), 'source': str(s), 'origin': origin}
            if p in result and result[p] != item:
                raise ValueError('teacher parent has conflicting label/source')
            result[p] = item
            view_counts[p] += 1
    return [result[p] | {'views': view_counts[p]} for p in sorted(result)]


def compare_population(old, current, cal_ids):
    known = {r['parent_id']: r for r in old}
    present = {r['parent_id']: r for r in current}
    if len(present) != len(current):
        raise ValueError('duplicate current parent')
    shared = known.keys() & present.keys()
    if any(known[p]['label'] != present[p]['label'] for p in shared):
        raise ValueError('teacher/current label direction changed')
    missing = [r | {'reason': 'protected_E51_CAL' if r['parent_id'] in cal_ids else
                    'outside_current_admission_do_not_restore_historical_roles'}
               for r in old if r['parent_id'] not in present]
    return {'parents': len(current), 'labels': dict(Counter(r['label'] for r in current)),
            'teacher_shared': len(shared), 'teacher_missing': missing,
            'teacher_missing_by_label': dict(Counter(r['label'] for r in missing)),
            'teacher_missing_by_reason': dict(Counter(r['reason'] for r in missing)),
            'teacher_shared_ai': sum(known[p]['label'] == 1 for p in shared)}


def unused_native(records, inventory, current):
    allowed = set(inventory['remaining_record_ids'])
    used = {r['parent_id'].removeprefix('e32:') for r in current}
    excluded = {p.removeprefix('e32:') for pair in inventory['internal_pairs']
                for p in (pair['train_parent'], pair['cal_parent'])}
    seen_groups = {(r['source_id'], r['role_group']) for r in records if r['record_id'] in used}
    unused = [r for r in records if r['record_id'] in allowed - used - excluded]
    unseen_groups = [r for r in unused if (r['source_id'], r['role_group']) not in seen_groups]
    return {'unused_parents': len(unused),
            'unused_sources': dict(Counter(r['source_id'] for r in unused)),
            'unused_recorded_group_parents': len(unseen_groups),
            'unused_recorded_group_labels': dict(Counter(r['label'] for r in unseen_groups)),
            'unused_recorded_group_sources': dict(Counter(r['source_id'] for r in unseen_groups)),
            'balanced_group_disjoint_development_available':
                {r['label'] for r in unseen_groups} == {'real', 'ai'},
            'limitation': 'Unused parents are not a fresh publisher test. No new role admission; '
                          'single-image groups do not establish unknown prompt independence.'}


def freeze():
    load_data()  # Verify the existing admitted TRAIN chain; no new role selection.
    crop = read(EVIDENCE / 'e54_crop_cache.json')
    pins = {REFERENCE: REFERENCE_SHA, E42_FEATURES: E42_FEATURES_SHA256,
            RR_FEATURES: RR_FEATURES_SHA256, TEACHER: crop['teacher_sha256'],
            INDEX: crop['index_sha256'], DATA_CONTRACT: crop['contract_sha256'],
            DATA_ROOT / 'e51/admission_v1.json': read(EVIDENCE / 'e51_admission.json')['admission_sha256'],
            DATA_ROOT / 'e53/native_inventory_result.json': read(EVIDENCE / 'e53_native_inventory.json')['report_sha256'],
            DATA_ROOT / 'e32/c3_role_manifest.json': '0b6656a25762dbb097d18634d4193282e7c03d2e6bbe257f31471434f67b91eb'}
    # The later blind reserve screens cover all 11,630 current TRAIN parents.
    for family, prefix in [('mnw', 'mnw'), ('hdrplus', 'hdrplus')]:
        receipt = read(EVIDENCE / f'e54_{prefix}_overlap.json')
        if receipt['cross_match_observations'] != 0:
            raise ValueError('later reserve overlaps current TRAIN')
        if family == 'mnw' and receipt['current_train_parents_covered'] != 11630:
            raise ValueError('later protection does not cover current TRAIN')
        pins[DATA_ROOT / f'e52/{family}_reserve_v1/overlap.json'] = receipt['report_sha256']
        pins[EVIDENCE / f'e54_{prefix}_overlap.json'] = digest(EVIDENCE / f'e54_{prefix}_overlap.json')
    inputs = {str(p): digest(verified(p, sha)) for p, sha in pins.items()}
    for rel in ['e42/parent_manifest.json', 'e43/rr_roles.json', 'e51/manifests/train_parents_unscored.json',
                'e51/development/manifest_unscored.json', 'e53/expansion_contract.json']:
        p = DATA_ROOT / rel
        inputs[str(p)] = digest(p)
    helpers = ['e42_train.py', 'e42_features.py', 'e43_train.py', 'e54_data.py']
    inputs.update({str(Path(__file__).with_name(n)): digest(Path(__file__).with_name(n)) for n in helpers})
    value = {'state': 'E60_audit_frozen_before_analysis', 'code_sha256': digest(__file__),
             'inputs': inputs, 'source_downloads': 0, 'model_scores_created': 0,
             'policy': 'Existing admitted E54 TRAIN only; missing historical teacher rows are '
                       'not automatically restored. No protected final scoring or new split admission.'}
    write(CONTRACT, value)
    write(EVIDENCE / 'e60_audit_contract.json', value | {'contract_sha256': digest(CONTRACT)})
    return {'state': value['state'], 'inputs': len(inputs)}


def verify_contract():
    value = read(CONTRACT)
    verified(__file__, value['code_sha256'])
    verified(CONTRACT, read(EVIDENCE / 'e60_audit_contract.json')['contract_sha256'])
    for path, sha in value['inputs'].items():
        verified(path, sha)
    return value


def run():
    verify_contract()
    old = reconstruct_fit(metadata(E42_FEATURES), metadata(RR_FEATURES))
    if len(old) != 8844 or sum(r['views'] for r in old) != 19648:
        raise ValueError('teacher FIT membership does not reproduce')
    admitted = read(DATA_ROOT / 'e51/admission_v1.json')
    data = read(DATA_CONTRACT)
    cal_ids = {r['parent_id'] for r in admitted['cal']}
    old_ids = {r['parent_id'] for r in old}
    rows = data['rows']
    if len(rows) != 11630 or any(r['role'] != 'TRAIN' or r['label'] not in (0, 1) for r in rows):
        raise ValueError('non-admitted role or label in current TRAIN')
    native = unused_native(read(DATA_ROOT / 'e32/c3_role_manifest.json')['records'],
                           read(DATA_ROOT / 'e53/native_inventory_result.json'), rows)
    comparisons = {name: compare_population(old, pool, cal_ids) for name, pool in
                   [('E51', admitted['train']), ('E54', rows)]}
    e43_by_parent = {r['parent_id']: r for r in read(DATA_ROOT / 'e42/parent_manifest.json')['rows']}
    shared = [r for r in rows if r['parent_id'] in e43_by_parent]
    if any(r['sha256'] != e43_by_parent[r['parent_id']]['sha256'] for r in shared):
        raise ValueError('shared parent encoded body differs from historical teacher')
    report = {'state': 'E60_exposure_audit_complete_no_fit', 'contract_sha256': digest(CONTRACT),
              'reference_sha256': REFERENCE_SHA, 'teacher_fit_parents': len(old),
              'teacher_fit_views': sum(r['views'] for r in old), 'teacher_fit': old,
              'teacher_labels': dict(Counter(r['label'] for r in old)),
              'teacher_source_counts': dict(Counter(r['source'] for r in old)),
              'populations': comparisons, 'shared_encoded_body_parity': len(shared),
              'e54_validation_teacher_seen_counts': [sum(role == 'VALIDATION' and r['parent_id'] in old_ids
                  for r, role in zip(rows, fold['roles'], strict=True)) for fold in data['folds']],
              'unused_native_audit': native, 'admitted_ai_replay_parents': sum(r['label'] == 1 for r in rows),
              'admitted_real_parents': sum(r['label'] == 0 for r in rows),
              'evaluation': {'fresh_balanced_group_disjoint_cached_dev': False,
                  'reason': 'E54 source folds consumed and overlap teacher FIT; E51 DEV consumed; '
                            'unused native recorded groups supply no REAL; E52/MNW/HDR+/Module2 protected.',
                  'permitted_next': 'One bounded TRAIN engineering correction. Consumed E49 can only '
                                    'be a preregistered regression diagnosis after candidate freeze.'},
              'image_inference': 0, 'source_downloads': 0, 'training_authorized_by_this_report': False,
              'limitations': ['Admission is inherited from bound audits, not a new exhaustive semantic duplicate screen.',
                             'All eligible replay means every AI parent in the admitted 11,630-parent pool; '
                             'remaining unadmitted native candidates and historical protected parents are not silently included.',
                             'No independent quality or serving-promotion claim is available from this audit.']}
    write(RESULT, report)
    compact = {k: v for k, v in report.items() if k not in {'teacher_fit', 'populations'}}
    compact['populations'] = {k: {a: b for a, b in v.items() if a != 'teacher_missing'}
                              for k, v in comparisons.items()}
    compact['report_sha256'] = digest(RESULT)
    write(EVIDENCE / 'e60_audit.json', compact)
    return compact


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=['freeze', 'run'])
    print(json.dumps({'freeze': freeze, 'run': run}[parser.parse_args().phase](), indent=2))
