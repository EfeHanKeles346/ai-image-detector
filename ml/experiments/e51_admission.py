"""Freeze score-blind E51 parent exclusions after completed local admission audits."""

from collections import Counter
import hashlib
import json

from experiments.e51_prefit_audit import load_inputs, ROOT, INPUTS
from experiments.e51_train_cal_realize import _write_atomic
from pixelproof.project_paths import ML_ROOT

OUTPUT = ROOT / 'admission_v1.json'
EVIDENCE = ML_ROOT.parent / 'evidence/e51_admission.json'


def verified_report(name):
    path = ROOT / 'audit' / name
    raw = path.read_bytes()
    report = json.loads(raw)
    evidence_name = {'protected_pixels_v1.json':'e51_protected_pixels.json',
                     'reserve_closure_v1.json':'e51_reserve_closure.json'}[name]
    recorded = json.loads((ML_ROOT.parent / 'evidence' / evidence_name).read_text())
    if hashlib.sha256(raw).hexdigest() != recorded['report_sha256']:
        raise ValueError('admission audit changed after completion')
    if report.get('model_scores_created') != 0:
        raise ValueError('admission report contains model scores')
    return report, hashlib.sha256(raw).hexdigest()


def filter_parents(train, cal, excluded):
    parent_sets = [{r['parent_id'] for r in rows} for rows in (train, cal)]
    if not excluded <= set.union(*parent_sets):
        raise ValueError('unknown admission exclusion')
    for parents in parent_sets:
        if len(parents & excluded) > .05 * len(parents):
            raise ValueError('admission exclusions exceed fixed 5% bound')
    kept_train = [r for r in train if r['parent_id'] not in excluded]
    kept_cal = [r for r in cal if r['parent_id'] not in excluded]
    return kept_train, kept_cal


def admit():
    if OUTPUT.exists() or EVIDENCE.exists():
        raise FileExistsError('admission already frozen')
    main, main_sha = verified_report('protected_pixels_v1.json')
    reserve, reserve_sha = verified_report('reserve_closure_v1.json')
    if (main.get('verified_locations') != 117898 or main.get('verified_query_observations') != 9098
            or reserve.get('unjoined_metadata_rows')):
        raise ValueError('local protected/reserve coverage not closed')
    train, cal = load_inputs()
    exclusions = set(reserve['reserved_identity_overlap'])
    for rows in [*main['matches'].values(), reserve['supplementary_matches']]:
        exclusions.update(r['train_parent'] for r in rows)
    train, cal = filter_parents(train, cal, exclusions)
    originals = [r for r in cal if r['condition'] == 'original']
    devices = Counter(r['device_id'] for r in originals if r['label'] == 0)
    generators = Counter(r['source'] for r in originals if r['label'] == 1)
    if len(devices) != 30 or min(devices.values()) < 30 or len(generators) != 18 or min(generators.values()) < 15:
        raise ValueError('CAL source/device coverage depleted')
    report = {'schema_version':1, 'state':'e51_admitted_before_features_and_fit',
              'training_authorized':True, 'model_scores_created':0,
              'input_manifest_sha256':{role:sha for role,(_,sha) in INPUTS.items()},
              'protected_pixels_sha256':main_sha, 'reserve_closure_sha256':reserve_sha,
              'excluded_parent_ids':sorted(exclusions), 'train':train, 'cal':cal,
              'train_parents':len(train), 'cal_parents':len(originals), 'cal_observations':len(cal),
              'cal_real_device_counts':dict(devices), 'cal_ai_source_counts':dict(generators),
              'unmaterialized_reserve_limitation':reserve['limitation'],
              'policy':'Score-blind conservative whole-parent exclusion; no class changes; original manifests preserved.'}
    raw = _write_atomic(OUTPUT, report)
    summary = {k:v for k,v in report.items() if k not in {'train','cal'}}
    summary['admission_sha256'] = hashlib.sha256(raw).hexdigest()
    _write_atomic(EVIDENCE, summary)
    return summary


if __name__ == '__main__':
    print(json.dumps(admit(), indent=2))
