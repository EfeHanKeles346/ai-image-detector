"""Metadata-only lineage audit; does not make historical CAL fresh or admit data."""
from collections import Counter
import json

from experiments.e65_acquisition import digest, read, write_once
from pixelproof.project_paths import DATA_ROOT, ML_ROOT


def main():
    paths = [DATA_ROOT / p for p in ['e54/data_contract_v2.json',
             'e72/training_manifest.json', 'e88/training_manifest.json']]
    train = [r for p in paths for r in read(p)['rows']]
    assert len(train) == 12269 and len({r['sha256'] for r in train}) == 12269
    assert all(r['role'].upper() == 'TRAIN' for r in train)
    train_sha = {r['sha256'] for r in train}
    reports = []
    for name in ['e32/c3_role_manifest.json', 'e32/r1b_role_manifest.json']:
        path = DATA_ROOT / name; paths.append(path)
        rows = read(path)['records']
        cal = [r for r in rows if r['role'] == 'CALIBRATION']
        selected_train = [r for r in rows if r['sha256'] in train_sha]
        shared = train_sha & {r['sha256'] for r in cal}
        groups = {}
        for key in ['role_group', 'parent_group', 'scene_group']:
            # Scope keys by source: unrelated publishers reuse simple group names.
            left = {(r['source_id'], str(r[key])) for r in selected_train if r.get(key)}
            right = {(r['source_id'], str(r[key])) for r in cal if r.get(key)}
            groups[key] = {'train_groups': len(left), 'cal_groups': len(right),
                           'shared_groups': len(left & right)}
        reports.append({'manifest': name, 'cal_rows': len(cal),
                        'cal_classes': dict(Counter(r['label'] for r in cal)),
                        'e92_train_rows_mapped_by_sha': len(selected_train),
                        'exact_train_cal_overlap': len(shared), 'group_overlap': groups,
                        'fresh_calibration': False,
                        'history': 'Previously consumed E32 model-selection CAL; not a new independent test.'})
    report = {'state': 'E98_metadata_role_audit_complete', 'e92_train_parents': len(train),
              'train_classes': dict(Counter(str(r['label']) for r in train)), 'cal_manifests': reports,
              'inputs': {str(p.relative_to(DATA_ROOT)): digest(p) for p in paths},
              'code_sha256': digest(__file__), 'pixel_reads': 0, 'model_scores': 0,
              'role_changes': 0, 'calibration_fit_allowed': False,
              'limitations': 'Mapped source-scoped groups and byte identities only; no new semantic '
                  'independence proof, no inference of missing identifiers. Older CAL remains protected '
                  'from TRAIN; the two CAL manifests overlap and their counts must not be added.'}
    write_once(ML_ROOT.parent / 'evidence/e98_role_audit.json', report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
