"""Supplement byte-only E98 checks with original identity lineage for transformed inputs."""
from collections import Counter
import json
from experiments.e65_acquisition import digest, read, write_once
from pixelproof.project_paths import DATA_ROOT, ML_ROOT


def main():
    first = read(ML_ROOT.parent/'evidence/e98_role_audit.json')
    names = ['e54/data_contract_v2.json', 'e32/r1b_role_manifest.json']
    for name in names:
        if digest(DATA_ROOT/name) != first['inputs'][name]:
            raise ValueError('E98 inputs changed')
    train = [r for r in read(DATA_ROOT/names[0])['rows'] if r['source'].startswith('e32:')]
    old = read(DATA_ROOT/names[1])['records']
    by_id = {r['record_id']: r for r in old}; by_sha = {r['sha256']: r for r in old}
    mapped = []
    for row in train:
        by_parent = by_id.get(row['parent_id'].removeprefix('e32:'))
        by_body = by_sha.get(row['sha256'])
        if by_parent and by_body and by_parent['record_id'] != by_body['record_id']:
            raise ValueError('Contradictory parent/body lineage')
        origin = by_parent or by_body
        if origin is None or 'e32:'+origin['source_id'] != row['source']:
            raise ValueError('Unresolved source lineage')
        if int(origin['label'] == 'ai') != row['label']:
            raise ValueError('Label changed across lineage')
        mapped.append(origin)
    cal = [r for r in old if r['role'] == 'CALIBRATION']
    shared = {}
    for field in ['role_group', 'parent_group', 'scene_group']:
        a = {(r['source_id'], str(r[field])) for r in mapped if r.get(field)}
        b = {(r['source_id'], str(r[field])) for r in cal if r.get(field)}
        shared[field] = len(a & b)
    result = {'state': 'E98_original_lineage_supplement_complete', 'e32_train_rows': len(train),
        'mapped_rows': len(mapped), 'unmapped_rows': 0,
        'original_roles': dict(Counter(r['role'] for r in mapped)), 'shared_CAL_groups': shared,
        'inputs': {name: digest(DATA_ROOT/name) for name in names}, 'code_sha256': digest(__file__),
        'pixel_reads': 0, 'model_scores': 0, 'role_changes': 0,
        'limits': 'Original record-id/body lineage resolves transforms missed by byte-only mapping. '
            'This does not make historical CAL fresh or prove semantic scene independence.'}
    write_once(ML_ROOT.parent/'evidence/e98_lineage_audit.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
