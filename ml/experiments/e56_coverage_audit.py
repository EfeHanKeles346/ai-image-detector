"""Metadata-only TRAIN coverage diagnosis; no new fit, image reads or test scores."""
from collections import Counter, defaultdict
import json

from experiments.e53_offline import EVIDENCE, digest, fixed_write, publisher
from experiments.e54_data import CONTRACT, ROOT, load


def summarize(rows, facts, folds):
    parents = [r['parent_id'] for r in rows]
    if len(set(parents)) != len(parents) or len(facts) != len(rows):
        raise ValueError('duplicate/incomplete parents')
    colour = {r['parent_id']: r for r in facts}
    if set(colour) != set(parents):
        raise ValueError('colour identity mismatch')
    for row in rows:
        fact = colour[row['parent_id']]
        if row['label'] not in (0, 1) or any(row[k] != fact[k] for k in ('label', 'source', 'native')):
            raise ValueError('colour/class/source metadata mismatch')
        if not isinstance(fact['near_monochrome_global_crop'], bool):
            raise ValueError('invalid colour descriptor')

    def count(population):
        out = {'parents': len(population), 'real': 0, 'ai': 0, 'monochrome': 0,
               'exact224': 0, 'square': 0, 'short_side_ge512': 0,
               'device_metadata_present': 0, 'scene_metadata_present': 0,
               'topic_metadata_present': 0}
        for row in population:
            w, h = row['width'], row['height']
            if not isinstance(w, int) or not isinstance(h, int) or min(w, h) <= 0:
                raise ValueError('missing verified geometry')
            out['real' if row['label'] == 0 else 'ai'] += 1
            out['monochrome'] += colour[row['parent_id']]['near_monochrome_global_crop']
            out['exact224'] += w == h == 224
            out['square'] += w == h
            out['short_side_ge512'] += min(w, h) >= 512
            out['device_metadata_present'] += bool(row.get('device'))
            out['scene_metadata_present'] += bool(row.get('scene_group'))
            out['topic_metadata_present'] += bool(row.get('topic'))
        return out

    by_publisher = defaultdict(list)
    for row in rows:
        by_publisher[publisher(row['source'])].append(row)
    groups = {p: count(rs) for p, rs in sorted(by_publisher.items())}
    exposure = []
    for fold in folds:
        if len(fold['roles']) != len(rows):
            raise ValueError('role count mismatch')
        role_publishers = defaultdict(set)
        strata = defaultdict(list)
        for row, role in zip(rows, fold['roles'], strict=True):
            if role not in {'FIT', 'CAL', 'VALIDATION', 'EXCLUDED'}:
                raise ValueError('unknown role')
            if role != 'EXCLUDED':
                role_publishers[publisher(row['source'])].add(role)
            strata[f'{role}:label{row["label"]}'].append(row)
        if any(len(rs) != 1 for rs in role_publishers.values()):
            raise ValueError('publisher crosses roles')
        fit = [r for r, role in zip(rows, fold['roles'], strict=True) if role == 'FIT']
        exposure.append({'fold': fold['fold'], 'strata': {k: count(rs) for k, rs in sorted(strata.items())},
                         'fit_publishers_by_class': {str(label): sorted({publisher(r['source']) for r in fit if r['label'] == label}) for label in (0, 1)}})
    single = [p for p, group in groups.items() if min(group['real'], group['ai']) == 0]
    return {'publisher_groups': groups, 'single_class_publishers': single,
            'parents_in_single_class_publishers': sum(groups[p]['parents'] for p in single),
            'classes': {str(label): count([r for r in rows if r['label'] == label]) for label in (0, 1)},
            'fold_exposure': exposure,
            'field_presence': dict(Counter(k for r in rows for k, v in r.items() if v is not None))}


def run():
    data = load()
    colour_receipt = json.loads((EVIDENCE/'e54_color_audit.json').read_text())
    path = ROOT/'color_descriptors.json'
    if digest(path) != colour_receipt['descriptors_sha256']:
        raise ValueError('colour input changed')
    result = {'state': 'E56_metadata_coverage_audit_complete', 'code_sha256': digest(__file__),
              'inputs': {str(CONTRACT): digest(CONTRACT), str(path): digest(path),
                         str(EVIDENCE/'e54_color_audit.json'): digest(EVIDENCE/'e54_color_audit.json')},
              'coverage': summarize(data['rows'], json.loads(path.read_text())['records'], data['folds']),
              'new_images_read': 0, 'new_model_scores': 0, 'candidate_saved': False,
              'limitations': ['Descriptive admitted TRAIN inventory, not causal or independent test evidence.',
                             'Absent fields mean unknown; field presence does not validate metadata accuracy.',
                             'Publisher grouping follows the existing frozen protocol, not semantic identity.',
                             'Source-class association alone does not prove the model exploits it.',
                             'Dimensions and monochrome flags are descriptors, never label rules.']}
    fixed_write(EVIDENCE/'e56_coverage_audit.json', result)
    return result


if __name__ == '__main__':
    print(json.dumps(run(), indent=2))
