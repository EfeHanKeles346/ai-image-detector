"""Original-parent overlap with the already protected DDA-COCO val2017 manifest."""
from collections import Counter, defaultdict
import json
from pathlib import Path
import re
from experiments.e65_acquisition import digest, read, write_once
from pixelproof.project_paths import DATA_ROOT, ML_ROOT


def original_id(row):
    parent = re.fullmatch(r'dda-coco:(\d+)', row['parent_id'])
    member = re.fullmatch(r'(?:DDA-COCO/[^/]+/)?val2017/(\d+)\.jpg', row['member'])
    if not parent or not member or int(parent.group(1)) != int(member.group(1)):
        raise ValueError('Protected parent/member COCO identity mismatch')
    return int(parent.group(1))


def audit():
    evidence = ML_ROOT.parent/'evidence'; root = DATA_ROOT/'e117'; root.mkdir(exist_ok=True)
    lineage = DATA_ROOT/'e116/lineage.json'
    if digest(lineage) != read(evidence/'e116_cocoglide_lineage.json')['lineage_sha256']:
        raise ValueError('Exact author lineage differs')
    protected_receipt = evidence/'e51_protected_inventory.json'
    candidates = [r for r in read(protected_receipt)['sources'] if '/e43_dda_coco/' in r['path']]
    if len(candidates) != 1: raise ValueError('One pinned protected DDA-COCO manifest required')
    pin = candidates[0]; protected_path = Path(pin['path'])
    if digest(protected_path) != pin['sha256']: raise ValueError('Protected DDA manifest changed')
    protected = read(protected_path)['rows']; parents = defaultdict(list)
    for row in protected:
        parents[original_id(row)].append(row['record_id'])
    rows = []
    for row in read(lineage)['rows']:
        matches = parents.get(row['coco_id'], [])
        rows.append({'compiled_index': row['compiled_index'], 'coco_id': row['coco_id'],
            'historically_exposed_first120': row['historically_exposed_first120'],
            'prior_perceptual_quarantine': row['overlap_quarantined'],
            'protected_original_parent_match': bool(matches), 'protected_derivative_records': len(matches),
            'training_allowed': False})
    write_once(root/'rows.json', {'rows': rows, 'lineage_sha256': digest(lineage),
        'protected_manifest_sha256': digest(protected_path)})
    matched = [r for r in rows if r['protected_original_parent_match']]
    result = {'state': 'E117_original_COCO_ancestry_overlap_audited', 'code_sha256': digest(Path(__file__)),
        'lineage_sha256': digest(lineage), 'protected_manifest_sha256': digest(protected_path),
        'rows_sha256': digest(root/'rows.json'), 'author_parents': len(rows),
        'protected_derivative_records': len(protected), 'protected_unique_COCO_parents': len(parents),
        'matched_original_parents': len(matched),
        'matched_first120': sum(r['historically_exposed_first120'] for r in matched),
        'newly_flagged_beyond_perceptual_audit': sum(not r['prior_perceptual_quarantine'] for r in matched),
        'derivatives_per_matching_parent': dict(Counter(r['protected_derivative_records'] for r in matched)),
        'image_pixels_read': 0, 'model_scores': 0, 'training_allowed': False,
        'limits': 'Exact original COCO IDs recovered by author real/edit/mask pixel triples. Different generated descendants can share the protected original even without perceptual similarity. This audit targets pinned DDA-COCO ancestry only; no negative comprehensive-independence claim.',
        'next': 'Exclude every matched original and all descendants from new Model2 TRAIN/CAL admission under current protected-parent policy. Retain only consumed research diagnostics. Seek eligible non-overlapping originals with explicit ancestry; do not weaken the rule to reuse this convenient corpus.'}
    write_once(root/'report.json', result); write_once(evidence/'e117_coco_ancestry.json', result)
    return result


if __name__ == '__main__':
    print(json.dumps(audit(), indent=2))
