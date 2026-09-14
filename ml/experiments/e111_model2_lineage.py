"""Score-blind Model2 body/perceptual overlap audit against existing role fingerprints."""
import argparse
from collections import Counter
import json
from pathlib import Path
import time
from experiments.e65_acquisition import digest, read, write_once
from experiments.e65_audit import fingerprint, cross_role_matches
from experiments.e100_audit import references
from pixelproof.project_paths import DATA_ROOT, LEGACY_DATA_ROOT, ML_ROOT, WORK_ROOT

ROOT = DATA_ROOT/'e111'; EVIDENCE = ML_ROOT.parent/'evidence'; CONTRACT = ROOT/'contract.json'
INVENTORY = WORK_ROOT/'model2_local_inventory_2026-09-14.json'


def freeze():
    previous = read(DATA_ROOT/'e100/audit_contract.json')
    if digest(DATA_ROOT/'e100/audit_contract.json') != read(EVIDENCE/'e100_audit_contract.json')['contract_sha256']:
        raise ValueError('Protected reference receipt differs')
    for p in previous['reference_files']:
        if digest(p) != previous['inputs'][p]: raise ValueError('Protected reference changed')
    if digest(DATA_ROOT/'e100/audit.json') != read(EVIDENCE/'e100_audit.json')['report_sha256'] or \
            digest(INVENTORY) != read(EVIDENCE/'model2_local_inventory_2026-09-14.json')['inventory_sha256'] or \
            digest(DATA_ROOT/'e105/download.json') != read(EVIDENCE/'e105_download.json')['receipt_sha256']:
        raise ValueError('Bound source receipt differs')
    refs = previous['reference_files']+[str(DATA_ROOT/'e100/audit.json')]
    files = [Path(__file__), Path(fingerprint.__code__.co_filename),
        Path(references.__code__.co_filename), INVENTORY, DATA_ROOT/'e105/download.json',
        DATA_ROOT/'e100/audit_contract.json']+list(map(Path, refs))
    c = {'state': 'E111_Model2_lineage_audit_registered', 'reference_files': refs,
        'inputs': {str(p): digest(p) for p in files},
        'selection': 'All512 CocoGlide edits plus their512 original pointers and all512 fixed DiffSeg30k images; no scores or refills.',
        'matching': 'Canonical native fingerprint: exact body/RGB or dHash<=4 AND pHash63<=4. Propagate edit/auth matches to CocoGlide parent; internal cross-parent matches recorded.',
        'limits': 'No model inference, masks not used for selection. Match-free does not establish COCO/prompt or generator independence; no automatic TRAIN/CAL/final admission.'}
    ROOT.mkdir(exist_ok=True); write_once(CONTRACT, c)
    write_once(EVIDENCE/'e111_model2_contract.json', {k:v for k,v in c.items() if k not in ('inputs', 'reference_files')} |
        {'contract_sha256': digest(CONTRACT)})
    return {'registered_images': 1536, 'reference_files': len(refs)}


def audit():
    c = read(CONTRACT)
    if digest(CONTRACT) != read(EVIDENCE/'e111_model2_contract.json')['contract_sha256']:
        raise ValueError('Contract differs')
    for p, sha in c['inputs'].items():
        if digest(p) != sha: raise ValueError('Input changed')
    selected = []
    for i, row in enumerate(read(INVENTORY)['rows']):
        for kind, path, sha, label in [('edit', row['image'], row['image_sha256'], 1),
                ('auth', str(LEGACY_DATA_ROOT/'manipulation_test'/row['auth']), row['auth_sha256'], 0)]:
            selected.append({'path': path, 'sha256': sha, 'label': label,
                'parent_id': f'CocoGlide:{i}:{kind}', 'source': 'CocoGlide',
                'ancestry_group': f'CocoGlide:{i}', 'condition': kind})
    for row in read(DATA_ROOT/'e105/download.json')['rows']:
        f = next(f for f in row['files'] if f['member'].endswith('.image.png'))
        selected.append({'path': f['path'], 'sha256': f['sha256'], 'label': 1,
            'parent_id': 'DiffSeg30k:'+row['id'], 'source': 'DiffSeg30k',
            'ancestry_group': 'DiffSeg30k:'+row['id'], 'condition': 'edit_unknown_ancestry'})
    start = time.monotonic(); facts = []; failures = []
    for row in selected:
        if digest(row['path']) != row['sha256']: raise ValueError('Image identity changed')
        try:
            facts.append(row | fingerprint(Path(row['path']).read_bytes(), max_pixels=32_000_000))
        except (ValueError, OSError) as exc:
            failures.append({'parent_id': row['parent_id'], 'reason': str(exc)})
    refs = references(c); cross = cross_role_matches(facts, refs)
    print(json.dumps({'E111_decoded': len(facts), 'reference_records': len(refs), 'cross_matches': len(cross)}), flush=True)
    groups = {r['parent_id']: r['ancestry_group'] for r in facts}
    internal = [m for m in cross_role_matches(facts, facts)
                if m['train_parent'] < m['cal_parent'] and groups[m['train_parent']] != groups[m['cal_parent']]]
    blocked = {groups[m['train_parent']] for m in cross}
    detail = {'records': facts, 'cross_matches': cross, 'internal_cross_parent_matches': internal,
        'failures': failures, 'contract_sha256': digest(CONTRACT)}
    write_once(ROOT/'audit.json', detail)
    result = {'state': 'E111_Model2_lineage_audit_complete', 'contract_sha256': digest(CONTRACT),
        'report_sha256': digest(ROOT/'audit.json'), 'selected_images': len(selected), 'decoded': len(facts),
        'decode_failures': len(failures), 'reference_records': len(refs),
        'cross_match_observations': len(cross), 'overlap_affected_candidate_groups': dict(Counter(x.split(':')[0] for x in blocked)),
        'internal_cross_parent_match_observations': len(internal), 'seconds': time.monotonic()-start,
        'model_scores': 0, 'training_allowed': False, 'limits': c['limits'],
        'next': 'Resolve original COCO/prompt ancestry and transitive parent groups before assigning research roles. Keep all matched parent derivatives out of new independent evaluation.'}
    write_once(ROOT/'summary.json', result); write_once(EVIDENCE/'e111_model2_lineage.json', result)
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('stage', choices=['freeze', 'audit'])
    print(json.dumps({'freeze': freeze, 'audit': audit}[p.parse_args().stage](), indent=2))
