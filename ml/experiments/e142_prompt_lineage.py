"""Recover declared prompt IDs for already-consumed E36 TRAIN AI, offline."""
import argparse
from collections import Counter, defaultdict
import fcntl
import json
from pathlib import Path
import socket
from experiments.e65_acquisition import digest, read, write_once
from pixelproof.project_paths import DATA_ROOT, ML_ROOT

ROOT = DATA_ROOT / 'e142'
EVIDENCE = ML_ROOT.parent / 'evidence'
CONTRACT = ROOT / 'contract.json'
TRAIN = DATA_ROOT / 'e131/contract.json'
CAL = DATA_ROOT / 'e36/cal_manifest.json'


def recover(train, cal):
    index = {}
    for row in cal:
        if row['label'] != 1:
            continue
        key = (row['source'], row['parent_id'])
        if key in index:
            raise ValueError('Ambiguous original source/prompt identity')
        index[key] = row
    overlay = []
    seen = set()
    for row in train:
        if row['label'] != 1 or not row['source'].startswith('e36:'):
            continue
        if row['parent_id'] in seen or str(row['role']).upper() != 'TRAIN':
            raise ValueError('Unique active TRAIN identities required')
        seen.add(row['parent_id'])
        source = row['source'][4:]
        prefix = 'e36:' + source + ':'
        if not row['parent_id'].startswith(prefix):
            raise ValueError('Unexpected active parent namespace')
        prior = index[(source, row['parent_id'][len(prefix):])]
        if prior['sha256'] != row['sha256'] or prior['role'] != 'cal':
            raise ValueError('Consumed source body/role identity mismatch')
        prompt = prior.get('prompt_id')
        if type(prompt) is not int or not 101 <= prompt <= 200 or prior['parent_id'] != f'qwen-bench:{prompt}':
            raise ValueError('Invalid declared CAL prompt identity')
        overlay.append(dict(parent_id=row['parent_id'], role='TRAIN', source=row['source'],
            sha256=row['sha256'], declared_prompt_group=f'e36:qwen-bench:{prompt}',
            declared_generator=source, base_generator_ancestry_verified=False,
            prompt_text_hash_available=False))
    if not overlay:
        raise ValueError('No eligible active TRAIN parents')
    return overlay


def freeze():
    if digest(TRAIN) != read(EVIDENCE / 'e131_source_holdout_contract.json')['contract_sha256'] or \
            digest(CAL) != read(EVIDENCE / 'e36_cal_manifest.json')['detailed_manifest_sha256']:
        raise ValueError('Frozen input metadata differs')
    files = [TRAIN, CAL, Path(__file__), EVIDENCE / 'e131_source_holdout_contract.json', EVIDENCE / 'e36_cal_manifest.json']
    c = dict(state='E142_prompt_lineage_registered', inputs={str(p): digest(p) for p in files},
        scope='Only current E131 E36 AI TRAIN rows join to E36 consumed CAL metadata by source/parent/body. Read no E36 FINAL or protected reserve records, images, scores, features or weights. Recover declared local prompt IDs, not prompt text or model ancestry. Preserve all old roles/folds.',
        downloads=0, fits=0, model_scores=0, independent_test_created=False)
    write_once(CONTRACT, c)
    write_once(EVIDENCE / 'e142_prompt_lineage_contract.json', {k:v for k,v in c.items() if k != 'inputs'} | {'contract_sha256':digest(CONTRACT)})
    return {'contract_sha256': digest(CONTRACT)}


def run():
    c = read(CONTRACT)
    if digest(CONTRACT) != read(EVIDENCE / 'e142_prompt_lineage_contract.json')['contract_sha256']:
        raise ValueError('Contract changed')
    for path, sha in c['inputs'].items():
        if digest(path) != sha: raise ValueError('Frozen metadata/code changed')
    train = read(TRAIN)
    overlay = recover(train['rows'], read(CAL)['rows'])
    write_once(ROOT / 'train_prompt_overlay.json', {'rows': overlay, 'contract_sha256':digest(CONTRACT)})
    groups = defaultdict(list)
    for row in overlay: groups[row['declared_prompt_group']].append(row['parent_id'])
    report = dict(state='E142_prompt_lineage_complete', contract_sha256=digest(CONTRACT),
        overlay_sha256=digest(ROOT / 'train_prompt_overlay.json'), parents=len(overlay),
        source_counts=dict(Counter(r['source'] for r in overlay)), declared_prompt_groups=len(groups),
        group_size_counts=dict(Counter(str(len(g)) for g in groups.values())),
        cross_component_groups=sum(len({train['components'][p] for p in g}) > 1 for g in groups.values()),
        cross_fold_groups=sum(len({train['outer_fold'][p] for p in g}) > 1 for g in groups.values()),
        downloads=0, fits=0, model_scores=0, independent_test_created=False,
        limits='Declared prompt IDs within one historical publisher namespace. No prompt text/hash recovery, semantic matching across corpora, verified base-generator ancestry, new training admission or score. Current consumed TRAIN only; no final reserve opened.')
    write_once(ROOT / 'report.json', report)
    write_once(EVIDENCE / 'e142_prompt_lineage.json', report)
    return report


if __name__ == '__main__':
    def denied(*args, **kwargs): raise RuntimeError('Metadata recovery is offline')
    socket.socket.connect = denied
    socket.socket.connect_ex = denied
    socket.create_connection = denied
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['freeze','run'])
    ROOT.mkdir(exist_ok=True)
    with (ROOT / 'execution.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        print(json.dumps({'freeze':freeze,'run':run}[parser.parse_args().stage](), indent=2))
