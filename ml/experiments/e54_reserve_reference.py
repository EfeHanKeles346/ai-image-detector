"""Reproduce the pinned MNW protected-reference snapshot for later blind audits."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
from io import BytesIO
import json
import sqlite3

from PIL import Image, ImageOps

from experiments.e51_prefit_audit import VERSION, fingerprint
from experiments.e51_train_cal_realize import _q75
from experiments.e53_offline import EVIDENCE, digest, fixed_write
from experiments.e54_mnw_audit import CONTRACT, DATABASE, NATIVE_DB, CLOSURE, DEV, ROUTE, RESULT
from pixelproof.project_paths import DATA_ROOT

SNAPSHOT = DATA_ROOT/'e52/protected_reference_v1.json'


def build():
    config = json.loads(CONTRACT.read_text())
    receipt = json.loads((EVIDENCE/'e54_mnw_overlap_contract.json').read_text())
    if digest(CONTRACT) != receipt['contract_sha256']:
        raise ValueError('reference contract changed')
    for path, expected in config['inputs'].items():
        if digest(path) != expected:
            raise ValueError('reference input changed')
    original_receipt = json.loads((EVIDENCE/'e54_mnw_overlap.json').read_text())
    if digest(RESULT) != original_receipt['report_sha256']:
        raise ValueError('reference audit changed')
    refs = {}
    def retain(fact):
        if fact['version'] != VERSION:
            raise ValueError('mixed fingerprint conventions')
        sha = fact['sha256']
        refs[sha] = {**fact, 'parent_id': 'protected:'+sha, 'label': -1, 'condition': 'protected'}
    for path, sql, args in ((DATABASE, 'SELECT facts FROM fingerprints WHERE version=?', (VERSION,)),
                            (NATIVE_DB, 'SELECT facts FROM fingerprints', ())):
        with sqlite3.connect(f'file:{path}?mode=ro', uri=True) as db:
            for (raw,) in db.execute(sql,args):
                retain(json.loads(raw))
    for fact in json.loads(CLOSURE.read_text())['additional_fingerprints']:
        retain(fact)
    selected = json.loads(DEV.read_text())['rows']
    for fact in selected:
        retain(fact)
    chosen = {r['parent_id'] for r in selected if r['label'] == 1}
    remaining = [r for r in json.loads(ROUTE.read_text())['roles']['development_ai']['rows'] if r['identity'] not in chosen]
    if len(remaining) != 120:
        raise ValueError('incomplete reserve closure')
    def verify(row):
        raw = (DATA_ROOT/'e51/development/ai_reserve'/(row['expected_sha256']+'.image')).read_bytes()
        if len(raw) != row['expected_bytes'] or hashlib.sha256(raw).hexdigest() != row['expected_sha256']:
            raise ValueError('reserve original changed')
        with Image.open(BytesIO(raw)) as im:
            q75 = _q75(ImageOps.exif_transpose(im).convert('RGB'))
        if q75 != (DATA_ROOT/'e51/development/q75'/(row['expected_sha256']+'.jpg')).read_bytes():
            raise ValueError('reserve Q75 changed')
        return [fingerprint(raw),fingerprint(q75)]
    with ThreadPoolExecutor(max_workers=4) as pool:
        for pair in pool.map(verify,remaining):
            for fact in pair:
                retain(fact)
    records = [refs[s] for s in sorted(refs)]
    sha = hashlib.sha256(json.dumps(records,sort_keys=True).encode()).hexdigest()
    if sha != original_receipt['reference_facts_sha256']:
        raise ValueError('reconstructed reference differs from frozen MNW audit')
    value = {'state':'canonical_reference_reproduced_exactly','records':records,
             'unique_bodies':len(records),'facts_sha256':sha,'code_sha256':digest(__file__),
             'mnw_overlap_contract_sha256':digest(CONTRACT),'model_scores_created':0}
    fixed_write(SNAPSHOT,value)
    summary = {k:v for k,v in value.items() if k != 'records'}
    fixed_write(EVIDENCE/'e54_reserve_reference.json',summary|{'snapshot_sha256':digest(SNAPSHOT)})
    return summary


if __name__ == '__main__':
    print(json.dumps(build(),indent=2))
