"""Read-only source-image audit for future E53 expansion; no training authorization.

Reads existing local bodies, stores only fingerprints/reports, never downloads images.
Conservatively excludes every identity used in the E51 protected inventory, even old FIT rows.
"""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import sqlite3

from experiments.e32_r0_input import source_path, PARQUET_SOURCES, _parquet_raw
from experiments.e51_prefit_audit import VERSION, fingerprint, cross_role_matches
from experiments.e51_protected_inventory import identity_rows, keys
from experiments.e51_protected_pixels import DATABASE
from experiments.e51_reserve_closure import PINS
from experiments.e53_offline import digest, fixed_write, ROOT, EVIDENCE
from pixelproof.project_paths import DATA_ROOT

CONTRACT = ROOT/'native_inventory_contract.json'
CACHE = ROOT/'native_fingerprints.sqlite3'
RESULT = ROOT/'native_inventory_result.json'


def run():
    if RESULT.exists():raise FileExistsError('native audit already complete')
    c3path=DATA_ROOT/'e32/c3_role_manifest.json'
    if digest(c3path)!='0b6656a25762dbb097d18634d4193282e7c03d2e6bbe257f31471434f67b91eb':raise ValueError('C3 changed')
    document=json.loads(c3path.read_text()); records=document['records']
    paths={str(c3path):digest(c3path)}; protected_keys=set()
    inventory=json.loads((DATA_ROOT/'e51/audit/protected_inventory_v1.json').read_text())
    for item in inventory['sources']:
        path=Path(item['path'])
        if digest(path)!=item['sha256']:raise ValueError('protected metadata changed')
        paths[str(path)]=item['sha256']
        for row in identity_rows(json.loads(path.read_text())):protected_keys.update(keys(row))
    for rel,sha in PINS.items():
        path=DATA_ROOT/rel
        if digest(path)!=sha:raise ValueError('reserve metadata changed')
        paths[str(path)]=sha
        for row in identity_rows(json.loads(path.read_text())):protected_keys.update(keys(row))
    for path,field in [(DATA_ROOT/'e51/admission_v1.json',None),
                       (DATA_ROOT/'e51/development/manifest_unscored.json','rows')]:
        paths[str(path)]=digest(path);doc=json.loads(path.read_text())
        for row in (doc[field] if field else doc['train']+doc['cal']):protected_keys.update(keys(row))
    def identity(row):return keys(row)|{('identity','e32:'+row['record_id'])}
    candidates=[r for r in records if r['role']=='TRAIN' and not identity(r)&protected_keys]
    protected_cal=[r for r in records if r['role']=='CALIBRATION']
    selected=candidates+protected_cal
    protocol={'schema_version':1,'state':'native_audit_before_pixels_and_scores','code_sha256':digest(__file__),
              'inputs':paths,'candidate_count':len(candidates),'protected_c3_cal_count':len(protected_cal),
              'selected_record_ids':[r['record_id'] for r in selected],
              'training_authorized':False,'model_scores_created':0,
              'policy':'Conservative all-old-role exclusion; original-image fingerprints before any expansion fit.'}
    fixed_write(CONTRACT,protocol)
    ROOT.mkdir(parents=True,exist_ok=True)
    db=sqlite3.connect(CACHE)
    db.execute('CREATE TABLE IF NOT EXISTS fingerprints(sha TEXT PRIMARY KEY,facts TEXT)')
    cache={s:json.loads(f) for s,f in db.execute('SELECT sha,facts FROM fingerprints')}
    allfacts={};seen=0
    def process(pair):
        row,raw=pair
        if hashlib.sha256(raw).hexdigest()!=row['sha256']:raise ValueError('original source body changed')
        fact=cache.get(row['sha256'])
        if fact is None:fact=fingerprint(raw)
        if fact['version']!=VERSION or fact['sha256']!=row['sha256']:raise ValueError('fingerprint convention changed')
        return row,fact
    def retain(batch):
        nonlocal seen
        with db:
            db.executemany('INSERT OR REPLACE INTO fingerprints VALUES (?,?)',
                           [(fact['sha256'],json.dumps(fact,sort_keys=True)) for _,fact in batch])
        for row,fact in batch:
            cache[fact['sha256']]=fact
            allfacts[row['record_id']]={**fact,'parent_id':'e32:'+row['record_id'],
                'condition':'original','label':int(row['label']=='ai'),'source':row['source_id']}
        seen+=len(batch)
        print(json.dumps({'phase':'native_original_audit','verified':seen,'total':len(selected)}),flush=True)
    with ThreadPoolExecutor(max_workers=4) as pool:
        loose=[r for r in selected if r['source_id'] not in PARQUET_SOURCES]
        for start in range(0,len(loose),128):
            def loose_process(row):return process((row,source_path(row['source_id'],row['source_key']).read_bytes()))
            retain(list(pool.map(loose_process,loose[start:start+128])))
        for source,(folder,column) in PARQUET_SOURCES.items():
            source_rows=[r for r in selected if r['source_id']==source]; batch=[]
            for row,raw in _parquet_raw(source_rows,folder,column):
                batch.append((row,raw))
                if len(batch)>=64:
                    retain(list(pool.map(process,batch)));batch=[]
            if batch:retain(list(pool.map(process,batch)))
    db.close()
    if len(allfacts)!=len(selected):raise ValueError('incomplete native audit')
    with sqlite3.connect(f'file:{DATABASE}?mode=ro',uri=True) as old:
        protected={sha:json.loads(f) for sha,f in old.execute('SELECT sha,facts FROM fingerprints WHERE version=?',(VERSION,))}
    # Snapshot predates this separate native cache. Keep all old canonical bodies protected.
    reference=[{**f,'parent_id':'protected:'+sha,'label':-1,'condition':'protected'} for sha,f in protected.items()]
    reference.extend(allfacts[r['record_id']] for r in protected_cal)
    reference.extend(json.loads((DATA_ROOT/'e51/audit/reserve_closure_v1.json').read_text())['additional_fingerprints'])
    reference.extend(json.loads((DATA_ROOT/'e51/development/manifest_unscored.json').read_text())['rows'])
    queries=[allfacts[r['record_id']] for r in candidates]
    matches=cross_role_matches(queries,reference)
    excluded={m['train_parent'] for m in matches}
    internal=[m for m in cross_role_matches(queries,queries) if m['train_parent']<m['cal_parent']]
    # Propagate protected overlap through every detected internal duplicate component.
    changed=True
    while changed:
        before=len(excluded)
        for m in internal:
            if {m['train_parent'],m['cal_parent']}&excluded:excluded.update([m['train_parent'],m['cal_parent']])
        changed=len(excluded)!=before
    kept=[r for r in candidates if 'e32:'+r['record_id'] not in excluded]
    report={'schema_version':1,'state':'native_inventory_audited_not_training_admitted',
            'contract_sha256':digest(CONTRACT),'original_bodies_verified':len(selected),
            'candidate_parents_before_pixels':len(candidates),'protected_c3_cal_parents':len(protected_cal),
            'remaining_candidates':len(kept),'source_counts':dict(Counter(r['source_id'] for r in kept)),
            'canonical_protected_references':len(reference),'protected_matches':matches,'internal_pairs':internal,
            'excluded_parent_ids':sorted(excluded),'remaining_record_ids':[r['record_id'] for r in kept],
            'image_bytes_downloaded':0,'model_scores_created':0,'training_authorized':False,
            'limitations':['Heuristic duplicate screen is not complete semantic independence.',
                          'Source licence, grouping, selection quotas and new feature contract still required.',
                          'Historical TRAIN sources are not fresh independent final publishers.',
                          'Original bodies were checked; existing 224px features are not native-resolution features.']}
    fixed_write(RESULT,report)
    compact={k:v for k,v in report.items() if k not in {'protected_matches','internal_pairs','remaining_record_ids','excluded_parent_ids'}}
    compact.update({'report_sha256':digest(RESULT),'protected_match_rows':len(matches),
                    'internal_pair_count':len(internal),'excluded_parents':len(excluded)})
    fixed_write(EVIDENCE/'e53_native_inventory.json',compact)
    return compact


if __name__=='__main__':print(json.dumps(run(),indent=2))
