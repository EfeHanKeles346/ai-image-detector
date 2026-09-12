"""Score-blind admission of unused local AI observations to limited E66 development."""
from __future__ import annotations
import argparse
from collections import Counter
import fcntl
import hashlib
import json
from pathlib import Path

from experiments.e65_acquisition import digest,read,write_once
from experiments.e65_audit import fingerprint,cross_role_matches,REFERENCES,VERSION
from experiments.e32_r0_input import source_path,_parquet_raw
from pixelproof.project_paths import DATA_ROOT,ML_ROOT

ROOT=DATA_ROOT/'e66'
EVIDENCE=ML_ROOT.parent/'evidence'
CONTRACT=ROOT/'ai_contract.json'
REPORT=ROOT/'ai_admission.json'


def select(records,inventory,current,n=80):
    allowed=set(inventory['remaining_record_ids'])
    used={r['parent_id'].removeprefix('e32:') for r in current}
    excluded={p.removeprefix('e32:') for pair in inventory['internal_pairs']
              for p in (pair['train_parent'],pair['cal_parent'])}
    seen={(r['source_id'],r['role_group']) for r in records if r['record_id'] in used}
    eligible=[r for r in records if r['record_id'] in allowed-used-excluded
              and (r['source_id'],r['role_group']) not in seen]
    selected=[]
    for source in ['gpt-image-1','nano-banana-local']:
        choices=[r for r in eligible if r['source_id']==source and r['label']=='ai' and r['role'].upper()=='TRAIN']
        if len(choices)<n:raise ValueError('insufficient unexposed source records')
        selected.extend(sorted(choices,key=lambda r:hashlib.sha256(('E66|'+r['record_id']).encode()).hexdigest())[:n])
    return selected,eligible


def freeze():
    pins={DATA_ROOT/'e32/c3_role_manifest.json':'0b6656a25762dbb097d18634d4193282e7c03d2e6bbe257f31471434f67b91eb',
          DATA_ROOT/'e53/native_inventory_result.json':read(EVIDENCE/'e53_native_inventory.json')['report_sha256'],
          DATA_ROOT/'e54/data_contract_v2.json':read(EVIDENCE/'e61_replay_contract.json')['inputs']['manifest']['sha256'],
          DATA_ROOT/'e60/audit.json':read(EVIDENCE/'e60_audit.json')['report_sha256'],
          DATA_ROOT/'e65/audit.json':read(EVIDENCE/'e65_audit.json')['report_sha256']}
    pins.update({DATA_ROOT/p:s for p,s in REFERENCES.items()})
    for p,s in pins.items():
        if digest(p)!=s:raise ValueError('prior population/audit changed')
    rows,eligible=select(read(DATA_ROOT/'e32/c3_role_manifest.json')['records'],
        read(DATA_ROOT/'e53/native_inventory_result.json'),read(DATA_ROOT/'e54/data_contract_v2.json')['rows'])
    old_ids={r['parent_id'] for r in read(DATA_ROOT/'e60/audit.json')['teacher_fit']}
    train=read(DATA_ROOT/'e54/data_contract_v2.json')['rows'];train_shas={r['sha256'] for r in train}
    if any('e32:'+r['record_id'] in old_ids or r['sha256'] in train_shas for r in rows):
        raise ValueError('selected observation already in E43/current TRAIN')
    for file in ['e66_native_dev.py','e65_acquisition.py','e65_audit.py','e32_r0_input.py']:
        p=Path(__file__).with_name(file);pins[p]=digest(p)
    for file in ['bitmind__nano-banana_metadata.json','a3xrfgb__gpt-image-mega-4k_metadata.json']:
        p=ROOT/file;pins[p]=digest(p)
    if read(ROOT/'bitmind__nano-banana_metadata.json')['cardData']['license']!='mit' or \
       read(ROOT/'a3xrfgb__gpt-image-mega-4k_metadata.json')['cardData']['license']!='cc-by-4.0':
        raise ValueError('source licence differs')
    c={'state':'E66_unused_AI_DEV_selection_registered','inputs':{str(p):s for p,s in pins.items()},
       'rows':rows,'eligible_unused_records':len(eligible),'reserved_from_future_train':[r['record_id'] for r in eligible],
       'role':'DEVELOPMENT','training_allowed':False,'model_scores_created':0,
       'policy':'80 hash-ranked per source, seedE66; exclude E60 current recorded groups, E53 internal duplicate '
                'endpoints and all previously protected old-role identities from the E53 audited inventory. '
                'Verify originals and repeat canonical overlap against complete snapshot plus later reserves/E65. '
                'The snapshot contains candidate native bodies: exempt ONLY these160 exact self-body SHA256 '
                'entries after prior eligibility and explicit E43/current-TRAIN disjoint checks; do not exempt '
                'same pixels under other hashes or newer reserve entries. Quarantine matches; no replacements.',
       'limitations':'Two previously seen generator families; unknown prompt dependencies, single-file groups '
                    'not proof of independence. Whole remaining2385 eligible pool reserved from future TRAIN. '
                    'Balanced DEV remains pending SIDD admission; never a fresh publisher/final claim.'}
    write_once(CONTRACT,c);write_once(EVIDENCE/'e66_ai_contract.json',{k:v for k,v in c.items()
        if k not in ['rows','reserved_from_future_train']}|{'selected_record_ids':[r['record_id'] for r in rows],
        'contract_sha256':digest(CONTRACT)})
    return {'state':c['state'],'selected':len(rows),'reserved_unused':len(eligible)}


def validate():
    c=read(CONTRACT)
    if digest(CONTRACT)!=read(EVIDENCE/'e66_ai_contract.json')['contract_sha256']:raise ValueError('AI contract changed')
    for p,s in c['inputs'].items():
        if digest(p)!=s:raise ValueError(f'AI admission bound input changed: {p}')
    return c


def audit():
    c=validate()
    if REPORT.exists():raise FileExistsError('AI admission already recorded')
    rows=c['rows'];(ROOT/'ai').mkdir(exist_ok=True);facts=[]
    def process(r,raw):
        sha=hashlib.sha256(raw).hexdigest()
        if sha!=r['sha256']:raise ValueError('native body changed')
        path=ROOT/'ai'/(r['record_id']+'.image')
        if path.exists():
            if digest(path)!=sha:raise ValueError('materialized native body changed')
        else:
            temp=path.with_suffix('.image.part');temp.write_bytes(raw);temp.replace(path)
        f=fingerprint(raw)
        if min(f['width'],f['height'])<224:raise ValueError('native source below224px')
        facts.append(r|f|{'path':str(path),'parent_id':'e32:'+r['record_id'],'label':1,'source':r['source_id'],
            'condition':'original','role':'DEVELOPMENT','training_allowed':False,
            'group_limitation':'recorded single-image group; prompt dependence unknown'})
        if len(facts)%20==0:print(json.dumps({'AI_originals_verified':len(facts),'total':len(rows)}),flush=True)
    for r in rows:
        if r['source_id']=='gpt-image-1':process(r,source_path(r['source_id'],r['source_key']).read_bytes())
    for r,raw in _parquet_raw([r for r in rows if r['source_id']=='nano-banana-local'],'bitmind__nano-banana','image'):
        process(r,raw)
    if len(facts)!=len(rows):raise ValueError('incomplete local native materialization')
    self_sha={r['sha256'] for r in facts}
    snapshot=read(DATA_ROOT/'e52/protected_reference_v1.json')['records']
    train_sha={r['sha256'] for r in read(DATA_ROOT/'e54/data_contract_v2.json')['rows']}
    if self_sha & train_sha or not train_sha <= {r['sha256'] for r in snapshot}:
        raise ValueError('snapshot TRAIN coverage or selected exposure changed')
    refs=[r for r in snapshot if r['sha256'] not in self_sha]
    for p in REFERENCES:
        if p!='e52/protected_reference_v1.json':refs.extend(read(DATA_ROOT/p)['records'])
    refs.extend(read(DATA_ROOT/'e65/audit.json')['records'])
    if any(r['version']!=VERSION for r in refs):raise ValueError('fingerprint convention differs')
    matches=cross_role_matches(facts,refs)
    internal=[r for r in cross_role_matches(facts,facts) if r['train_parent']<r['cal_parent']]
    bad={r['train_parent'] for r in matches}|{p for m in internal for p in [m['train_parent'],m['cal_parent']]}
    admitted=sorted([r for r in facts if r['parent_id'] not in bad],key=lambda r:r['parent_id'])
    result={'state':'E66_local_AI_DEV_audit_complete','contract_sha256':digest(CONTRACT),'records':admitted,
        'selected':len(rows),'admitted':len(admitted),'source_counts':dict(Counter(r['source'] for r in admitted)),
        'self_snapshot_entries_exempted':len(snapshot)-sum(r['sha256'] not in self_sha for r in snapshot),
        'reference_records':len(refs),'cross_matches':matches,'internal_matches':internal,
        'quarantined_parents':sorted(bad),'model_scores_created':0,'training_allowed':False,
        'balanced_development_admitted':False,'independent_final_admitted':False,'limitations':c['limitations']}
    write_once(REPORT,result)
    summary={k:v for k,v in result.items() if k not in ['records','cross_matches','internal_matches']}
    summary.update(report_sha256=digest(REPORT),cross_match_observations=len(matches),internal_match_pairs=len(internal))
    write_once(EVIDENCE/'e66_ai_admission.json',summary);return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=('freeze','audit'))
    with (ROOT/'ai_execution.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        print(json.dumps({'freeze':freeze,'audit':audit}[parser.parse_args().stage](),indent=2))
