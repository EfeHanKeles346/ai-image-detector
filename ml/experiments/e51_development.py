"""Offline, score-blind realization of fixed IEEE/Datapoint DEVELOPMENT for E51-A."""

from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
from io import BytesIO
import json
from pathlib import Path
import sqlite3

import numpy as np
from PIL import Image, ImageOps
import pyarrow.parquet as pq

from experiments.e42_features import _sha256_file
from experiments.e51_pipeline import ROOT, RESULT, RESULT_EVIDENCE
from experiments.e51_prefit_audit import VERSION, fingerprint, cross_role_matches, load_inputs
from experiments.e51_protected_pixels import DATABASE, LOCATOR_SHA256
from experiments.e51_offline_bodies import OUTPUT as LOCATORS
from experiments.e51_train_cal_realize import _q75, _write_atomic
from pixelproof.project_paths import ML_ROOT

OUT = ROOT/'development'
MANIFEST = OUT/'manifest_unscored.json'
EVIDENCE = ML_ROOT.parent/'evidence/e51_development_manifest.json'
PINS = {
    'route/contract_untransferred.json':'975e8164477c7234292ba87449007f0ee4c8b65eb582f25a8b0d81140ec315e4',
    'receipts/ieee_spcup_download_unscored.json':'09188d4966a19f70a2f9ed34dab052f2d5a9b69f8819b6578fe5421e548b3794',
    'receipts/datapoint_shards_download_unscored.json':'18b8326a7972d19094b48ebc1e67d9ece8114ae36baeb258575f803f3a90bad1',
}


def pinned(path,sha):
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=sha: raise ValueError(f'bound input changed: {path}')
    return json.loads(raw)


def select_shared_prompts(rows, excluded):
    models = {r['model_id'] for r in rows}
    if len(models)!=5 or len(rows)!=920: raise ValueError('AI reserve population changed')
    by_prompt = defaultdict(list)
    for row in rows: by_prompt[(row['prompt_category'],row['prompt_ordinal'])].append(row)
    categories = defaultdict(list)
    for (category,ordinal), group in by_prompt.items():
        if len(group)!=5 or {r['model_id'] for r in group}!=models:
            raise ValueError('shared AI prompt group is incomplete')
        if not any(r['parent_id'] in excluded for r in group):
            categories[category].append((group[0]['rank'],ordinal,group))
    if len(categories)!=8 or any(len(v)<20 for v in categories.values()):
        raise ValueError('score-blind AI reserve cannot fill 20 prompts per category')
    return [r for category in sorted(categories)
            for _,_,group in sorted(categories[category])[:20] for r in group]


def materialize(raw,path):
    if path.exists():
        if path.read_bytes()!=raw: raise ValueError(f'local realized bytes changed: {path}')
    else:
        path.parent.mkdir(parents=True,exist_ok=True)
        temporary = path.with_suffix(path.suffix+'.part')
        temporary.write_bytes(raw); temporary.replace(path)


def paired(row):
    raw = Path(row['path']).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=row['sha256']:
        raise ValueError('DEVELOPMENT body changed')
    original = {**row,**fingerprint(raw),'condition':'original',
                'record_id':row['parent_id']+'|original'}
    with Image.open(BytesIO(raw)) as opened:
        q75 = _q75(ImageOps.exif_transpose(opened).convert('RGB'))
    path = OUT/'q75'/(row['sha256']+'.jpg')
    materialize(q75,path)
    child = {**row,**fingerprint(q75),'path':str(path),'condition':'q75',
             'record_id':row['parent_id']+'|q75'}
    return [original,child]


def protected_facts():
    locators = pinned(LOCATORS,LOCATOR_SHA256)
    train,cal = load_inputs()
    expected = {r['sha256'] for r in locators['rows']+train+cal}
    with sqlite3.connect(f'file:{DATABASE}?mode=ro',uri=True) as db:
        cache = {sha:json.loads(f) for sha,f in db.execute(
            'SELECT sha,facts FROM fingerprints WHERE version=?',(VERSION,))}
    if not expected<=cache.keys(): raise ValueError('protected canonical cache incomplete')
    facts = []
    for sha in sorted(expected):
        row = cache[sha]
        if row['sha256']!=sha or row['version']!=VERSION: raise ValueError('cache identity changed')
        facts.append({**row,'parent_id':sha,'label':-1,'condition':'protected'})
    reserve_summary = json.loads((ML_ROOT.parent/'evidence/e51_reserve_closure.json').read_text())
    reserve = pinned(ROOT/'audit/reserve_closure_v1.json',reserve_summary['report_sha256'])
    facts.extend(reserve['additional_fingerprints'])
    return facts


def realize():
    if MANIFEST.exists() or EVIDENCE.exists(): raise FileExistsError('DEVELOPMENT already admitted')
    cal = json.loads(RESULT.read_text())
    if cal!=json.loads(RESULT_EVIDENCE.read_text()) or cal['selected']!='A' or not cal['candidates']['A']['passed']:
        raise ValueError('fresh DEVELOPMENT requires the frozen CAL-selected A')
    inputs = {rel:pinned(ROOT/rel,sha) for rel,sha in PINS.items()}
    contract = {'state':'development_realization_bound_before_image_columns',
                'candidate_sha256':cal['candidates']['A']['artifact_sha256'],
                'cal_result_sha256':_sha256_file(RESULT),'inputs':PINS,
                'code_sha256':_sha256_file(Path(__file__)),
                'target_real':2640,'target_ai':800,'reserved_ai':920,
                'exclusion':'All IEEE parents required. Any protected or cross-parent internal candidate involving IEEE aborts. For AI, reject entire shared prompt group; take first 20 valid ranked prompts per category across all five models.',
                'scores_created':0,'downloads':0}
    contract_path = OUT/'realization_contract.json'
    if contract_path.exists():
        if json.loads(contract_path.read_text())!=contract: raise ValueError('DEV realization code/contract changed')
    else: _write_atomic(contract_path,contract)
    route = inputs['route/contract_untransferred.json']
    real = inputs['receipts/ieee_spcup_download_unscored.json']['rows']
    if len(real)!=2640 or Counter(r['transport_cell'] for r in real)!={'unaltered':1320,'postprocessed':1320}:
        raise ValueError('IEEE population changed')
    real = [{**r,'parent_id':r['identity'],'group':'IEEE:'+r['transport_cell']} for r in real]
    requested = {r['image_key']:r for r in route['roles']['development_ai']['rows']}
    ai = {}
    for shard in inputs['receipts/datapoint_shards_download_unscored.json']['rows']:
        path = Path(shard['path'])
        if path.stat().st_size!=shard['bytes'] or _sha256_file(path)!=shard['sha256']:
            raise ValueError('Datapoint source shard changed')
        table = pq.ParquetFile(path)
        for batch in table.iter_batches(batch_size=32):
            for item in batch.to_pylist():
                key = item['image_key']
                if key not in requested: continue
                row = requested[key]
                raw = item['image']['bytes']
                if (key in ai or row['source_shard']!=shard['remote_path']
                        or item['model_id']!=row['model_id'] or len(raw)!=row['expected_bytes']
                        or hashlib.sha256(raw).hexdigest()!=row['expected_sha256']):
                    raise ValueError('Datapoint reserved identity/bytes changed')
                facts = fingerprint(raw)
                if (facts['width'],facts['height'])!=(row['width'],row['height']):
                    raise ValueError('Datapoint reserved geometry changed')
                destination = OUT/'ai_reserve'/(facts['sha256']+'.image')
                materialize(raw,destination)
                ai[key] = {**row,**facts,'path':str(destination),'parent_id':row['identity'],'group':row['source']}
        print(json.dumps({'phase':'DEV_AI_local_decode','reserved_images':len(ai),'target':920}),flush=True)
    if set(ai)!=set(requested): raise ValueError('Datapoint reserve coverage missing')
    parents = real+list(ai.values())
    rows = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for i,pack in enumerate(pool.map(paired,parents),1):
            rows.extend(pack)
            if i%200==0: print(json.dumps({'phase':'DEV_pairing','parents':i,'total':len(parents)}),flush=True)
    protected = protected_facts()
    external = cross_role_matches(rows,protected)
    internal = [m for m in cross_role_matches(rows,rows) if m['train_parent']!=m['cal_parent']]
    excluded = {m['train_parent'] for m in external+internal}
    excluded.update(m['cal_parent'] for m in internal)
    audit = {'state':'development_model_blind_identity_audit','model_scores_created':0,
             'protected_matches':external,'internal_cross_parent_matches':internal,
             'excluded_parent_ids':sorted(excluded),'protected_bodies':len(protected),
             'protected_fingerprint_digest':hashlib.sha256(json.dumps(protected,sort_keys=True).encode()).hexdigest(),
             'candidate_sha256':contract['candidate_sha256'],'realization_contract_sha256':_sha256_file(contract_path)}
    _write_atomic(OUT/'identity_audit.json',audit)
    if excluded & {r['parent_id'] for r in real}:
        raise ValueError('IEEE fixed DEV population has duplicate/overlap candidates; no scores authorized')
    selected_ai = select_shared_prompts(list(ai.values()),excluded)
    selected_ids = {r['parent_id'] for r in real+selected_ai}
    selected = [r for r in rows if r['parent_id'] in selected_ids]
    if len(selected)!=6880: raise ValueError('DEVELOPMENT paired count changed')
    report = {'state':'E51_DEVELOPMENT_frozen_unscored','model_scores_created':0,
              'candidate_sha256':contract['candidate_sha256'],'cal_result_sha256':contract['cal_result_sha256'],
              'identity_audit_sha256':_sha256_file(OUT/'identity_audit.json'),
              'observations':len(selected),'parents':len(selected_ids),'rows':selected,
              'source_counts_original':dict(Counter(r['group'] for r in selected if r['condition']=='original')),
              'limits':['IEEE 512px publisher images and hidden device ids; real transport cells are not device-disjoint guarantees.',
                        'Five current AI generators share 160 prompts; uncertainty must group by prompt, not 800 independent prompts.',
                        'Passed DEV does not replace a new independent E52 final; no threshold or weight changes allowed.']}
    raw = _write_atomic(MANIFEST,report)
    summary = {k:v for k,v in report.items() if k!='rows'}
    summary['manifest_sha256'] = hashlib.sha256(raw).hexdigest()
    _write_atomic(EVIDENCE,summary)
    return summary


if __name__=='__main__':
    print(json.dumps(realize(),indent=2))
