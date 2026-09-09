"""Score-blind grouping amendment after the preserved IEEE duplicate-screen stop."""

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path

from experiments.e42_features import _sha256_file
from experiments.e51_development import (
    ROOT, OUT, MANIFEST, EVIDENCE, PINS, pinned, paired, protected_facts, select_shared_prompts,
)
from experiments.e51_pipeline import RESULT, RESULT_EVIDENCE
from experiments.e51_prefit_audit import cross_role_matches
from experiments.e51_train_cal_realize import _write_atomic

AUDIT_SHA = '975569d361f43f17ce3e2788d8fc4c7b6889d7e5504d47c81f38d85a0ec258e8'


def real_clusters(real, matches):
    ids = {r['parent_id'] for r in real}
    links = {i:i for i in ids}
    def find(i):
        while links[i]!=i:
            links[i] = links[links[i]]; i = links[i]
        return i
    for m in matches:
        if m['train_label']==m['cal_label']==0:
            a,b = m['train_parent'],m['cal_parent']
            if a not in ids or b not in ids: raise ValueError('unknown REAL cluster member')
            ra,rb = find(a),find(b)
            links[max(ra,rb)] = min(ra,rb)
    return {i:find(i) for i in ids}


def normalized_matches(rows):
    return Counter(json.dumps(r,sort_keys=True) for r in rows)


def amend():
    if MANIFEST.exists() or EVIDENCE.exists(): raise FileExistsError('DEV already admitted')
    prior = pinned(OUT/'identity_audit.json',AUDIT_SHA)
    if prior['model_scores_created']!=0 or any(m['train_label']==0 for m in prior['protected_matches']):
        raise ValueError('cannot amend protected REAL overlap or scored population')
    cal = json.loads(RESULT.read_text())
    if cal!=json.loads(RESULT_EVIDENCE.read_text()) or cal['selected']!='A':
        raise ValueError('CAL-selected candidate changed')
    if (OUT/'scores.partial.jsonl').exists(): raise ValueError('DEV scores must not exist before amendment')
    documents = {rel:pinned(ROOT/rel,sha) for rel,sha in PINS.items()}
    real = [{**r,'parent_id':r['identity'],'group':'IEEE:'+r['transport_cell']}
            for r in documents['receipts/ieee_spcup_download_unscored.json']['rows']]
    clusters = real_clusters(real,prior['internal_cross_parent_matches'])
    sizes = Counter(clusters.values())
    if len(clusters)!=2640 or len(sizes)!=2635 or Counter(sizes.values())!={1:2630,2:5}:
        raise ValueError('reviewed five REAL pairs changed')
    contract = {'state':'score_blind_internal_scene_grouping_amendment',
        'superseded_stop_audit_sha256':AUDIT_SHA,'candidate_sha256':prior['candidate_sha256'],
        'cal_result_sha256':_sha256_file(RESULT),'code_sha256':_sha256_file(Path(__file__)),
        'inputs':PINS,'real_images':2640,'real_detected_scene_groups':2635,'model_scores_created':0,
        'review':'All ten REAL bodies visually inspected before scores: four visibly same-scene variant pairs; one smooth grey pair ambiguous and conservatively grouped. Keep every image, label and cutoff. No outside-role overlap exception.',
        'grouping':{k:v for k,v in clusters.items() if sizes[v]>1}}
    contract_path = OUT/'grouping_amendment.json'
    if contract_path.exists():
        if json.loads(contract_path.read_text())!=contract: raise ValueError('grouping amendment changed')
    else: _write_atomic(contract_path,contract)
    ai = []
    for r in documents['route/contract_untransferred.json']['roles']['development_ai']['rows']:
        path = OUT/'ai_reserve'/(r['expected_sha256']+'.image')
        ai.append({**r,'path':str(path),'sha256':r['expected_sha256'],'parent_id':r['identity'],'group':r['source']})
    parents = real+ai
    rows = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for i,pack in enumerate(pool.map(paired,parents),1):
            rows.extend(pack)
            if i%400==0: print(json.dumps({'phase':'DEV_amendment_reverify','parents':i,'total':3560}),flush=True)
    protected = protected_facts()
    external = cross_role_matches(rows,protected)
    internal = [m for m in cross_role_matches(rows,rows) if m['train_parent']!=m['cal_parent']]
    if (normalized_matches(external)!=normalized_matches(prior['protected_matches'])
            or normalized_matches(internal)!=normalized_matches(prior['internal_cross_parent_matches'])):
        raise ValueError('model-blind audit changed; no admission')
    excluded_ai = {r for r in prior['excluded_parent_ids'] if r not in clusters}
    selected_ai = select_shared_prompts(ai,excluded_ai)
    ids = {r['parent_id'] for r in real+selected_ai}
    selected = [{**r,'scene_cluster':clusters[r['parent_id']] if r['label']==0
                 else 'datapoint_prompt:'+str(r['prompt_ordinal'])} for r in rows if r['parent_id'] in ids]
    if len(selected)!=6880: raise ValueError('grouping amendment cannot drop DEV rows')
    report = {'state':'E51_DEVELOPMENT_frozen_unscored','model_scores_created':0,
        'candidate_sha256':contract['candidate_sha256'],'cal_result_sha256':contract['cal_result_sha256'],
        'identity_audit_sha256':AUDIT_SHA,'grouping_amendment_sha256':_sha256_file(contract_path),
        'observations':6880,'parents':3440,'real_detected_scene_groups':2635,'ai_prompt_groups':160,
        'rows':selected,'source_counts_original':dict(Counter(r['group'] for r in selected if r['condition']=='original')),
        'limits':['IEEE 512px publisher images; hidden devices, no worst-device or native-resolution proof.',
                  'All 2,640 IEEE images retained, but five pairs share scene clusters; unknown further scene dependence may remain.',
                  '800 AI images share 160 prompts across five generators; interval resampling must keep prompt clusters together.',
                  'Observed DEV pass does not establish a fresh E52 final pass. No fitting or threshold change.']}
    raw = _write_atomic(MANIFEST,report)
    summary = {k:v for k,v in report.items() if k!='rows'}
    summary['manifest_sha256'] = hashlib.sha256(raw).hexdigest()
    _write_atomic(EVIDENCE,summary)
    return summary


if __name__=='__main__':
    print(json.dumps(amend(),indent=2))
