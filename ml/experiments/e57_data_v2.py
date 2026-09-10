"""Explicit pre-score metadata eligibility amendment; original selection/decoder unchanged."""
import argparse
from collections import Counter
import fcntl
import json
from pathlib import Path
import subprocess
import time

from experiments import e57_data as original
from experiments.e53_offline import EVIDENCE,digest,fixed_write
from experiments.e51_prefit_audit import fingerprint,cross_role_matches
from experiments.e56_fivek_pilot import decoder,PYTHON,ROOT as PILOT

ROOT=original.ROOT/'v2'
CONTRACT=ROOT/'data_contract.json'
MANIFEST=ROOT/'admitted.json'
PROBE=Path(__file__).with_name('e57_raw_probe.py')
safe=original.safe


def freeze():
    data=original.load();safe(time.monotonic()+3600)
    value={'state':'E57_v2_metadata_eligibility_frozen_before_further_download_or_scores',
        'code_sha256':digest(__file__),'original_contract_sha256':digest(original.CONTRACT),
        'probe_sha256':digest(PROBE),'original_code_sha256':digest(original.__file__),
        'predicate':'Four finite as-shot WB values, first three strictly positive; unchanged decoder requirement.',
        'exclusion':'All predicate failures quarantined without replacement; all other failures stop.',
        'prior_failure':{'parent':'a1854-kme_290.dng','camera_whitebalance':[0.,1.,0.,0.],
                         'completed_v1_parents':61,'new_model_scores':0},
        'selected_parents':len(data['rows']),'image_bytes_expected':data['image_bytes_expected'],
        'decoder_changed':False,'runtime':data['runtime'],'model_scores':0}
    fixed_write(CONTRACT,value);fixed_write(EVIDENCE/'e57_v2_data_contract.json',value|{'contract_sha256':digest(CONTRACT)})
    return value


def load():
    config=original.load();value=json.loads(CONTRACT.read_text())
    if digest(__file__)!=value['code_sha256'] or digest(PROBE)!=value['probe_sha256'] or digest(original.__file__)!=value['original_code_sha256'] or digest(original.CONTRACT)!=value['original_contract_sha256']:
        raise ValueError('amended source implementation changed')
    if digest(CONTRACT)!=json.loads((EVIDENCE/'e57_v2_data_contract.json').read_text())['contract_sha256']:raise ValueError('v2 contract changed')
    return config


def download():
    config=load();deadline=time.monotonic()+3600;safe(deadline);binding=digest(CONTRACT)
    old={r['filename']:r for r in json.loads((PILOT/'download.json').read_text())['records']}
    records=[];rejected=[];(ROOT/'records').mkdir(exist_ok=True)
    for name in ('LicenseAdobe.txt','LicenseAdobeMIT.txt'):
        path=ROOT/name
        if not path.exists():path.write_bytes((original.META/name).read_bytes())
        if digest(path)!=config['inputs'][str(original.META/name)]:raise ValueError('licence copy changed')
    for row in config['rows']:
        safe(deadline);prior=None;record_path=ROOT/'records'/(row['filename']+'.json')
        if record_path.exists():
            record=json.loads(record_path.read_text())
            if record['amendment_sha256']!=binding or digest(record['original_path'])!=record['original_sha256']:raise ValueError('v2 cached raw changed')
            if not record['eligible_decode']:rejected.append(record)
            else:
                if digest(record['path'])!=record['sha256']:raise ValueError('v2 cached pixels changed')
                records.append(record)
            continue
        previous_path=original.ROOT/'records'/(row['filename']+'.json')
        if previous_path.exists():
            prior=json.loads(previous_path.read_text());rawpath=Path(prior['original_path']);sha=prior['original_sha256']
            if prior['contract_sha256']!=digest(original.CONTRACT):raise ValueError('v1 record contract changed')
        elif row['filename'] in old:
            prior=old[row['filename']];rawpath=PILOT/row['filename'];sha=prior['original_sha256']
        else:rawpath,sha=original.acquire_raw(row,digest(original.CONTRACT),deadline)
        if digest(rawpath)!=sha or rawpath.stat().st_size!=row['bytes']:raise ValueError('raw reuse mismatch')
        if prior is not None and (row['etag']!=prior['etag'] or row['last_modified']!=prior['last_modified']):raise ValueError('reuse HTTP identity changed')
        run=subprocess.run([str(PYTHON),str(PROBE),str(rawpath)],capture_output=True,text=True,timeout=30,check=True)
        probe=json.loads(run.stdout)
        if probe['rawpy']!=config['runtime']['rawpy'] or probe['libraw']!=config['runtime']['libraw']:raise ValueError('probe runtime changed')
        if not probe['valid_as_shot_wb']:
            record=row|{'original_path':str(rawpath),'original_sha256':sha,'original_bytes':row['bytes'],
                'eligible_decode':False,'probe':probe,'amendment_sha256':binding,'reason':'missing_valid_as_shot_WB'}
            rejected.append(record)
        else:
            png=Path(prior['path']) if prior else original.ROOT/'png'/Path(row['filename']).with_suffix('.png')
            if prior:
                result=prior['decode']
                if digest(png)!=prior['sha256']:raise ValueError('v1/pilot pixels changed')
            else:result=decoder([rawpath,png],deadline)
            if result['versions']!=config['runtime']:raise ValueError('decoder changed')
            facts=fingerprint(png.read_bytes())
            if facts['pixel_sha256']!=result['pixel_sha256']:raise ValueError('pixel mismatch')
            record=row|facts|{'original_path':str(rawpath),'original_sha256':sha,'original_bytes':row['bytes'],
                'path':str(png),'decode':result,'eligible_decode':True,'probe':probe,
                'condition':'fixed_raw_development','amendment_sha256':binding}
            records.append(record)
        fixed_write(record_path,record)
        n=len(records)+len(rejected)
        if n%10==0:print(json.dumps({'phase':'E57_v2_acquire','processed':n,'valid':len(records),'rejected':len(rejected),'total':201}),flush=True)
    value={'state':'E57_v2_acquired_quarantine','contract_sha256':binding,'records':records,'rejected':rejected,
        'original_bytes':sum(r['original_bytes'] for r in records+rejected),'png_bytes':sum(r['bytes'] for r in records),
        'pilot_reused_parents':sum(Path(r['original_path']).parent==PILOT for r in records+rejected),
        'new_original_bytes':sum(r['original_bytes'] for r in records+rejected if Path(r['original_path']).parent!=PILOT),
        'lost_strata':dict(Counter(r['subject']+'|'+r['light'] for r in rejected)),
        'model_scores':0,'training_admitted':False}
    fixed_write(ROOT/'download.json',value);fixed_write(EVIDENCE/'e57_v2_download.json',{k:v for k,v in value.items() if k not in {'records','rejected'}}|
        {'valid_parents':len(records),'metadata_rejected_parents':len(rejected),'report_sha256':digest(ROOT/'download.json')})
    return {'valid':len(records),'rejected':len(rejected)}


def audit():
    load();deadline=time.monotonic()+3600;safe(deadline);path=ROOT/'download.json'
    if digest(path)!=json.loads((EVIDENCE/'e57_v2_download.json').read_text())['report_sha256']:raise ValueError('download changed')
    data=json.loads(path.read_text());rows=data['records']
    for row in rows:
        safe(deadline)
        if digest(row['path'])!=row['sha256'] or digest(row['original_path'])!=row['original_sha256']:raise ValueError('bodies changed')
    refs=json.loads(original.SNAPSHOT.read_text())['records']+json.loads((original.MNW/'download.json').read_text())['records']+json.loads((original.HDR/'download.json').read_text())['records']
    cross=cross_role_matches(rows,refs);internal=[r for r in cross_role_matches(rows,rows) if r['train_parent']<r['cal_parent']]
    raw_hashes={r['sha256'] for r in refs};excluded={r['parent_id'] for r in rows if r['original_sha256'] in raw_hashes}
    excluded.update(r['train_parent'] for r in cross)
    for r in internal:excluded.update((r['train_parent'],r['cal_parent']))
    admitted=[r|{'role':'FIT_ONLY_RESEARCH'} for r in rows if r['parent_id'] not in excluded]
    if not admitted:raise ValueError('empty supplement')
    value={'state':'E57_v2_supplement_admitted_for_FIT','contract_sha256':digest(CONTRACT),
        'download_sha256':digest(path),'rows':admitted,'excluded_overlap_parents':sorted(excluded),
        'metadata_exclusions':[{'parent_id':r['parent_id'],'reason':r['reason']} for r in data['rejected']],
        'cross_matches':cross,'internal_pairs':internal,'reference_observations':len(refs),'model_scores':0,
        'limits':['Supported-as-shot-WB subset; no claim for missing metadata or modern phones.',
                  'Heuristic overlap is not exhaustive semantic deduplication.']}
    fixed_write(MANIFEST,value);fixed_write(EVIDENCE/'e57_admission.json',{k:v for k,v in value.items() if k not in {'rows','cross_matches','internal_pairs'}}|
        {'parents':len(admitted),'cross_count':len(cross),'internal_count':len(internal),'manifest_sha256':digest(MANIFEST)})
    return {'admitted':len(admitted),'overlap_exclusions':len(excluded),'metadata_exclusions':len(data['rejected'])}


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['freeze','download','audit']);args=parser.parse_args()
    with (EVIDENCE.parent/'ml/work/e57.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);print(json.dumps(globals()[args.phase](),indent=2))
