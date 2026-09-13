"""Score-blind MIDD TRAIN admission, protected-reference audit and duplicate components."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import fcntl
import hashlib
import json
from pathlib import Path
import re
import socket
import time
import PIL
from PIL import Image
from experiments.e65_acquisition import digest,read,write_once
from experiments.e65_audit import fingerprint,cross_role_matches,REFERENCES,VERSION
from experiments.e72_acquisition import ROOT,EVIDENCE,RECEIPT,resource_check,validate as validate_acquisition
from pixelproof.project_paths import DATA_ROOT

CONTRACT=ROOT/'audit_contract.json'
REPORT=ROOT/'audit.json'
MANIFEST=ROOT/'training_manifest.json'


def components(rows, matches):
    """Transitive similarity/time links; protect an entire component if any member overlaps."""
    parents={r['parent_id']:r['parent_id'] for r in rows}
    if len(parents)!=len(rows):raise ValueError('duplicate candidate parent')
    def find(p):
        while parents[p]!=p:
            parents[p]=parents[parents[p]];p=parents[p]
        return p
    def join(a,b):
        if a not in parents or b not in parents:raise ValueError('unknown component endpoint')
        a,b=find(a),find(b)
        if a!=b:parents[max(a,b)]=min(a,b)
    for m in matches:join(m['train_parent'],m['cal_parent'])
    times={}
    for row in rows:
        stamp=row.get('capture_second')
        if stamp:
            key=(row['sensor'],stamp)
            if key in times:join(row['parent_id'],times[key])
            else:times[key]=row['parent_id']
    groups=defaultdict(list)
    for p in sorted(parents):groups[find(p)].append(p)
    return sorted(groups.values())


def resolve(rows, matches, rejected):
    groups=components(rows,matches);kept=[];discarded=[];bad=set(rejected);lineage={}
    for group in groups:
        identity='MIDD:component:'+hashlib.sha256('|'.join(group).encode()).hexdigest()
        lineage.update({p:identity for p in group})
        if set(group)&bad:bad.update(group);continue
        chosen=min(group,key=lambda p:hashlib.sha256(('E72|'+p).encode()).hexdigest())
        kept.append(chosen);discarded.extend(p for p in group if p!=chosen)
    return {'kept':sorted(kept),'quarantined':sorted(bad),'duplicate_members_removed':sorted(discarded),
            'components':groups,'lineage':lineage}


def freeze():
    validate_acquisition()
    if digest(RECEIPT)!=read(EVIDENCE/'e72_download.json')['receipt_sha256']:
        raise ValueError('completed MIDD receipt changed')
    expected={DATA_ROOT/'e66/ai_contract.json':read(EVIDENCE/'e66_ai_contract.json')['contract_sha256'],
              DATA_ROOT/'e32/c3_role_manifest.json':'0b6656a25762dbb097d18634d4193282e7c03d2e6bbe257f31471434f67b91eb',
              DATA_ROOT/'e54/data_contract_v2.json':read(EVIDENCE/'e61_replay_contract.json')['inputs']['manifest']['sha256']}
    for p,s in expected.items():
        if digest(p)!=s:raise ValueError('prior population/reservation identity changed')
    receipt=read(RECEIPT)
    if receipt['images']!=512 or receipt['training_allowed'] or receipt['model_scores']:
        raise ValueError('complete unscored quarantine population required')
    refs={DATA_ROOT/p:s for p,s in REFERENCES.items()}
    refs[DATA_ROOT/'e65/audit.json']=read(EVIDENCE/'e65_audit.json')['report_sha256']
    dev=DATA_ROOT/'e66/development_manifest.json'
    refs[dev]=read(EVIDENCE/'e66_development.json')['manifest_sha256']
    inputs={str(p):s for p,s in refs.items()}
    for p in [Path(__file__),Path(__file__).with_name('e65_audit.py'),Path(__file__).with_name('e65_acquisition.py'),
              RECEIPT,DATA_ROOT/'e66/ai_contract.json',DATA_ROOT/'e32/c3_role_manifest.json',
              DATA_ROOT/'e54/data_contract_v2.json']:
        inputs[str(p)]=digest(p)
    for p,s in inputs.items():
        if digest(p)!=s:raise ValueError('protected input changed')
    # Verify both current TRAIN and the complete E66-reserved AI pool are covered by the old snapshot.
    snapshot_sha={r['sha256'] for r in read(DATA_ROOT/'e52/protected_reference_v1.json')['records']}
    train=read(DATA_ROOT/'e54/data_contract_v2.json')['rows']
    reserve=set(read(DATA_ROOT/'e66/ai_contract.json')['reserved_from_future_train'])
    native=[r for r in read(DATA_ROOT/'e32/c3_role_manifest.json')['records'] if r['record_id'] in reserve]
    if len(native)!=2385 or len({r['record_id'] for r in native})!=len(reserve) or \
            not {r['sha256'] for r in native+train}<=snapshot_sha:
        raise ValueError('current TRAIN or entire reserved AI pool missing from protected snapshot')
    c={'state':'E72_MIDD_score_blind_TRAIN_audit_registered','inputs':inputs,
       'reference_files':[str(p) for p in refs], 'fingerprint_version':VERSION,
       'pillow_version':PIL.__version__,'max_pixels':100000000,'max_seconds':3600,
       'policy':'Verify every original SHA/length. Quarantine failed decodes, below224px and all body/RGB '
                'or dHash<=4 AND pHash63<=4 protected matches. No self exemptions. Build internal transitive '
                'components from same canonical matches plus exact nonempty sensor+EXIF DateTimeOriginal '
                'second. Propagate protected overlap to entire component; keep one SHA256(E72|parent_id) '
                'representative per surviving component. No model score filtering or quota refill.',
       'reserved_AI_snapshot_coverage':2385,'dev_status':'E66 consumed DEVELOPMENT; all320 explicit references.',
       'metadata':'Read only make/model/software/ISO and DateTimeOriginal for grouping; no GPS or serial output.',
       'limits':'Perceptual matches and second-level capture links are incomplete scene detection. No claim '
                'of512 independent scenes/cameras. Whole MIDD publisher reserved exclusively for research '
                'TRAIN, including unselected official-test/denoised images; not DEV/final. Publisher originals '
                'are JPEG renderings, not necessarily unprocessed sensor data.',
       'role_after_audit':'TRAIN','model_scoring_allowed':False,'promotion_allowed':False,
       'license':'CC BY-NC-SA4.0 research only; no automatic serving or redistribution.'}
    write_once(CONTRACT,c);write_once(EVIDENCE/'e72_audit_contract.json',c|{'contract_sha256':digest(CONTRACT)})
    return {'state':c['state'],'reserved_AI_covered':2385}


def validate():
    validate_acquisition();c=read(CONTRACT)
    if digest(CONTRACT)!=read(EVIDENCE/'e72_audit_contract.json')['contract_sha256']:
        raise ValueError('audit contract changed')
    for p,s in c['inputs'].items():
        if digest(p)!=s:raise ValueError('audit input changed: '+p)
    if PIL.__version__!=c['pillow_version']:raise ValueError('decoder version differs')
    return c


def audit():
    c=validate()
    if REPORT.exists() or MANIFEST.exists():raise FileExistsError('audit already recorded')
    deadline=time.monotonic()+c['max_seconds'];facts=[];failed=[]
    for i,row in enumerate(read(RECEIPT)['rows']):
        resource_check(deadline)
        raw=Path(row['path']).read_bytes()
        if hashlib.sha256(raw).hexdigest()!=row['sha256'] or len(raw)!=row['bytes']:
            raise ValueError('acquired source body changed')
        try:
            f=fingerprint(raw,max_pixels=c['max_pixels'])
            if min(f['width'],f['height'])<224:raise ValueError('source below224px')
            with Image.open(row['path']) as im:
                exif=im.getexif();detail={str(k):str(exif[k]) for k in [271,272,305] if k in exif}
                nested=exif.get_ifd(34665) if 34665 in exif else {}
                if 34855 in nested:detail['ISO']=str(nested[34855])
                stamp=str(nested.get(36867,exif.get(36867,'')))
            stamp=stamp if re.fullmatch(r'\d{4}:\d{2}:\d{2} \d{2}:\d{2}:\d{2}',stamp) else None
            facts.append(row|f|{'exif':detail,'capture_second':stamp})
        except (OSError,ValueError) as error:failed.append({'parent_id':row['parent_id'],'reason':str(error)})
        if (i+1)%32==0:print(json.dumps({'decoded':len(facts),'failed':len(failed),'processed':i+1}),flush=True)
    refs=[]
    for path in c['reference_files']:
        item=read(path);refs.extend(item['rows'] if path.endswith('development_manifest.json') else item['records'])
    if any(r['version']!=VERSION for r in refs):raise ValueError('reference fingerprint convention differs')
    cross=cross_role_matches(facts,refs)
    internal=[m for m in cross_role_matches(facts,facts) if m['train_parent']<m['cal_parent']]
    selection=resolve(facts,internal,{r['parent_id'] for r in failed}|{m['train_parent'] for m in cross})
    keep=set(selection['kept'])
    admitted=[r|{'role':'TRAIN','training_allowed':True,'scene_group':selection['lineage'][r['parent_id']],
                 'scene_independence_verified':False} for r in facts if r['parent_id'] in keep]
    admitted.sort(key=lambda r:r['parent_id'])
    counts=dict(Counter(r['sensor'] for r in admitted))
    report={'state':'E72_MIDD_TRAIN_audit_complete','contract_sha256':digest(CONTRACT),'records':admitted,
        'selected':512,'admitted':len(admitted),'sensor_counts':counts,'reference_records':len(refs),
        'decode_failures':failed,'cross_matches':cross,'internal_matches':internal,'selection':selection,
        'dimensions':dict(Counter(f"{r['width']}x{r['height']}" for r in facts)),
        'exif_make_model':dict(Counter(r['exif'].get('271','')+'|'+r['exif'].get('272','') for r in facts)),
        'timestamp_records':sum(bool(r['capture_second']) for r in facts),
        'training_allowed':bool(admitted),'model_scores':0,'independent_final_admitted':False,'limitations':c['limits']}
    write_once(REPORT,report)
    manifest={'state':'E72_MIDD_audited_research_TRAIN','rows':admitted,'audit_sha256':digest(REPORT),
        'class_counts':{'0':len(admitted),'1':0},'publisher':'MIDD','license':c['license'],
        'whole_publisher_role':'RESEARCH_TRAIN_ONLY','independent_scenes_verified':False,'model_scores':0}
    write_once(MANIFEST,manifest)
    summary={k:v for k,v in report.items() if k not in ['records','cross_matches','internal_matches','selection']}
    summary.update(report_sha256=digest(REPORT),manifest_sha256=digest(MANIFEST),
        cross_match_observations=len(cross),internal_match_pairs=len(internal),
        quarantined=len(selection['quarantined']),duplicate_members_removed=len(selection['duplicate_members_removed']))
    write_once(EVIDENCE/'e72_audit.json',summary);return summary


if __name__=='__main__':
    def denied(*a,**kw):raise RuntimeError('E72 audit is offline')
    socket.socket.connect=denied;socket.socket.connect_ex=denied;socket.create_connection=denied
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=['freeze','audit'])
    with (ROOT/'audit_execution.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        print(json.dumps({'freeze':freeze,'audit':audit}[parser.parse_args().stage](),indent=2))
