"""Complete16-parent numerical correction replay after the fixed E120 branch decision."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import socket
import time
import numpy as np
from experiments import e120_inpaint_numerics as probe, e119_inpainting_pilot as original, e118_inpaint_assets as assets
from experiments.e65_acquisition import digest,read,write_once
from pixelproof.project_paths import DATA_ROOT,ML_ROOT

ROOT=DATA_ROOT/'e122';EVIDENCE=ML_ROOT.parent/'evidence';CONTRACT=ROOT/'contract.json'


def freeze():
    previous=probe.validate();report=read(probe.ROOT/'report.json')
    if digest(probe.ROOT/'report.json')!=digest(EVIDENCE/'e120_inpaint_numerics.json'):
        raise ValueError('Complete numerical probe receipt differs')
    branch=next((b for b in ('fp16_sdpa','fp32_sliced') if report['branch_passes'][b]),None)
    if branch is None or branch!=report['next_complete_replay_branch']:raise ValueError('No qualified numerical correction')
    rows=read(original.ROOT/'prepared.json')['rows']
    if len(rows)!=16 or [r['index'] for r in rows]!=list(range(16)):raise ValueError('Unchanged entire16-parent pilot required')
    files=[Path(__file__),Path(probe.__file__),probe.CONTRACT,probe.ROOT/'report.json']
    c={'state':'E122_complete_numerical_replay_registered','branch':branch,
        'inputs':previous['inputs']|{str(p):digest(p) for p in files},
        'parents':16,'seeds':list(range(119000,119016)),'max_seconds':2400,'mps_limit_bytes':10*1024**3,
        'change':'Apply the prespecified passing numerical branch to ALL16 original E119 parents. Keep source weights,512px geometry,masks,prompt,30 steps,guidance7.5,strength1 and safety checker. Retain finite tensor guards and every original acceptance check. No selection/refill or parameter sweep.',
        'labels':'Composites carry intended AI region masks; original/classic controls carry zero AI masks. Raw generator images are diagnostic outputs, not localized-ground-truth training pairs.',
        'limits':'Engineering correction of numerical failure, not improved detector performance or a semantic-quality audit. Only after16/16 pass may a separate TRAIN-only spatial learning experiment be registered. All original research-only licences and ancestry roles remain.',
        'downloads':0,'detector_scores':0,'training_allowed':False,'promotion_allowed':False}
    ROOT.mkdir(exist_ok=True);write_once(CONTRACT,c)
    write_once(EVIDENCE/'e122_inpaint_replay_contract.json',{k:v for k,v in c.items() if k!='inputs'}|{'contract_sha256':digest(CONTRACT)})
    return {'branch':branch,'parents':16}


def replay():
    c=read(CONTRACT)
    if digest(CONTRACT)!=read(EVIDENCE/'e122_inpaint_replay_contract.json')['contract_sha256']:raise ValueError('Replay contract differs')
    for p,s in c['inputs'].items():
        if digest(p)!=s:raise ValueError('Replay input differs')
    original.validate()
    if (ROOT/'report.json').exists():raise FileExistsError('Full numerical replay already complete')
    for r in read(assets.CONTRACT)['files']:assets.verify(assets.ROOT/'model'/r['rfilename'],r)
    start=time.monotonic()
    rows,peak=probe.run_cases(c['branch'],read(original.ROOT/'prepared.json')['rows'],ROOT,start+c['max_seconds'],c['mps_limit_bytes'])
    result={'state':'E122_complete16_numerical_replay_finished','contract_sha256':digest(CONTRACT),'branch':c['branch'],
        'passed':len(rows)==16 and all(r['passed'] for r in rows),'parents':len(rows),'passed_parents':sum(r['passed'] for r in rows),
        'rows':rows,'peak_mps_bytes':peak,'seconds':time.monotonic()-start,'detector_scores':0,'training_allowed':False,
        'promotion_allowed':False,'limits':c['limits']}
    write_once(ROOT/'report.json',result)
    public={k:v for k,v in result.items() if k!='rows'}|{'report_sha256':digest(ROOT/'report.json'),
        'mean_valid_raw_background_changed_fraction':float(np.mean([r['raw_background_changed_fraction'] for r in rows if r['passed']])) if any(r['passed'] for r in rows) else None}
    write_once(EVIDENCE/'e122_inpaint_replay.json',public)
    text='\n### E122 complete numerical replay result\n\n'+json.dumps(public,sort_keys=True)+'\n\nOriginal E119 remains failed; this is a separately registered correction. No detector performance or independent evidence is claimed.\n'
    for path in (ML_ROOT.parent/'PLAN.md',ML_ROOT.parent/'HISTORY.md',ML_ROOT/'EXPERIMENTS.md',ML_ROOT.parent/'DATASETS.md'):
        with path.open('a') as f:fcntl.flock(f,fcntl.LOCK_EX);f.write(text)
    return public


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['freeze','replay']);args=p.parse_args()
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    def denied(*a,**kw):raise RuntimeError('Complete numerical replay is offline')
    socket.socket.connect=denied;socket.socket.connect_ex=denied;socket.create_connection=denied
    ROOT.mkdir(exist_ok=True)
    with (ROOT/'execution.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        print(json.dumps({'freeze':freeze,'replay':replay}[args.stage](),indent=2))
