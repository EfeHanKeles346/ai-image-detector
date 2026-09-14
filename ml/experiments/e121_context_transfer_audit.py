"""Read-only failed context-transfer diagnosis from locked scores and TRAIN geometry."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import numpy as np
from scipy.special import logit
from experiments.e65_acquisition import digest, read, write_once
from experiments.e114_context_development import pair_identity, comparison
from experiments.e92_model import AI_CUT, REAL_CUT
from pixelproof.e32_candidate import INPUT_SIZE, RESIZE_SHORT_SIDE
from pixelproof.project_paths import DATA_ROOT, ML_ROOT

ROOT=DATA_ROOT/'e121';EVIDENCE=ML_ROOT.parent/'evidence';CONTRACT=ROOT/'contract.json'


def center_fraction(width,height):
    if width<=0 or height<=0:raise ValueError('Positive source geometry required')
    # Aspect-ratio based area of the historical center window, before JPEG90.
    scale=RESIZE_SHORT_SIDE/min(width,height)
    return INPUT_SIZE**2/(round(width*scale)*round(height*scale))


def summarize(rows):
    groups=defaultdict(list)
    for row in rows:groups[(row['condition'],row['label'],row['source'])].append(row)
    out=[]
    for (condition,label,source),group in sorted(groups.items()):
        old=np.array([r['E103_score'] for r in group]);new=np.array([r['score'] for r in group])
        if not np.isfinite(old).all() or not np.isfinite(new).all() or np.any((old<0)|(old>1)|(new<0)|(new>1)):
            raise ValueError('Locked probabilities must be finite unit scores')
        shift=logit(np.clip(new,1e-15,1-1e-15))-logit(np.clip(old,1e-15,1-1e-15))
        out.append({'condition':condition,'label':label,'source':source,'views':len(group),
            'logit_shift_quantiles_0_25_50_75_100':np.quantile(shift,[0,.25,.5,.75,1]).tolist(),
            'nonnegative_correction_views':int(np.sum(shift>=0)),
            'changed_AI_decisions':int(np.sum((old>=AI_CUT)!=(new>=AI_CUT))),
            'changed_REAL_decisions':int(np.sum((old>=REAL_CUT)!=(new>=REAL_CUT)))})
    return out


def freeze():
    files=[Path(__file__),ML_ROOT/'src/pixelproof/e32_candidate.py',
        ML_ROOT/'experiments/e114_context_development.py',ML_ROOT/'experiments/e92_model.py',
        DATA_ROOT/'e112/contract.json',DATA_ROOT/'e113/report.json',DATA_ROOT/'e114/report.json',DATA_ROOT/'e114/scores.json']
    if digest(files[-3])!=digest(EVIDENCE/'e113_context_fit.json') or \
            digest(files[-2])!=digest(EVIDENCE/'e114_context_development.json') or \
            digest(files[-1])!=read(files[-2])['scores_sha256']:
        raise ValueError('Frozen paired results differ')
    if any(r['passes_consumed_DEV_screen'] for r in read(files[-2])['summary'].values()):raise ValueError('Both failed branches required')
    if digest(files[-4])!=read(EVIDENCE/'e112_context_contract.json')['contract_sha256']:raise ValueError('TRAIN manifest differs')
    c={'state':'E121_posthoc_readonly_transfer_audit_registered','inputs':{str(p):digest(p) for p in files},
        'scope':'All locked640 consumed DEV scores for both branches; all12525 TRAIN metadata rows. Aggregate by complete source/class/condition, no image/prompt identity export. No new scoring, threshold proposals, counterfactual head removal or fitting.',
        'geometry':'Historical center branch only:224px square after short-side256 resize. Estimate retained source area by metadata aspect ratio; report unknown sizes explicitly. This is NOT the union of center and two texture crops, and not proof of the missed AI cause.',
        'limits':'Posthoc descriptive diagnostic of consumed development. TRAIN separability does not establish OOD transfer. Counts can remain constant despite score changes; compare individual decisions and complete guard outputs.',
        'model_inference':0,'training_allowed':False,'downloads':0}
    ROOT.mkdir(exist_ok=True);write_once(CONTRACT,c);write_once(EVIDENCE/'e121_transfer_contract.json',c|{'contract_sha256':digest(CONTRACT)})
    return {'paired_consumed_views':1280,'TRAIN_metadata_parents':12525}


def audit():
    c=read(CONTRACT)
    if digest(CONTRACT)!=read(EVIDENCE/'e121_transfer_contract.json')['contract_sha256']:raise ValueError('Audit contract differs')
    for p,s in c['inputs'].items():
        if digest(p)!=s:raise ValueError('Locked audit input differs')
    scores=read(DATA_ROOT/'e114/scores.json')['branches'];names=list(scores);pair_identity(scores[names[0]],scores[names[1]])
    reports=read(DATA_ROOT/'e114/report.json');analysis={}
    for name,rows in scores.items():
        if comparison(rows)!=reports['results'][name]:raise ValueError('Complete DEV guard replay differs')
        analysis[name]=summarize(rows)
    geometry=defaultdict(lambda:{'known':[],'unknown':0})
    for r in read(DATA_ROOT/'e112/contract.json')['rows']:
        if r['role']!='TRAIN':raise ValueError('Only admitted TRAIN geometry')
        key=f'{r["label"]}:{r["source"]}'
        if r.get('width',0)>0 and r.get('height',0)>0:geometry[key]['known'].append(center_fraction(r['width'],r['height']))
        else:geometry[key]['unknown']+=1
    geometry={k:{'known_geometry_parents':len(v['known']),'unknown_geometry_parents':v['unknown'],
        'center_area_fraction_min_median_max':np.quantile(v['known'],[0,.5,1]).tolist() if v['known'] else None} for k,v in sorted(geometry.items())}
    result={'state':'E121_failed_context_transfer_audited','contract_sha256':digest(CONTRACT),
        'locked_DEV_guards_reproduce':True,'branch_source_summaries':analysis,'TRAIN_geometry':geometry,
        'next_hypothesis':'Evaluate uncropped full-frame information on TRAIN only with a separately registered feasibility probe, preserving old scores and every role/retention guard. Both context heads remain rejected; do not choose a new cut for the known missed image.',
        'model_inference':0,'training_allowed':False,'limits':c['limits']}
    write_once(ROOT/'report.json',result);write_once(EVIDENCE/'e121_context_transfer.json',result)
    return {'state':result['state'],'source_geometry_groups':len(geometry),'locked_DEV_guards_reproduce':True}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['freeze','audit'])
    print(json.dumps({'freeze':freeze,'audit':audit}[p.parse_args().stage](),indent=2))
