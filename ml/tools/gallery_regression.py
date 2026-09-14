"""Freeze/compare private owner REAL + consumed E66 AI replay without fitting."""
import argparse
import json
from pathlib import Path
from pixelproof.project_paths import DATA_ROOT, ML_ROOT, WORK_ROOT
from pixelproof.e92_demo import AI_CUT, digest
from pixelproof.development_regression import compare_development

ROOT=WORK_ROOT/'e95_owner_gallery'
OUT=ML_ROOT.parent/'evidence'
BASELINE=ROOT/'regression_baseline.json'


def read(p):return json.loads(p.read_text())

def write(p,v):
    with p.open('x') as f:json.dump(v,f,indent=2,sort_keys=True);f.write('\n')


def freeze():
    gallery=ROOT/'scores.json';ai=DATA_ROOT/'e92/dev_scores.json'
    if digest(gallery)!=read(OUT/'e95_gallery_scores_receipt.json')['scores_sha256'] or \
            digest(ai)!=read(OUT/'e92_dev_scores.json')['scores_sha256']:
        raise ValueError('Locked input scores changed')
    parents=list({r['sha256']:r for r in read(gallery)['rows']}.values());rows=[]
    for r in parents:
        for condition,key in [('publisher_original','original_score'),('social_q75','social_q75_score')]:
            rows.append({'parent_id':'owner:'+r['sha256'],'sha256':r['sha256'],'label':0,
                         'source':'owner_gallery','role':'CONSUMED_DEVELOPMENT',
                         'condition':condition,'score':r[key]})
    rows += [{k:r[k] for k in ['parent_id','sha256','label','source','condition','score']} |
             {'role':'CONSUMED_DEVELOPMENT'} for r in read(ai)['rows'] if r['label']==1]
    if len(rows)!=732:raise ValueError('Expected206 REAL +160 AI parents in two conditions')
    result=compare_development(rows,rows,AI_CUT)
    write(BASELINE,{'rows':rows,'ai_cut':AI_CUT,'transform_code_sha256':digest(ML_ROOT/'src/pixelproof/e92_demo.py')})
    write(OUT/'e96_gallery_regression.json',{'state':'consumed_regression_baseline_frozen',
        'baseline_sha256':digest(BASELINE),'ai_cut':AI_CUT,'source_score_sha256':[digest(gallery),digest(ai)],
        'self_replay':result,'note':'Private gallery view identity uses original file SHA plus condition and frozen social transform; no gallery bytes/names exported. Future comparisons require all732 rows. No fitting/threshold selection on this replay.'})
    return result


def compare(path):
    receipt=read(OUT/'e96_gallery_regression.json')
    if digest(BASELINE)!=receipt['baseline_sha256']:raise ValueError('Baseline changed')
    old=read(BASELINE)
    return compare_development(old['rows'],read(path)['rows'],old['ai_cut'])


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['freeze','compare']);p.add_argument('--candidate',type=Path)
    args=p.parse_args()
    if args.stage=='compare' and args.candidate is None:p.error('--candidate required')
    result=freeze() if args.stage=='freeze' else compare(args.candidate)
    print(json.dumps(result,indent=2))
    if not result['passes_paired_regression']:
        raise SystemExit(2)
