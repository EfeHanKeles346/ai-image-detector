"""One frozen E51-A DEVELOPMENT evaluation; no fit, cut selection, or second candidate."""

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np

from experiments import e42_features as dino
from experiments.e49_final_score import _append
from experiments.e51_development import MANIFEST, EVIDENCE as MANIFEST_EVIDENCE, OUT, pinned
from experiments.e51_pipeline import ROOT, RESULT, RESULT_EVIDENCE, prepare, metrics
from experiments.e51_train_cal_realize import _write_atomic
from pixelproof.project_paths import ML_ROOT

EVIDENCE = ML_ROOT.parent/'evidence/e51_development_result.json'


def validate_rows(rows):
    if len(rows)!=6880 or len({r['record_id'] for r in rows})!=6880:
        raise ValueError('DEV observation count/identity changed')
    maps = {c:{r['parent_id']:(r['label'],r['group']) for r in rows if r['condition']==c}
            for c in ('original','q75')}
    if len(maps['original'])!=3440 or maps['original']!=maps['q75']:
        raise ValueError('DEV pairing changed')
    originals = [r for r in rows if r['condition']=='original']
    if sum(r['label']==0 for r in originals)!=2640 or sum(r['label']==1 for r in originals)!=800:
        raise ValueError('DEV class counts changed')
    if {r['role'] for r in rows}!={'DEVELOPMENT'}:
        raise ValueError('DEV role changed')


def intervals(rows,scores,threshold):
    labels = np.asarray([r['label'] for r in rows])
    real_strata = []
    for cell in ('unaltered','postprocessed'):
        clusters = {}
        for i,r in enumerate(rows):
            if r['label']==0 and r['transport_cell']==cell:
                clusters.setdefault(r.get('scene_cluster',f'row:{i}'),[]).append(i)
        real_strata.append(list(clusters.values()))
    ai_clusters = []
    for category in sorted({r['prompt_category'] for r in rows if r['label']==1}):
        ordinals = sorted({r['prompt_ordinal'] for r in rows if r['label']==1 and r['prompt_category']==category})
        blocks = [[i for i,r in enumerate(rows) if r['label']==1 and r['prompt_category']==category
                   and r['prompt_ordinal']==ordinal] for ordinal in ordinals]
        if len(blocks)!=20 or any(len(b)!=5 for b in blocks):
            raise ValueError('AI prompt bootstrap structure changed')
        ai_clusters.append(np.asarray(blocks))
    rng = np.random.default_rng(51)
    values = []
    predicted = scores>=threshold
    for _ in range(2000):
        real = np.concatenate([np.concatenate([s[j] for j in rng.integers(0,len(s),len(s))]) for s in real_strata])
        ai = np.concatenate([b[rng.integers(0,len(b),len(b))].ravel() for b in ai_clusters])
        fp,tp = predicted[real].mean(),predicted[ai].mean()
        values.append([fp,tp,(1-fp+tp)/2])
    return {k:[float(v) for v in np.quantile(np.asarray(values)[:,i],[.025,.975])]
            for i,k in enumerate(('real_false_ai','ai_recall','balanced_accuracy'))}


def run(batch_size=16):
    if EVIDENCE.exists() or (OUT/'result.json').exists():
        raise FileExistsError('DEVELOPMENT already evaluated; no second attempt')
    summary = json.loads(MANIFEST_EVIDENCE.read_text())
    manifest = pinned(MANIFEST,summary['manifest_sha256'])
    cal = json.loads(RESULT.read_text())
    if (cal!=json.loads(RESULT_EVIDENCE.read_text()) or cal['selected']!='A'
            or not cal['candidates']['A']['passed'] or manifest['model_scores_created']!=0
            or manifest['state']!='E51_DEVELOPMENT_frozen_unscored'
            or manifest['cal_result_sha256']!=dino._sha256_file(RESULT)):
        raise ValueError('DEV admission/candidate freeze changed')
    rows = manifest['rows']; validate_rows(rows)
    path = ROOT/'training/e51_A.joblib'
    if dino._sha256_file(path)!=manifest['candidate_sha256'] or manifest['candidate_sha256']!=cal['candidates']['A']['artifact_sha256']:
        raise ValueError('DEV candidate changed')
    artifact = joblib.load(path)
    if (artifact['threshold']!=cal['candidates']['A']['threshold']
            or artifact['real_cut']!=cal['candidates']['A']['real_cut']
            or list(artifact['head'].classes_)!=[0,1]):
        raise ValueError('DEV decision/label mapping changed')
    contract = {'candidate_sha256':manifest['candidate_sha256'],
                'manifest_sha256':summary['manifest_sha256'],'cal_result_sha256':manifest['cal_result_sha256'],
                'threshold':artifact['threshold'],'real_cut':artifact['real_cut'],
                'code_sha256':dino._sha256_file(Path(__file__)),
                'preprocessing_sha256':dino._sha256_file(Path(dino.__file__)),
                'policy':'One A evaluation; no fitting, threshold selection or candidate substitution.',
                'scores_created':0}
    contract_path = OUT/'score_contract.json'
    if contract_path.exists():
        if json.loads(contract_path.read_text())!=contract: raise ValueError('DEV score contract changed')
    else: _write_atomic(contract_path,contract)
    partial = OUT/'scores.partial.jsonl'
    done = []
    if partial.exists():
        raw = partial.read_bytes()
        if raw and not raw.endswith(b'\n'): raise ValueError('truncated DEV score checkpoint')
        done = [json.loads(line) for line in raw.splitlines()]
        if len(done)>len(rows): raise ValueError('DEV score prefix longer than population')
        for score,row in zip(done,rows):
            if (any(score[k]!=row[k] for k in ('record_id','parent_id','label','condition','group'))
                    or not np.isfinite(score['score']) or not 0<=score['score']<=1
                    or dino._sha256_file(Path(row['path']))!=row['sha256']):
                raise ValueError('DEV cached score/input changed')
    import torch
    model,means,stds,_ = dino._load_small()
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    model = model.to(device).eval()
    mean = torch.tensor(means,device=device).view(1,3,1,1)
    std = torch.tensor(stds,device=device).view(1,3,1,1)
    with torch.inference_mode(),ThreadPoolExecutor(max_workers=4) as pool:
        for start in range(len(done),len(rows),batch_size):
            group = rows[start:start+batch_size]
            prepared = list(pool.map(prepare,[{**r,'role':'CAL'} for r in group]))
            tensor = torch.from_numpy(np.stack([c for p in prepared for c in p[0]])).to(device)
            tensor = tensor.permute(0,3,1,2).float().div_(255)
            blocks = model.forward_intermediates((tensor-mean)/std,indices=list(dino.BLOCKS['small']),
                       return_prefix_tokens=True,norm=True,intermediates_only=True)
            tokens = torch.stack([b[1][:,0,:] for b in blocks],dim=1)
            scores = artifact['head'].predict_proba(dino.aggregate_tokens(tokens.float().cpu().numpy(),len(group)))[:,1]
            if not np.isfinite(scores).all(): raise ValueError('invalid DEV scores')
            output = [{**{k:r[k] for k in ('record_id','parent_id','label','condition','group')},'score':float(s)}
                      for r,s in zip(group,scores,strict=True)]
            _append(partial,output); done.extend(output)
            print(json.dumps({'phase':'fresh_DEVELOPMENT','observations':len(done),'total':len(rows)}),flush=True)
    results = {}
    for condition in ('original','q75'):
        index = [i for i,r in enumerate(rows) if r['condition']==condition]
        subset = [rows[i] for i in index]
        labels = np.asarray([r['label'] for r in subset])
        scores = np.asarray([done[i]['score'] for i in index])
        groups = np.asarray([r['group'] for r in subset])
        result = metrics(labels,scores,groups,artifact['threshold'],artifact['real_cut'])
        result['observed_group_checks_passed'] = result.pop('passed')
        result['score_coverage'] = 1.0
        result['false_ai_count'] = int(((labels==0)&(scores>=artifact['threshold'])).sum())
        result['missed_ai_count'] = int(((labels==1)&(scores<artifact['threshold'])).sum())
        result['source_rates'] = {g:{'rows':int((groups==g).sum()),'label':int(labels[groups==g][0]),
             'ai_prediction_rate':float((scores[groups==g]>=artifact['threshold']).mean())} for g in sorted(set(groups))}
        result['cluster_bootstrap_95pct_intervals'] = intervals(subset,scores,artifact['threshold'])
        results[condition] = result
    report = {'state':'E51_fresh_DEVELOPMENT_complete_no_retuning','candidate':'A','contract':contract,
              'raw_score_sha256':dino._sha256_file(partial),'parents':3440,'observations':6880,
              'conditions':results,'observed_checks_passed':all(r['observed_group_checks_passed'] for r in results.values()),
              'limits':manifest['limits']+['Bootstrap groups detected REAL scenes and each AI prompt across its five generators; additional real scene dependence remains unknown.'],
              'independent_final_passed':False,'serving_changed':False}
    _write_atomic(OUT/'result.json',report); _write_atomic(EVIDENCE,report)
    return report


if __name__=='__main__':
    print(json.dumps(run(),indent=2))
