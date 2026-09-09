"""Frozen E51 A/B versus E43 on consumed E49; diagnostic, never winner selection."""

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np

from experiments import e42_features as dino
from experiments.e43_train import CANDIDATE as OLD_ARTIFACT
from experiments.e48_score import GENERALIST_SHA256
from experiments.e49_evaluation import BINARY_THRESHOLD, REAL_CUT, validate_paired_final
from experiments.e49_failure_diagnosis import MANIFEST, MANIFEST_SHA256, SCORES, SCORES_SHA256
from experiments.e49_final_score import _append
from experiments.e51_features import candidate_matrix
from experiments.e51_pipeline import ROOT, RESULT, RESULT_EVIDENCE, metrics, prepare
from experiments.e51_train_cal_realize import _write_atomic
from pixelproof.project_paths import ML_ROOT

OUT = ROOT/'regression'
EVIDENCE = ML_ROOT.parent/'evidence/e51_e49_regression.json'


def read_pinned(path, expected):
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=expected:
        raise ValueError(f'regression input changed: {path}')
    return raw


def compare(labels, old, new, groups, old_threshold, new_threshold, old_real_cut, new_real_cut):
    result = {'old':metrics(labels,old,groups,old_threshold,old_real_cut),
              'new':metrics(labels,new,groups,new_threshold,new_real_cut)}
    for name,scores,threshold in [('old',old,old_threshold),('new',new,new_threshold)]:
        pred = scores>=threshold
        result[name]['false_ai_count'] = int(((labels==0)&pred).sum())
        result[name]['missed_ai_count'] = int(((labels==1)&~pred).sum())
        result[name]['sources'] = {str(g):{
            'rows':int((groups==g).sum()), 'label':int(labels[groups==g][0]),
            'ai_prediction_rate':float(pred[groups==g].mean())} for g in sorted(set(groups))}
    # Paired stratified-parent bootstrap within this transport. No stacking paired images
    # as independent parents, and no threshold refits in bootstrap samples.
    rng = np.random.default_rng(51)
    deltas = []
    strata = [np.flatnonzero((labels==label)&(groups==g))
              for label in (0,1) for g in sorted(set(groups[labels==label]))]
    for _ in range(2000):
        index = np.concatenate([rng.choice(s,len(s),replace=True) for s in strata])
        y = labels[index]
        old_pred,new_pred = old[index]>=old_threshold,new[index]>=new_threshold
        dfp = new_pred[y==0].mean()-old_pred[y==0].mean()
        dtp = new_pred[y==1].mean()-old_pred[y==1].mean()
        deltas.append([dfp,dtp,(dtp-dfp)/2])
    result['paired_delta_95pct_intervals'] = {
        name:[float(v) for v in np.quantile(np.asarray(deltas)[:,i],[.025,.975])]
        for i,name in enumerate(('real_false_ai','ai_recall','balanced_accuracy'))}
    result['interval_scope'] = '2,000 paired parent bootstraps within fixed source strata; not uncertainty over new publishers.'
    return result


def run(batch_size=16):
    if EVIDENCE.exists() or (OUT/'result.json').exists():
        raise FileExistsError('regression already archived; do not silently rerun')
    cal_raw = RESULT.read_bytes()
    cal = json.loads(cal_raw)
    if cal != json.loads(RESULT_EVIDENCE.read_text()) or cal['state']!='E51_CAL_complete':
        raise ValueError('E51 CAL freeze changed')
    rows = json.loads(read_pinned(MANIFEST,MANIFEST_SHA256))['rows']
    validate_paired_final(rows)
    baseline = [json.loads(line) for line in read_pinned(SCORES,SCORES_SHA256).splitlines() if line]
    by_id = {r['record_id']:r for r in baseline}
    if len(by_id)!=len(rows) or any(r['record_id'] not in by_id for r in rows):
        raise ValueError('baseline coverage changed')
    for row in rows:
        if any(row[k]!=by_id[row['record_id']][k] for k in ('parent_id','condition','label','source')):
            raise ValueError('baseline identity/label changed')
    artifacts = {}
    for name in ('A','B'):
        path = ROOT/f'training/e51_{name}.joblib'
        read_pinned(path,cal['candidates'][name]['artifact_sha256'])
        artifact = joblib.load(path)
        if (artifact['candidate']!=name or artifact['positive_label']!=1
                or artifact['threshold']!=cal['candidates'][name]['threshold']
                or artifact['real_cut']!=cal['candidates'][name]['real_cut']):
            raise ValueError('frozen candidate decision changed')
        artifacts[name] = artifact
    read_pinned(OLD_ARTIFACT,GENERALIST_SHA256)
    old_head = joblib.load(OLD_ARTIFACT)['head']
    contract = {'purpose':'consumed E49 paired regression only; forbidden for selection or refitting',
                'cal_sha256':hashlib.sha256(cal_raw).hexdigest(),
                'manifest_sha256':MANIFEST_SHA256,'old_scores_sha256':SCORES_SHA256,
                'code_sha256':dino._sha256_file(Path(__file__)),
                'pipeline_sha256':dino._sha256_file(Path(__file__).with_name('e51_pipeline.py')),
                'feature_sha256':dino._sha256_file(Path(__file__).with_name('e51_features.py')),
                'old_artifact_sha256':GENERALIST_SHA256,
                'candidates':{k:v['artifact_sha256'] for k,v in cal['candidates'].items()},
                'CAL_selected':cal['selected'],'development_access':False,'network_transfer':False}
    if (OUT/'contract.json').exists():
        if json.loads((OUT/'contract.json').read_text())!=contract:
            raise ValueError('regression contract changed')
    else:
        _write_atomic(OUT/'contract.json',contract)
    partial = OUT/'scores.partial.jsonl'
    done = []
    if partial.exists():
        raw = partial.read_bytes()
        if raw and not raw.endswith(b'\n'):
            raise ValueError('truncated regression checkpoint')
        done = [json.loads(line) for line in raw.splitlines()]
        if len(done)>len(rows): raise ValueError('score prefix exceeds population')
        for score,row in zip(done,rows):
            if (any(score[k]!=row[k] for k in ('record_id','parent_id','label','source','condition'))
                    or any(not np.isfinite(score[k]) or not 0<=score[k]<=1 for k in ('A','B','old_reproduced'))):
                raise ValueError('regression checkpoint changed')
            read_pinned(Path(row['path']),row['sha256'])
    import torch
    model,means,stds,_ = dino._load_small()
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    model = model.to(device).eval()
    mean = torch.tensor(means,device=device).view(1,3,1,1)
    std = torch.tensor(stds,device=device).view(1,3,1,1)
    with torch.inference_mode(),ThreadPoolExecutor(max_workers=4) as pool:
        for start in range(len(done),len(rows),batch_size):
            group = rows[start:start+batch_size]
            # Both original and Q75 are already materialized; CAL role here means clean
            # decoding of a bound file, NOT permission to calibrate on this old test.
            prepared = list(pool.map(prepare,[{**r,'role':'CAL'} for r in group]))
            arrays = np.stack([crop for p in prepared for crop in p[0]])
            tensor = torch.from_numpy(arrays).to(device).permute(0,3,1,2).float().div_(255)
            blocks = model.forward_intermediates((tensor-mean)/std,indices=list(dino.BLOCKS['small']),
                      return_prefix_tokens=True,norm=True,intermediates_only=True)
            tokens = torch.stack([b[1][:,0,:] for b in blocks],dim=1)
            base = dino.aggregate_tokens(tokens.float().cpu().numpy(),len(group))
            residual = np.stack([p[1] for p in prepared])
            old_scores = old_head.predict_proba(base)[:,1]
            predictions = {n:a['head'].predict_proba(candidate_matrix(base,residual,n))[:,1]
                           for n,a in artifacts.items()}
            output = [{**{k:r[k] for k in ('record_id','parent_id','label','source','condition')},
                       'old_reproduced':float(old_scores[i]),
                       **{n:float(s[i]) for n,s in predictions.items()}} for i,r in enumerate(group)]
            _append(partial,output); done.extend(output)
            print(json.dumps({'phase':'E49_regression','observations':len(done),'total':len(rows)}),flush=True)
    maximum_difference = max(abs(r['old_reproduced']-by_id[r['record_id']]['score']) for r in done)
    # A feature/preprocessing drift must not masquerade as a classifier improvement.
    if maximum_difference>5e-5:
        raise ValueError(f'old scorer reproduction failed: max difference {maximum_difference}')
    labels = np.asarray([r['label'] for r in rows])
    groups = np.asarray([r['source'] for r in rows])
    conditions = np.asarray([r['condition'] for r in rows])
    old = np.asarray([by_id[r['record_id']]['score'] for r in rows])
    results = {}
    for name,artifact in artifacts.items():
        new = np.asarray([r[name] for r in done])
        results[name] = {c:compare(labels[m:=conditions==c],old[m],new[m],groups[m],
                         BINARY_THRESHOLD,artifact['threshold'],REAL_CUT,artifact['real_cut'])
                         for c in sorted(set(conditions))}
    report = {'schema_version':1,'state':'consumed_E49_regression_complete_not_independent_final',
              'contract':contract,'score_sha256':dino._sha256_file(partial),
              'observations':len(done),'parents':len({r['parent_id'] for r in rows}),
              'old_reproduction_max_score_difference':maximum_difference,
              'candidates':results,'winner_selection_from_this_report':False,'serving_changed':False}
    _write_atomic(OUT/'result.json',report); _write_atomic(EVIDENCE,report)
    return report


if __name__=='__main__':
    print(json.dumps(run(),indent=2))
