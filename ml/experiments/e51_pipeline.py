"""Fixed offline E51 extraction and TRAIN/CAL A/B experiment; no DEVELOPMENT access."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
import warnings

import joblib
import numpy as np
from PIL import Image, ImageOps
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

from experiments import e42_features as dino
from experiments.e43_train import E42_FEATURES, E42_FEATURES_SHA256, _load, parent_source_weights
from experiments.e51_admission import OUTPUT as ADMISSION, EVIDENCE as ADMISSION_EVIDENCE
from experiments.e51_features import aggregate_crops, candidate_matrix
from experiments.e51_train_cal_realize import _write_atomic
from pixelproof.project_paths import DATA_ROOT, ML_ROOT

ROOT = DATA_ROOT / 'e51'
FEATURES = ROOT / 'features/train_cal_v1.npz'
FEATURE_EVIDENCE = ML_ROOT.parent / 'evidence/e51_features.json'
RESULT = ROOT / 'training/cal_result_v1.json'
RESULT_EVIDENCE = ML_ROOT.parent / 'evidence/e51_cal_result.json'
SEEDS = (42,43,44)


def admission():
    raw = ADMISSION.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    if sha != json.loads(ADMISSION_EVIDENCE.read_text())['admission_sha256']:
        raise ValueError('E51 admission changed')
    value = json.loads(raw)
    if value.get('training_authorized') is not True or value.get('model_scores_created') != 0:
        raise ValueError('E51 training not admitted')
    return value, sha


def views(value):
    result = []
    for row in value['train']:
        for condition in ('clean', dino.assigned_transport(row['parent_id']), 'q75'):
            result.append({**row,'role':'TRAIN','condition':condition,
                           'record_id':f"TRAIN|{row['parent_id']}|{condition}"})
    for row in value['cal']:
        result.append({**row,'role':'CAL','record_id':f"CAL|{row['parent_id']}|{row['condition']}"})
    return result


def prepare(row):
    raw = Path(row['path']).read_bytes()
    if hashlib.sha256(raw).hexdigest() != row['sha256']:
        raise ValueError(f"E51 image changed: {row['parent_id']}")
    with Image.open(BytesIO(raw)) as opened:
        image = ImageOps.exif_transpose(opened).convert('RGB')
    condition = row['condition']
    if row['role'] == 'CAL':
        condition = 'clean'  # CAL Q75 is already encoded on disk.
    elif condition == 'q75':
        output = BytesIO()
        image.save(output,format='JPEG',quality=75,subsampling=2,optimize=False)
        with Image.open(BytesIO(output.getvalue())) as compressed:
            image = compressed.convert('RGB')
        condition = 'clean'
    crops = dino.texture_crops(dino.transport_image(image, condition))
    return crops, aggregate_crops(crops)


def extract(batch_size=24):
    if FEATURES.exists() or FEATURE_EVIDENCE.exists():
        raise FileExistsError('E51 features already frozen')
    value, admission_sha = admission()
    rows = views(value)
    old = _load(E42_FEATURES, E42_FEATURES_SHA256)
    old_index = {(str(p),str(c),int(y)):i for i,(p,c,y) in enumerate(zip(
        old['parent_ids'],old['conditions'],old['labels'],strict=True))}
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    import torch
    model, means, stds, weight_sha = dino._load_small()
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    model = model.to(device).eval()
    mean = torch.tensor(means,device=device).view(1,3,1,1)
    std = torch.tensor(stds,device=device).view(1,3,1,1)
    code_sha = hashlib.sha256(Path(__file__).read_bytes() + Path(dino.__file__).read_bytes()
                             + Path(__file__).with_name('e51_features.py').read_bytes()).hexdigest()
    binding = hashlib.sha256(f'{admission_sha}|{weight_sha}|{code_sha}'.encode()).hexdigest()
    chunks, reused = [], 0
    chunk_root = ROOT / 'features/chunks_v1'
    with torch.inference_mode(), ThreadPoolExecutor(max_workers=4) as pool:
        for start in range(0,len(rows),batch_size):
            group = rows[start:start+batch_size]
            path = chunk_root / f'{start:06d}.npz'
            ids = np.asarray([r['record_id'] for r in group])
            if path.exists():
                with np.load(path,allow_pickle=False) as saved:
                    if str(saved['binding']) != binding or not np.array_equal(saved['ids'],ids):
                        raise ValueError('E51 feature chunk binding changed')
                    values = {k:saved[k] for k in saved.files}
                # Do not reuse cached features for changed source bytes.
                for row in group:
                    if dino._sha256_file(Path(row['path'])) != row['sha256']:
                        raise ValueError('cached E51 input bytes changed')
            else:
                prepared = list(pool.map(prepare,group))
                base = np.empty((len(group),3072),dtype=np.float32)
                missing, arrays = [], []
                for i,row in enumerate(group):
                    condition = ('clean' if row['role']=='CAL' and row['condition']=='original'
                                 else row['condition'])
                    key = (row['parent_id'],condition,int(row['label']))
                    if key in old_index:
                        base[i] = old['features'][old_index[key]]
                        reused += 1
                    else:
                        missing.append(i); arrays.extend(prepared[i][0])
                if missing:
                    tensor = torch.from_numpy(np.stack(arrays)).to(device).permute(0,3,1,2).float().div_(255)
                    blocks = model.forward_intermediates((tensor-mean)/std,indices=list(dino.BLOCKS['small']),
                        return_prefix_tokens=True,norm=True,intermediates_only=True)
                    tokens = torch.stack([item[1][:,0,:] for item in blocks],dim=1)
                    base[missing] = dino.aggregate_tokens(tokens.float().cpu().numpy(),len(missing))
                values = {'dino':base,'residual':np.stack([p[1] for p in prepared]),
                          'binding':np.asarray(binding),'ids':ids}
                dino._save_npz(path,values)
            chunks.append(values)
            print(json.dumps({'phase':'features','views':start+len(group),'total':len(rows),'device':str(device)}),flush=True)
    values = {'dino':np.concatenate([c['dino'] for c in chunks]),
              'residual':np.concatenate([c['residual'] for c in chunks]),
              'binding':np.asarray(binding),'admission_sha256':np.asarray(admission_sha)}
    for name,key in [('ids','record_id'),('parents','parent_id'),('roles','role'),('conditions','condition'),('sources','source')]:
        values[name] = np.asarray([r[key] for r in rows])
    values['groups'] = np.asarray([f"SCMI:{r['device_id']}" if r['role']=='CAL' and r['label']==0 else r['source'] for r in rows])
    values['labels'] = np.asarray([r['label'] for r in rows],dtype=np.int8)
    if not np.isfinite(values['dino']).all() or not np.isfinite(values['residual']).all():
        raise ValueError('non-finite E51 features')
    dino._save_npz(FEATURES,values)
    report = {'schema_version':1,'admission_sha256':admission_sha,'feature_binding':binding,
              'feature_sha256':dino._sha256_file(FEATURES),'views':len(rows),'DINO_dimensions':3072,
              'residual_dimensions':32,'model_scores_created':0,'state':'features_complete_before_fit',
              'reused_old_DINO_views_this_run':reused,'device':str(device)}
    _write_atomic(FEATURE_EVIDENCE,report)
    return report


def validate_scores(labels,scores,*columns):
    if (labels.ndim != 1 or scores.shape != labels.shape or len(labels)==0
            or set(labels.tolist())!={0,1} or not np.isfinite(scores).all()
            or np.any((scores<0)|(scores>1))
            or any(c.shape!=labels.shape for c in columns)):
        raise ValueError('invalid E51 score population')


def real_safe_threshold(labels,scores,groups,conditions):
    validate_scores(labels,scores,groups,conditions)
    cuts = [0.0]
    for condition in sorted(set(conditions)):
        validate_scores(labels[conditions==condition],scores[conditions==condition])
        real = (conditions==condition)&(labels==0)
        populations = [(scores[real],.10)]
        populations.extend((scores[real&(groups==group)],.20) for group in sorted(set(groups[real])))
        for values,budget in populations:
            allowed = int(np.floor(len(values)*budget + 1e-12))
            cuts.append(float(np.nextafter(np.sort(values)[len(values)-allowed-1],np.inf)))
    return max(cuts)


def metrics(labels,scores,groups,threshold,real_cut):
    validate_scores(labels,scores,groups)
    if not (0<=real_cut<=threshold and np.isfinite(threshold)):
        raise ValueError('invalid decision band')
    real,ai = labels==0,labels==1
    predicted = scores>=threshold
    auto = (scores<real_cut)|predicted
    correct = (real&(scores<real_cut))|(ai&predicted)
    real_rates = [float(predicted[real&(groups==g)].mean()) for g in set(groups[real])]
    ai_rates = [float(predicted[ai&(groups==g)].mean()) for g in set(groups[ai])]
    result = {'auc':float(roc_auc_score(labels,scores)),
              'balanced_accuracy':float(((~predicted[real]).mean()+predicted[ai].mean())/2),
              'real_false_ai':float(predicted[real].mean()),'worst_real_source':max(real_rates),
              'ai_recall':float(predicted[ai].mean()),'worst_ai_source':min(ai_rates),
              'automatic_coverage':float(auto.mean()),'covered_accuracy':float(correct.sum()/max(1,auto.sum())),
              'uncertainty':float((~auto).mean())}
    result['passed'] = (result['auc']>=.90 and result['balanced_accuracy']>=.85
        and result['real_false_ai']<=.10+1e-12 and result['worst_real_source']<=.20+1e-12
        and result['ai_recall']>=.80 and result['worst_ai_source']>=.60
        and result['automatic_coverage']>=.80 and result['covered_accuracy']>=.95
        and result['uncertainty']<=.20+1e-12)
    return result


def choose_real_cut(labels,scores,conditions,threshold):
    validate_scores(labels,scores,conditions)
    candidates = np.unique(np.r_[0.0,scores[scores<threshold],threshold])
    coverage = np.ones(len(candidates)); valid = np.ones(len(candidates),dtype=bool)
    for condition in sorted(set(conditions)):
        mask = conditions==condition
        s,y = scores[mask],labels[mask]
        real_count = np.searchsorted(np.sort(s[y==0]),candidates,side='left')
        false_real = np.searchsorted(np.sort(s[y==1]),candidates,side='left')
        predicted = s>=threshold
        automatic = real_count+false_real+predicted.sum()
        accuracy = (real_count+((y==1)&predicted).sum())/np.maximum(automatic,1)
        valid &= accuracy>=.95
        coverage = np.minimum(coverage,automatic/len(s))
    if not valid.any():
        return 0.0
    objective = np.where(valid,coverage,-1)
    return float(candidates[np.argmax(objective)])


def fit():
    if RESULT.exists() or RESULT_EVIDENCE.exists():
        raise FileExistsError('E51 CAL result already frozen')
    _, admission_sha = admission()
    feature_report = json.loads(FEATURE_EVIDENCE.read_text())
    if feature_report['admission_sha256']!=admission_sha or dino._sha256_file(FEATURES)!=feature_report['feature_sha256']:
        raise ValueError('E51 fitting feature binding changed')
    with np.load(FEATURES,allow_pickle=False) as saved:
        data = {k:saved[k] for k in saved.files}
    train,cal = data['roles']=='TRAIN',data['roles']=='CAL'
    if set(data['parents'][train]) & set(data['parents'][cal]):
        raise ValueError('CAL parent entered fitting')
    weights = parent_source_weights(data['labels'][train],data['sources'][train],data['parents'][train])
    results = {}
    ROOT.joinpath('training').mkdir(parents=True,exist_ok=True)
    for candidate in ('A','B'):
        checkpoint = ROOT / f'training/candidate_{candidate}.json'
        artifact_path = ROOT / f'training/e51_{candidate}.joblib'
        score_path = ROOT / f'training/cal_scores_{candidate}.npz'
        if checkpoint.exists():
            prior = json.loads(checkpoint.read_text())
            if (prior['feature_sha256']!=feature_report['feature_sha256']
                    or prior['artifact_sha256']!=dino._sha256_file(artifact_path)
                    or prior['score_sha256']!=dino._sha256_file(score_path)):
                raise ValueError('candidate checkpoint changed')
            results[candidate] = prior
            continue
        if artifact_path.exists() or score_path.exists():
            raise ValueError('incomplete candidate freeze requires explicit recovery')
        x = candidate_matrix(data['dino'],data['residual'],candidate)
        seed_scores, head = [], None
        for seed in SEEDS:
            fitted = make_pipeline(StandardScaler(),LogisticRegression(C=.01,max_iter=1000,solver='lbfgs',random_state=seed))
            with warnings.catch_warnings():
                warnings.simplefilter('error',ConvergenceWarning)
                fitted.fit(x[train],data['labels'][train],standardscaler__sample_weight=weights,
                           logisticregression__sample_weight=weights)
            if list(fitted.classes_) != [0,1]:
                raise ValueError('E51 positive-label mapping changed')
            seed_scores.append(fitted.predict_proba(x[cal])[:,1])
            if seed==42: head = fitted
            print(json.dumps({'phase':'fit','candidate':candidate,'seed':seed}),flush=True)
        scores = seed_scores[0]
        labels,groups,conditions = data['labels'][cal],data['groups'][cal],data['conditions'][cal]
        threshold = real_safe_threshold(labels,scores,groups,conditions)
        real_cut = choose_real_cut(labels,scores,conditions,threshold)
        by_condition = {c:metrics(labels[conditions==c],scores[conditions==c],groups[conditions==c],threshold,real_cut)
                        for c in sorted(set(conditions))}
        artifact = {'head':head,'candidate':candidate,'positive_label':1,'threshold':threshold,'real_cut':real_cut,
                    'feature_binding':feature_report['feature_binding'],'status':'research_CAL_only_not_served',
                    'feature_contract':'E42 global+texture DINO-S; B additionally fixed E51 residual32'}
        temporary = artifact_path.with_suffix('.joblib.part')
        joblib.dump(artifact,temporary)
        temporary.replace(artifact_path)
        dino._save_npz(score_path,{'ids':data['ids'][cal],
                       'scores':np.stack(seed_scores),'labels':labels,'groups':groups,'conditions':conditions})
        results[candidate] = {'conditions':by_condition,'threshold':threshold,'real_cut':real_cut,
                             'passed':all(m['passed'] for m in by_condition.values()),
                             'seed_max_score_difference':float(np.max(np.abs(np.stack(seed_scores)-scores))),
                             'artifact_sha256':dino._sha256_file(artifact_path),
                             'feature_sha256':feature_report['feature_sha256'],
                             'score_sha256':dino._sha256_file(score_path)}
        _write_atomic(checkpoint,results[candidate])
    selected = next((name for name in ('A','B') if results[name]['passed']),None)
    report = {'schema_version':1,'state':'E51_CAL_complete','admission_sha256':admission_sha,
              'feature_sha256':feature_report['feature_sha256'],'candidates':results,'selected':selected,
              'seed_note':'Deterministic lbfgs repetitions, not independent data trials.',
              'development_scores_created':0,'old_test_scores_created':0,'serving_changed':False}
    _write_atomic(RESULT,report);_write_atomic(RESULT_EVIDENCE,report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('extract','fit'))
    args = parser.parse_args()
    print(json.dumps(extract() if args.command=='extract' else fit(),indent=2))
