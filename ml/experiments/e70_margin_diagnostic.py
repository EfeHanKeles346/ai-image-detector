"""Describe fixed E70 margins on TRAIN and its already-consumed DEV scores."""
import argparse
import json
from pathlib import Path
import time
import joblib
import numpy as np
from scipy.special import logit
from threadpoolctl import threadpool_limits
from experiments.e65_acquisition import digest, read, write_once
from experiments import e70_fit as prior, e70_model as model
from experiments.e71_features import resource_check

CONTRACT = prior.ROOT/'margin_diagnostic_contract.json'
REPORT = prior.ROOT/'margin_diagnostic.json'
QUANTILES = [0, .01, .05, .25, .5, .75, .95, .99, 1]


def margins(before, after):
    before = np.asarray(before, dtype=np.float64); after = np.asarray(after, dtype=np.float64)
    if before.ndim != 1 or before.shape != after.shape or not np.isfinite(before).all() or not np.isfinite(after).all():
        raise ValueError('finite aligned margins required')
    caught = before >= 0; old = before[caught]; new = after[caught]
    return {'views': len(before), 'previously_caught': int(caught.sum()),
            'new_misses': int(np.sum(new < 0)),
            'near_boundary_after': {str(t): int(np.sum((new >= 0) & (new <= t))) for t in [1e-6, .01, .1, .5]},
            'old_quantiles': np.quantile(old, QUANTILES).tolist() if len(old) else None,
            'new_quantiles': np.quantile(new, QUANTILES).tolist() if len(new) else None,
            'mean_shift': float((new-old).mean()) if len(old) else None}


def freeze():
    c = prior.validate(); result = read(prior.REPORT)
    if digest(prior.CANDIDATE) != result['candidate_sha256'] or digest(prior.REPORT) != digest(prior.EVIDENCE/'e70_fit.json'):
        raise ValueError('E70 candidate/fit identity changed')
    exposure = read(prior.ROOT/'dev_report.json')
    if exposure['passes_limited_dev_screen'] or exposure['scores_sha256'] != digest(prior.ROOT/'dev_scores.json') or \
            digest(prior.ROOT/'dev_report.json') != digest(prior.EVIDENCE/'e70_development.json'):
        raise ValueError('locked failed consumed DEV required')
    paths = [Path(__file__), prior.CONTRACT, prior.REPORT, prior.CANDIDATE, prior.ROOT/'dev_report.json',
             prior.ROOT/'dev_scores.json', Path(__file__).with_name('e71_features.py')]
    contract = {'state': 'E70_fixed_margin_diagnostic_registered', 'inputs': {str(p): digest(p) for p in paths},
        'fit_contract_sha256': digest(prior.CONTRACT), 'quantiles': QUANTILES,
        'near_boundary_logit_distances': [1e-6, .01, .1, .5], 'max_seconds': 600,
        'scope': 'Describe all previously caught AI margins by TRAIN condition/source and consumed E66 DEV '
                 'condition/source. Report every existing DEV new AI miss score/margin. No new candidate, '
                 'fit, threshold selection, image extraction, E49 access or E71 score access.',
        'limitations': 'Descriptive post-fit analysis; TRAIN margins cannot certify unseen AI retention. '
                      'Dependent DEV observations and small counts preclude causal/generalization claims.'}
    write_once(CONTRACT, contract)
    write_once(prior.EVIDENCE/'e70_margin_diagnostic_contract.json', contract|{'contract_sha256': digest(CONTRACT)})
    return {'state': contract['state']}


def run():
    c = read(CONTRACT); fit = prior.validate()
    if digest(CONTRACT) != read(prior.EVIDENCE/'e70_margin_diagnostic_contract.json')['contract_sha256']:
        raise ValueError('margin protocol changed')
    for p, sha in c['inputs'].items():
        if digest(p) != sha: raise ValueError('bound diagnostic input changed')
    if REPORT.exists(): raise FileExistsError('margin diagnostic already complete')
    deadline = time.monotonic()+c['max_seconds']; resource_check(deadline)
    rows = read(fit['inputs']['manifest']['path'])['rows']
    if len(rows) != 11630 or any(r['role'].upper() != 'TRAIN' for r in rows): raise ValueError('TRAIN only')
    with np.load(fit['inputs']['features']['path'], allow_pickle=False) as a: original = a['features'].reshape(-1,3072)
    with np.load(fit['inputs']['blur']['path'], allow_pickle=False) as a: blurred = a['features'].reshape(-1,3072)
    with np.load(prior.CANDIDATE, allow_pickle=False) as a: arrays = {k:a[k] for k in a.files}
    head = joblib.load(fit['inputs']['reference']['path'])['head']
    with threadpool_limits(limits=2):
        before = head.decision_function(original).astype(np.float64)-logit(model.AI_CUT)
        after = before+model.project(head,original,blurred,arrays)@arrays['weights']
    resource_check(deadline)
    labels = np.repeat([r['label'] for r in rows],3); sources = np.repeat([r['source'] for r in rows],3)
    conditions = np.tile(fit['conditions'],len(rows)); groups = {}
    for condition in fit['conditions']:
        use = (labels==1)&(conditions==condition)
        groups[condition] = {'pooled': margins(before[use],after[use]), 'sources': {}}
        for source in sorted(set(sources[use])):
            selected = use&(sources==source)
            groups[condition]['sources'][source] = margins(before[selected],after[selected])
    dev = read(prior.ROOT/'dev_scores.json')['rows']; dev_groups = {}; new_misses = []
    for condition in sorted({r['condition'] for r in dev}):
        selected = [r for r in dev if r['condition']==condition and r['label']==1]
        def describe(items):
            old = logit(np.clip([r['reference_score'] for r in items],1e-15,1-1e-15))-logit(model.AI_CUT)
            new = logit(np.clip([r['score'] for r in items],1e-15,1-1e-15))-logit(model.AI_CUT)
            return margins(old,new)
        dev_groups[condition] = {'pooled': describe(selected), 'sources': {s:describe([r for r in selected if r['source']==s]) for s in sorted({r['source'] for r in selected})}}
        new_misses.extend({k:r[k] for k in ['parent_id','condition','source','reference_score','score']}
                          for r in selected if r['reference_score']>=model.AI_CUT and r['score']<model.AI_CUT)
    result = {'state': 'E70_fixed_margin_diagnostic_complete', 'contract_sha256': digest(CONTRACT),
              'train': groups, 'consumed_dev': dev_groups, 'existing_dev_new_misses': new_misses,
              'new_fits': 0, 'new_image_inference': 0, 'e49_rows_read': 0, 'e71_scores_read': 0,
              'limitations': c['limitations']}
    write_once(REPORT,result); write_once(prior.EVIDENCE/'e70_margin_diagnostic.json',result)
    return {'train': {k:v['pooled'] for k,v in groups.items()}, 'consumed_dev': {k:v['pooled'] for k,v in dev_groups.items()}}


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=('freeze','run'))
    print(json.dumps({'freeze':freeze,'run':run}[parser.parse_args().stage](),indent=2))
