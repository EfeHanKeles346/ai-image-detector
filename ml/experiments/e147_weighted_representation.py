"""Fixed FIT-weighted normalization/PCA control on the consumed E131 source folds."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import socket
import time
import numpy as np
from threadpoolctl import threadpool_limits
from experiments.e65_acquisition import digest, read, write_once
from experiments.e71_features import save_npz
from experiments.e72_acquisition import resource_check
from experiments import e131_source_holdout as previous
from experiments import e138_source_risk as paired
from pixelproof import holdout_linear, source_holdout, weighted_holdout
from pixelproof.project_paths import DATA_ROOT, ML_ROOT

ROOT = DATA_ROOT / 'e147'
EVIDENCE = ML_ROOT.parent / 'evidence'
CONTRACT = ROOT / 'contract.json'
BRANCHES = previous.BRANCHES
COMMON = ('dino', 'clip', 'dear')


def fit_project(values, fit_mask, width, weights):
    """Held-out rows can be projected but never enter fitted statistics."""
    fit_mask = np.asarray(fit_mask)
    if fit_mask.dtype != bool or fit_mask.shape != (len(values),) or not (~fit_mask).any():
        raise ValueError('Explicit aligned FIT and held-out masks required')
    mapping = weighted_holdout.fit_map(values[fit_mask], width, weights)
    return mapping, holdout_linear.project(values, mapping)


def map_fields(saved, prefix=''):
    return {k: saved[prefix + k] for k in ('center', 'scale', 'mean', 'components', 'latent_scale')}


def check_replay(expected, actual):
    if expected.shape != actual.shape or not np.isfinite(expected).all() or not np.isfinite(actual).all():
        raise ValueError('Complete finite replay required')
    error = float(np.abs(expected - actual).max())
    if error > 1e-10 or not np.array_equal(expected >= .5, actual >= .5):
        raise ValueError('Saved map/head scores do not replay')
    return error


def freeze():
    prior = previous.validate()
    report = read(previous.ROOT / 'report.json')
    locked = read(previous.ROOT / 'locked_scores.json')
    if digest(previous.ROOT / 'report.json') != digest(EVIDENCE / 'e131_source_holdout.json') or \
            digest(previous.ROOT / 'locked_scores.json') != report['locked_scores_sha256'] or \
            digest(previous.ROOT / 'scores.npz') != locked['scores_sha256']:
        raise ValueError('Complete bound E131 comparator required')
    files = [Path(__file__), Path(weighted_holdout.__file__), Path(paired.__file__),
             Path(previous.__file__), Path(source_holdout.__file__), Path(holdout_linear.__file__),
             previous.CONTRACT, previous.ROOT / 'report.json', previous.ROOT / 'locked_scores.json',
             previous.ROOT / 'scores.npz', EVIDENCE / 'e131_source_holdout.json']
    for fold in range(3):
        files.append(previous.ROOT / f'fold{fold}_shared.npz')
        for branch in BRANCHES:
            path = previous.ROOT / f'fold{fold}_{branch}.npz'
            if digest(path) != locked['fits'][f'{fold}_{branch}']['artifact_sha256']:
                raise ValueError('Frozen comparator artifact differs')
            files.append(path)
    c = {'state': 'E147_weighted_representation_registered',
         'inputs': prior['inputs'] | {str(p): digest(p) for p in files},
         'parents': prior['parents'], 'conditions': prior['conditions'], 'branches': BRANCHES,
         'folds': 3, 'max_seconds': 7200,
         'change': 'FIT-only weighted normalization and PCA using exactly the classifier class/component/parent/view weights. Weighted population mean/variance; covariance denominator 1-sum(w*w) recovers n-1 sample convention for uniform weights, not a claim of independent views.',
         'fixed': 'E131 source folds, all four conditions, DINO/CLIP/DEAR64 PCs each and branch128 PCs, whitening floors, randomized SVD seed131/power3/oversamples10. Same zero-start weighted BCE+.005||w||^2, unpenalized intercept, L-BFGS500/ftol1e-12/gtol1e-7, gradient<=1e-5. Six heads, no rank/weight/cut search. Fixed0.5 diagnostic cut, not E92 serving cuts.',
         'replay': 'Replay old saved maps/heads against every E131 held-out score and replay new saved shared/branch maps and heads; max error<=1e-10 and identical0.5 decisions.',
         'evaluation': 'Lock all12525x4 scores for both branches before any candidate metrics. Report each condition/fold/component, pooled different-head metrics, worst groups and paired new/rescued errors. Require zero new AI misses and zero new REAL alerts in every condition for individual nonregression. No automatic promotion even on an internal pass.',
         'downloads': 0, 'new_pixels_read': 0, 'external_DEV_or_gallery_reads': 0,
         'promotion_allowed': False, 'generator_family_holdout_supported': False,
         'limits': prior['limits'] + ' Adaptive control chosen after E146 and the representation audit; all folds are consumed development data. This is a weighted representation bundle, not separate attribution to scaling versus PCA. Unknown corpus ancestry remains; this is not E92 accuracy or independent unseen-family evidence.'}
    ROOT.mkdir(exist_ok=True)
    write_once(CONTRACT, c)
    write_once(EVIDENCE / 'e147_weighted_representation_contract.json',
               {k: v for k, v in c.items() if k != 'inputs'} | {'contract_sha256': digest(CONTRACT)})
    return {'parents': c['parents'], 'fits': 6, 'contract_sha256': digest(CONTRACT)}


def validate():
    c = read(CONTRACT)
    if digest(CONTRACT) != read(EVIDENCE / 'e147_weighted_representation_contract.json')['contract_sha256']:
        raise ValueError('Weighted representation contract differs')
    for path, sha in c['inputs'].items():
        if digest(path) != sha:
            raise ValueError('Frozen weighted representation input differs')
    return c


def fit():
    c = validate()
    start = time.monotonic()
    deadline = start + c['max_seconds']
    resource_check(deadline)
    binding = digest(CONTRACT)
    old_binding = digest(previous.CONTRACT)
    write_once(ROOT / 'started.json', {'contract_sha256': binding})
    prior = read(previous.CONTRACT)
    rows = prior['rows']
    features = previous.load_features(rows)
    labels = np.asarray([r['label'] for r in rows])
    folds = np.asarray([prior['outer_fold'][r['parent_id']] for r in rows])
    groups = np.asarray([prior['components'][r['parent_id']] for r in rows])
    scores = {b: np.full((len(rows), 4), np.nan) for b in BRANCHES}
    with np.load(previous.ROOT / 'scores.npz', allow_pickle=False) as data:
        if str(data['contract_sha256']) != old_binding or list(data['parents']) != [r['parent_id'] for r in rows] or \
                list(data['conditions']) != c['conditions'] or not np.array_equal(data['outer_fold'], folds) or \
                str(data['global_role']) != 'TRAIN' or str(data['usage']) != 'INTERNAL_HELD_OUT':
            raise ValueError('Comparator score identity differs')
        baseline = {b: data[b].copy() for b in BRANCHES}
    fits = {}
    with threadpool_limits(limits=2):
        for fold in range(3):
            parent_fit = folds != fold
            view_fit = np.repeat(parent_fit, 4)
            fit_rows = [r for r, take in zip(rows, parent_fit, strict=True) if take]
            weights = source_holdout.balanced_weights(fit_rows,
                {r['parent_id']: prior['components'][r['parent_id']] for r in fit_rows}, 4)
            with np.load(previous.ROOT / f'fold{fold}_shared.npz', allow_pickle=False) as saved:
                if str(saved['contract_sha256']) != old_binding:
                    raise ValueError('Comparator shared binding differs')
                old_common = np.column_stack([holdout_linear.project(features[k][~view_fit], map_fields(saved, k+'_')) for k in COMMON])
            old_errors = {}
            for branch in BRANCHES:
                with np.load(previous.ROOT / f'fold{fold}_{branch}.npz', allow_pickle=False) as saved:
                    if str(saved['contract_sha256']) != old_binding:
                        raise ValueError('Comparator branch binding differs')
                    x = np.column_stack([old_common, holdout_linear.project(features[branch][~view_fit], map_fields(saved))])
                    old_errors[branch] = check_replay(baseline[branch][~parent_fit], holdout_linear.predict(x, saved['parameters']).reshape(-1, 4))
            del x, old_common
            shared, common = {}, []
            for name in COMMON:
                resource_check(deadline)
                mapping, projected = fit_project(features[name], view_fit, 64, weights)
                shared.update({name+'_'+k: v for k, v in mapping.items()})
                common.append(projected)
                print(json.dumps({'E147_fold': fold, 'map': name, 'seconds': round(time.monotonic()-start)}), flush=True)
            common = np.column_stack(common)
            shared_path = ROOT / f'fold{fold}_shared.npz'
            save_npz(shared_path, **shared, contract_sha256=binding)
            with np.load(shared_path, allow_pickle=False) as saved:
                saved_common = np.column_stack([holdout_linear.project(features[k][~view_fit], map_fields(saved, k+'_')) for k in COMMON])
            for branch in BRANCHES:
                resource_check(deadline)
                mapping, projected = fit_project(features[branch], view_fit, 128, weights)
                x = np.column_stack([common, projected])
                parameters, solver = holdout_linear.fit_head(x[view_fit], np.repeat(labels[parent_fit], 4), weights, lambda: resource_check(deadline))
                artifact = ROOT / f'fold{fold}_{branch}.npz'
                save_npz(artifact, **mapping, parameters=parameters, contract_sha256=binding)
                prediction = holdout_linear.predict(x[~view_fit], parameters).reshape(-1, 4)
                with np.load(artifact, allow_pickle=False) as saved:
                    replay_x = np.column_stack([saved_common, holdout_linear.project(features[branch][~view_fit], map_fields(saved))])
                    replay = holdout_linear.predict(replay_x, saved['parameters']).reshape(-1, 4)
                error = check_replay(prediction, replay)
                scores[branch][~parent_fit] = prediction
                fits[f'{fold}_{branch}'] = {'solver': solver, 'old_score_max_error': old_errors[branch],
                    'saved_score_max_error': error, 'artifact_sha256': digest(artifact), 'shared_sha256': digest(shared_path),
                    'FIT_parents': int(parent_fit.sum()), 'held_parents': int((~parent_fit).sum()),
                    'FIT_REAL_representation_mass': float(weights[np.repeat(labels[parent_fit], 4) == 0].sum())}
                print(json.dumps({'E147_fold': fold, 'branch': branch, 'seconds': round(time.monotonic()-start)}), flush=True)
                del x, replay_x, projected
    if any(not np.isfinite(s).all() for s in scores.values()):
        raise ValueError('Incomplete candidate scores')
    save_npz(ROOT / 'scores.npz', **scores, parents=np.asarray([r['parent_id'] for r in rows]), outer_fold=folds,
             conditions=c['conditions'], global_role='TRAIN', usage='INTERNAL_HELD_OUT', contract_sha256=binding)
    write_once(ROOT / 'locked_scores.json', {'scores_sha256': digest(ROOT / 'scores.npz'), 'contract_sha256': binding, 'fits': fits})
    reports = {}
    for branch in BRANCHES:
        reports[branch] = {}
        for j, condition in enumerate(c['conditions']):
            components = []
            for group in sorted(set(groups.tolist())):
                take = groups == group
                components.append(previous.metrics(labels[take], scores[branch][take, j]) | {
                    'sources': sorted({r['source'] for r, t in zip(rows, take, strict=True) if t}),
                    'baseline': previous.metrics(labels[take], baseline[branch][take, j]),
                    'paired': paired.transitions(labels[take], baseline[branch][take, j], scores[branch][take, j])})
            reports[branch][condition] = {
                'pooled_different_fold_models': previous.metrics(labels, scores[branch][:, j]),
                'baseline': previous.metrics(labels, baseline[branch][:, j]),
                'folds': [previous.metrics(labels[folds == f], scores[branch][folds == f, j]) for f in range(3)],
                'components': components,
                'REAL_component_worst_FPR': max(r['REAL_FPR'] for r in components if r['REAL_FPR'] is not None),
                'AI_component_worst_recall': min(r['AI_recall'] for r in components if r['AI_recall'] is not None),
                'paired': paired.transitions(labels, baseline[branch][:, j], scores[branch][:, j])}
    nonregression = {b: all(r['paired']['zero_new_AI_misses'] and r['paired']['zero_new_REAL_false_alerts'] for r in reports[b].values()) for b in BRANCHES}
    report = {'state': 'E147_weighted_representation_complete', 'contract_sha256': binding,
        'reports': reports, 'passes_internal_individual_nonregression': nonregression,
        'seconds': time.monotonic()-start, 'locked_scores_sha256': digest(ROOT / 'locked_scores.json'),
        'parents': len(rows), 'downloads': 0, 'new_pixels_read': 0, 'promotion_allowed': False, 'limits': c['limits']}
    write_once(ROOT / 'report.json', report)
    write_once(EVIDENCE / 'e147_weighted_representation.json', report)
    return {k: v for k, v in report.items() if k != 'reports'}


if __name__ == '__main__':
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
    def denied(*args, **kwargs):
        raise RuntimeError('E147 is offline')
    socket.socket.connect = denied
    socket.socket.connect_ex = denied
    socket.create_connection = denied
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['freeze', 'fit'])
    ROOT.mkdir(exist_ok=True)
    with (ROOT / 'execution.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        print(json.dumps({'freeze': freeze, 'fit': fit}[parser.parse_args().stage](), indent=2))
