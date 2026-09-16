"""One fixed source-pixel residual addition to the consumed E131 source folds."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import socket
import time
import numpy as np
import scipy
import sklearn
from threadpoolctl import threadpool_limits
from experiments.e65_acquisition import digest, read, write_once
from experiments.e71_features import save_npz
from experiments.e72_acquisition import resource_check
from experiments import e131_source_holdout as previous
from experiments import e138_source_risk as paired
from pixelproof import holdout_linear, source_holdout
from experiments import e150b_source_pixel_features as pixels
from experiments.e147_weighted_representation import map_fields, check_replay
from pixelproof.project_paths import DATA_ROOT, ML_ROOT

ROOT = DATA_ROOT / 'e151'
EVIDENCE = ML_ROOT.parent / 'evidence'
CONTRACT = ROOT / 'contract.json'
BRANCHES = previous.BRANCHES
COMMON = ('dino', 'clip', 'dear')


def versions():
    return pixels.previous.versions() | {'scipy': scipy.__version__, 'sklearn': sklearn.__version__}


def fit_project(values, fit_mask, width):
    """Held-out rows can be projected but never enter fitted statistics."""
    fit_mask = np.asarray(fit_mask)
    if fit_mask.dtype != bool or fit_mask.shape != (len(values),) or not (~fit_mask).any():
        raise ValueError('Explicit aligned FIT and held-out masks required')
    mapping = holdout_linear.fit_map(values[fit_mask], width)
    return mapping, holdout_linear.project(values, mapping)


def residual_coordinates(saved, rows, binding, conditions):
    if str(saved['binding']) != binding or str(saved['role']) != 'TRAIN' or \
            list(saved['parents']) != [r['parent_id'] for r in rows] or list(saved['conditions']) != list(conditions):
        raise ValueError('Source-pixel feature role/parent/condition/binding differs')
    value = saved['features']
    if value.shape != (len(rows), 4, 2, 300) or value.dtype != np.float32 or \
            not np.isfinite(value).all() or (value < 0).any() or \
            not np.allclose(value.reshape(len(rows), 4, 2, 12, 25).sum(axis=-1), 1, rtol=0, atol=1e-6):
        raise ValueError('Complete finite source/control histograms required')
    return np.concatenate([value[:, :, 0], value[:, :, 1] - value[:, :, 0]], axis=-1).reshape(len(rows)*4, 600)


def freeze():
    prior = previous.validate()
    pixel_contract = pixels.validate()
    pixel_report = read(pixels.ROOT / 'report.json')
    public_pixel_report = read(EVIDENCE / 'e150b_source_pixel_features.json')
    if digest(pixels.ROOT / 'report.json') != public_pixel_report['report_sha256'] or \
            digest(pixels.ROOT / 'features.npz') != pixel_report['features_sha256'] or \
            pixel_report['contract_sha256'] != digest(pixels.FULL) or \
            digest(pixels.FULL) != read(EVIDENCE / 'e150b_source_pixel_full_contract.json')['contract_sha256'] or \
            not pixel_report['exact_probe_and_serialization_replay'] or pixel_report['parents'] != prior['parents']:
        raise ValueError('Complete bound E150B full extraction required')
    report = read(previous.ROOT / 'report.json')
    locked = read(previous.ROOT / 'locked_scores.json')
    if digest(previous.ROOT / 'report.json') != digest(EVIDENCE / 'e131_source_holdout.json') or \
            digest(previous.ROOT / 'locked_scores.json') != report['locked_scores_sha256'] or \
            digest(previous.ROOT / 'scores.npz') != locked['scores_sha256']:
        raise ValueError('Complete bound E131 comparator required')
    files = [Path(__file__), Path(paired.__file__), ML_ROOT / 'experiments/e147_weighted_representation.py',
             Path(pixels.__file__), pixels.CONTRACT, pixels.FULL, pixels.ROOT / 'report.json',
             pixels.ROOT / 'features.npz', EVIDENCE / 'e150b_source_pixel_features.json',
             EVIDENCE / 'e150b_source_pixel_full_contract.json', DATA_ROOT / 'e148/private_records.json',
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
    c = {'state': 'E151_source_pixel_comparison_registered',
         'inputs': prior['inputs'] | pixel_contract['inputs'] | {str(p): digest(p) for p in files},
         'parents': prior['parents'], 'conditions': prior['conditions'], 'branches': BRANCHES,
         'folds': 3, 'max_seconds': 7200,
         'change': 'Append one FIT-only unweighted normalized/whitened PCA64 of[source300,control-minus-source300] from the frozen E150B source-pixel features to the frozen E131320-coordinate maps. No refit of semantic maps; no metadata enters features.',
         'fixed': 'E131 source folds/all four conditions; old DINO/CLIP/DEAR64 PCs each and branch128 PCs plus new64 residual PCs. Existing floors, seed131/power3. Same zero-start weighted BCE+.005||w||^2, unpenalized intercept, L-BFGS500/ftol1e-12/gtol1e-7, gradient<=1e-5. Three new maps/six heads; no rank/weight/cut search. Fixed0.5 diagnostic cut, not E92 serving cuts. CPU2; AC/20GiB reserve and7200seconds.',
         'replay': 'Replay old saved maps/heads against every E131 held-out score and replay new saved residual maps and heads combined with immutable E131 maps; max error<=1e-10 and identical0.5 decisions.',
         'evaluation': 'Lock all12525x4 scores for both branches before any candidate metrics. Report each condition/fold/component and recorded processing-history group, pooled different-head metrics, worst groups and paired new/rescued errors. Require zero new AI misses and zero new REAL alerts in every condition for individual nonregression. No automatic promotion even on an internal pass.',
         'downloads': 0, 'new_pixels_read': 0, 'external_DEV_or_gallery_reads': 0,
         'versions': versions(),
         'promotion_allowed': False, 'generator_family_holdout_supported': False,
         'limits': prior['limits'] + ' Adaptive mechanism chosen after E147/E148. All folds are consumed development data. Source-pixel plus paired down/up-control is tested as one bundle, not separate attribution. Processing-history strata are observational and source/class confounded. Unknown corpus ancestry remains; this is not E92 accuracy or independent unseen-family evidence.'}
    ROOT.mkdir(exist_ok=True)
    write_once(CONTRACT, c)
    write_once(EVIDENCE / 'e151_source_pixel_comparison_contract.json',
               {k: v for k, v in c.items() if k != 'inputs'} | {'contract_sha256': digest(CONTRACT)})
    return {'parents': c['parents'], 'fits': 6, 'contract_sha256': digest(CONTRACT)}


def validate():
    c = read(CONTRACT)
    if digest(CONTRACT) != read(EVIDENCE / 'e151_source_pixel_comparison_contract.json')['contract_sha256']:
        raise ValueError('Source-pixel comparison contract differs')
    if c['versions'] != versions():
        raise ValueError('Frozen runtime differs')
    for path, sha in c['inputs'].items():
        if digest(path) != sha:
            raise ValueError('Frozen comparison input differs')
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
    with np.load(pixels.ROOT / 'features.npz', allow_pickle=False) as saved:
        residual = residual_coordinates(saved, rows, digest(pixels.FULL), c['conditions'])
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
            resource_check(deadline)
            with np.load(previous.ROOT / f'fold{fold}_shared.npz', allow_pickle=False) as saved:
                if str(saved['contract_sha256']) != old_binding:
                    raise ValueError('Comparator shared binding differs')
                common = np.column_stack([holdout_linear.project(features[k], map_fields(saved, k+'_')) for k in COMMON])
            mapping, projected = fit_project(residual, view_fit, 64)
            map_path = ROOT / f'fold{fold}_residual.npz'
            save_npz(map_path, **mapping, contract_sha256=binding)
            with np.load(map_path, allow_pickle=False) as saved:
                if str(saved['contract_sha256']) != binding:
                    raise ValueError('Saved residual map binding differs')
                replay_residual = holdout_linear.project(residual[~view_fit], map_fields(saved))
            for branch in BRANCHES:
                resource_check(deadline)
                with np.load(previous.ROOT / f'fold{fold}_{branch}.npz', allow_pickle=False) as saved:
                    if str(saved['contract_sha256']) != old_binding:
                        raise ValueError('Comparator branch binding differs')
                    branch_map = map_fields(saved)
                    base = np.column_stack([common, holdout_linear.project(features[branch], branch_map)])
                    old_error = check_replay(baseline[branch][~parent_fit], holdout_linear.predict(base[~view_fit], saved['parameters']).reshape(-1, 4))
                x = np.column_stack([base, projected])
                parameters, solver = holdout_linear.fit_head(x[view_fit], np.repeat(labels[parent_fit], 4), weights, lambda: resource_check(deadline))
                artifact = ROOT / f'fold{fold}_{branch}.npz'
                save_npz(artifact, parameters=parameters, contract_sha256=binding, residual_map_sha256=digest(map_path),
                         base_shared_sha256=digest(previous.ROOT / f'fold{fold}_shared.npz'),
                         base_branch_sha256=digest(previous.ROOT / f'fold{fold}_{branch}.npz'))
                prediction = holdout_linear.predict(x[~view_fit], parameters).reshape(-1, 4)
                with np.load(artifact, allow_pickle=False) as saved:
                    if str(saved['contract_sha256']) != binding or str(saved['residual_map_sha256']) != digest(map_path):
                        raise ValueError('Saved head binding differs')
                    with np.load(previous.ROOT / f'fold{fold}_shared.npz', allow_pickle=False) as shared:
                        replay_common = np.column_stack([holdout_linear.project(features[k][~view_fit], map_fields(shared, k+'_')) for k in COMMON])
                    with np.load(previous.ROOT / f'fold{fold}_{branch}.npz', allow_pickle=False) as old:
                        replay_x = np.column_stack([replay_common, holdout_linear.project(features[branch][~view_fit], map_fields(old)), replay_residual])
                    replay = holdout_linear.predict(replay_x, saved['parameters']).reshape(-1, 4)
                error = check_replay(prediction, replay)
                scores[branch][~parent_fit] = prediction
                fits[f'{fold}_{branch}'] = {'solver': solver, 'old_score_max_error': old_error,
                    'saved_score_max_error': error, 'artifact_sha256': digest(artifact), 'residual_map_sha256': digest(map_path),
                    'FIT_parents': int(parent_fit.sum()), 'held_parents': int((~parent_fit).sum()),
                    'FIT_REAL_loss_mass': float(weights[np.repeat(labels[parent_fit], 4) == 0].sum())}
                print(json.dumps({'E151_fold': fold, 'branch': branch, 'seconds': round(time.monotonic()-start)}), flush=True)
                del x, replay_x, replay_common, base
            del common, projected, replay_residual
    if any(not np.isfinite(s).all() for s in scores.values()):
        raise ValueError('Incomplete candidate scores')
    save_npz(ROOT / 'scores.npz', **scores, parents=np.asarray([r['parent_id'] for r in rows]), outer_fold=folds,
             conditions=c['conditions'], global_role='TRAIN', usage='INTERNAL_HELD_OUT', contract_sha256=binding)
    write_once(ROOT / 'locked_scores.json', {'scores_sha256': digest(ROOT / 'scores.npz'), 'contract_sha256': binding, 'fits': fits})
    history = {r['parent_id']: r['processing_history'] for r in read(DATA_ROOT / 'e148/private_records.json')['rows']}
    histories = np.asarray([history[r['parent_id']] for r in rows])
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
                'folds': [previous.metrics(labels[folds == f], scores[branch][folds == f, j]) | {
                    'baseline': previous.metrics(labels[folds == f], baseline[branch][folds == f, j]),
                    'paired': paired.transitions(labels[folds == f], baseline[branch][folds == f, j], scores[branch][folds == f, j])} for f in range(3)],
                'processing_history': {h: previous.metrics(labels[histories == h], scores[branch][histories == h, j]) | {
                    'baseline': previous.metrics(labels[histories == h], baseline[branch][histories == h, j]),
                    'paired': paired.transitions(labels[histories == h], baseline[branch][histories == h, j], scores[branch][histories == h, j])} for h in sorted(set(histories.tolist()))},
                'components': components,
                'REAL_component_worst_FPR': max(r['REAL_FPR'] for r in components if r['REAL_FPR'] is not None),
                'AI_component_worst_recall': min(r['AI_recall'] for r in components if r['AI_recall'] is not None),
                'paired': paired.transitions(labels, baseline[branch][:, j], scores[branch][:, j])}
    nonregression = {b: all(r['paired']['zero_new_AI_misses'] and r['paired']['zero_new_REAL_false_alerts'] for r in reports[b].values()) for b in BRANCHES}
    report = {'state': 'E151_source_pixel_comparison_complete', 'contract_sha256': binding,
        'reports': reports, 'passes_internal_individual_nonregression': nonregression,
        'seconds': time.monotonic()-start, 'locked_scores_sha256': digest(ROOT / 'locked_scores.json'),
        'parents': len(rows), 'downloads': 0, 'new_pixels_read': 0, 'promotion_allowed': False, 'limits': c['limits']}
    write_once(ROOT / 'report.json', report)
    write_once(EVIDENCE / 'e151_source_pixel_comparison.json', report)
    return {k: v for k, v in report.items() if k != 'reports'}


if __name__ == '__main__':
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
    def denied(*args, **kwargs):
        raise RuntimeError('E151 is offline')
    socket.socket.connect = denied
    socket.socket.connect_ex = denied
    socket.create_connection = denied
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['freeze', 'fit'])
    ROOT.mkdir(exist_ok=True)
    with (ROOT / 'execution.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        print(json.dumps({'freeze': freeze, 'fit': fit}[parser.parse_args().stage](), indent=2))
