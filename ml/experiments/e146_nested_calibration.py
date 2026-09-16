"""FIT-only fresh heads with source-separated CAL cuts and untouched-per-fit EVAL."""
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
from pixelproof import holdout_linear, nested_calibration, source_holdout
from pixelproof.project_paths import DATA_ROOT, ML_ROOT

ROOT = DATA_ROOT / 'e146'
EVIDENCE = ML_ROOT.parent / 'evidence'
CONTRACT = ROOT / 'contract.json'
FEATURES = [('dino', 64), ('clip', 64), ('dear', 64), ('full_frame', 128)]


def freeze():
    prior = previous.validate()
    split = read(EVIDENCE / 'e145_nested_split.json')
    observed = nested_calibration.split_roster(prior['rows'], prior['components'], prior['outer_fold'])
    if split['prior_contract_sha256'] != digest(previous.CONTRACT) or any(split[k] != observed[k] for k in observed) or \
            split['helper_sha256'] != digest(Path(nested_calibration.__file__)):
        raise ValueError('Nested split evidence differs')
    files = [Path(__file__), previous.CONTRACT, Path(previous.__file__),
             Path(holdout_linear.__file__), Path(nested_calibration.__file__),
             Path(source_holdout.__file__), Path(save_npz.__code__.co_filename),
             Path(resource_check.__code__.co_filename), EVIDENCE / 'e145_nested_split.json']
    c = dict(state='E146_source_separated_calibration_registered',
        inputs=prior['inputs'] | {str(p): digest(p) for p in files},
        assignments=split['assignments'], parents=prior['parents'], conditions=prior['conditions'],
        features=FEATURES, fits=3, max_seconds=3600,
        method='For each existing FIT fold, refit every raw-feature standardizer/PCA and fresh weighted logistic head on that fold only. Same E131 ranks, whitening seed131, L2 .01, optimizer and four views. Three fresh full-frame heads; six ordered FIT/CAL/EVAL assignments. Reuse no learned E131 transform/head. Bound prior raw caches; legacy loaded reference is discarded and never used as an input feature/coefficient.',
        selection='One common CAL-only cutoff across four conditions. Smallest cut meeting REAL FPR<=10% pooled and<=20% each declared REAL component in every condition, by exact tied-score order statistics. Require >=95% AI recall and zero lost .5-caught CAL AI in every condition; otherwise reject the cut without applying it to EVAL. No grid, hyperparameter or outer-score search.',
        evaluation='Lock all three non-FIT score arrays, then all six CAL selections before any EVAL metric. Report each assignment separately (no pooling repeated parents), .5 baseline, CAL feasibility, accepted-cut EVAL component/condition metrics and paired individual errors. Require EVAL budgets, >=95% AI recall, zero new AI misses and zero new REAL alerts in every condition. Failing assignments stay in the report. No full-TRAIN or serving candidate.',
        limits='Adaptive consumed TRAIN development. One AI-bearing component per role and unresolved RR/CF upstream overlap; no unseen-family or independent validation claim. Less FIT data than E131; only same-new-head .5 vs CAL-cut comparisons isolate cutoff changes. Baseline heads are distinct from E92. Empirical CAL budgets are not statistical guarantees for new sources.',
        downloads=0, new_pixels_read=0, protected_reserve_reads=0, promotion_allowed=False)
    write_once(CONTRACT, c)
    write_once(EVIDENCE / 'e146_nested_calibration_contract.json',
               {k: v for k, v in c.items() if k != 'inputs'} | {'contract_sha256': digest(CONTRACT)})
    return dict(contract_sha256=digest(CONTRACT), fits=c['fits'], assignments=len(c['assignments']))


def validate():
    c = read(CONTRACT)
    if digest(CONTRACT) != read(EVIDENCE / 'e146_nested_calibration_contract.json')['contract_sha256']:
        raise ValueError('Nested calibration contract differs')
    for path, sha in c['inputs'].items():
        if digest(path) != sha:
            raise ValueError('Bound nested-calibration input differs')
    return c


def component_metrics(rows, labels, scores, groups, cut, conditions):
    result = []
    for group in sorted(set(groups)):
        selected = groups == group
        sources = sorted({r['source'] for r, take in zip(rows, selected, strict=True) if take})
        y = labels[selected]
        for j, condition in enumerate(conditions):
            p = scores[selected, j] >= cut
            base = scores[selected, j] >= .5
            real, ai = y == 0, y == 1
            result.append(dict(sources=sources, condition=condition, parents=int(selected.sum()),
                REAL_parents=int(real.sum()), AI_parents=int(ai.sum()),
                REAL_FP=int(p[real].sum()), AI_TP=int(p[ai].sum()),
                REAL_FPR=float(p[real].mean()) if real.any() else None,
                AI_recall=float(p[ai].mean()) if ai.any() else None,
                new_AI_misses=int((base & ~p & ai).sum()), rescued_AI=int((~base & p & ai).sum()),
                new_REAL_alerts=int((~base & p & real).sum()), rescued_REAL=int((base & ~p & real).sum())))
    return result


def fit():
    c = validate()
    start = time.monotonic()
    deadline = start + c['max_seconds']
    resource_check(deadline)
    write_once(ROOT / 'started.json', {'contract_sha256': digest(CONTRACT)})
    prior = read(previous.CONTRACT)
    rows = prior['rows']
    features = previous.load_features(rows)
    # The center branch is unused; release it before the new FIT-only PCA work.
    del features['center_control']
    n = len(rows)
    labels = np.asarray([r['label'] for r in rows])
    folds = np.asarray([prior['outer_fold'][r['parent_id']] for r in rows])
    groups = np.asarray([prior['components'][r['parent_id']] for r in rows])
    predictions = np.full((3, n, 4), np.nan, dtype=np.float64)
    fits = []
    with threadpool_limits(limits=2):
        for fold in range(3):
            mask = folds == fold
            view_fit = np.repeat(mask, 4)
            fit_rows = [r for r, take in zip(rows, mask, strict=True) if take]
            weights = source_holdout.balanced_weights(fit_rows,
                {r['parent_id']: prior['components'][r['parent_id']] for r in fit_rows}, 4)
            blocks = []
            transforms = {}
            for name, width in FEATURES:
                resource_check(deadline)
                transform = holdout_linear.fit_map(features[name][view_fit], width)
                blocks.append(holdout_linear.project(features[name], transform))
                transforms.update({name + '_' + k: v for k, v in transform.items()})
                print(json.dumps(dict(fit_fold=fold, PCA=name, seconds=round(time.monotonic() - start))), flush=True)
            x = np.column_stack(blocks)
            parameters, solver = holdout_linear.fit_head(x[view_fit], np.repeat(labels[mask], 4),
                weights, lambda: resource_check(deadline))
            path = ROOT / f'fit{fold}.npz'
            save_npz(path, **transforms, parameters=parameters, fit_fold=fold, contract_sha256=digest(CONTRACT))
            predicted = holdout_linear.predict(x[~view_fit], parameters).reshape(-1, 4)
            with np.load(path, allow_pickle=False) as saved:
                replay_x = np.column_stack([holdout_linear.project(features[name][~view_fit],
                    {k: saved[name + '_' + k] for k in ('center', 'scale', 'mean', 'components', 'latent_scale')}) for name, _ in FEATURES])
                replay = holdout_linear.predict(replay_x, saved['parameters']).reshape(-1, 4)
            error = float(np.abs(predicted - replay).max())
            if error > 1e-10 or not np.array_equal(predicted >= .5, replay >= .5):
                raise ValueError('Saved nested head replay differs')
            predictions[fold, ~mask] = replay
            fits.append(dict(FIT=fold, FIT_parents=int(mask.sum()), scored_parents=int((~mask).sum()),
                solver=solver, artifact_sha256=digest(path), replay_max_error=error))
            del x, blocks, replay_x
    for fold in range(3):
        if not np.isnan(predictions[fold, folds == fold]).all() or not np.isfinite(predictions[fold, folds != fold]).all():
            raise ValueError('Unexpected FIT prediction or incomplete held-out scores')
    save_npz(ROOT / 'scores.npz', scores=predictions, parents=np.asarray([r['parent_id'] for r in rows]),
             folds=folds, conditions=c['conditions'], global_role='TRAIN', contract_sha256=digest(CONTRACT))
    write_once(ROOT / 'locked_scores.json', dict(scores_sha256=digest(ROOT / 'scores.npz'), fits=fits,
        contract_sha256=digest(CONTRACT)))
    selections = []
    for assignment in c['assignments']:
        mask = folds == assignment['CAL']
        selection = nested_calibration.select_cut(labels[mask], predictions[assignment['FIT'], mask], groups[mask])
        selections.append(assignment | {'selection': selection})
    write_once(ROOT / 'locked_calibration.json', dict(assignments=selections,
        contract_sha256=digest(CONTRACT), locked_scores_sha256=digest(ROOT / 'locked_scores.json')))
    results = []
    for assignment in selections:
        mask = folds == assignment['EVAL']
        y, g = labels[mask], groups[mask]
        scores = predictions[assignment['FIT'], mask]
        selected_rows = [r for r, take in zip(rows, mask, strict=True) if take]
        result = assignment | dict(baseline_half=nested_calibration.outcomes(y, scores, g, .5),
            baseline_components=component_metrics(selected_rows, y, scores, g, .5, c['conditions']),
            calibrated=None, EVAL_pass=False)
        if assignment['selection']['accepted']:
            cut = assignment['selection']['cut']
            result['calibrated'] = nested_calibration.outcomes(y, scores, g, cut)
            result['calibrated_components'] = component_metrics(selected_rows, y, scores, g, cut, c['conditions'])
            result['EVAL_pass'] = all(r['REAL_FPR'] <= .10 + 1e-12 and r['worst_REAL_component_FPR'] <= .20 + 1e-12
                and r['AI_recall'] >= .95 and r['new_AI_misses'] == 0 and r['new_REAL_alerts'] == 0
                for r in result['calibrated'])
        results.append(result)
    report = dict(state='E146_nested_calibration_complete', contract_sha256=digest(CONTRACT),
        locked_scores_sha256=digest(ROOT / 'locked_scores.json'),
        locked_calibration_sha256=digest(ROOT / 'locked_calibration.json'),
        conditions=c['conditions'], parents=n, fits=fits, assignments=results,
        accepted_CAL_assignments=sum(r['selection']['accepted'] for r in results),
        passed_EVAL_assignments=sum(r['EVAL_pass'] for r in results),
        seconds=time.monotonic() - start, downloads=0, new_pixels_read=0,
        protected_reserve_reads=0, promotion_allowed=False, limits=c['limits'])
    write_once(ROOT / 'report.json', report)
    write_once(EVIDENCE / 'e146_nested_calibration.json', report)
    return {k: v for k, v in report.items() if k not in ('fits', 'assignments')}


if __name__ == '__main__':
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
    def denied(*args, **kwargs):
        raise RuntimeError('Nested calibration is offline')
    socket.socket.connect = denied
    socket.socket.connect_ex = denied
    socket.create_connection = denied
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['freeze', 'fit'])
    ROOT.mkdir(exist_ok=True)
    with (ROOT / 'execution.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        print(json.dumps({'freeze': freeze, 'fit': fit}[parser.parse_args().stage](), indent=2))
