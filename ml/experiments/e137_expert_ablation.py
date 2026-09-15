"""Registered leave-one-expert-out refits on frozen E131 source-excluding maps."""
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
from experiments import e136_transport_consistency as consistency
from pixelproof import holdout_linear, source_holdout
from pixelproof.project_paths import DATA_ROOT, ML_ROOT

ROOT = DATA_ROOT / 'e137'
EVIDENCE = ML_ROOT.parent / 'evidence'
CONTRACT = ROOT / 'contract.json'
BRANCHES = previous.BRANCHES
EXPERTS = ('dino', 'clip', 'dear', 'context')


def without(blocks, omitted):
    if set(blocks) != set(EXPERTS) or omitted not in EXPERTS:
        raise ValueError('Exactly four named experts and one registered omission required')
    values = [np.asarray(blocks[name]) for name in EXPERTS]
    if any(v.ndim != 2 or not v.shape[1] or not np.isfinite(v).all() for v in values) or \
            len({len(v) for v in values}) != 1:
        raise ValueError('Finite aligned expert blocks required')
    return np.column_stack([blocks[name] for name in EXPERTS if name != omitted])


def freeze():
    prior = consistency.validate()
    if digest(consistency.ROOT / 'report.json') != digest(EVIDENCE / 'e136_transport_consistency.json'):
        raise ValueError('Complete bound E136 outcome required before next diagnostic')
    files = [Path(__file__), Path(consistency.__file__), consistency.CONTRACT,
             consistency.ROOT / 'report.json', EVIDENCE / 'e136_transport_consistency.json']
    c = dict(state='E137_Model1_expert_ablation_registered',
             inputs=prior['inputs'] | {str(p): digest(p) for p in files},
             parents=prior['parents'], branches=BRANCHES, omitted_experts=EXPERTS,
             conditions=prior['conditions'], folds=3, fits=24, max_seconds=3600,
             change='Drop exactly one of DINO64, pooledCLIP64, DEAR64, context128; refit remaining coordinates from zero. Fixed four omissions, both geometries, three E131 source folds. No combinations, search or E136 consistency penalty.',
             fitting='Exact frozen E131 FIT-only PCA/whitening maps, class/component/parent/view weights, BCE+.005||w||^2, zero-start L-BFGS500/ftol1e-12/gtol1e-7 and final gradient<=1e-5. Dimensionality changes by design; no retuned regularization.',
             evaluation='Replay E131 held-out scores/cuts within1e-10, lock all24 fit outputs before metrics. Fixed0.5 cut, source/fold/condition metrics and individual new/rescued errors versus E131. Context-omitted branches must replay each other within1e-10. Report every omission; no winner selection or automatic promotion.',
             downloads=0, new_pixels_read=0, external_DEV_or_gallery_reads=0,
             promotion_allowed=False, generator_family_holdout_supported=False,
             limits=prior['limits'] + ' Adaptive repeated-fold expert-ablation diagnostic. Refit differences measure conditional utility of an expert plus changed capacity, not causal dataset bias, standalone expert quality or independent selection evidence.')
    ROOT.mkdir(exist_ok=True)
    write_once(CONTRACT, c)
    write_once(EVIDENCE / 'e137_expert_ablation_contract.json',
               {k: v for k, v in c.items() if k != 'inputs'} | {'contract_sha256': digest(CONTRACT)})
    return {'parents': c['parents'], 'fits': c['fits'], 'contract_sha256': digest(CONTRACT)}


def validate():
    c = read(CONTRACT)
    if digest(CONTRACT) != read(EVIDENCE / 'e137_expert_ablation_contract.json')['contract_sha256']:
        raise ValueError('Ablation contract differs')
    for path, sha in c['inputs'].items():
        if digest(path) != sha:
            raise ValueError('Frozen ablation input differs')
    return c


def fit():
    c = validate(); start = time.monotonic(); deadline = start + c['max_seconds']
    resource_check(deadline)
    write_once(ROOT / 'started.json', {'contract_sha256': digest(CONTRACT)})
    prior = read(previous.CONTRACT); rows = prior['rows']; n = len(rows)
    features = previous.load_features(rows)
    labels = np.asarray([r['label'] for r in rows])
    folds = np.asarray([prior['outer_fold'][r['parent_id']] for r in rows])
    groups = np.asarray([prior['components'][r['parent_id']] for r in rows])
    scores = {f'{branch}_without_{expert}': np.full((n, 4), np.nan)
              for branch in BRANCHES for expert in EXPERTS}
    with np.load(previous.ROOT / 'scores.npz', allow_pickle=False) as a:
        if str(a['contract_sha256']) != digest(previous.CONTRACT) or \
                list(a['parents']) != [r['parent_id'] for r in rows] or \
                list(a['conditions']) != c['conditions'] or not np.array_equal(a['outer_fold'], folds) or \
                str(a['global_role']) != 'TRAIN' or str(a['usage']) != 'INTERNAL_HELD_OUT':
            raise ValueError('E131 comparator parent/fold/condition/role identity differs')
        baseline = {b: a[b].copy() for b in BRANCHES}
    fits = {}; baseline_replays = {}
    with threadpool_limits(limits=2):
        for fold in range(3):
            parent_fit = folds != fold; view_fit = np.repeat(parent_fit, 4)
            selected = [r for r, take in zip(rows, parent_fit, strict=True) if take]
            weights = source_holdout.balanced_weights(selected,
                {r['parent_id']: prior['components'][r['parent_id']] for r in selected}, 4)
            blocks = {}
            with np.load(previous.ROOT / f'fold{fold}_shared.npz', allow_pickle=False) as maps:
                if str(maps['contract_sha256']) != digest(previous.CONTRACT):
                    raise ValueError('Shared map contract differs')
                for expert in EXPERTS[:-1]:
                    resource_check(deadline)
                    transform = {k.removeprefix(expert + '_'): maps[k]
                                 for k in maps.files if k.startswith(expert + '_')}
                    blocks[expert] = holdout_linear.project(features[expert], transform)
            for branch in BRANCHES:
                with np.load(previous.ROOT / f'fold{fold}_{branch}.npz', allow_pickle=False) as a:
                    if str(a['contract_sha256']) != digest(previous.CONTRACT):
                        raise ValueError('Context map contract differs')
                    blocks['context'] = holdout_linear.project(features[branch], a)
                    parameters = a['parameters'].copy()
                complete = np.column_stack([blocks[name] for name in EXPERTS])
                replay = holdout_linear.predict(complete[~view_fit], parameters).reshape(-1, 4)
                old = baseline[branch][~parent_fit]
                error = float(np.abs(replay - old).max())
                if error > 1e-10 or not np.array_equal(replay >= .5, old >= .5):
                    raise ValueError('E131 baseline replay differs')
                baseline_replays[f'{fold}_{branch}'] = error
                del complete
                for omitted in EXPERTS:
                    resource_check(deadline)
                    x = without(blocks, omitted); name = f'{branch}_without_{omitted}'
                    parameters, solver = holdout_linear.fit_head(x[view_fit],
                        np.repeat(labels[parent_fit], 4), weights, lambda: resource_check(deadline))
                    artifact = ROOT / f'fold{fold}_{name}.npz'
                    save_npz(artifact, parameters=parameters, contract_sha256=digest(CONTRACT),
                             omitted_expert=omitted, branch=branch, outer_fold=fold)
                    prediction = holdout_linear.predict(x[~view_fit], parameters).reshape(-1, 4)
                    with np.load(artifact, allow_pickle=False) as saved:
                        replay = holdout_linear.predict(x[~view_fit], saved['parameters']).reshape(-1, 4)
                    error = float(np.abs(replay - prediction).max())
                    if error > 1e-10 or not np.array_equal(replay >= .5, prediction >= .5):
                        raise ValueError('Saved ablation score/cut replay differs')
                    scores[name][~parent_fit] = prediction
                    fits[f'{fold}_{name}'] = dict(solver=solver, artifact_sha256=digest(artifact),
                        saved_score_max_error=error, fit_parents=int(parent_fit.sum()),
                        heldout_parents=int((~parent_fit).sum()), coordinates=x.shape[1])
                    del x
                print(json.dumps({'E137_fold': fold, 'branch': branch,
                                  'seconds': round(time.monotonic() - start)}), flush=True)
    if any(not np.isfinite(v).all() for v in scores.values()):
        raise ValueError('Incomplete ablation scores')
    left = scores['center_control_without_context']; right = scores['full_frame_without_context']
    identical = float(np.abs(left - right).max())
    if identical > 1e-10 or not np.array_equal(left >= .5, right >= .5):
        raise ValueError('Context-omitted geometries must be identical')
    save_npz(ROOT / 'scores.npz', **scores, parents=np.asarray([r['parent_id'] for r in rows]),
             conditions=c['conditions'], outer_fold=folds, global_role='TRAIN',
             usage='INTERNAL_HELD_OUT', contract_sha256=digest(CONTRACT))
    write_once(ROOT / 'locked_scores.json', dict(scores_sha256=digest(ROOT / 'scores.npz'),
        contract_sha256=digest(CONTRACT), fits=fits, baseline_replays=baseline_replays,
        context_omitted_geometry_max_error=identical))
    reports = {}; passes = {}
    for branch in BRANCHES:
        for omitted in EXPERTS:
            name = f'{branch}_without_{omitted}'; reports[name] = {}
            for j, condition in enumerate(c['conditions']):
                component = []
                for group in sorted(set(groups.tolist())):
                    take = groups == group
                    component.append(previous.metrics(labels[take], scores[name][take, j]) | {
                        'sources': sorted({r['source'] for r, t in zip(rows, take, strict=True) if t}),
                        'paired': consistency.transitions(labels[take], baseline[branch][take, j], scores[name][take, j])})
                fp = [r['REAL_FPR'] for r in component if r['REAL_FPR'] is not None]
                recall = [r['AI_recall'] for r in component if r['AI_recall'] is not None]
                reports[name][condition] = dict(
                    pooled_different_fold_models=previous.metrics(labels, scores[name][:, j]),
                    folds=[previous.metrics(labels[folds == f], scores[name][folds == f, j]) for f in range(3)],
                    components=component, REAL_component_worst_FPR=max(fp), AI_component_worst_recall=min(recall),
                    paired=consistency.transitions(labels, baseline[branch][:, j], scores[name][:, j]))
            passes[name] = all(r['paired']['zero_new_AI_misses'] and r['paired']['zero_new_REAL_false_alerts']
                               for r in reports[name].values())
    report = dict(state='E137_Model1_expert_ablation_complete', contract_sha256=digest(CONTRACT),
        parents=n, fits=len(fits), reports=reports, passes_internal_individual_nonregression=passes,
        context_omitted_geometry_max_error=identical, locked_scores_sha256=digest(ROOT / 'locked_scores.json'),
        seconds=time.monotonic() - start, downloads=0, new_pixels_read=0,
        promotion_allowed=False, generator_family_holdout_supported=False, limits=c['limits'])
    write_once(ROOT / 'report.json', report)
    write_once(EVIDENCE / 'e137_expert_ablation.json', report)
    summary = {k: v for k, v in report.items() if k != 'reports'}
    note = '\n### E137 Model1 expert-ablation result\n\n' + json.dumps(summary, sort_keys=True) + \
        '\n\nAll eight branch/omission reports: evidence/e137_expert_ablation.json. No serving change.\n'
    for path in (ML_ROOT.parent / 'HISTORY.md', ML_ROOT / 'EXPERIMENTS.md', ML_ROOT.parent / 'DATASETS.md'):
        with path.open('a') as f:
            fcntl.flock(f, fcntl.LOCK_EX); f.write(note)
    return summary


if __name__ == '__main__':
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
    def denied(*args, **kwargs):
        raise RuntimeError('Model1 expert-ablation experiment is offline')
    socket.socket.connect = denied; socket.socket.connect_ex = denied; socket.create_connection = denied
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['freeze', 'fit'])
    ROOT.mkdir(exist_ok=True)
    with (ROOT / 'execution.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        print(json.dumps({'freeze': freeze, 'fit': fit}[parser.parse_args().stage](), indent=2))
