"""Registered paired patch head: internal sensor-held-out research, never promotion."""
import argparse
import fcntl
from importlib.metadata import version
import json
import os
from pathlib import Path
import socket
import time
import numpy as np
from PIL import Image
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits
from experiments.e65_acquisition import digest, read, write_once
from experiments.e71_features import save_npz
from experiments.e72_acquisition import resource_check
from experiments import e130_patch_drift_audit as baseline
from pixelproof import patch_learning, holdout_linear, spatial_evaluation, patch_drift
from pixelproof.project_paths import DATA_ROOT, ML_ROOT

ROOT = DATA_ROOT / 'e132'
EVIDENCE = ML_ROOT.parent / 'evidence'
CONTRACT = ROOT / 'contract.json'
CONDITIONS = ('original', 'jpeg75')
VARIANTS = ('authentic', 'classical_edit', 'ai_composite')


def freeze():
    previous = baseline.validate()
    report = read(DATA_ROOT / 'e130/report.json')
    if digest(DATA_ROOT / 'e130/report.json') != digest(EVIDENCE / 'e130_patch_drift_audit.json') or \
            report['accepted_composites'] != 16 or report['excluded_generations'] != 0:
        raise ValueError('Complete bound sixteen-parent diagnostic required')
    generation = read(EVIDENCE / 'e129_context_replay.json')
    if not generation['comparison']['full16_engineering_gate_passed']:
        raise ValueError('Original full16 generation gate required')
    scoring = read(DATA_ROOT / 'e130/scoring.json')
    if digest(DATA_ROOT / 'e130/scoring.json') != report['scoring_sha256'] or \
            digest(DATA_ROOT / 'e130/tokens.npz') != scoring['tokens_sha256']:
        raise ValueError('Bound tokens differ')
    rows = read(DATA_ROOT / 'e128/prepared.json')['rows']
    folds = patch_learning.source_folds(rows)
    expected = {f"{r['index']:03d}_{v}_{c}" for r in rows for v in VARIANTS for c in CONDITIONS}
    if {r['key'] for r in scoring['identities']} != expected or len(scoring['identities']) != len(expected):
        raise ValueError('Complete paired identity population required')
    for identity in scoring['identities']:
        row = rows[identity['index']]
        if identity['parent_id'] != row['parent_id'] or identity['role'] != 'TRAIN_RESEARCH_PILOT':
            raise ValueError('Token ancestry differs')
    for row in rows:
        with Image.open(row['files']['mask']['path']) as im:
            raw = np.asarray(im)
        if not np.isin(raw, [0, 255]).all():
            raise ValueError('Exact binary masks required')
        patch_learning.pure_patches(raw == 255)
    files = [Path(__file__), Path(patch_learning.__file__), Path(holdout_linear.__file__),
             Path(spatial_evaluation.__file__), Path(patch_drift.__file__),
             Path(digest.__code__.co_filename), Path(save_npz.__code__.co_filename),
             Path(resource_check.__code__.co_filename),
             DATA_ROOT / 'e130/contract.json', DATA_ROOT / 'e130/report.json',
             DATA_ROOT / 'e130/scoring.json', DATA_ROOT / 'e130/tokens.npz',
             EVIDENCE / 'e129_context_replay.json', EVIDENCE / 'e130_patch_drift_audit.json']
    c = {'state': 'E132_paired_patch_learning_registered',
         'inputs': previous['inputs'] | {str(p): digest(p) for p in files},
         'rows': rows, 'folds': folds.tolist(), 'parents': len(rows),
         'fold_count': len(set(folds.tolist())), 'conditions': CONDITIONS,
         'runtime_versions': {name: version(name) for name in ('numpy', 'scipy', 'scikit-learn', 'Pillow', 'threadpoolctl')},
         'representation': 'Frozen E130 original384-D tokens; no positional coordinates, drift, masks or reference image enter inference. FIT-only weighted mean/std; no PCA.',
         'split': 'Leave one declared sensor component out, joining known scene/body ancestry. Both conditions and all variants stay with parent. Already consumed research population, not fresh validation.',
         'targets': 'Pure16px cells wholly inside fixed8px-eroded mask positive; pure exterior background negative. Other composite cells excluded from fitting; all cells scored. Authentic and classical controls all AI-negative.',
         'objective': 'Unit weight: half AI interior, one sixth each composite background/authentic/classical; equal parent and condition within each. Zero-start BCE + .005||w||^2; fixed tested L-BFGS-B500/ftol1e-12/gtol1e-7, final gradient<=1e-5.',
         'diagnostic_cut': .5, 'max_seconds': 3600,
         'scope_admission': 'Only these16 paired sets may be used in E132 TRAIN research. Global roles unchanged; no final full-data head.',
         'metrics': 'Lock all OOF maps before metrics. Mean per-parent full/interior-background AUC, IoU and negative flagged area at.5. Paired AI-authentic/classical aligned AUC contrasts and fixed E130/radial/constant comparisons. No independent pixel intervals or tuning.',
         'downloads': 0, 'promotion_allowed': False,
         'limits': 'Sixteen inspected MIDD parents, one SD1.5 editor, hard composites, simple masks and two views. Global token context may carry synthetic cues into background. Intended generation masks are not semantic-change annotations. Unknown scene ancestry and encoder pretraining remain. Fold heads are not one deployable model; raw scores are not calibrated probabilities.'}
    ROOT.mkdir(exist_ok=True); write_once(CONTRACT, c)
    write_once(EVIDENCE / 'e132_patch_learning_contract.json',
               {k: v for k, v in c.items() if k not in ('inputs', 'rows')} | {'contract_sha256': digest(CONTRACT)})
    return {'parents': len(rows), 'folds': c['fold_count'], 'contract_sha256': digest(CONTRACT)}


def validate():
    c = read(CONTRACT)
    if digest(CONTRACT) != read(EVIDENCE / 'e132_patch_learning_contract.json')['contract_sha256']:
        raise ValueError('Patch-learning contract differs')
    for path, sha in c['inputs'].items():
        if digest(path) != sha:
            raise ValueError('Registered input differs')
    return c


def fit():
    c = validate(); start = time.monotonic(); deadline = start + c['max_seconds']
    resource_check(deadline)
    write_once(ROOT / 'started.json', {'contract_sha256': digest(CONTRACT)})
    rows = c['rows']; folds = np.asarray(c['folds']); masks = {}
    with np.load(DATA_ROOT / 'e130/tokens.npz', allow_pickle=False) as data:
        if str(data['contract_sha256']) != digest(DATA_ROOT / 'e130/contract.json'):
            raise ValueError('Token contract differs')
        tokens = {key: data[key] for key in data.files if key != 'contract_sha256'}
    for row in rows:
        with Image.open(row['files']['mask']['path']) as im:
            masks[row['index']] = np.asarray(im) == 255
    maps = {}; fits = {}
    with threadpool_limits(limits=2):
        for fold in sorted(set(folds.tolist())):
            resource_check(deadline)
            train = [r for r, f in zip(rows, folds, strict=True) if f != fold]
            held = [r for r, f in zip(rows, folds, strict=True) if f == fold]
            x, y, weights = patch_learning.training_arrays(train, tokens, masks)
            center, scale = patch_learning.fit_normalizer(x, weights)
            x = (x - center) / scale
            parameters, solver = holdout_linear.fit_head(x, y, weights, lambda: resource_check(deadline))
            artifact = ROOT / f'fold{fold}.npz'
            save_npz(artifact, center=center, scale=scale, parameters=parameters, contract_sha256=digest(CONTRACT))
            parity = 0.
            with np.load(artifact, allow_pickle=False) as saved:
                for row in held:
                    for variant in VARIANTS:
                        for condition in CONDITIONS:
                            key = f"{row['index']:03d}_{variant}_{condition}"
                            value = tokens[key].reshape(-1, 384)
                            pred = holdout_linear.predict((value-center)/scale, parameters)
                            replay = holdout_linear.predict((value-saved['center'])/saved['scale'], saved['parameters'])
                            parity = max(parity, float(np.max(np.abs(pred-replay))))
                            if parity > 1e-10 or not np.array_equal(pred >= .5, replay >= .5):
                                raise ValueError('Saved-head prediction parity differs')
                            maps[key] = pred.reshape(32, 32)
            fits[str(fold)] = {'train_parents': [r['index'] for r in train], 'held_parents': [r['index'] for r in held],
                              'solver': solver, 'artifact_sha256': digest(artifact), 'max_replay_error': parity}
            print(json.dumps({'E132_fold_complete': fold, 'seconds': round(time.monotonic()-start)}), flush=True)
    if len(maps) != len(rows)*len(VARIANTS)*len(CONDITIONS):
        raise ValueError('Incomplete held-out maps')
    save_npz(ROOT / 'scores.npz', **maps, contract_sha256=digest(CONTRACT))
    write_once(ROOT / 'locked_scores.json', {'scores_sha256': digest(ROOT/'scores.npz'), 'fits': fits,
                                          'contract_sha256': digest(CONTRACT)})
    results = []
    for row in rows:
        mask = masks[row['index']]
        for condition in CONDITIONS:
            dense = {v: patch_drift.expand_grid(maps[f"{row['index']:03d}_{v}_{condition}"], (512, 512)) for v in VARIANTS}
            metrics = spatial_evaluation.evaluate_triplet(*(dense[v] for v in VARIANTS), mask,
                                                         threshold=.5, boundary_width=8)
            aligned = {v: float(roc_auc_score(mask.ravel(), dense[v].ravel())) for v in VARIANTS}
            results.append({'index': row['index'], 'source': row['source'], 'fold': c['folds'][row['index']],
                            'condition': condition, 'metrics': metrics, 'aligned_auc': aligned})
    write_once(ROOT/'measurements.json', {'results': results, 'contract_sha256': digest(CONTRACT)})
    reference = read(EVIDENCE/'e130_patch_drift_audit.json')['summary']; summary = {}
    for condition in CONDITIONS:
        group = [r for r in results if r['condition'] == condition]
        summary[condition] = {}
        for variant in VARIANTS:
            values = [r['metrics']['maps'][variant] for r in group]
            summary[condition][variant] = {
                'mean_flagged_area': float(np.mean([v['flagged_area_fraction'] for v in values])),
                'mean_iou': float(np.mean([v['iou'] for v in values if v['iou'] is not None])) if any(v['iou'] is not None for v in values) else None,
                'mean_pixel_auc': float(np.mean([v['all_pixel_ranking']['auc'] for v in values])) if variant == 'ai_composite' else None,
                'mean_interior_background_auc': float(np.mean([v['interior_background_ranking']['auc'] for v in values])) if variant == 'ai_composite' else None}
        summary[condition]['AI_minus_authentic_aligned_auc'] = float(np.mean([r['aligned_auc']['ai_composite']-r['aligned_auc']['authentic'] for r in group]))
        summary[condition]['AI_minus_classical_aligned_auc'] = float(np.mean([r['aligned_auc']['ai_composite']-r['aligned_auc']['classical_edit'] for r in group]))
        summary[condition]['reference_rankings'] = reference[condition]['accepted_parent_ranking_baselines'] | {
            'E130_drift_mean_pixel_auc': reference[condition]['ai_composite']['metrics']['pixel_auc']['mean']}
    report = {'state': 'E132_internal_patch_learning_complete', 'contract_sha256': digest(CONTRACT),
              'parents': len(rows), 'folds': c['fold_count'], 'summary': summary,
              'seconds': time.monotonic()-start, 'locked_scores_sha256': digest(ROOT/'locked_scores.json'),
              'measurements_sha256': digest(ROOT/'measurements.json'), 'downloads': 0,
              'promotion_allowed': False, 'limits': c['limits']}
    write_once(ROOT/'report.json', report); write_once(EVIDENCE/'e132_patch_learning.json', report)
    note = '\n### E132 internal patch-learning result\n\n'+json.dumps(report, sort_keys=True)+'\n\nNo serving change, calibrated probability or independent generalization claim.\n'
    for path in (ML_ROOT.parent/'HISTORY.md', ML_ROOT/'EXPERIMENTS.md', ML_ROOT.parent/'DATASETS.md'):
        with path.open('a') as f:
            fcntl.flock(f, fcntl.LOCK_EX); f.write(note)
    return report


if __name__ == '__main__':
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
    def denied(*args, **kwargs):
        raise RuntimeError('Patch-learning experiment is offline')
    socket.socket.connect = denied; socket.socket.connect_ex = denied; socket.create_connection = denied
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('stage', choices=['freeze', 'fit'])
    ROOT.mkdir(exist_ok=True)
    with (ROOT/'execution.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        print(json.dumps({'freeze': freeze, 'fit': fit}[parser.parse_args().stage](), indent=2))
