"""One consumed E49 comparison of immutable E62; no fresh-final or tuning claim."""
from __future__ import annotations
import argparse
import fcntl
import json
import os
from pathlib import Path
import socket
import time

if __name__ == '__main__':
    def denied(*args, **kwargs):
        raise RuntimeError('E62 network connections disabled')
    socket.socket.connect = denied
    socket.socket.connect_ex = denied
    socket.create_connection = denied
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_HUB_DISABLE_TELEMETRY='1')

import joblib
import numpy as np
from threadpoolctl import threadpool_limits
from experiments import e62_constrained as correction
from experiments.e62_run import ROOT, EVIDENCE, CONTRACT, CANDIDATE, RESULT, validate, resource_check
from experiments.e61_replay_gate import digest, read, write_once as write, REFERENCE_SHA
from experiments.e60_run import pair_rates
from pixelproof.project_paths import DATA_ROOT, ML_ROOT

REFERENCE = DATA_ROOT / 'e43/e43_small_predev.joblib'
REGRESSION_CONTRACT = ROOT / 'regression_contract.json'
REGRESSION_SCORES = ROOT / 'e49_scores.json'
REGRESSION_RESULT = ROOT / 'e49_report.json'


def verified(path, sha):
    if digest(path) != sha:
        raise ValueError(f'bound input changed: {path}')
    return Path(path)


def freeze():
    validate()
    fitted = read(RESULT)
    if not fitted['consumed_regression_permitted']:
        raise ValueError('E62 TRAIN guard failed; evaluation denied')
    verified(CANDIDATE, fitted['candidate_sha256'])
    manifest = DATA_ROOT / 'e49/final/manifest_unscored.json'
    old_scores = DATA_ROOT / 'e49/final/generalist_scores.jsonl'
    manifest_sha = '9744a9d2385ef2f105b7a132bfee76a7099d280ac9d075d81220f572429c5909'
    old_sha = '249f005c83acbc65ac987c9fdeea91a9a3c7dfc4a388fdd4e20a436bc53610a8'
    verified(manifest, manifest_sha); verified(old_scores, old_sha)
    from experiments.e49_evaluation import validate_paired_final
    validate_paired_final(read(manifest)['rows'])
    files = [Path(__file__), Path(__file__).with_name('e60_run.py'),
             Path(__file__).with_name('e42_features.py'), Path(__file__).with_name('e49_evaluation.py'),
             ML_ROOT / 'src/pixelproof/benchmark_metrics.py', ML_ROOT / 'src/pixelproof/e32_candidate.py']
    config = {'state': 'E62_consumed_E49_comparison_registered',
              'candidate_sha256': digest(CANDIDATE), 'fit_sha256': digest(RESULT),
              'contract_sha256': digest(CONTRACT), 'inputs': {str(p): digest(p) for p in files},
              'manifest': str(manifest), 'manifest_sha256': manifest_sha,
              'old_scores': str(old_scores), 'old_scores_sha256': old_sha,
              'batch_views': 16, 'max_seconds': 900, 'reference_max_score_error': 5e-5,
              'reference_binary_selective_decision_changes': 0,
              'comparison': 'All20 frozen E49 gates plus E43 pooled/source AI recall no-loss '
                            'and paired source-cluster intervals; no parameter selection.',
              'intervals': '20000 paired source-cluster resamples stratified by label; seed62; '
                           'Bonferroni4 intervals quantiles .00625/.99375.',
              'fresh_final_claim_allowed': False, 'promotion_allowed': False, 'downloads': 0}
    write(REGRESSION_CONTRACT, config)
    write(EVIDENCE / 'e62_regression_contract.json', config | {'regression_contract_sha256': digest(REGRESSION_CONTRACT)})
    return {'state': config['state'], 'candidate_sha256': config['candidate_sha256']}


def regression_inputs():
    validate()
    c = read(REGRESSION_CONTRACT)
    verified(REGRESSION_CONTRACT, read(EVIDENCE / 'e62_regression_contract.json')['regression_contract_sha256'])
    verified(CANDIDATE, c['candidate_sha256']); verified(RESULT, c['fit_sha256'])
    for path, sha in c['inputs'].items():
        verified(path, sha)
    verified(c['manifest'], c['manifest_sha256']); verified(c['old_scores'], c['old_scores_sha256'])
    rows = read(c['manifest'])['rows']
    baseline = {r['record_id']: r for r in map(json.loads, Path(c['old_scores']).read_text().splitlines())}
    if len(rows) != 4000 or len(baseline) != len(rows):
        raise ValueError('incomplete benchmark coverage')
    for r in rows:
        if any(r[k] != baseline[r['record_id']][k] for k in ('parent_id', 'label', 'source', 'condition')):
            raise ValueError('reference/manifest identity mismatch')
    return c, rows, baseline


# Evaluation mechanics copied from frozen E60; new bindings, candidate and namespace.
def regression_score():
    c, rows, baseline = regression_inputs()
    if REGRESSION_SCORES.exists():
        raise FileExistsError('complete regression scores already exist')
    from concurrent.futures import ThreadPoolExecutor
    from PIL import Image
    import torch
    from experiments import e42_features as dino
    torch.set_num_threads(2)
    started = time.monotonic(); deadline = started+c['max_seconds']; resource_check(deadline)
    head = joblib.load(REFERENCE)['head']
    with np.load(CANDIDATE, allow_pickle=False) as a:
        if str(a['contract_sha256']) != digest(CONTRACT) or str(a['reference_sha256']) != REFERENCE_SHA:
            raise ValueError('candidate binding changed')
        weights = {k: a[k] for k in ('weights', 'mean', 'components', 'scales')}
    model, means, stds, _ = dino._load_small()  # Cached pinned local weights; network denied.
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    model = model.to(device).eval()
    mean = torch.tensor(means, device=device).view(1, 3, 1, 1)
    std = torch.tensor(stds, device=device).view(1, 3, 1, 1)
    partial = ROOT / 'e49_scores.partial.jsonl'
    done = [json.loads(line) for line in partial.read_text().splitlines()] if partial.exists() else []
    if len(done) > len(rows):
        raise ValueError('invalid score prefix')
    for result, row in zip(done, rows, strict=False):
        if any(result[k] != row[k] for k in ('record_id', 'parent_id', 'label', 'source', 'condition')):
            raise ValueError('resumed score identities differ')
        if result['binding'] != digest(REGRESSION_CONTRACT) or not all(
                np.isfinite(result[k]) and 0 <= result[k] <= 1 for k in ('old_reproduced', 'score')):
            raise ValueError('resumed score contract differs')
        verified(row['path'], row['sha256'])
    binding = digest(REGRESSION_CONTRACT)
    def prepare(row):
        verified(row['path'], row['sha256'])
        with Image.open(row['path']) as im:
            return dino.texture_crops(dino.transport_image(im, 'clean'))
    with torch.inference_mode(), ThreadPoolExecutor(max_workers=4) as pool, threadpool_limits(limits=2):
        for start in range(len(done), len(rows), c['batch_views']):
            resource_check(deadline)
            batch = rows[start:start+c['batch_views']]
            arrays = np.stack([crop for prepared in pool.map(prepare, batch) for crop in prepared])
            tensor = torch.from_numpy(arrays).to(device).permute(0, 3, 1, 2).float().div_(255)
            blocks = model.forward_intermediates((tensor-mean)/std, indices=list(dino.BLOCKS['small']),
                return_prefix_tokens=True, norm=True, intermediates_only=True)
            tokens = torch.stack([b[1][:, 0, :] for b in blocks], dim=1).float().cpu().numpy()
            features = dino.aggregate_tokens(tokens, len(batch))
            old = head.predict_proba(features)[:, 1]
            new = correction.predict(head, features, weights)
            results = [dict({k: r[k] for k in ('record_id', 'parent_id', 'label', 'source', 'condition')},
                           old_reproduced=float(old[i]), score=float(new[i]), status='ok', binding=binding)
                       for i, r in enumerate(batch)]
            with partial.open('a') as stream:
                for r in results:
                    stream.write(json.dumps(r, sort_keys=True)+'\n')
                stream.flush(); os.fsync(stream.fileno())
            done.extend(results)
            if len(done) % 160 == 0:
                print(json.dumps({'phase': 'E62_frozen_E49_scores', 'rows': len(done), 'total': len(rows),
                                  'seconds': round(time.monotonic()-started)}), flush=True)
    differences = [abs(r['old_reproduced']-baseline[r['record_id']]['score']) for r in done]
    changes = {str(cut): sum((r['old_reproduced'] >= cut) != (baseline[r['record_id']]['score'] >= cut)
               for r in done) for cut in (correction.AI_CUT, correction.REAL_CUT)}
    if max(differences) > c['reference_max_score_error'] or any(changes.values()):
        raise ValueError('E43 replay drift prevents regression interpretation')
    write(REGRESSION_SCORES, {'state': 'E62_scores_locked_before_metrics', 'rows': done,
        'contract_sha256': binding, 'reference_max_score_error': max(differences),
        'reference_decision_changes': changes, 'seconds': time.monotonic()-started})
    write(EVIDENCE / 'e62_regression_scores.json', {'rows': len(done), 'sha256': digest(REGRESSION_SCORES),
        'reference_max_score_error': max(differences), 'reference_decision_changes': changes,
        'metrics_opened': False, 'source_downloads': 0, 'contract_sha256': binding})
    return {'rows': len(done), 'reference_max_score_error': max(differences)}


def regression_report():
    c, manifest, baseline = regression_inputs()
    verified(REGRESSION_SCORES, read(EVIDENCE / 'e62_regression_scores.json')['sha256'])
    scored = read(REGRESSION_SCORES)['rows']
    from experiments.e49_evaluation import evaluate_final, validate_paired_final
    validate_paired_final(scored)
    full = evaluate_final(scored)
    parents = sorted({r['parent_id'] for r in manifest})
    by_key = {(r['parent_id'], r['condition']): r for r in scored}
    conditions = ('publisher_original', 'social_q75')
    labels = np.array([by_key[p, conditions[0]]['label'] for p in parents])
    sources = np.array([by_key[p, conditions[0]]['source'] for p in parents])
    new = np.array([[by_key[p, cond]['score'] for cond in conditions] for p in parents])
    old = np.array([[baseline[by_key[p, cond]['record_id']]['score'] for cond in conditions] for p in parents])
    comparisons = {cond: pair_rates(labels, sources, old[:, j], new[:, j]) for j, cond in enumerate(conditions)}
    delta = (new >= correction.AI_CUT).astype(float) - (old >= correction.AI_CUT)
    rng = np.random.default_rng(62); samples = []
    for y in (0, 1):
        groups = sorted(set(sources[labels == y]))
        counts = np.array([np.sum((labels == y) & (sources == s)) for s in groups])
        changes = np.array([delta[(labels == y) & (sources == s)].sum(axis=0) for s in groups])
        draws = rng.integers(len(groups), size=(20000, len(groups)))
        samples.append(changes[draws].sum(axis=1) / counts[draws].sum(axis=1)[:, None])
    intervals = {name: np.quantile(np.concatenate(samples, axis=1)[:, j], [.00625, .99375]).tolist()
                 for j, name in enumerate(('real_clean', 'real_q75', 'ai_clean', 'ai_q75'))}
    checks = {
        'real_fpr_lower_both': all(v['new']['real_false_ai'] < v['old']['real_false_ai'] for v in comparisons.values()),
        'ai_recall_no_point_loss_both': all(v['new']['ai_recall'] >= v['old']['ai_recall'] for v in comparisons.values()),
        'supported_ai_sources_no_point_loss': all(g['new_ai_minus_old'] >= 0 for v in comparisons.values()
            for key, g in v['by_source'].items() if key.startswith('1:') and g['parents_or_views'] >= 50),
        'real_improvement_intervals_both': all(intervals[f'real_{cond}'][1] < 0 for cond in ('clean', 'q75')),
        'ai_preservation_intervals_both': all(intervals[f'ai_{cond}'][0] >= 0 for cond in ('clean', 'q75')),
        'absolute_selective_20_checks': full['gate']['passed']}
    result = {'state': 'E62_consumed_E49_regression_complete_not_fresh_final', 'checks': checks,
        'preservation_and_absolute_checks_passed': all(checks.values()), 'comparisons': comparisons,
        'source_cluster_paired_intervals': intervals, 'candidate_e49_diagnostics': full,
        'candidate_sha256': c['candidate_sha256'], 'contract_sha256': digest(REGRESSION_CONTRACT),
        'raw_scores_sha256': digest(REGRESSION_SCORES), 'reference_sha256': digest(REFERENCE),
        'serving_changed': False, 'independent_final_passed': False, 'winner_selection_allowed': False,
        'limitations': ['Consumed E49 diagnostic, never a new validation/final or tuning set.',
                       'Intervals condition on observed camera/generator sources; unknown prompt dependence remains.',
                       'Candidate remains research-only even if these diagnostic checks pass.']}
    write(REGRESSION_RESULT, result); write(EVIDENCE / 'e62_regression.json', result)
    return {'checks': checks, 'comparisons': {k: {a: b for a, b in v.items() if a != 'by_source'}
                                             for k, v in comparisons.items()}, 'intervals': intervals}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('freeze', 'score', 'report'))
    args = parser.parse_args()
    with (ROOT / 'execution.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        print(json.dumps({'freeze': freeze, 'score': regression_score,
                          'report': regression_report}[args.stage](), indent=2))
