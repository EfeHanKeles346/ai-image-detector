"""One consumed E49 comparison, inaccessible until immutable E86 TRAIN and E66 DEV pass."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import socket
import time

import joblib
import numpy as np
from PIL import Image
import torch
from threadpoolctl import threadpool_limits
from experiments import e86_model as model, e42_features as dino
from experiments.e86_development import validate as validate_dev, REPORT as DEV_REPORT, CONTRACT as DEV_CONTRACT
from experiments.e86_fit import ROOT, EVIDENCE, CANDIDATE, CONTRACT as FIT_CONTRACT, REPORT as FIT_REPORT
from experiments.e65_acquisition import digest, read, write_once
from experiments.e72_acquisition import resource_check
from experiments.e71_features import save_npz
from experiments.e71_development import transitions
from experiments.e84_features import Encoders
from experiments.e60_run import pair_rates
from experiments.e49_evaluation import evaluate_final, validate_paired_final, CONDITIONS
from pixelproof.project_paths import DATA_ROOT, ML_ROOT

CONTRACT = ROOT / 'regression_contract.json'
SCORES = ROOT / 'e49_scores.json'
REPORT = ROOT / 'e49_report.json'
WIDTHS = {'dino': 3072, 'clip': 1536, 'dear': 1640}
MANIFEST_SHA = '9744a9d2385ef2f105b7a132bfee76a7099d280ac9d075d81220f572429c5909'
OLD_SHA = '249f005c83acbc65ac987c9fdeea91a9a3c7dfc4a388fdd4e20a436bc53610a8'


def require_development_pass():
    validate_dev(); report = read(DEV_REPORT)
    if digest(DEV_REPORT) != digest(EVIDENCE / 'e86_development.json') or \
            not report['passes_limited_dev_screen'] or not report['consumed_regression_may_be_registered'] or \
            not all(all(v.values()) for v in report['checks'].values()):
        raise ValueError('E86 full TRAIN/runtime and all consumed DEV guards required before any E49 access')
    if digest(CANDIDATE) != report['candidate_sha256']:
        raise ValueError('passed development candidate changed')


def freeze():
    require_development_pass()
    manifest = DATA_ROOT / 'e49/final/manifest_unscored.json'
    old_scores = DATA_ROOT / 'e49/final/generalist_scores.jsonl'
    if digest(manifest) != MANIFEST_SHA or digest(old_scores) != OLD_SHA:
        raise ValueError('immutable consumed E49 identities required')
    validate_paired_final(read(manifest)['rows'])
    reference = read(DEV_CONTRACT)['reference']
    files = [Path(__file__), Path(model.__file__), Path(dino.__file__),
             Path(__file__).with_name('e83_model.py'), Path(__file__).with_name('e84_features.py'),
             Path(__file__).with_name('e80_development.py'), Path(__file__).with_name('e71_features.py'),
             Path(__file__).with_name('e78_model.py'), Path(__file__).with_name('e71_development.py'),
             Path(__file__).with_name('e60_run.py'), Path(__file__).with_name('e49_evaluation.py'),
             ML_ROOT / 'src/pixelproof/benchmark_metrics.py', ML_ROOT / 'src/pixelproof/e32_candidate.py',
             FIT_CONTRACT, FIT_REPORT, DEV_CONTRACT, DEV_REPORT, CANDIDATE]
    c = {'state': 'E86_single_consumed_E49_regression_registered', 'inputs': {str(p): digest(p) for p in files},
         'manifest': str(manifest), 'manifest_sha256': MANIFEST_SHA, 'old_scores': str(old_scores),
         'old_scores_sha256': OLD_SHA, 'reference': reference, 'views': 4000, 'batch_views': 8,
         'max_seconds': 7200, 'mps_driver_limit_bytes': 6 * 1024**3, 'reference_max_score_error': 5e-5,
         'encoding': 'Exact E84/E83 DINO/CLIP/DEAR encoder operations and E43 clean crop pipeline on each '
                     'already-realized original/Q75 image. No second social transform. Frozen E86 CPU model, '
                     'runtime batch8. Immutable window chunks bind images/order/candidate contract and '
                     'CONSUMED_REGRESSION role. Resume completed chunks after full identity/hash/model replay.',
         'comparison': 'All20 frozen E49 numeric gates, lower pooled REAL FPR in both conditions, '
                       'pooled/source AI recall no-loss and zero newly missed E43 AI per source/condition. '
                       'Paired source-cluster intervals, unchanged two cuts. Scores locked before metrics.',
         'intervals': 'Same E63 paired source-cluster method:20000 resamples stratified by label, fixed seed86, '
                      'Bonferroni4 quantiles.00625/.99375. Conditional on observed sources, unknown prompt dependence.',
         'resource': 'Two-hour per-execution ceiling based on E83 640-view cost and E84 fixed-batch probe; '
                     'AC,20GiB reserve,6GiB MPS limit. No automatic budget extension or quality-rule change.',
         'limitations': 'Previously consumed E49 regression, not fresh final or tuning data. No parameter/threshold '
                        'selection, retraining, winner sweep or automatic serving promotion. Existing lineage/licence '
                        'limitations remain; even a pass is research-only consumed-test evidence.',
         'downloads': 0, 'training_allowed': False, 'promotion_allowed': False, 'independent_final_passed': False}
    write_once(CONTRACT, c)
    write_once(EVIDENCE / 'e86_regression_contract.json', c | {'regression_contract_sha256': digest(CONTRACT)})
    return {'state': c['state'], 'views': c['views']}


def inputs():
    require_development_pass(); c = read(CONTRACT)
    if digest(CONTRACT) != read(EVIDENCE / 'e86_regression_contract.json')['regression_contract_sha256']:
        raise ValueError('regression contract changed')
    for p, sha in c['inputs'].items():
        if digest(p) != sha:
            raise ValueError('regression input changed: ' + p)
    for path, sha in [(c['manifest'], c['manifest_sha256']), (c['old_scores'], c['old_scores_sha256']),
                      (c['reference']['path'], c['reference']['sha256'])]:
        if digest(path) != sha:
            raise ValueError('frozen benchmark/reference changed')
    rows = read(c['manifest'])['rows']
    baseline_rows = [json.loads(line) for line in Path(c['old_scores']).read_text().splitlines()]
    baseline = {r['record_id']: r for r in baseline_rows}
    if len(rows) != c['views'] or len(baseline) != len(rows) or len(baseline_rows) != len(rows):
        raise ValueError('complete unique4000-view reference required')
    validate_paired_final(rows)
    for r in rows:
        if r['record_id'] not in baseline or any(r[k] != baseline[r['record_id']][k]
                for k in ['parent_id', 'label', 'source', 'condition']):
            raise ValueError('reference/manifest pairing differs')
    return c, rows, baseline


def body_sha(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def check_chunk(a, rows, binding):
    if str(a['binding']) != binding or list(a['record_ids']) != [r['record_id'] for r in rows] or \
            list(a['image_sha256']) != [r['sha256'] for r in rows] or \
            list(a['roles']) != ['CONSUMED_REGRESSION'] * len(rows):
        raise ValueError('regression chunk role/order/image identity differs')
    values = {}
    for key in list(WIDTHS) + ['reference_score', 'score']:
        expected = (len(rows), WIDTHS[key]) if key in WIDTHS else (len(rows),)
        dtype = np.float32 if key in WIDTHS else np.float64
        value = a[key]
        if value.shape != expected or value.dtype != dtype or not np.isfinite(value).all() or \
                body_sha(value) != str(a[key + '_sha256']):
            raise ValueError('regression chunk finite body/hash differs')
        if key not in WIDTHS and np.any((value < 0) | (value > 1)):
            raise ValueError('invalid cached regression probabilities')
        values[key] = value
    return values


def reference_check(actual, expected, tolerance):
    if actual.shape != expected.shape or not len(actual) or not np.isfinite(actual).all() or not np.isfinite(expected).all():
        raise ValueError('complete finite reference scores required')
    error = float(np.max(np.abs(actual - expected)))
    if error > tolerance or any(np.any((actual >= cut) != (expected >= cut)) for cut in [model.AI_CUT, model.REAL_CUT]):
        raise ValueError('E43 score/cut drift prevents interpretation')
    return error


def score():
    c, rows, baseline = inputs()
    if SCORES.exists():
        raise FileExistsError('complete E86 regression scores already locked')
    torch.set_num_threads(2); started = time.monotonic(); deadline = started + c['max_seconds']
    resource_check(deadline); binding = digest(CONTRACT); folder = ROOT / 'regression_chunks'; folder.mkdir(exist_ok=True)
    head = joblib.load(c['reference']['path'])['head']
    with np.load(CANDIDATE, allow_pickle=False) as candidate:
        if str(candidate['contract_sha256']) != digest(FIT_CONTRACT) or str(candidate['reference_sha256']) != c['reference']['sha256']:
            raise ValueError('candidate bindings differ')
        arrays = {k: candidate[k] for k in candidate.files}
    enc = Encoders(); done, chunks = [], {}; maximum_error = 0.; peak = enc.memory(); created = 0
    with threadpool_limits(limits=2):
        for start in range(0, len(rows), c['batch_views']):
            resource_check(deadline); batch = rows[start:start + c['batch_views']]; path = folder / f'{start:05d}.npz'
            for row in batch:
                if digest(row['path']) != row['sha256']:
                    raise ValueError('frozen regression image body changed')
            if not path.exists():
                crops = []
                for row in batch:
                    with Image.open(row['path']) as image:
                        crops.append(dino.texture_crops(dino.transport_image(image, 'clean')))
                values = enc.encode(np.stack(crops))
                values['reference_score'] = head.predict_proba(values['dino'])[:, 1].astype(np.float64)
                values['score'] = model.predict(head, values['dino'], values['clip'], values['dear'], arrays).astype(np.float64)
                reference_check(values['reference_score'], np.array([baseline[r['record_id']]['score'] for r in batch]),
                                c['reference_max_score_error'])
                save_npz(path, **values, **{k + '_sha256': np.array(body_sha(v)) for k, v in values.items()},
                    record_ids=np.array([r['record_id'] for r in batch]), image_sha256=np.array([r['sha256'] for r in batch]),
                    roles=np.repeat('CONSUMED_REGRESSION', len(batch)), binding=np.array(binding))
                created += len(batch)
            with np.load(path, allow_pickle=False) as a:
                values = check_chunk(a, batch, binding)
            actual_old = head.predict_proba(values['dino'])[:, 1]
            actual_new = model.predict(head, values['dino'], values['clip'], values['dear'], arrays)
            if not np.array_equal(actual_old, values['reference_score']) or not np.array_equal(actual_new, values['score']):
                raise ValueError('cached regression model replay differs')
            maximum_error = max(maximum_error, reference_check(actual_old,
                np.array([baseline[r['record_id']]['score'] for r in batch]), c['reference_max_score_error']))
            for index, row in enumerate(batch):
                done.append({k: row[k] for k in ['record_id', 'parent_id', 'label', 'source', 'condition']} |
                    {'sha256': row['sha256'], 'reference_score': float(actual_old[index]),
                     'score': float(actual_new[index]), 'status': 'ok', 'binding': binding})
            chunks[path.name] = digest(path); peak = max(peak, enc.memory())
            if peak > c['mps_driver_limit_bytes']:
                raise RuntimeError('frozen6GiB regression MPS budget exceeded')
            if len(done) % 128 == 0:
                print(json.dumps({'E86_regression_views': len(done), 'total': len(rows), 'created_this_run': created,
                                  'seconds': round(time.monotonic() - started)}), flush=True)
    validate_paired_final(done)
    result = {'state': 'E86_consumed_E49_scores_locked_before_metrics', 'rows': done, 'chunks': chunks,
              'contract_sha256': binding, 'reference_max_score_error': maximum_error, 'reference_decision_changes': 0,
              'seconds': time.monotonic() - started, 'created_views_this_run': created, 'mps_driver_peak_bytes': peak,
              'training_allowed': False, 'downloads': 0}
    write_once(SCORES, result)
    write_once(EVIDENCE / 'e86_regression_scores.json', {k: v for k, v in result.items() if k not in ['rows', 'chunks']} |
               {'scores_sha256': digest(SCORES), 'views': len(done), 'chunks': len(chunks), 'metrics_opened': False})
    return {'views': len(done), 'reference_max_score_error': maximum_error, 'seconds': result['seconds']}


def paired_intervals(labels, sources, old, new):
    delta = (new >= model.AI_CUT).astype(float) - (old >= model.AI_CUT)
    rng = np.random.default_rng(86); samples = []
    for label in [0, 1]:
        groups = sorted(set(sources[labels == label]))
        if not groups:
            raise ValueError('both labels and source groups required')
        counts = np.array([np.sum((labels == label) & (sources == group)) for group in groups])
        changes = np.array([delta[(labels == label) & (sources == group)].sum(axis=0) for group in groups])
        draws = rng.integers(len(groups), size=(20000, len(groups)))
        samples.append(changes[draws].sum(axis=1) / counts[draws].sum(axis=1)[:, None])
    combined = np.concatenate(samples, axis=1)
    return {name: np.quantile(combined[:, j], [.00625, .99375]).tolist()
            for j, name in enumerate(['real_clean', 'real_q75', 'ai_clean', 'ai_q75'])}


def report():
    c, manifest, baseline = inputs()
    if digest(SCORES) != read(EVIDENCE / 'e86_regression_scores.json')['scores_sha256']:
        raise ValueError('locked regression scores changed')
    scored = read(SCORES)['rows']; validate_paired_final(scored); full = evaluate_final(scored)
    parents = sorted({r['parent_id'] for r in manifest}); by_key = {(r['parent_id'], r['condition']): r for r in scored}
    labels = np.array([by_key[p, CONDITIONS[0]]['label'] for p in parents])
    sources = np.array([by_key[p, CONDITIONS[0]]['source'] for p in parents])
    new = np.array([[by_key[p, condition]['score'] for condition in CONDITIONS] for p in parents])
    old = np.array([[baseline[by_key[p, condition]['record_id']]['score'] for condition in CONDITIONS] for p in parents])
    comparisons = {condition: pair_rates(labels, sources, old[:, j], new[:, j]) for j, condition in enumerate(CONDITIONS)}
    changes = {condition: transitions([r | {'reference_score': baseline[r['record_id']]['score']}
               for r in scored if r['condition'] == condition]) for condition in CONDITIONS}
    intervals = paired_intervals(labels, sources, old, new)
    checks = {
        'absolute_selective_20_checks': full['gate']['passed'],
        'real_fpr_lower_both': all(v['new']['real_false_ai'] < v['old']['real_false_ai'] for v in comparisons.values()),
        'ai_recall_no_point_loss_both': all(v['new']['ai_recall'] >= v['old']['ai_recall'] for v in comparisons.values()),
        'supported_ai_sources_no_point_loss': all(g['new_ai_minus_old'] >= 0 for v in comparisons.values()
            for key, g in v['by_source'].items() if key.startswith('1:') and g['parents_or_views'] >= 50),
        'zero_new_ai_misses_every_source_condition': all(v['new_ai_misses'] == 0 for group in changes.values() for v in group.values()),
        'real_improvement_intervals_both': all(intervals['real_' + condition][1] < 0 for condition in ['clean', 'q75']),
        'ai_preservation_intervals_both': all(intervals['ai_' + condition][0] >= 0 for condition in ['clean', 'q75']),
    }
    result = {'state': 'E86_consumed_E49_regression_complete_not_fresh_final', 'checks': checks,
              'preservation_and_absolute_checks_passed': all(checks.values()), 'comparisons': comparisons,
              'transitions': changes, 'source_cluster_paired_intervals': intervals, 'candidate_e49_diagnostics': full,
              'candidate_sha256': digest(CANDIDATE), 'contract_sha256': digest(CONTRACT), 'scores_sha256': digest(SCORES),
              'independent_final_passed': False, 'serving_changed': False, 'winner_selection_allowed': False,
              'limitations': c['limitations']}
    write_once(REPORT, result); write_once(EVIDENCE / 'e86_regression.json', result)
    return {'checks': checks, 'preservation_and_absolute_checks_passed': all(checks.values())}


if __name__ == '__main__':
    def denied(*args, **kwargs):
        raise RuntimeError('E86 regression is offline')
    socket.socket.connect = denied; socket.socket.connect_ex = denied; socket.create_connection = denied
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_HUB_DISABLE_TELEMETRY='1')
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('stage', choices=['freeze', 'score', 'report'])
    ROOT.mkdir(exist_ok=True)
    with (ROOT / 'regression_execution.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        print(json.dumps({'freeze': freeze, 'score': score, 'report': report}[parser.parse_args().stage](), indent=2))
