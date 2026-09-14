"""Read-only E92 audit: no image reads, model deserialization or new inference."""
from __future__ import annotations

import ast
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import socket
import subprocess


def denied(*args, **kwargs):
    raise RuntimeError('E92 readiness audit is offline')


socket.socket.connect = denied
socket.socket.connect_ex = denied
socket.create_connection = denied

from experiments.e49_evaluation import AI_CUT, REAL_CUT, GATES, evaluate_condition
from experiments.e71_development import validate_pairs
from pixelproof.project_paths import DATA_ROOT, ML_ROOT

REPO = ML_ROOT.parent
EVIDENCE = REPO / 'evidence'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def constants(source):
    tree = ast.parse(source)
    return {node.targets[0].id: ast.literal_eval(node.value)
            for node in tree.body if isinstance(node, ast.Assign)
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id in {'GATES', 'BINARY_THRESHOLD', 'REAL_CUT'}}


def independent_metrics(rows):
    """Direct count and pairwise-rank formulas, independent of benchmark helpers."""
    assert all(r.get('status', 'ok') == 'ok' and math.isfinite(r['score'])
               and 0 <= r['score'] <= 1 for r in rows)
    ai = [r['score'] for r in rows if r['label'] == 1]
    real = [r['score'] for r in rows if r['label'] == 0]
    tp = sum(s >= AI_CUT for s in ai)
    fp = sum(s >= AI_CUT for s in real)
    automatic = [r for r in rows if r['score'] < REAL_CUT or r['score'] >= AI_CUT]
    correct = sum((r['score'] >= AI_CUT) == r['label'] for r in automatic)
    return {'confusion': {'tp': tp, 'fn': len(ai)-tp, 'fp': fp, 'tn': len(real)-fp},
            'auc': sum((a > b) + 0.5*(a == b) for a in ai for b in real) / (len(ai)*len(real)),
            'balanced_accuracy': (tp/len(ai) + 1-fp/len(real))/2,
            'automatic_coverage': len(automatic)/len(rows),
            'covered_accuracy': correct/len(automatic),
            'uncertain_rate': (len(rows)-len(automatic))/len(rows)}


def audit():
    contract = read(EVIDENCE / 'e92_dev_contract.json')
    result = read(EVIDENCE / 'e92_development.json')
    score_path = DATA_ROOT / 'e92/dev_scores.json'
    assert sha(score_path) == result['scores_sha256']
    assert sha(DATA_ROOT / 'e92/correction.npz') == result['candidate_sha256']
    assert sha(contract['manifest']) == contract['manifest_sha256']
    scores = read(score_path)['rows']
    dev = read(contract['manifest'])['rows']
    validate_pairs(scores, dev)
    paths = [DATA_ROOT / 'e54/data_contract_v2.json',
             DATA_ROOT / 'e72/training_manifest.json',
             DATA_ROOT / 'e88/training_manifest.json']
    train = [row for path in paths for row in read(path)['rows']]
    assert len(train) == 12269 and all(r['role'].upper() == 'TRAIN' for r in train)
    assert len(dev) == 320 and all(r['role'] == 'DEVELOPMENT' for r in dev)
    intersections = {}
    for key in ['parent_id', 'sha256', 'pixel_sha256', 'original_sha256']:
        lhs = {r[key] for r in train if r.get(key)}
        rhs = {r[key] for r in dev + scores if r.get(key)}
        intersections[key] = {'train_available': sum(bool(r.get(key)) for r in train),
                              'dev_parents_available': sum(bool(r.get(key)) for r in dev),
                              'matches': len(lhs & rhs)}
        assert not lhs & rhs, f'observed TRAIN/DEV {key} intersection'
    old_code = subprocess.check_output(
        ['git', 'show', 'b173d5b:ml/experiments/e49_evaluation.py'], cwd=REPO, text=True)
    current = constants((ML_ROOT / 'experiments/e49_evaluation.py').read_text())
    assert constants(old_code) == current
    conditions = {}
    for condition in contract['conditions']:
        selected = [r for r in scores if r['condition'] == condition]
        metrics = evaluate_condition(selected)
        assert metrics == result['reports'][condition]['new']
        manual = independent_metrics(selected)
        assert manual['confusion'] == metrics['binary_metrics']['confusion']
        assert abs(manual['auc']-metrics['binary_metrics']['roc_auc']) < 1e-12
        assert abs(manual['balanced_accuracy']-metrics['binary_metrics']['balanced_accuracy']) < 1e-12
        assert all(abs(manual[k]-metrics['selective'][k]) < 1e-12
                   for k in ['automatic_coverage', 'covered_accuracy', 'uncertain_rate'])
        scenes = []
        for scene in sorted({r['scene_group'] for r in selected if r['label'] == 0}):
            rows = [r for r in selected if r['label'] == 0 and r['scene_group'] == scene]
            errors = sum(r['score'] >= AI_CUT for r in rows)
            scenes.append({'scene': scene, 'parents': len(rows), 'false_ai': errors,
                           'false_ai_rate': errors / len(rows)})
        sensitivity = metrics['binary_rates']['pooled_ai_recall']
        fpr = metrics['binary_rates']['pooled_real_false_ai']
        n = metrics['selective']['automatic_rows']
        correct = round(n * metrics['selective']['covered_accuracy'])
        conditions[condition] = {
            'metric_replay_exact': True, 'gate': metrics['gate'],
            'independent_count_and_pairwise_rank_check': manual,
            'binary_metrics': {k: metrics['binary_metrics'][k] for k in
                               ['confusion', 'roc_auc', 'balanced_accuracy']},
            'rates': metrics['binary_rates'], 'selective': metrics['selective'],
            'ai_below_real_cut': sum(r['label'] == 1 and r['score'] < REAL_CUT for r in selected),
            'scenes': scenes,
            'scene_macro_real_fpr_descriptive_only': sum(s['false_ai_rate'] for s in scenes) / len(scenes),
            'worst_scene_real_fpr_descriptive_only': max(s['false_ai_rate'] for s in scenes),
            'covered_accuracy_one_more_error_same_coverage': (correct - 1) / n,
            'prevalence_scenarios_not_deployment_estimates': [
                {'assumed_ai_prevalence': p,
                 'implied_ai_positive_precision_if_rates_transfer': sensitivity * p / (sensitivity * p + fpr * (1-p))}
                for p in [.01, .10, .50]],
        }
    return {
        'state': 'E92_locked_evidence_readiness_audit_complete',
        'model_inference': 0, 'training_runs': 0, 'image_reads': 0, 'downloads': 0,
        'gates_unchanged_since': 'b173d5b (2026-09-04)', 'gates': GATES,
        'acceptance_complete': result['passes_limited_dev_screen'],
        'train_parents': len(train), 'train_label_counts': dict(Counter(r['label'] for r in train)),
        'train_source_counts': dict(Counter(r['source'] for r in train)),
        'dev_parent_count': len(dev), 'dev_view_count': len(scores),
        'dev_real_scene_count': len({r['scene_group'] for r in dev if r['label'] == 0}),
        'dev_source_counts': dict(Counter(r['source'] for r in dev)),
        'observed_identity_intersections': intersections,
        'conditions': conditions,
        'limitations': [
            'Zero observed exact intersections does not prove semantic, prompt or pretrained-corpus separation.',
            'Canonical pixel hashes are unavailable for some legacy TRAIN rows; no new perceptual audit is claimed.',
            'Scene metrics reuse locked consumed DEV scores: descriptive sensitivity only, no new model selection or confidence guarantee.',
            'Prevalence scenarios assume unchanged TPR/FPR and are algebraic illustrations, not market measurements.',
            'No independent E92 final or end-to-end serving benchmark was performed.'],
        'source_sha256': {str(p.relative_to(REPO)) if p.is_relative_to(REPO) else str(p): sha(p)
                          for p in [*paths, Path(contract['manifest']), score_path,
                                    EVIDENCE/'e92_development.json', Path(__file__)]},
    }


if __name__ == '__main__':
    print(json.dumps(audit(), indent=2, sort_keys=True))
