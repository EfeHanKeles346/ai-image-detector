"""Reload all 72 saved fold heads and reproduce their already-consumed predictions."""
import json
from pathlib import Path

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

from experiments.e51_pipeline import FEATURES
from experiments.e53_offline import EVIDENCE, contract, digest, fixed_write
from experiments.e53_coverage import ARMS, load as coverage_contract
from experiments.e53_head_controls import feature_map


def compare(saved, replayed, threshold, tolerance=5e-7):
    if len(saved) != len(replayed) or not len(saved) or not np.isfinite(replayed).all():
        raise ValueError('invalid replay population')
    expected = np.asarray([r['score'] for r in saved])
    decisions = np.asarray([r['predicted_ai'] for r in saved])
    if not np.isfinite(expected).all() or not np.array_equal(expected >= threshold, decisions):
        raise ValueError('saved prediction/cut mismatch')
    error = float(np.max(np.abs(expected-replayed)))
    flips = int(np.count_nonzero((replayed >= threshold) != decisions))
    if error > tolerance or flips:
        raise ValueError(f'replay drift: error={error}, flips={flips}')
    return error


def replay():
    base, _ = contract()
    coverage_contract()
    if digest(FEATURES) != base['features_sha256']:
        raise ValueError('source feature archive changed')
    with np.load(FEATURES, allow_pickle=False) as archive:
        mask = (archive['roles'] == 'TRAIN') & np.isin(archive['conditions'], ['clean', 'q75'])
        features = archive['dino'][mask]
        indices = {(str(p), str(c)): i for i, (p, c) in enumerate(zip(
            archive['parents'][mask], archive['conditions'][mask], strict=True))}
    if len(indices) != len(features):
        raise ValueError('duplicate cached observation')
    results, receipts = [], {}
    for name in ('e53_source_held_out_expanded_controls_result.json', 'e53_coverage_result.json'):
        path = EVIDENCE/name
        receipts[name] = digest(path)
        summary = json.loads(path.read_text())
        if len(summary['result_bindings']) != 36:
            raise ValueError('all 36 fold heads per protocol required')
        for filename, expected in summary['result_bindings'].items():
            if digest(filename) != expected:
                raise ValueError('frozen result changed')
            result = json.loads(Path(filename).read_text())
            artifact_path = Path(filename).with_suffix('.joblib')
            if digest(artifact_path) != result['artifact_sha256']:
                raise ValueError('saved fold head changed')
            artifact = joblib.load(artifact_path)
            if artifact['threshold'] != result['threshold']:
                raise ValueError('saved cut changed')
            width, _, l2, _ = ARMS[result['arm']]
            selection = [indices[(o['parent_id'], o['condition'])] for o in result['observations']]
            values = feature_map(features[selection], width, l2)
            with threadpool_limits(limits=2):
                scores = artifact['head'].predict_proba(values)[:, 1]
            error = compare(result['observations'], scores, result['threshold'])
            results.append({'result': filename, 'artifact_sha256': result['artifact_sha256'],
                            'observations': len(scores), 'max_score_error': error, 'decision_flips': 0})
    output = {'state': 'all_72_fold_artifacts_replayed', 'code_sha256': digest(__file__),
              'summary_bindings': receipts, 'feature_sha256': base['features_sha256'],
              'heads': len(results), 'prediction_observations': sum(r['observations'] for r in results),
              'max_score_error': max(r['max_score_error'] for r in results), 'decision_flips': 0,
              'results': results, 'serving_changed': False, 'independent_test_opened': False,
              'limit': 'Serialization/numerical replay only; not new accuracy evidence or another final.'}
    fixed_write(EVIDENCE/'e53_artifact_replay.json', output)
    return {k: v for k, v in output.items() if k != 'results'}


if __name__ == '__main__':
    print(json.dumps(replay(), indent=2))
