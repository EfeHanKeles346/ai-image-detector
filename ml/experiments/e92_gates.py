"""Keep every previous gate and add explicit SID and combined-population checks."""
import numpy as np
from experiments.e85_data import CONDITIONS
from experiments.e86_gates import population_gates as previous_population_gates
from experiments.e49_evaluation import evaluate_condition
from experiments.e64_constrained import AI_CUT


def population_gates(rows, scores, legacy_count, previous_count=12141):
    if scores.shape != (len(rows), 4) or not np.isfinite(scores).all() or \
            not 0 < legacy_count < previous_count < len(rows):
        raise ValueError('complete finite legacy/MIDD/SID TRAIN populations required')
    previous = previous_population_gates(rows[:previous_count], scores[:previous_count], legacy_count)
    real = np.array([r['label'] == 0 for r in rows]); sid = np.arange(len(rows)) >= previous_count
    sources = np.array([r['source'] for r in rows])
    if not np.all(real[sid]) or any(not s.startswith('SID:') for s in sources[sid]):
        raise ValueError('new cohort must be SID REAL')
    detected = scores >= AI_CUT; result = {}
    for i, condition in enumerate(CONDITIONS):
        cameras = {s: float(detected[sid & (sources == s), i].mean()) for s in sorted(set(sources[sid]))}
        rates = {'all_real': float(detected[real, i].mean()), 'new_sid_real': float(detected[sid, i].mean()),
                 'new_sid_by_camera': cameras}
        checks = {'all_previous_population_guards': previous[condition]['passed'],
                  'all_real_lte_10': rates['all_real'] <= .10,
                  'new_sid_real_lte_10': rates['new_sid_real'] <= .10,
                  'worst_sid_camera_lte_20': max(cameras.values()) <= .20}
        result[condition] = {'previous_E86_populations': previous[condition], 'rates': rates,
                             'checks': checks, 'passed': all(checks.values())}
    return result


def full_training_gates(rows, scores, legacy_count, previous_count=12141):
    population_gates(rows, scores, legacy_count, previous_count); result = {}
    for i, condition in enumerate(CONDITIONS):
        views = [{'parent_id': r['parent_id'], 'label': r['label'], 'source': r['source'],
                  'score': float(scores[j, i]), 'role': 'TRAIN'} for j, r in enumerate(rows)]
        result[condition] = {'legacy_TRAIN': evaluate_condition(views[:legacy_count]),
                             'previous_TRAIN': evaluate_condition(views[:previous_count]),
                             'expanded_TRAIN': evaluate_condition(views)}
    return result
