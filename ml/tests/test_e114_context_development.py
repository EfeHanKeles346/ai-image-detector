import copy
import pytest
from experiments.e114_context_development import eligible, comparison, pair_identity
from experiments.e113_context_fit import BRANCHES
from experiments.e49_evaluation import CONDITIONS


def report(passing):
    return {'state': 'E113_paired_TRAIN_ablation_complete', 'reports': {name: {
        'dev_scoring_permitted': name in passing,
        'state': 'E113_TRAIN_guard_passed' if name in passing else 'E113_TRAIN_guard_failed',
        'solver': {'success': True, 'max_constraint_violation': 0.},
        'runtime': {'passed': True}, 'retention': {'passes_train_retention': True}}
        for name in BRANCHES}}


def test_failed_or_incomplete_training_cannot_open_dev():
    with pytest.raises(ValueError, match='Neither'): eligible(report([]))
    incomplete = report(BRANCHES); del incomplete['reports'][BRANCHES[1]]
    with pytest.raises(ValueError, match='Both'): eligible(incomplete)
    assert eligible(report(BRANCHES)) == list(BRANCHES)
    assert eligible(report([BRANCHES[1]])) == [BRANCHES[1]]


@pytest.mark.parametrize('field,value', [('success', False), ('max_constraint_violation', .01),
                                       ('max_constraint_violation', float('nan'))])
def test_permission_flag_does_not_override_failed_solver(field, value):
    r = report(BRANCHES); r['reports'][BRANCHES[0]]['solver'][field] = value
    with pytest.raises(ValueError, match='Inconsistent'): eligible(r)


def test_equal_pooled_counts_cannot_hide_a_lost_ai_or_new_real_error():
    rows = []
    for condition in CONDITIONS:
        for i, (label, previous, new) in enumerate([(0, .001, .9), (0, .9, .001),
                                                   (1, .9, .001), (1, .001, .9)]):
            rows.append({'parent_id': str(i), 'role': 'DEVELOPMENT', 'source': str(label),
                'label': label, 'condition': condition, 'score': new,
                'reference_score': previous, 'E92_score': previous, 'E103_score': previous})
    for result in comparison(rows).values():
        assert not result['passed']
        assert not result['checks']['zero_lost_E43_AI']
        assert not result['checks']['zero_new_E92_REAL_errors']
        assert not result['checks']['zero_new_E103_REAL_errors']


def test_predecessor_transport_identity_must_be_preserved():
    row = dict(parent_id='p', condition='publisher_original', sha256='body',
               role='DEVELOPMENT', source='s', label=0, record_id='p|original')
    pair_identity([row], [copy.deepcopy(row)])
    with pytest.raises(ValueError): pair_identity([row], [row | {'sha256': 'different'}])
