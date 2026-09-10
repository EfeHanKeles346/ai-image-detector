import pytest

from experiments.e58_score_diagnosis import envelope, transitions


def test_perfect_and_reversed():
    assert envelope([0, 0, 1, 1], [0, .1, .8, 1])['oracle_95recall_10fpr_feasible']
    bad = envelope([0, 0, 1, 1], [1, .8, .1, 0])
    assert bad['max_ai_recall_at_real_fpr_10pct'] == 0
    assert bad['min_real_fpr_at_ai_recall_80pct'] == 1


def test_ties_indivisible_and_permutation_invariant():
    a = envelope([0, 1, 1, 0], [.5, .5, .5, .5])
    assert a == envelope([1, 0, 0, 1], [.5, .5, .5, .5])
    assert a['max_ai_recall_at_real_fpr_10pct'] == 0
    assert a['min_real_fpr_at_ai_recall_95pct'] == 1


def test_invalid_scores():
    with pytest.raises(ValueError):
        envelope([0, 1], [0, float('nan')])
    with pytest.raises(ValueError):
        envelope([1, 1], [.2, .8])


def test_paired_transitions_and_label_integrity():
    b = [dict(parent_id='a', condition='clean', source='s', label=0, predicted_ai=True),
         dict(parent_id='b', condition='clean', source='s', label=0, predicted_ai=False)]
    c = [dict(b[0], predicted_ai=False), dict(b[1], predicted_ai=True)]
    out = transitions(b, c)[0]
    assert out['rescued_errors'] == out['new_errors'] == 1
    with pytest.raises(ValueError):
        transitions(b, [dict(c[0], label=1), c[1]])
