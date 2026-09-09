import pytest

from experiments.e53_diagnostics import describe, transitions


def row(label, score, source='x', cut=.5):
    return {'label': label, 'score': score, 'source': source, 'predicted_ai': score >= cut}


def test_rates_are_conditional_on_label_even_for_mixed_source():
    result = describe([row(0, .1), row(0, .8), row(1, .9), row(1, .7)], .5)
    assert result['fixed_cut_real_fpr'] == .5
    assert result['fixed_cut_ai_recall'] == 1
    assert result['diagnostic_tpr_at_fpr_le_10pct'] == .5
    assert result['by_class']['real']['sources']['x']['parents'] == 2


def test_ties_cannot_manufacture_a_low_false_positive_operating_point():
    result = describe([row(0, .5), row(1, .5)], .5)
    assert result['diagnostic_tpr_at_fpr_le_10pct'] == 0


@pytest.mark.parametrize('rows', [[row(0, .1)], [row(0, .1), row(1, float('nan'))]])
def test_invalid_population_rejected(rows):
    with pytest.raises(ValueError):
        describe(rows, .5)


def test_cut_mismatch_rejected():
    with pytest.raises(ValueError):
        describe([row(0, .2), row(1, .8)], .9)


def test_transitions_separate_real_rescue_from_ai_loss():
    old = {'r': row(0, .8), 'a': row(1, .8)}
    new = {'r': row(0, .2), 'a': row(1, .2)}
    result = transitions(new, old)
    assert result['0:x'] == {'parents': 1, 'rescued': 1, 'new_errors': 0}
    assert result['1:x'] == {'parents': 1, 'rescued': 0, 'new_errors': 1}
    with pytest.raises(ValueError):
        transitions({'r': new['r']}, old)
