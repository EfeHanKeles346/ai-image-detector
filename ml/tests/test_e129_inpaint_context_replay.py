import pytest
from experiments.e129_inpaint_context_replay import paired_outcomes


def cases():
    return [dict(index=i, seed=119000+i, branch='fp16_sdpa', passed=i<11) for i in range(16)]


def test_net_success_count_cannot_hide_new_failed_attempts():
    previous = cases(); current = cases(); current[0]['passed'] = False; current[15]['passed'] = True
    result = paired_outcomes(previous, current)
    assert result['context_passed'] == result['previous_passed'] == 11
    assert result['rescued_attempts'] == result['new_failed_attempts'] == 1
    assert not result['full16_engineering_gate_passed']
    assert paired_outcomes(previous, [r | {'passed': True} for r in current])['full16_engineering_gate_passed']


@pytest.mark.parametrize('field,value', [('seed', 1), ('branch', 'fp32_sliced'), ('passed', 1), ('index', 99)])
def test_paired_recipe_cannot_change_or_drop_failures(field, value):
    previous = cases(); current = cases(); current[0][field] = value
    with pytest.raises(ValueError): paired_outcomes(previous, current)
    with pytest.raises(ValueError): paired_outcomes(previous, previous[:-1])
