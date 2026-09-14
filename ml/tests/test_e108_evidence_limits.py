import pytest
from experiments.e108_evidence_limits import bounds, assess, zero_error_size


def test_exact_limits_and_prospective_sample_size():
    assert bounds(0, 10)['upper'] == pytest.approx(1-.05**.1)
    assert bounds(10, 10)['lower'] == pytest.approx(.05**.1)
    assert zero_error_size() == 299
    assert bounds(0, 298)['upper'] > .01 >= bounds(0, 299)['upper']
    assert zero_error_size(claims=20) == 597


def test_perfect_consumed_or_dependent_results_cannot_certify():
    assert not assess(0, 10000, .01, independent=True, fresh=False, complete=True)['proof_supported']
    assert not assess(0, 10000, .01, independent=False, fresh=True, complete=True)['proof_supported']
    assert not assess(0, 10000, .01, independent=True, fresh=True, complete=False)['proof_supported']
    assert assess(0, 299, .01, independent=True, fresh=True, complete=True)['proof_supported']


@pytest.mark.parametrize('k,n', [(1, 0), (-1, 10), (11, 10), (1.5, 10), (True, 10)])
def test_invalid_evidence_rejected(k, n):
    with pytest.raises(ValueError):
        bounds(k, n)
