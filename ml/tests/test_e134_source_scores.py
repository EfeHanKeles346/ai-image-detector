import numpy as np
import pytest
from experiments.e134_source_scores import distribution,paired


def test_fixed_cut_includes_ties_and_preserves_quantiles():
    d=distribution([0,.25,.5,.75,1])
    assert d['alerts_at_fixed_half']==3
    assert d['quantiles'][3]==.5
    assert d['mean']==.5


def test_paired_error_direction_depends_on_class():
    before=[.1,.7,.3,.9];after=[.8,.2,.3,.9]
    for label in (0,1):
        r=paired(before,after,label)
        assert r['new_errors']==1 and r['rescued_errors']==1
        assert r['to_alert']==1 and r['to_no_alert']==1
    assert paired([.2],[.8],0)['new_errors']==1
    assert paired([.2],[.8],1)['rescued_errors']==1
    assert paired([.8],[.8],1)['mean_score_shift']==0


@pytest.mark.parametrize('value',[[],[np.nan],[1.1],[-.1],[[.1,.2]]])
def test_invalid_scores_fail_closed(value):
    with pytest.raises(ValueError):distribution(value)


def test_pairs_require_same_size_and_binary_class():
    with pytest.raises(ValueError):paired([.1],[.2,.3],0)
    with pytest.raises(ValueError):paired([.1],[.2],True)
