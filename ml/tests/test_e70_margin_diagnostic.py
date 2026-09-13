import numpy as np
import pytest
from experiments.e70_margin_diagnostic import margins


def test_rescues_do_not_cancel_new_misses_and_only_previous_catches_count():
    result=margins([1.,2.,-3.],[1e-7,-.5,5.])
    assert result['previously_caught']==2 and result['new_misses']==1
    assert result['near_boundary_after']['1e-06']==1
    assert result['new_quantiles'][-1]==1e-7


def test_empty_and_invalid_margin_groups_are_explicit():
    result=margins([-1.],[-2.])
    assert result['old_quantiles'] is None and result['mean_shift'] is None
    with pytest.raises(ValueError):margins([1.],[np.nan])
    with pytest.raises(ValueError):margins([1.],[1.,2.])
