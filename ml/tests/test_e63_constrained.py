import numpy as np
import pytest
from experiments.e63_constrained import kernels


def test_gaussian_basis_is_symmetric_and_decays_with_distance():
    x = np.array([[0., 0.], [1., 0.], [3., 0.]])
    k = kernels(x, x, 1.)
    np.testing.assert_allclose(k, k.T)
    np.testing.assert_allclose(np.diag(k), 1.)
    assert 0 < k[0, 2] < k[0, 1] < k[0, 0]


@pytest.mark.parametrize('sigma', [0., -1., np.nan, np.inf])
def test_degenerate_bandwidth_fails_closed(sigma):
    with pytest.raises(ValueError, match='bandwidth'):
        kernels(np.zeros((2, 2)), np.zeros((1, 2)), sigma)
