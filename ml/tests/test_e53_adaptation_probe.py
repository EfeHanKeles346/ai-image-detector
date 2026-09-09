import numpy as np
import pytest
import torch

from experiments.e42_features import aggregate_tokens
from experiments.e53_adaptation_probe import differentiable_aggregate


def test_differentiable_aggregation_matches_population_std_layout():
    values = np.random.default_rng(53).normal(size=(6, 4, 8)).astype('float32')
    tokens = torch.tensor(values, requires_grad=True)
    result = differentiable_aggregate(tokens)
    np.testing.assert_allclose(result.detach().numpy(), aggregate_tokens(values, 2), atol=2e-6)
    result.sum().backward()
    assert torch.isfinite(tokens.grad).all()


def test_duplicate_crops_have_finite_gradients():
    tokens = torch.ones((3, 4, 8), requires_grad=True)
    differentiable_aggregate(tokens).sum().backward()
    assert torch.isfinite(tokens.grad).all()


def test_invalid_crop_population_rejected():
    with pytest.raises(ValueError):
        differentiable_aggregate(torch.ones((4, 4, 8)))
