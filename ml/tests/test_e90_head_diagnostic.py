import numpy as np
import torch
from experiments.e82_representation import build_network, export_network
from experiments.e90_head_diagnostic import scalar_logits, summarize


def test_saved_scalar_head_matches_complete_network_cpu64():
    torch.manual_seed(82); network = build_network().eval()
    values = np.random.default_rng(90).normal(size=(8, 385))
    arrays = export_network(network) | {'input_center': np.zeros(385), 'input_scale': np.ones(385)}
    network = network.double()
    with torch.inference_mode():
        expected = network(torch.from_numpy(values)).numpy().ravel()
    assert np.allclose(scalar_logits(values, arrays), expected, rtol=0, atol=1e-12)


def test_native_zero_boundary_has_fixed_sign_without_tuning():
    report = summarize(np.array([-1., 0., 1.]))
    assert report['native_BCE_AI_sign_count'] == 2
    assert report['native_BCE_REAL_sign_count'] == 1
    assert report['logit_min_median_max'] == [-1., 0., 1.]
