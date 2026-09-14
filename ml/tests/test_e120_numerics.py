import numpy as np
import pytest
import torch
from experiments.e120_inpaint_numerics import check_array, NonfiniteTensor, instrument


def test_nonfinite_output_is_not_a_valid_black_image():
    check_array(np.zeros((4,4,3),dtype=np.uint8),'render')
    for v in (np.nan,np.inf,-np.inf):
        with pytest.raises(NonfiniteTensor,match='before_safety'):check_array([v],'before_safety')


def test_nested_tensor_guard_handles_diffusers_tuple_outputs():
    from types import SimpleNamespace
    class VAE:
        def encode(self,x):return SimpleNamespace(latent_dist=SimpleNamespace(parameters=x))
        def decode(self,x):return (x,)
    pipe=SimpleNamespace(text_encoder=torch.nn.Identity(),unet=torch.nn.Identity(),vae=VAE(),run_safety_checker=lambda *a:None)
    finite,counts=instrument(pipe,torch)
    with pytest.raises(NonfiniteTensor,match='vae_encode'):pipe.vae.encode(torch.tensor([float('nan')]))
    with pytest.raises(NonfiniteTensor,match='vae_decode'):pipe.vae.decode(torch.tensor([float('inf')]))
    with pytest.raises(NonfiniteTensor,match='unet'):pipe.unet(torch.tensor([float('nan')]))
    finite({'nested':(torch.ones(2),)},'latent')
    assert counts['latent']==1
