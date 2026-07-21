import torch

from pixelproof.models import create_model


def test_small_cnn_returns_one_logit_per_image():
    model = create_model("small_cnn")
    assert model(torch.randn(4, 3, 32, 32)).shape == (4,)
