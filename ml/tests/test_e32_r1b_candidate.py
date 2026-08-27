from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from pixelproof.e32_r1b_candidate import ARTIFACT_SHA256, E32R1bCandidate


def test_r1b_candidate_artifact_identity_is_full_sha() -> None:
    assert len(ARTIFACT_SHA256) == 64


def test_r1b_single_image_path_uses_standardized_input_and_probability_head() -> None:
    class Processor:
        def __init__(self):
            self.images = None

        def __call__(self, *, images, return_tensors):
            self.images = images
            assert return_tensors == "pt"
            return {"pixel_values": torch.zeros((1, 3, 224, 224))}

    class Model:
        def vit(self, *, pixel_values):
            assert pixel_values.shape == (1, 3, 224, 224)
            return SimpleNamespace(last_hidden_state=torch.ones((1, 2, 3)))

    class Head:
        def predict_proba(self, values):
            assert values.shape == (1, 3)
            return np.asarray([[0.18, 0.82]])

    candidate = E32R1bCandidate.__new__(E32R1bCandidate)
    candidate.model = Model()
    candidate.processor = Processor()
    candidate.device = torch.device("cpu")
    candidate.head = Head()
    candidate.threshold = 0.125

    score = candidate.score_image(Image.new("RGB", (480, 320), "white"))

    assert score == pytest.approx(0.82)
    assert len(candidate.processor.images) == 1
    assert candidate.processor.images[0].size == (224, 224)
