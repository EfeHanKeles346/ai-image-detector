import numpy as np
import pytest
from experiments.e109_diffseg_audit import mask_labels


def test_generator_ids_are_positive_and_empty_not_fabricated():
    x = np.arange(9, dtype=np.uint8).reshape(3, 3)
    assert mask_labels(x).sum() == 8
    assert not mask_labels(np.zeros((3, 3), np.uint8)).any()
    with pytest.raises(ValueError):
        mask_labels(np.full((3, 3), 255, np.uint8))
