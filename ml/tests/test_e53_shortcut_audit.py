import numpy as np
import pytest

from experiments.e53_shortcut_audit import summarize


def test_geometry_and_dispersion_are_class_conditional():
    rows = [{'label': 0, 'source': 'mixed', 'width': 224, 'height': 224},
            {'label': 1, 'source': 'mixed', 'width': 1024, 'height': 512}]
    features = np.ones((2, 3072), dtype='float32')
    features[0, 1536:] = 0
    result = summarize(rows, features)
    assert result['class:0']['exact_224_square'] == 1
    assert result['class:0']['near_zero_crop_dispersion'] == 1
    assert result['class:1']['exact_224_square'] == 0
    assert result['class:1']['near_zero_crop_dispersion'] == 0


def test_missing_geometry_rejected():
    with pytest.raises(ValueError):
        summarize([{'label': 'real', 'source_id': 'x'}], np.ones((1, 3072)))


def test_misaligned_features_rejected():
    with pytest.raises(ValueError):
        summarize([], np.ones((1, 3072)))
