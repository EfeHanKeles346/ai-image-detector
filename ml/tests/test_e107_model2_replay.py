import numpy as np
import pytest
from experiments.e107_model2_replay import positions, dense_map, metrics, baselines


def test_irregular_edges_are_all_covered_and_constant_preserved():
    size = (301, 257)
    boxes = [(x, y) for y in positions(size[1]) for x in positions(size[0])]
    assert positions(301)[-1] == 173
    assert np.allclose(dense_map(np.full(len(boxes), .3), boxes, size), .3)
    with pytest.raises(ValueError, match='Uncovered'):
        dense_map([.3], [(0, 0)], size)


def test_tiny_mask_retained_without_oracle_threshold():
    truth = np.zeros((128, 128), bool); truth[0, 0] = True
    result = metrics(truth, np.full(truth.shape, .49))
    assert result['iou_at_fixed_half'] == 0
    assert result['auc'] == .5
    assert result['ap'] == pytest.approx(1/16384)
    assert metrics(truth, truth.astype(float))['iou_at_fixed_half'] == 1


def test_random_baseline_reproducible_and_no_mask_input():
    a, b = baselines((131, 129), 'parent'), baselines((131, 129), 'parent')
    assert np.array_equal(a['seeded_random_tiles'], b['seeded_random_tiles'])
    assert not np.array_equal(a['seeded_random_tiles'], baselines((131, 129), 'other')['seeded_random_tiles'])
