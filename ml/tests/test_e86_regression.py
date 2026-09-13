import numpy as np
import pytest
from experiments import e86_regression as regression


@pytest.mark.parametrize('flags,checks', [
    (False, {'original': {'zero_new_ai_misses': True}}),
    (True, {'original': {'zero_new_ai_misses': False}}),
])
def test_failed_development_blocks_e49_before_any_benchmark_read(monkeypatch, flags, checks):
    monkeypatch.setattr(regression, 'validate_dev', lambda: None)
    monkeypatch.setattr(regression, 'digest', lambda _: 'same')
    def read(path):
        assert path == regression.DEV_REPORT, 'benchmark accessed before guard'
        return {'passes_limited_dev_screen': flags, 'consumed_regression_may_be_registered': flags, 'checks': checks}
    monkeypatch.setattr(regression, 'read', read)
    with pytest.raises(ValueError, match='before any E49 access'):
        regression.freeze()


def test_reference_tiny_cut_flip_rejected_despite_small_absolute_error():
    cut = regression.model.AI_CUT
    with pytest.raises(ValueError, match='score/cut drift'):
        regression.reference_check(np.array([cut + 1e-10]), np.array([cut - 1e-10]), 5e-5)


def test_float64_score_hash_detects_change_below_float32_resolution():
    first = np.array([.5], dtype=np.float64)
    second = np.array([.5 + 1e-12], dtype=np.float64)
    assert np.array_equal(first.astype(np.float32), second.astype(np.float32))
    assert regression.body_sha(first) != regression.body_sha(second)


def test_paired_intervals_preserve_zero_ai_change_and_real_improvement():
    labels = np.array([0, 0, 1, 1]); sources = np.array(['r1', 'r2', 'a1', 'a2'])
    old = np.full((4, 2), .9); new = old.copy(); new[:2] = 0
    intervals = regression.paired_intervals(labels, sources, old, new)
    assert intervals == {'real_clean': [-1., -1.], 'real_q75': [-1., -1.],
                         'ai_clean': [0., 0.], 'ai_q75': [0., 0.]}
