import numpy as np
import pytest
from experiments.e112_context_features import pooled, verify_chunk
from experiments.e71_features import array_sha


def test_resumed_chunk_rejects_role_identity_corruption_and_changed_aggregate():
    raw = np.random.default_rng(112).normal(size=(4, 3, 768)).astype(np.float32)
    row = {'parent_id': 'parent', 'sha256': 'source', 'cohort': 1}
    a = {'raw': raw, 'binding': 'b', 'parent': 'parent', 'source_sha256': 'source',
         'role': 'TRAIN', 'raw_sha256': array_sha(raw)}
    assert verify_chunk(a, row, 'b', pooled(raw))[1] == 0
    for field, value in [('role', 'CAL'), ('parent', 'other'), ('source_sha256', 'changed'), ('binding', 'other')]:
        with pytest.raises(ValueError): verify_chunk(a | {field: value}, row, 'b', pooled(raw))
    with pytest.raises(ValueError): verify_chunk(a, row, 'b', pooled(raw)+.01)
    corrupt = raw.copy(); corrupt[0, 0, 0] += 1
    with pytest.raises(ValueError): verify_chunk(a | {'raw': corrupt}, row, 'b', pooled(corrupt))


def test_legacy_social_chunk_must_contain_exactly_one_view():
    raw = np.zeros((4, 3, 768), np.float32)
    row = {'parent_id': 'p', 'sha256': 's', 'cohort': 0}
    with pytest.raises(ValueError):
        verify_chunk({'raw': raw, 'binding': 'b', 'parent': 'p', 'source_sha256': 's',
                      'role': 'TRAIN', 'raw_sha256': array_sha(raw)}, row, 'b', pooled(raw))
