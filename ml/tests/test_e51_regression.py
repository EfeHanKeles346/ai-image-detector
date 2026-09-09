import numpy as np
import pytest

from experiments.e51_regression import compare, read_pinned


def test_identical_predictions_have_zero_paired_deltas():
    labels = np.r_[np.zeros(10,dtype=int),np.ones(10,dtype=int)]
    scores = np.r_[np.full(10,.1),np.full(10,.9)]
    groups = np.r_[np.repeat('camera',10),np.repeat('generator',10)]
    result = compare(labels,scores,scores,groups,.5,.5,.2,.2)
    assert result['old']==result['new']
    assert result['new']['false_ai_count']==result['new']['missed_ai_count']==0
    assert all(v==[0.,0.] for v in result['paired_delta_95pct_intervals'].values())


def test_changed_pinned_file_fails(tmp_path):
    path = tmp_path/'artifact'; path.write_bytes(b'changed')
    with pytest.raises(ValueError): read_pinned(path,'wrong')
