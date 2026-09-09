import numpy as np
import pytest

from experiments.e51_development_score import intervals, validate_rows


def test_missing_development_rows_fail_closed():
    with pytest.raises(ValueError): validate_rows([])


def test_clustered_prompt_intervals_preserve_perfect_predictions():
    rows = [{'label':0,'transport_cell':c} for c in ('unaltered','postprocessed') for _ in range(10)]
    rows += [{'label':1,'prompt_category':str(c),'prompt_ordinal':p}
             for c in range(8) for p in range(20) for _ in range(5)]
    scores = np.asarray([r['label'] for r in rows],dtype=float)
    result = intervals(rows,scores,.5)
    assert result=={'real_false_ai':[0.,0.],'ai_recall':[1.,1.],'balanced_accuracy':[1.,1.]}
