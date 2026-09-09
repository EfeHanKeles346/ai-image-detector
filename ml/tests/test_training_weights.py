import numpy as np
import pytest
from experiments.e43_train import parent_source_weights
from pixelproof.training_weights import balanced_parent_weights


def test_exact_legacy_equivalence_with_unequal_views_sources_and_parent_counts():
    rows=[]
    for label in (0,1):
        for source in range(label+2):
            for parent in range(source+3):
                rows.extend([(label,f'{label}:{source}',f'{label}:{source}:{parent}')]*(parent+1))
    rng=np.random.default_rng(53);rng.shuffle(rows)
    y=np.asarray([r[0] for r in rows]);s=np.asarray([r[1] for r in rows]);p=np.asarray([r[2] for r in rows])
    expected=parent_source_weights(y,s,p);actual=balanced_parent_weights(y,s,p)
    assert np.array_equal(actual,expected)
    assert np.isclose(actual[y==0].sum(),actual[y==1].sum())


@pytest.mark.parametrize('y,s,p',[
    ([0],['source'],['parent']),([0,.7,1],['a','b','c'],['a','b','c']),
    ([0,1],['a','b'],['same','same']),([0,1],['','b'],['a','b']),
    ([0,1],['a'],['a','b']),
])
def test_invalid_or_conflicting_metadata_fails_closed(y,s,p):
    with pytest.raises(ValueError):balanced_parent_weights(y,s,p)
