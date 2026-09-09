import numpy as np
import pytest
from experiments.e53_crop_dedup import unique_crops


def test_stable_dedup_exactly_reconstructs_original_batch():
    a=np.zeros((4,4,3),dtype=np.uint8);b=a.copy();b[0,0,0]=1
    batch=np.stack([a,b,a,b,a]);unique,inverse=unique_crops(batch)
    assert len(unique)==2
    assert inverse.tolist()==[0,1,0,1,0]
    assert np.array_equal(unique[inverse],batch)


@pytest.mark.parametrize('bad',[np.zeros((0,4,4,3),dtype=np.uint8),np.zeros((4,4,3),dtype=np.uint8),np.zeros((2,4,4,3))])
def test_invalid_crop_arrays_fail_closed(bad):
    with pytest.raises(ValueError):unique_crops(bad)
