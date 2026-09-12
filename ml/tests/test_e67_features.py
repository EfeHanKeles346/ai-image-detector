import hashlib
import numpy as np
import pytest
from experiments.e67_features import blur_crops,load_crops


def test_blur_preserves_crop_identity_order_and_constant_colours():
    crops=np.empty((3,3,224,224,3),dtype=np.uint8)
    for condition in range(3):
        for crop in range(3):crops[condition,crop]=20*(3*condition+crop)
    assert np.array_equal(blur_crops(crops),crops)


def test_blur_changes_edges_without_cross_crop_mixing_or_input_mutation():
    crops=np.zeros((3,3,224,224,3),dtype=np.uint8);crops[1,2,:,112:]=255;copy=crops.copy()
    out=blur_crops(crops)
    assert np.array_equal(crops,copy)
    assert 0<int(out[1,2,100,111,0])<255
    assert not out[0].any() and not out[2].any()
    with pytest.raises(ValueError):blur_crops(crops.astype(np.float32))


def test_crop_identity_and_population_checked_before_use(tmp_path):
    p=tmp_path/'crops.npz';np.savez(p,crops=np.zeros((3,3,224,224,3),dtype=np.uint8),binding=np.array('population'))
    r={'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
    assert load_crops(r,'population').shape==(3,3,224,224,3)
    with pytest.raises(ValueError,match='binding'):load_crops(r,'different')
    p.write_bytes(b'replaced')
    with pytest.raises(ValueError,match='body changed'):load_crops(r,'population')
