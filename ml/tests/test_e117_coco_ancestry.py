import pytest
from experiments.e117_coco_ancestry import original_id


def test_coco_numeric_identity_joins_zero_padding_but_rejects_other_split_or_mismatch():
    row = {'parent_id': 'dda-coco:000000139', 'member': 'DDA-COCO/FLUX.1/val2017/000000000139.jpg'}
    assert original_id(row) == 139
    assert original_id(row | {'member': 'val2017/000000000139.jpg'}) == 139
    with pytest.raises(ValueError): original_id(row | {'parent_id': 'dda-coco:140'})
    with pytest.raises(ValueError): original_id(row | {'member': 'DDA-COCO/FLUX.1/train2017/000000000139.jpg'})
