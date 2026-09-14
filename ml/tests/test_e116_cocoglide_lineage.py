import pytest
from experiments.e116_cocoglide_lineage import unique_join


def test_join_uses_all_three_exact_pixel_identities_not_filename_or_order():
    a = {'auth_pixel_sha256': 'auth1', 'edit_pixel_sha256': 'edit1', 'mask_pixel_sha256': 'mask1', 'coco_id': 7}
    b = {'auth_pixel_sha256': 'auth2', 'edit_pixel_sha256': 'edit2', 'mask_pixel_sha256': 'mask2', 'coco_id': 9}
    assert [r['coco_id'] for r in unique_join([a, b], [b, a])] == [9, 7]
    with pytest.raises(ValueError): unique_join([a], [a | {'mask_pixel_sha256': 'other'}])
    with pytest.raises(ValueError): unique_join([a, a], [a])
    with pytest.raises(ValueError): unique_join([a], [a, a])
