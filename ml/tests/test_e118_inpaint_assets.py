import hashlib
import pytest
from experiments.e118_inpaint_assets import git_blob, verify


def test_asset_publisher_identities_reject_equal_size_corruption(tmp_path):
    p = tmp_path/'asset'; p.write_bytes(b'abc')
    for row in ({'size': 3, 'blobId': git_blob(b'abc')},
                {'size': 3, 'lfs': {'sha256': hashlib.sha256(b'abc').hexdigest()}}):
        assert verify(p, row) == hashlib.sha256(b'abc').hexdigest()
        p.write_bytes(b'abd')
        with pytest.raises(ValueError, match='differs'): verify(p, row)
        p.write_bytes(b'abc')
    with pytest.raises(ValueError, match='byte count'): verify(p, {'size': 4})
