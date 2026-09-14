from io import BytesIO
import numpy as np
from PIL import Image
import pytest
from experiments import e101_features as current
from experiments import e89_features as previous


def test_new_camera_cache_preserves_the_frozen_four_view_representation():
    pixels = np.random.default_rng(101).integers(0, 256, (800, 1200, 3), dtype=np.uint8)
    stream = BytesIO(); Image.fromarray(pixels).save(stream, format='JPEG', quality=95)
    a, ah = current.four_crops(stream.getvalue(), 'MIDD:test:1')
    b, bh = previous.four_crops(stream.getvalue(), 'MIDD:test:1')
    assert np.array_equal(a, b) and ah == bh


def test_cache_rejects_nontrain_role_and_changed_original_identity():
    row = {'parent_id': 'MIDD:test:1', 'sha256': 'jpeg', 'original_sha256': 'jpeg'}
    base = {'binding': np.array('contract'), 'parents': np.array(['MIDD:test:1']),
            'source_sha256': np.array(['jpeg']), 'original_sha256': np.array(['jpeg']),
            'roles': np.array(['TRAIN']), 'conditions': np.array(current.CONDITIONS)}
    for change in [{'roles': np.array(['CALIBRATION'])}, {'original_sha256': np.array(['changed'])}]:
        with pytest.raises(ValueError, match='TRAIN identities'):
            current.check_chunk(base | change, [row], 'contract')
