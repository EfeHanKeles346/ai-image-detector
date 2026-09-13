from io import BytesIO
import numpy as np
from PIL import Image
import pytest
from experiments import e89_features as features


def test_added_social_view_preserves_three_existing_transforms():
    pixels = np.random.default_rng(89).integers(0, 256, (800, 1200, 3), dtype=np.uint8)
    stream = BytesIO(); Image.fromarray(pixels).save(stream, format='PNG'); body = stream.getvalue()
    original = features.crop_views(body, 'SID:test-parent')
    result, sha = features.four_crops(body, 'SID:test-parent')
    assert result.shape == (4, 3, 224, 224, 3)
    assert np.array_equal(result[:3], original)
    assert not np.array_equal(result[2], result[3])
    assert len(sha) == 64


def test_active_predecessor_blocks_gpu_before_validation(monkeypatch, tmp_path):
    monkeypatch.setattr(features, 'DATA_ROOT', tmp_path)
    def forbidden():
        raise AssertionError('encoder/validation reached before preceding experiment completed')
    monkeypatch.setattr(features, 'validate', forbidden)
    monkeypatch.setattr(features, 'Encoders', forbidden)
    with pytest.raises(ValueError, match='finish active E84B'):
        features.extract()


def test_chunk_rejects_development_role():
    row = {'parent_id': 'sid', 'sha256': 'png', 'original_sha256': 'raw'}
    chunk = {'binding': np.array('contract'), 'parents': np.array(['sid']),
             'source_sha256': np.array(['png']), 'original_sha256': np.array(['raw']),
             'roles': np.array(['DEVELOPMENT']), 'conditions': np.array(features.CONDITIONS)}
    with pytest.raises(ValueError, match='TRAIN identities'):
        features.check_chunk(chunk, [row], 'contract')
