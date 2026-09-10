import numpy as np
import pytest

from experiments.e59_features import aggregate, array_sha, read_chunk


def test_mean_population_std():
    raw = np.broadcast_to(np.arange(3, dtype=np.float32)[:, None], (3, 768))
    value = aggregate(raw)
    assert value.shape == (1536,)
    np.testing.assert_array_equal(value[:768], np.ones(768))
    np.testing.assert_allclose(value[768:], np.sqrt(2/3), rtol=1e-6)
    with pytest.raises(ValueError):
        aggregate(np.zeros((2, 768)))


def test_chunk_roundtrip_and_identity(tmp_path):
    raw = np.ones((3, 3, 768), dtype=np.float32)
    features = aggregate(raw)
    p = tmp_path / 'chunk.npz'
    np.savez(p, raw=raw, features=features, binding='b', parent_id='p',
             raw_sha256=array_sha(raw), features_sha256=array_sha(features))
    np.testing.assert_array_equal(read_chunk(p, 'b', 'p'), features)
    with pytest.raises(ValueError):
        read_chunk(p, 'other', 'p')
    with pytest.raises(ValueError):
        read_chunk(p, 'b', 'other')
    np.savez(p, raw=raw+1, features=features, binding='b', parent_id='p',
             raw_sha256=array_sha(raw), features_sha256=array_sha(features))
    with pytest.raises(ValueError):
        read_chunk(p, 'b', 'p')


def test_interrupted_finalization_requires_exact_reconstruction(tmp_path, monkeypatch):
    from experiments import e59_features as f
    root = tmp_path / 'external'
    evidence = tmp_path / 'evidence'
    (root / 'chunks').mkdir(parents=True)
    evidence.mkdir()
    monkeypatch.setattr(f, 'ROOT', root)
    monkeypatch.setattr(f, 'FEATURES', root / 'clip_features.npz')
    monkeypatch.setattr(f, 'RECEIPT', root / 'features.json')
    monkeypatch.setattr(f, 'EVIDENCE', evidence)
    monkeypatch.setattr(f, 'safe', lambda deadline: None)
    raw = np.ones((3, 3, 768), dtype=np.float32)
    features = aggregate(raw)
    np.savez(root / 'chunks/00000.npz', binding='b', parent_id='p', raw=raw,
             features=features, raw_sha256=array_sha(raw), features_sha256=array_sha(features))
    np.savez(f.FEATURES, binding='b', parents=np.asarray(['p']), features=features[None])
    before = f.digest(f.FEATURES)
    f.finalize({'rows': [{'parent_id': 'p'}]}, {}, 'b', 0)
    assert f.digest(f.FEATURES) == before
    assert f.RECEIPT.exists()
    np.savez(f.FEATURES, binding='b', parents=np.asarray(['p']), features=features[None]+1)
    with pytest.raises(ValueError):
        f.finalize({'rows': [{'parent_id': 'p'}]}, {}, 'b', 0)
