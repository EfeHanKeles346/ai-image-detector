import numpy as np
import pytest
from experiments import e71_features as f


def payload():
    raw = np.random.default_rng(71).normal(size=(3, 3, 768)).astype(np.float32)
    features = f.aggregate(raw)
    return dict(raw=raw, features=features, binding=np.array('legacy'), parent_id=np.array('p'),
                raw_sha256=np.array(f.array_sha(raw)), features_sha256=np.array(f.array_sha(features)))


def test_parent_aggregation_has_expected_mean_and_population_deviation():
    raw = np.broadcast_to(np.arange(3, dtype=np.float32)[None, :, None], (3, 3, 768)).copy()
    result = f.aggregate(raw)
    np.testing.assert_allclose(result[:, :768], 1)
    np.testing.assert_allclose(result[:, 768:], np.sqrt(2/3), rtol=1e-6)
    with pytest.raises(ValueError): f.aggregate(raw.astype(np.float64))
    with pytest.raises(ValueError): f.aggregate(raw*np.nan)


def test_chunk_reuse_validates_body_parent_binding_and_keeps_bytes(tmp_path):
    p = tmp_path/'chunk.npz'; f.save_npz(p, **payload()); before = f.digest(p)
    raw, features = f.read_chunk(p, 'legacy', 'p', before)
    np.testing.assert_array_equal(f.aggregate(raw), features)
    assert f.digest(p) == before
    with pytest.raises(ValueError, match='parent/binding'): f.read_chunk(p, 'legacy', 'other')
    with pytest.raises(ValueError, match='parent/binding'): f.read_chunk(p, 'other', 'p')
    with p.open('ab') as stream: stream.write(b'changed')
    with pytest.raises(ValueError, match='body changed'): f.read_chunk(p, 'legacy', 'p', before)


def test_self_hash_does_not_hide_inconsistent_aggregation(tmp_path):
    arrays = payload(); arrays['features'] = arrays['features']+.25
    arrays['features_sha256'] = np.array(f.array_sha(arrays['features']))
    p = tmp_path/'bad.npz'; f.save_npz(p, **arrays)
    with pytest.raises(ValueError, match='aggregation'): f.read_chunk(p, 'legacy', 'p')


def test_chunk_raw_digest_and_immutable_output(tmp_path):
    arrays = payload(); arrays['raw'][0, 0, 0] += 1
    p = tmp_path/'bad.npz'; f.save_npz(p, **arrays)
    with pytest.raises(ValueError, match='array digest'): f.read_chunk(p, 'legacy', 'p')
    with pytest.raises(FileExistsError): f.save_npz(p, **payload())


def test_parity_selection_covers_cached_sources_only():
    rows = [dict(parent_id=f'p{i}', source=s) for i,s in enumerate(['a','a','b','c'])]
    chosen = f.choose_parity(rows, {'0':{}, '1':{}, '2':{}})
    assert len(chosen) == 2 and 'p2' in chosen and 'p3' not in chosen
    assert chosen == f.choose_parity(rows, {'2':{}, '1':{}, '0':{}})


def test_complete_both_class_train_population_is_required():
    rows = [dict(parent_id=f'p{i}', label=int(i<4595), role='TRAIN') for i in range(11630)]
    f.check_population(rows)
    with pytest.raises(ValueError): f.check_population(rows[:-1])
    rows[-1]['role'] = 'DEVELOPMENT'
    with pytest.raises(ValueError): f.check_population(rows)
    rows[-1]['role'] = 'TRAIN'; rows[0]['label'] = 0
    with pytest.raises(ValueError, match='both full TRAIN classes'): f.check_population(rows)
