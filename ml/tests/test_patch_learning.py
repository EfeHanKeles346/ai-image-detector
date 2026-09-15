import numpy as np
import pytest
from pixelproof.patch_learning import source_folds, pure_patches, training_arrays, fit_normalizer


def rows():
    return [dict(index=i, parent_id=str(i), source=f'sensor{i//2}',
                 scene_group=f'scene{i}', source_body_sha256=f'body{i}') for i in range(4)]


def test_sensor_and_transitive_ancestry_never_cross_folds():
    r = rows()
    assert source_folds(r).tolist() == [0, 0, 1, 1]
    r += [dict(index=4, parent_id='4', source='sensor2', scene_group='scene2', source_body_sha256='body4')]
    assert source_folds(r).tolist() == [0, 0, 1, 1, 1]
    r[2]['source_body_sha256'] = 'body0'
    with pytest.raises(ValueError, match='No source-disjoint'):
        source_folds(r)


def test_missing_or_duplicate_parent_ancestry_fails():
    r = rows(); r[1]['parent_id'] = r[0]['parent_id']
    with pytest.raises(ValueError, match='Unique'):
        source_folds(r)
    r = rows(); r[1]['scene_group'] = ''
    with pytest.raises(ValueError, match='Complete'):
        source_folds(r)


def test_training_discards_entire_boundary_cells():
    mask = np.zeros((512, 512), dtype=bool); mask[128:256, 128:256] = True
    pos, neg = pure_patches(mask)
    assert pos.sum() == 36
    assert pos[9:15, 9:15].all()
    assert not pos[8].any() and not pos[15].any()
    assert not neg[7:17, 7:17].any()
    assert not (pos & neg).any()
    with pytest.raises(ValueError):
        pure_patches(np.zeros((512, 512), dtype=bool))


def test_equal_class_parent_condition_and_negative_variant_mass():
    r = rows()[:2]; tokens = {}; masks = {}
    for row in r:
        mask = np.zeros((512, 512), dtype=bool); mask[128:256, 128:256] = True
        masks[row['index']] = mask
        for ci, condition in enumerate(('original', 'jpeg75')):
            for vi, variant in enumerate(('authentic', 'classical_edit', 'ai_composite')):
                value = np.zeros((32, 32, 384), dtype=np.float32)
                value[:, :, 0] = row['index']; value[:, :, 1] = ci; value[:, :, 2] = vi
                tokens[f"{row['index']:03d}_{variant}_{condition}"] = value
    x, y, w = training_arrays(r, tokens, masks)
    assert w.sum() == pytest.approx(1)
    assert w[y == 1].sum() == pytest.approx(.5)
    for i in range(2):
        assert w[x[:, 0] == i].sum() == pytest.approx(.5)
        assert w[x[:, 1] == i].sum() == pytest.approx(.5)
    for i in range(3):
        assert w[(y == 0) & (x[:, 2] == i)].sum() == pytest.approx(1 / 6)
    center, scale = fit_normalizer(x, w)
    assert center[0] == pytest.approx(.5)
    assert center[1] == pytest.approx(.5)
    assert scale[0] == pytest.approx(.5)
    assert np.isfinite(scale).all() and (scale > 0).all()


def test_nonfinite_normalization_rejected():
    with pytest.raises(ValueError):
        fit_normalizer(np.array([[np.nan], [1.]]), np.array([.5, .5]))
