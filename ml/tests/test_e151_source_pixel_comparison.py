import numpy as np
import pytest
from experiments.e151_source_pixel_comparison import fit_project, residual_coordinates


def test_held_out_changes_never_change_fit_map_or_fit_projection():
    values = np.random.default_rng(151).normal(size=(90, 12))
    mask = np.arange(90) % 3 != 0
    mapping, projected = fit_project(values, mask, 4)
    altered = values.copy()
    altered[~mask] = altered[~mask] * 1000 + 20000
    other, altered_projection = fit_project(altered, mask, 4)
    for key in mapping:
        np.testing.assert_array_equal(mapping[key], other[key])
    np.testing.assert_array_equal(projected[mask], altered_projection[mask])
    assert not np.allclose(projected[~mask], altered_projection[~mask])
    with pytest.raises(ValueError):
        fit_project(values, np.ones(90, dtype=bool), 4)
    with pytest.raises(ValueError):
        fit_project(values, mask.astype(int), 4)


def fixture():
    rng = np.random.default_rng(151)
    hist = rng.uniform(size=(3, 4, 2, 12, 25)).astype(np.float32)
    hist /= hist.sum(axis=-1, keepdims=True)
    saved = dict(features=hist.reshape(3, 4, 2, 300), binding=np.asarray('bound'),
                 role=np.asarray('TRAIN'), parents=np.asarray(['a','b','c']),
                 conditions=np.asarray(['clean','assigned_transport','q75','social_q75']))
    return saved, [{'parent_id': p} for p in saved['parents']]


def test_control_is_difference_with_parent_major_view_order_preserved():
    saved, rows = fixture()
    coords = residual_coordinates(saved, rows, 'bound', saved['conditions'])
    assert coords.shape == (12, 600)
    assert coords.dtype == np.float32
    for i in range(3):
        for j in range(4):
            np.testing.assert_array_equal(coords[4*i+j,:300], saved['features'][i,j,0])
            np.testing.assert_array_equal(coords[4*i+j,300:], saved['features'][i,j,1]-saved['features'][i,j,0])


@pytest.mark.parametrize('damage', ['binding','role','parents','conditions','shape','dtype','nan','negative','normalization'])
def test_reject_bad_feature_identity_and_invalid_histograms(damage):
    saved, rows = fixture()
    conditions = saved['conditions'].copy()
    if damage in ('binding','role'):
        saved[damage] = np.asarray('other')
    elif damage in ('parents','conditions'):
        saved[damage] = saved[damage][::-1]
    elif damage == 'shape':
        saved['features'] = saved['features'][:2]
    elif damage == 'dtype':
        saved['features'] = saved['features'].astype(np.float64)
    else:
        saved['features'][0,0,0,0] = {'nan':np.nan,'negative':-1,'normalization':2}[damage]
    with pytest.raises(ValueError):
        residual_coordinates(saved, rows, 'bound', conditions)
