import copy
import pytest
from experiments.e99_acquisition import SENSORS, select


def entries(sensor):
    return [{'filename': f'{sensor}/training_set/original/{n}.jpg', 'bytes': 100,
             'compressed_bytes': 90, 'compression': 8, 'directory': False} for n in range(100)]


def test_selection_is_order_invariant_and_excludes_test_and_denoised():
    sensor = SENSORS[0]; rows = entries(sensor)
    expected = select(sensor, rows)
    extra = [{'filename': f'{sensor}/{split}/{kind}/1.jpg', 'bytes': 100,
              'compressed_bytes': 90, 'compression': 8, 'directory': False}
             for split, kind in [('test_set', 'original'), ('training_set', 'denoised')]]
    assert len(expected) == 64
    assert select(sensor, list(reversed(rows)) + extra) == expected


def test_invalid_archive_and_capacity_fail_without_refill():
    sensor = SENSORS[0]; rows = entries(sensor)
    for invalid in [rows[:63], rows+[rows[0]], rows+[dict(rows[0], filename='../escape.jpg')]]:
        with pytest.raises(ValueError):
            select(sensor, invalid)
    damaged = copy.deepcopy(rows)
    selected_name = select(sensor, rows)[0]['filename']
    next(r for r in damaged if r['filename'] == selected_name)['bytes'] = 33*1024**2
    with pytest.raises(ValueError):
        select(sensor, damaged)
    with pytest.raises(ValueError):
        select('Sony_IMX258', rows)
