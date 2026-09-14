import pytest
from experiments.e105_acquisition import select


def entries():
    return [{'filename': f'train_copy/{i:08d}.{kind}.png', 'directory':False,
             'bytes':100, 'compressed_bytes':90, 'compression':8}
            for i in range(10) for kind in ['image','mask']]


def test_paired_selection_stable_and_train_only():
    rows = entries(); selected = select(rows, 4)
    assert selected == select(list(reversed(rows)), 4)
    assert all(len(p['members']) == 2 for p in selected)
    foreign = dict(rows[0], filename='validation/00000000.image.png')
    assert selected == select(rows+[foreign], 4)


def test_missing_selected_mask_or_unsafe_name_does_not_refill():
    rows = entries(); chosen = select(rows, 4)[0]['members'][1]['filename']
    with pytest.raises(ValueError, match='mask missing'):
        select([r for r in rows if r['filename'] != chosen], 4)
    with pytest.raises(ValueError, match='Unsafe'):
        select(rows+[dict(rows[0], filename='../escape.png')], 4)
