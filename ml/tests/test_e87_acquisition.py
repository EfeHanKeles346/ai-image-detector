import pytest

from experiments.e87_acquisition import select


def source(camera='Sony'):
    ext = 'ARW' if camera == 'Sony' else 'RAF'
    names = [f'{camera}/long/0{i:04d}_00_10s.{ext}' for i in range(80)]
    catalog = {'camera': camera, 'image_members_opened': 0,
               'url': f'https://storage.googleapis.com/isl-datasets/SID/{camera}2025.zip',
               'entries': [{'filename': n, 'directory': False, 'bytes': 200, 'compressed_bytes': 100,
                            'compression': 8} for n in names]}
    body = '\n'.join(f'./{camera}/short/0{i:04d}_00_0.1s.{ext} ./{n} ISO200 F8' for i, n in enumerate(names))
    return catalog, body


@pytest.mark.parametrize('camera', ['Sony', 'Fuji'])
def test_fixed_selection_ignores_list_order_and_repeated_short_bursts(camera):
    catalog, body = source(camera)
    expected = select(camera, catalog, body)
    assert len(expected) == 64
    assert select(camera, catalog, '\n'.join(reversed(body.splitlines())) + '\n' + body) == expected


def test_published_test_member_cannot_enter_selection():
    catalog, body = source()
    with pytest.raises(ValueError, match='published TRAIN'):
        select('Sony', catalog, body.replace('long/00000_', 'long/10000_'))


def test_two_long_references_for_same_scene_are_rejected():
    catalog, body = source()
    with pytest.raises(ValueError, match='one RAW'):
        select('Sony', catalog, body + '\n' + body.splitlines()[0].replace('_00_10s.', '_00_20s.'))


def test_nonofficial_archive_rejected():
    catalog, body = source()
    catalog['url'] = 'https://example.org/mirror.zip'
    with pytest.raises(ValueError, match='official'):
        select('Sony', catalog, body)
