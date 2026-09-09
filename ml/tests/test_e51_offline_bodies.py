from pathlib import Path
import zipfile

import pytest

from experiments.e51_offline_bodies import relative_body_path, safe_zip_info, resolve


def test_relative_paths_use_manifest_parent_not_working_directory():
    assert relative_body_path(Path('/dataset/e36/cal.json'), 'cal/real/001.jpg') == Path('/dataset/e36/cal/real/001.jpg')
    with pytest.raises(ValueError, match='traverses'):
        relative_body_path(Path('/dataset/m.json'), '../elsewhere.jpg')


def test_local_body_size_is_checked(tmp_path):
    path = tmp_path / 'image.jpg'
    path.write_bytes(b'abc')
    row = {'path': 'image.jpg', 'bytes': 3, 'parent_id': 'a'}
    assert resolve(row, tmp_path / 'manifest.json', {})['bytes'] == 3
    with pytest.raises(ValueError, match='size differs'):
        resolve({**row, 'bytes': 4}, tmp_path / 'manifest.json', {})


def test_zip_member_is_resolved_without_extraction(tmp_path):
    archive = tmp_path / 'images.zip'
    with zipfile.ZipFile(archive, 'w') as bundle:
        bundle.writestr('images/test.jpg', b'abc')
    with zipfile.ZipFile(archive) as bundle:
        assert safe_zip_info(bundle, 'images/test.jpg').file_size == 3
        with pytest.raises(ValueError, match='unsafe'):
            safe_zip_info(bundle, '../test.jpg')
    assert not (tmp_path / 'images').exists()


def test_unknown_archive_route_cannot_trigger_a_download(tmp_path):
    with pytest.raises(ValueError, match='unbound archive'):
        resolve({'member': 'a.jpg', 'archive': 'unknown'}, tmp_path / 'manifest.json', {})
