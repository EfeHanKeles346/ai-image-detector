from io import BytesIO
import hashlib
import zipfile

from PIL import Image
import pytest

from experiments.e51_protected_pixels import cached_fingerprint, read_body, protected_fingerprint
import experiments.e51_protected_pixels as pixels


def image_bytes():
    stream = BytesIO()
    Image.new('RGB', (32, 32), 'black').save(stream, format='PNG')
    return stream.getvalue()


def test_cached_fingerprint_does_not_accept_changed_bytes():
    raw = image_bytes()
    facts = cached_fingerprint(raw)
    assert cached_fingerprint(raw, facts) == facts
    with pytest.raises(ValueError, match='binding mismatch'):
        cached_fingerprint(raw + b'changed', facts)


def test_size_exception_is_exact_hash_scoped_and_geometry_checked(monkeypatch):
    raw = image_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    monkeypatch.setattr(pixels, 'LARGE_PROTECTED', {sha: 1024})
    assert protected_fingerprint(raw)['width'] == 32
    monkeypatch.setattr(pixels, 'LARGE_PROTECTED', {sha: 1025})
    with pytest.raises(ValueError, match='geometry changed'):
        protected_fingerprint(raw)


def test_file_is_hashed_even_when_same_size(tmp_path):
    path = tmp_path / 'file'
    path.write_bytes(b'abc')
    row = {'kind': 'file', 'path': str(path), 'bytes': 3, 'sha256': hashlib.sha256(b'abc').hexdigest()}
    assert read_body(row, {}) == b'abc'
    path.write_bytes(b'xyz')
    with pytest.raises(ValueError, match='bytes changed'):
        read_body(row, {})


def test_zip_crc_and_content_are_both_checked(tmp_path):
    path = tmp_path / 'file.zip'
    with zipfile.ZipFile(path, 'w') as bundle:
        bundle.writestr('image.png', b'abc')
    with zipfile.ZipFile(path) as bundle:
        row = {'kind':'zip','archive':str(path),'member':'image.png','bytes':3,
               'crc32':f"{bundle.getinfo('image.png').CRC:08x}", 'sha256':hashlib.sha256(b'abc').hexdigest()}
        assert read_body(row, {str(path): bundle}) == b'abc'
        with pytest.raises(ValueError, match='CRC changed'):
            read_body({**row,'crc32':'00000000'}, {str(path):bundle})
