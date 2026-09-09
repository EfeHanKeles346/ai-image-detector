import hashlib
import pytest

from experiments.e54_mnw_reserve import parse_pointer,MAX_IMAGE


def pointer(size):
    raw=f"version https://git-lfs.github.com/spec/v1\noid sha256:{'a'*64}\nsize {size}\n".encode()
    sha=hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()
    return raw,sha


def test_verified_lfs_size_not_small_pointer_size():
    assert parse_pointer(*pointer(1024))==('a'*64,1024)


def test_oversized_selection_cannot_be_silently_admitted():
    with pytest.raises(ValueError):parse_pointer(*pointer(MAX_IMAGE+1))


def test_pointer_integrity_is_checked():
    raw,sha=pointer(123)
    with pytest.raises(ValueError):parse_pointer(raw+b'\n',sha)
