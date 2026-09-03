import zipfile

import pytest

from experiments.e45_mediaeval_acquisition import (
    EXPECTED_BYTES,
    EXPECTED_ETAG,
    EXPECTED_LAST_MODIFIED,
    classify_member,
    validate_headers,
)


def test_mediaeval_headers_are_immutable_and_resumable():
    result = validate_headers({
        "Content-Length": str(EXPECTED_BYTES),
        "ETag": EXPECTED_ETAG,
        "Last-Modified": EXPECTED_LAST_MODIFIED,
        "Accept-Ranges": "bytes",
    })
    assert result["bytes"] == EXPECTED_BYTES
    assert result["accept_ranges"] == "bytes"


@pytest.mark.parametrize("field", ["content-length", "etag", "last-modified", "accept-ranges"])
def test_mediaeval_header_change_fails_closed(field):
    headers = {
        "content-length": str(EXPECTED_BYTES),
        "etag": EXPECTED_ETAG,
        "last-modified": EXPECTED_LAST_MODIFIED,
        "accept-ranges": "bytes",
    }
    headers[field] = "changed"
    with pytest.raises((ValueError, TypeError)):
        validate_headers(headers)


def test_mediaeval_member_classification_accepts_declared_layouts():
    assert classify_member("0_real/a.jpg") == ("0_real", "image")
    assert classify_member("dataset/1_fake/b.webp") == ("1_fake", "image")


@pytest.mark.parametrize(
    "name",
    ["../escape.jpg", "/absolute.jpg", "other/a.jpg", "0_real/a.txt", "a.jpg"],
)
def test_mediaeval_member_classification_fails_closed(name):
    with pytest.raises(ValueError):
        classify_member(name)


def test_encrypted_member_fact_is_representable():
    info = zipfile.ZipInfo("0_real/a.jpg")
    info.flag_bits |= 0x1
    assert info.flag_bits & 0x1
