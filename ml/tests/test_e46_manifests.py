import hashlib

import pytest

from experiments.e46_manifests import (
    TRUEFAKE_NAMESPACE,
    _identity_audit,
    _path_dhash,
    _rank,
    parse_truefake_member,
)
from PIL import Image


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("Facebook/Real/FFHQ/00001.jpg", (0, "FFHQ", "real")),
        ("Facebook/Real/FORLAB/00001.jpg", (0, "FORLAB", "real")),
        ("Facebook/Fake/FLUX.1/animals/00001.jpg", (1, "FLUX.1", "animals")),
        ("Facebook/Fake/StyleGAN/00001.jpg", (1, "StyleGAN", "faces")),
    ],
)
def test_parse_truefake_member(name, expected):
    assert parse_truefake_member(name) == expected


@pytest.mark.parametrize(
    "name",
    ["../escape.jpg", "/absolute.jpg", "Facebook/Real/other/a.jpg",
     "Facebook/Fake/FLUX.1/a.txt", "Facebook/Fake/unknown/a.jpg"],
)
def test_parse_truefake_member_fails_closed(name):
    with pytest.raises(ValueError):
        parse_truefake_member(name)


def test_rank_is_namespaced_and_deterministic():
    name = "Facebook/Real/FFHQ/00001.jpg"
    assert _rank(name, "FFHQ") == hashlib.sha256(f"{TRUEFAKE_NAMESPACE}|FFHQ|{name}".encode()).hexdigest()


def test_identity_audit_excludes_protected_and_repeated_exact_bytes():
    rows = [
        {"record_id": "b", "label": 0, "sha256": "same", "dhash": "d1"},
        {"record_id": "a", "label": 0, "sha256": "same", "dhash": "d1"},
        {"record_id": "c", "label": 1, "sha256": "other", "dhash": "protected"},
    ]
    audit = _identity_audit(rows, set(), {"protected"})
    assert audit["excluded_record_ids"] == ["b", "c"]
    assert audit["exclusion_reasons"]["b"] == ["same_label_exact_duplicate_of:a"]
    assert audit["exclusion_reasons"]["c"] == ["protected_dhash_overlap"]


def test_path_dhash_decodes_image(tmp_path):
    path = tmp_path / "sample.png"
    Image.new("RGB", (32, 32), (20, 40, 80)).save(path)
    value = _path_dhash(path)
    assert isinstance(value, str)
    assert len(value) == 16
