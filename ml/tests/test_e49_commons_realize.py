from __future__ import annotations

from PIL import Image

from experiments.e49_commons_realize import audit_originals, device_evidence, realize_originals


def test_commons_audit_is_order_independent_and_excludes_protected():
    rows = [
        {"identity": "a", "rank": "1", "sha256": "sha-a", "dhash": "d-a"},
        {"identity": "b", "rank": "2", "sha256": "sha-b", "dhash": "d-b"},
        {"identity": "c", "rank": "3", "sha256": "sha-c", "dhash": "d-c"},
    ]
    for row in rows:
        row.update({"source": "Canon EOS R5", "exif_make": "", "exif_model": ""})
    first = audit_originals(rows, {"sha-b"}, {"d-c"})
    second = audit_originals(list(reversed(rows)), {"sha-b"}, {"d-c"})
    assert first == second
    assert first == {"b": ["protected_exact_overlap"], "c": ["protected_dhash_overlap"]}


def test_commons_audit_marks_later_internal_duplicates():
    rows = [
        {"identity": "a", "rank": "1", "sha256": "same", "dhash": "d-a"},
        {"identity": "b", "rank": "2", "sha256": "same", "dhash": "d-b"},
        {"identity": "c", "rank": "3", "sha256": "sha-c", "dhash": "d-a"},
    ]
    for row in rows:
        row.update({"source": "Canon EOS R5", "exif_make": "", "exif_model": ""})
    reasons = audit_originals(rows, set(), set())
    assert reasons["b"] == ["internal_exact_duplicate_of:a"]
    assert reasons["c"] == ["internal_dhash_duplicate_of:a"]


def test_device_evidence_accepts_aliases_and_rejects_wrong_camera():
    assert device_evidence({"source": "iPhone 14 Pro", "exif_make": "Apple",
                            "exif_model": "iPhone15,2"}) == "exif_device_match"
    assert device_evidence({"source": "Google Pixel 8 Pro", "exif_make": "Pixel 8 Pro (husky)",
                            "exif_model": "Google"}) == "exif_device_match"
    assert device_evidence({"source": "Nikon Z 8", "exif_make": "NIKON CORPORATION",
                            "exif_model": "NIKON D70"}) == "exif_device_mismatch"
    assert device_evidence({"source": "Nikon Z 8", "exif_make": "",
                            "exif_model": ""}) == "category_only_missing_exif"


def test_realize_originals_derives_dhash_after_receipt_validation(tmp_path):
    import hashlib

    path = tmp_path / "original.jpg"
    Image.new("RGB", (32, 24), (20, 30, 40)).save(path, format="JPEG")
    raw = path.read_bytes()
    rows = [{"identity": "commons:1", "path": str(path),
             "sha256": hashlib.sha256(raw).hexdigest(), "width": 32, "height": 24}]
    found = realize_originals(rows)
    assert len(found[0]["dhash"]) == 16
