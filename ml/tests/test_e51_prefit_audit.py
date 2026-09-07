from io import BytesIO

from PIL import Image
import pytest

from experiments.e51_prefit_audit import fingerprint, cross_role_matches, validate_roles, verify_row


def encoded(color, fmt="PNG"):
    output = BytesIO()
    Image.new("RGB", (32, 32), color).save(output, format=fmt)
    return output.getvalue()


def row(parent, role, facts, condition="original", label=0):
    return {"parent_id": parent, "role": role, "audit_role": role, "source": "test",
            "condition": condition, "label": label, **facts}


def test_equal_pixels_are_not_complementary_hashes():
    f = fingerprint(encoded("black"))
    assert f["canonical_dhash"] == "0000000000000000"


def test_decoded_pixel_hash_catches_different_encodings():
    a, b = fingerprint(encoded("black")), fingerprint(encoded("black", "BMP"))
    assert a["sha256"] != b["sha256"]
    assert a["pixel_sha256"] == b["pixel_sha256"]
    matches = cross_role_matches([row("a", "TRAIN", a)], [row("b", "CAL", b)])
    assert matches[0]["exact_fields"] == ["pixel_sha256"]


def test_phash_far_candidate_is_not_declared_duplicate():
    a = fingerprint(encoded("black"))
    b = {**a, "sha256": "b", "pixel_sha256": "b", "phash63": "7fffffffffffffff"}
    assert cross_role_matches([row("a", "TRAIN", a)], [row("b", "CAL", b)]) == []


def test_cross_role_parent_and_malformed_pair_fail_closed():
    a = row("a", "TRAIN", {})
    c = row("a", "CAL", {})
    with pytest.raises(ValueError, match="parent overlap"):
        validate_roles([a], [c])
    with pytest.raises(ValueError, match="pairing"):
        validate_roles([a], [row("b", "CAL", {})])
    validate_roles([a], [row("b", "CAL", {}), row("b", "CAL", {}, "q75")])


def test_changed_payload_fails_even_with_cache(tmp_path):
    raw = encoded("black")
    path = tmp_path / "image.png"
    path.write_bytes(raw)
    r = {**row("a", "TRAIN", fingerprint(raw)), "path": str(path)}
    verify_row(r, tmp_path / "cache")
    path.write_bytes(encoded("white"))
    with pytest.raises(ValueError, match="payload identity changed"):
        verify_row(r, tmp_path / "cache")


def test_historical_train_parent_without_transport_is_original(tmp_path):
    raw = encoded("black")
    path = tmp_path / "image.png"
    path.write_bytes(raw)
    r = {**row("a", "TRAIN", fingerprint(raw)), "path": str(path)}
    del r["condition"]
    assert verify_row(r, tmp_path / "cache")["condition"] == "original"
