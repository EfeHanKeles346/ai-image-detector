import pytest

from experiments.e45_mediaeval_manifest import audit_rows, parse_member


@pytest.mark.parametrize(
    ("member", "expected"),
    [
        ("ITW-SM/0_real/Facebook_real_12.jpg", (0, "Facebook")),
        ("ITW-SM/0_real/Linkedin_real_7.jpg", (0, "LinkedIn")),
        ("ITW-SM/1_fake/instagram_2.jpg", (1, "Instagram")),
        ("ITW-SM/1_fake/x_8.jpg", (1, "X")),
    ],
)
def test_parse_member_preserves_label_and_platform(member, expected):
    assert parse_member(member) == expected


@pytest.mark.parametrize(
    "member",
    [
        "../0_real/a.jpg",
        "ITW-SM/0_real/x_1.jpg",
        "ITW-SM/1_fake/X_real_1.jpg",
        "ITW-SM/2_unknown/x_1.jpg",
    ],
)
def test_parse_member_fails_closed(member):
    with pytest.raises(ValueError):
        parse_member(member)


def test_audit_excludes_protected_and_same_label_duplicate_rows():
    rows = [
        {"record_id": "a", "label": 0, "sha256": "same", "dhash": "d1"},
        {"record_id": "b", "label": 0, "sha256": "same", "dhash": "d2"},
        {"record_id": "c", "label": 1, "sha256": "other", "dhash": "protected"},
    ]
    audit = audit_rows(rows, prior_exact=set(), prior_dhash={"protected"})
    assert audit["excluded_record_ids"] == ["b", "c"]
    assert audit["cross_label_exact_groups"] == []


def test_audit_excludes_both_sides_of_cross_label_exact_duplicate():
    rows = [
        {"record_id": "a", "label": 0, "sha256": "same", "dhash": "d1"},
        {"record_id": "b", "label": 1, "sha256": "same", "dhash": "d2"},
    ]
    audit = audit_rows(rows, prior_exact=set(), prior_dhash=set())
    assert audit["excluded_record_ids"] == ["a", "b"]
    assert len(audit["cross_label_exact_groups"]) == 1
