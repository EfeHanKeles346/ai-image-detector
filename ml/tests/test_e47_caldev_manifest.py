from experiments.e47_caldev_manifest import QUOTAS, identity_exclusions


def test_registered_roles_are_balanced_and_total_2400():
    by_role_label = {}
    for (role, _), (label, quota) in QUOTAS.items():
        by_role_label[(role, label)] = by_role_label.get((role, label), 0) + quota
    assert by_role_label == {
        ("CAL", 0): 600,
        ("CAL", 1): 600,
        ("DEVELOPMENT", 0): 600,
        ("DEVELOPMENT", 1): 600,
    }


def test_identity_audit_removes_protected_and_internal_duplicates():
    rows = [
        {"record_id": "a", "rank": "0", "sha256": "x", "dhash": "d1"},
        {"record_id": "b", "rank": "1", "sha256": "x", "dhash": "d2"},
        {"record_id": "c", "rank": "2", "sha256": "z", "dhash": "d1"},
        {"record_id": "d", "rank": "3", "sha256": "protected", "dhash": "d4"},
    ]
    result = identity_exclusions(rows, {"protected"}, set())
    assert "internal_exact_duplicate_of:a" in result["b"]
    assert "internal_dhash_duplicate_of:a" in result["c"]
    assert result["d"] == ["protected_exact_overlap"]
