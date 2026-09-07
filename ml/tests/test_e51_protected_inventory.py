from experiments.e51_protected_inventory import identity_rows, keys


def test_nested_reserves_are_not_silently_ignored():
    payload = {"commons": {"rows": [{"identity": "commons:1", "sha256": "a"}]},
               "aigc": {"rows": [{"record_id": "aigc:2"}]}}
    rows = list(identity_rows(payload))
    assert len(rows) == 2
    assert ("identity", "aigc:2") in keys(rows[1])


def test_parent_identity_namespaces_match_but_hash_algorithms_do_not():
    assert keys({"parent_id": "a", "sha256": "abc"}) & keys({"identity": "a"})
    assert not keys({"sha256": "abc"}) & keys({"sha1": "abc"})
