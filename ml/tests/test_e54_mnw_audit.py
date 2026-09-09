from experiments.e54_mnw_audit import rejected_parents


def test_rejects_both_internal_endpoints_and_cross_query_only():
    cross = [{'train_parent': 'new-a', 'cal_parent': 'old-protected'}]
    internal = [{'train_parent': 'new-b', 'cal_parent': 'new-c'}]
    assert rejected_parents(cross, internal) == {'new-a', 'new-b', 'new-c'}


def test_no_matches_does_not_exclude_anything():
    assert rejected_parents([], []) == set()
