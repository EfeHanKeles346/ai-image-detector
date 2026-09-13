from experiments.e88_audit import raw_body_matches, resolve


def test_raw_source_match_survives_reference_rendering_change():
    rows = [{'parent_id': 'SID:Sony:a', 'sha256': 'raw-a'}]
    refs = [{'parent_id': 'protected', 'sha256': 'different-png', 'original_sha256': 'raw-a'},
            {'parent_id': 'unrelated', 'sha256': 'other'}]
    assert raw_body_matches(rows, refs) == [
        {'train_parent': 'SID:Sony:a', 'cal_parent': 'protected', 'match': 'original_RAW_body_sha256'}]


def test_protected_overlap_propagates_across_camera_component():
    rows = [{'parent_id': p, 'sensor': 'Sony' if p == 'a' else 'Fuji', 'capture_second': None}
            for p in ['a', 'b', 'c', 'd']]
    matches = [{'train_parent': 'a', 'cal_parent': 'b'}, {'train_parent': 'b', 'cal_parent': 'c'}]
    out = resolve(rows, matches, {'a', 'failed-decode'})
    assert out['quarantined'] == ['a', 'b', 'c', 'failed-decode']
    assert out['kept'] == ['d']
    assert out['lineage']['a'] == out['lineage']['c']
    assert out['lineage']['a'].startswith('SID:component:')
