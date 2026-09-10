import numpy as np
import pytest

from experiments.e60_audit import reconstruct_fit, compare_population, unused_native


def test_exposure_reconstructs_consumed_e42_clean_but_excludes_rr_cal():
    e42 = {'roles': np.array(['development', 'development']),
           'parent_ids': np.array(['p', 'p']), 'labels': np.array([1, 1]),
           'sources': np.array(['s', 's']), 'conditions': np.array(['clean', 'not_assigned'])}
    rr = {'roles': np.array(['train', 'calibration', 'development']),
          'parent_ids': np.array(['r', 'c', 'd']), 'labels': np.array([0, 1, 0]),
          'sources': np.array(['rr', 'rr', 'rr'])}
    assert {r['parent_id'] for r in reconstruct_fit(e42, rr)} == {'p', 'r'}


def test_missing_teacher_cal_is_not_restored_and_label_flip_fails():
    old = [{'parent_id': 'a', 'label': 1}, {'parent_id': 'b', 'label': 0}]
    result = compare_population(old, [{'parent_id': 'b', 'label': 0}], {'a'})
    assert result['teacher_missing'][0]['reason'] == 'protected_E51_CAL'
    with pytest.raises(ValueError, match='label direction'):
        compare_population(old, [{'parent_id': 'b', 'label': 1}], set())


def test_unused_parents_in_seen_group_are_not_group_disjoint_dev():
    records = [{'record_id': p, 'source_id': s, 'role_group': g, 'label': y}
               for p, s, g, y in [('a', 'camera', 'device1', 'real'),
                                  ('b', 'camera', 'device1', 'real'),
                                  ('c', 'ai', 'prompt1', 'ai')]]
    result = unused_native(records, {'remaining_record_ids': ['b', 'c'], 'internal_pairs': []},
                           [{'parent_id': 'e32:a'}])
    assert result['unused_parents'] == 2
    assert result['unused_recorded_group_labels'] == {'ai': 1}
    assert not result['balanced_group_disjoint_development_available']
