import numpy as np
import pytest
from pixelproof.nested_calibration import split_roster, select_cut


def test_no_component_or_class_leakage_in_nested_roles():
    rows = [dict(parent_id=str(i), label=i % 2, role='TRAIN', source=str(i // 2)) for i in range(6)]
    groups = {r['parent_id']: str(i // 2) for i, r in enumerate(rows)}
    folds = {p: int(g) for p, g in groups.items()}
    report = split_roster(rows, groups, folds)
    assert len(report['assignments']) == 6
    assert all(set(r.values()) == {0, 1, 2} for r in report['assignments'])
    with pytest.raises(ValueError):
        split_roster(rows, groups, {**folds, '0': 1})
    with pytest.raises(ValueError):
        split_roster([{**r, 'label': 0} for r in rows], groups, folds)
    with pytest.raises(ValueError):
        split_roster([{**rows[0], 'role': 'FINAL'}, *rows[1:]], groups, folds)


def test_cal_cut_handles_ties_and_preserves_ai():
    labels = np.array([0] * 10 + [1] * 20)
    scores = np.repeat(np.array([.2] * 9 + [.4] + [.9] * 20)[:, None], 4, axis=1)
    result = select_cut(labels, scores, np.array(['r'] * 10 + ['a'] * 20))
    assert result['accepted'] and result['cut'] == np.nextafter(.2, np.inf)
    assert all(r['REAL_FP'] == 1 and r['AI_TP'] == 20 for r in result['conditions'])


def test_infeasible_cal_cut_rejected_without_sacrificing_ai():
    labels = np.array([0] * 10 + [1] * 20)
    scores = np.repeat(np.array([.9] * 10 + [.8] * 20)[:, None], 4, axis=1)
    result = select_cut(labels, scores, np.array(['r'] * 10 + ['a'] * 20))
    assert not result['accepted']
    assert 'CAL_AI_retention_failed' in result['rejection_reasons']
    assert all(r['new_AI_misses'] == 20 for r in result['conditions'])


def test_worst_component_and_every_view_constrain_common_cut():
    labels = np.array([0] * 100 + [1] * 20)
    groups = np.array(['small'] * 5 + ['large'] * 95 + ['ai'] * 20)
    scores = np.full((120, 4), .1)
    scores[:5, 3] = .7
    scores[100:] = .9
    result = select_cut(labels, scores, groups)
    assert result['accepted'] and result['cut'] == np.nextafter(.7, np.inf)
    assert all(r['worst_REAL_component_FPR'] == 0 for r in result['conditions'])
    with pytest.raises(ValueError):
        select_cut(labels, scores[:, :3], groups)
