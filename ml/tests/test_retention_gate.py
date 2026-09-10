import copy

import numpy as np
import pytest

from pixelproof.retention_gate import check_train_retention


def fixture():
    parents = [{'parent_id': str(i), 'label': y, 'source': s, 'role': 'TRAIN'}
               for i, (y, s) in enumerate([(0, 'camera'), (1, 'gen_a'), (1, 'gen_b')])]
    return parents, np.array([[.8, .8], [.8, .5], [.1, .1]])


def check(parents, old, new):
    return check_train_retention(parents, old, new, conditions=('clean', 'q75'), ai_cut=.5)


def test_unchanged_reference_pass_is_not_promotion():
    parents, old = fixture()
    result = check(parents, old, old.copy())
    assert result['passes_train_retention']
    assert not result['promotion_allowed']
    assert not result['independent_quality_claim']


def test_equal_pooled_recall_hides_source_regression_and_is_rejected():
    parents, old = fixture()
    new = old.copy()
    new[0] = .1  # REAL improvement cannot pay for an AI miss.
    new[1, 1] = np.nextafter(.5, 0)
    new[2, 1] = .9  # Another generator's rescue cannot pay for it either.
    result = check(parents, old, new)
    q75 = result['comparisons']['q75']['ai']
    assert q75['old_ai_count'] == q75['new_ai_count']
    assert not result['passes_train_retention']
    assert result['new_ai_miss_views'] == result['new_ai_miss_parents'] == 1
    assert result['misses_by_condition'] == {'clean': 0, 'q75': 1}
    assert result['comparisons']['q75']['by_source']['1:gen_a']['ai_to_non_ai'] == 1


def test_duplicate_views_do_not_inflate_parent_loss_count():
    parents, old = fixture()
    new = old.copy(); new[1] = 0
    result = check(parents, old, new)
    assert result['new_ai_miss_views'] == 2
    assert result['new_ai_miss_parents'] == 1


@pytest.mark.parametrize('bad', [np.nan, np.inf, -.1, 1.1])
def test_invalid_score_fails_closed(bad):
    parents, old = fixture()
    new = old.copy(); new[1, 0] = bad
    with pytest.raises(ValueError, match='finite probabilities'):
        check(parents, old, new)


def test_missing_condition_and_duplicate_parent_fail_closed():
    parents, old = fixture()
    with pytest.raises(ValueError, match='complete parent-by-condition'):
        check(parents, old, old[:, :1])
    parents[1]['parent_id'] = parents[2]['parent_id']
    with pytest.raises(ValueError, match='duplicate parent'):
        check(parents, old, old)


@pytest.mark.parametrize('role', ['CAL', 'DEV', 'TEST', 'E52', ''])
def test_protected_roles_fail_closed(role):
    parents, old = fixture(); parents[1]['role'] = role
    with pytest.raises(ValueError, match='only admitted TRAIN'):
        check(parents, old, old)


def test_vacuous_empty_and_single_class_checks_fail_closed():
    parents, old = fixture()
    with pytest.raises(ValueError, match='catches no AI'):
        check(parents, np.zeros_like(old), old)
    with pytest.raises(ValueError, match='nonempty'):
        check([], np.empty((0, 2)), np.empty((0, 2)))
    parents = copy.deepcopy(parents)
    for row in parents:
        row['label'] = 1
    with pytest.raises(ValueError, match='both REAL and AI'):
        check(parents, old, old)
