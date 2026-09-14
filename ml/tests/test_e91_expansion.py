import json
import numpy as np
import pytest
from experiments import e91_data as data, e91_representation as representation
from experiments import e85_representation as previous_representation
from experiments import e92_fit as fit, e92_development as dev, e92_gates as gates
from experiments import e86_gates as previous_gates


def test_sid_append_preserves_all_previous_conditions_and_rejects_wrong_roles():
    rng = np.random.default_rng(91)
    previous = {k: rng.normal(size=(3, 4, w)).astype(np.float32) for k, w in data.WIDTHS.items()}
    rows = [{'parent_id': 'SID:a', 'role': 'TRAIN', 'label': 0}]
    new = {k: rng.normal(size=(1, 4, w)).astype(np.float32) for k, w in data.WIDTHS.items()}
    new.update(binding=np.array('bound'), parents=np.array(['SID:a']), roles=np.array(['TRAIN']),
               conditions=np.array(data.CONDITIONS))
    result = data.append_sid(previous, new, rows, 'bound')
    for k in data.WIDTHS:
        assert np.array_equal(result[k][:3], previous[k])
        assert np.array_equal(result[k][3:], new[k])
    for field, value in [('roles', np.array(['DEVELOPMENT'])), ('parents', np.array(['SID:b'])),
                         ('conditions', np.array(data.CONDITIONS[::-1]))]:
        with pytest.raises(ValueError, match='TRAIN'):
            data.append_sid(previous, new | {field: value}, rows, 'bound')


def test_admission_cannot_refill_duplicate_or_non_train_sid():
    old = [{'parent_id': f'old:{i}', 'sha256': f'oldsha:{i}', 'role': 'TRAIN',
            'label': int(i < 4595)} for i in range(12141)]
    sid = [{'parent_id': f'SID:{i}', 'sha256': f'sidsha:{i}', 'original_sha256': f'raw:{i}',
            'role': 'TRAIN', 'label': 0, 'training_allowed': True,
            'source': 'SID:Sony' if i < 64 else 'SID:Fuji'} for i in range(128)]
    assert len(data.combine(old, sid)) == 12269
    for change in [{'parent_id': old[0]['parent_id']}, {'sha256': old[0]['sha256']},
                   {'role': 'DEVELOPMENT'}, {'training_allowed': False}, {'source': 'SID:Sony'}]:
        modified = [*sid[:-1], sid[-1] | change]
        with pytest.raises(ValueError, match='disjoint'):
            data.combine(old, modified)


def test_sid_success_cannot_mask_previous_population_or_condition_failure():
    rows = [{'parent_id': 'a', 'label': 0, 'source': 'legacy'},
            {'parent_id': 'b', 'label': 1, 'source': 'AI'},
            {'parent_id': 'c', 'label': 0, 'source': 'MIDD:A'},
            {'parent_id': 'd', 'label': 0, 'source': 'SID:Sony'},
            {'parent_id': 'e', 'label': 0, 'source': 'SID:Fuji'}]
    scores = np.zeros((5, 4)); scores[1] = .99; scores[0, 0] = .99
    actual = gates.population_gates(rows, scores, 2, 3)
    previous = previous_gates.population_gates(rows[:3], scores[:3], 2)
    assert {c: v['previous_E86_populations'] for c, v in actual.items()} == previous
    assert not actual['clean']['passed'] and actual['social_q75']['passed']
    metrics = gates.full_training_gates(rows, scores, 2, 3)
    assert sum(v['gate']['total_checks'] for conditions in metrics.values() for v in conditions.values()) == 120
    assert not metrics['clean']['legacy_TRAIN']['gate']['passed']
    assert metrics['social_q75']['expanded_TRAIN']['gate']['passed']
    scores[0, 0] = 0; scores[4, 3] = .99
    actual = gates.population_gates(rows, scores, 2, 3)
    assert actual['social_q75']['previous_E86_populations']['passed']
    assert not actual['social_q75']['checks']['worst_sid_camera_lte_20']
    assert not actual['social_q75']['passed']


@pytest.mark.parametrize('module,current,previous', [
    (representation, 'e91_representation.json', 'e85_representation.json'),
    (fit, 'e92_fit.json', 'e86_fit.json')])
def test_new_receipts_preserve_predecessors(monkeypatch, tmp_path, module, current, previous):
    (tmp_path / previous).write_text('immutable predecessor')
    monkeypatch.setattr(module, 'EVIDENCE', tmp_path)
    monkeypatch.setattr(module, 'REPORT', tmp_path / 'new_report.json')
    module.write_report({'state': 'synthetic receipt'})
    assert (tmp_path / previous).read_text() == 'immutable predecessor'
    assert json.loads((tmp_path / current).read_text()) == {'state': 'synthetic receipt'}
    with pytest.raises(FileExistsError):
        module.write_report({'state': 'replacement'})


def test_expansion_keeps_weighting_network_and_order_functions():
    assert representation.loss_weights is previous_representation.loss_weights
    assert representation.build_network is previous_representation.build_network
    assert representation.epoch_order is previous_representation.epoch_order


def test_failed_train_prevents_all_development_cache_access(monkeypatch):
    monkeypatch.setattr(dev, 'validate_fit', lambda: None)
    monkeypatch.setattr(dev, 'read', lambda path: {'dev_scoring_permitted': False})
    monkeypatch.setattr(dev, 'digest', lambda path: 'same')
    monkeypatch.setattr(dev, 'validate_cache', lambda: pytest.fail('DEV cache opened after failed TRAIN'))
    with pytest.raises(ValueError, match='TRAIN/runtime'):
        dev.freeze()


def test_successful_train_freezes_dev_with_parent_manifest(monkeypatch, tmp_path):
    manifest = [{'parent_id': 'real', 'source': 'SIDD:test', 'label': 0},
                {'parent_id': 'ai', 'source': 'AI:test', 'label': 1}]
    rows = [r | {'condition': c, 'sha256': r['parent_id'] + c, 'role': 'DEVELOPMENT'}
            for r in manifest for c in dev.CONDITIONS]
    manifest_path = tmp_path / 'manifest.json'
    cache = {'manifest': manifest_path, 'manifest_sha256': 'same', 'reference': {}, 'limits': ''}
    values = {
        dev.FIT_REPORT: {'dev_scoring_permitted': True, 'candidate_sha256': 'same'},
        dev.CACHE_SCORES: {'rows': rows, 'features_sha256': 'same'},
        dev.PREVIOUS_SCORES: {'rows': rows},
        dev.PREVIOUS_REPORT: {'scores_sha256': 'same'},
        dev.DATA_ROOT / 'e83/dev_report.json': {'scores_sha256': 'same'},
        manifest_path: {'rows': manifest},
    }
    writes = {}
    monkeypatch.setattr(dev, 'validate_fit', lambda: None)
    monkeypatch.setattr(dev, 'validate_cache', lambda: cache)
    monkeypatch.setattr(dev, 'digest', lambda path: 'same')
    monkeypatch.setattr(dev, 'read', lambda path: values[path])
    monkeypatch.setattr(dev, 'write_once', lambda path, value: writes.setdefault(path, value))
    result = dev.freeze()
    assert result['state'] == 'E92_single_consumed_E66_cache_comparison_registered'
    assert writes[dev.CONTRACT]['manifest'] == manifest_path


def test_dev_predecessor_cannot_reorder_or_substitute_cached_images():
    manifest = [{'parent_id': 'real', 'source': 'SIDD:test', 'label': 0}]
    rows = [manifest[0] | {'condition': c, 'sha256': c, 'role': 'DEVELOPMENT'}
            for c in dev.CONDITIONS]
    dev.validate_predecessor(rows, rows, manifest)
    with pytest.raises(ValueError, match='identities/order'):
        dev.validate_predecessor(rows[::-1], rows, manifest)
    with pytest.raises(ValueError, match='identities/order'):
        dev.validate_predecessor([rows[0] | {'sha256': 'substituted'}, rows[1]], rows, manifest)
    with pytest.raises(ValueError, match='duplicate'):
        dev.validate_predecessor([rows[0], rows[0]], rows, manifest)
