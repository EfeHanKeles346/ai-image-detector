import copy
from io import BytesIO
import numpy as np
from PIL import Image
import pytest
from experiments import e126_fullframe_development as dev
from experiments.e123_fullframe_probe import transported_images, fullframe_array
from experiments.e42_features import texture_crops
from experiments.e112_context_features import pooled


def fit_report(passing):
    return {'state': 'E125_paired_TRAIN_ablation_complete', 'reports': {name: {
        'dev_scoring_permitted': name in passing,
        'state': 'E125_TRAIN_guard_passed' if name in passing else 'E125_TRAIN_guard_failed',
        'solver': {'success': True, 'max_constraint_violation': 0., 'minimum_ai_logit_shift': 0.},
        'runtime': {'passed': True}, 'retention': {'passes_train_retention': True}}
        for name in dev.BRANCHES}}


def test_registration_rejects_failed_training_before_any_dev_read(monkeypatch):
    monkeypatch.setattr(dev, 'validate_fit', lambda: None)
    monkeypatch.setattr(dev, 'digest', lambda _: 'matching')
    reads = []
    def read(path):
        assert path == dev.DATA_ROOT/'e125/report.json', 'DEV accessed before TRAIN passed'
        reads.append(path)
        return fit_report([])
    monkeypatch.setattr(dev, 'read', read)
    with pytest.raises(ValueError, match='Neither'): dev.freeze()
    assert len(reads) == 1


def test_complete_paired_fit_is_required_even_when_one_branch_passes():
    report = fit_report([dev.BRANCHES[1]])
    assert dev.eligible(report) == [dev.BRANCHES[1]]
    del report['reports'][dev.BRANCHES[0]]
    with pytest.raises(ValueError, match='Both'): dev.eligible(report)
    assert dev.eligible(fit_report([]), allow_none=True) == []


@pytest.mark.parametrize('field,value', [('success', False), ('max_constraint_violation', .01),
    ('max_constraint_violation', float('nan')), ('minimum_ai_logit_shift', -.01),
    ('minimum_ai_logit_shift', float('nan'))])
def test_permission_cannot_override_solver_failure(field, value):
    report = fit_report(dev.BRANCHES)
    report['reports'][dev.BRANCHES[0]]['solver'][field] = value
    with pytest.raises(ValueError, match='Inconsistent'): dev.eligible(report)


@pytest.mark.parametrize('permission', [None, 1, 'true', False])
def test_inconsistent_permission_is_not_silently_skipped(permission):
    report = fit_report(dev.BRANCHES)
    report['reports'][dev.BRANCHES[0]]['dev_scoring_permitted'] = permission
    with pytest.raises(ValueError, match='Inconsistent'): dev.eligible(report, allow_none=True)


def test_dev_views_match_train_recipe_without_double_social_compression():
    pixels = np.random.default_rng(126).integers(0, 256, (290, 790, 3), dtype=np.uint8)
    stream = BytesIO(); Image.fromarray(pixels).save(stream, format='PNG')
    transported = transported_images(stream.getvalue(), 'synthetic-parent')
    for image in (transported[0], transported[3]):
        center, full, repeated = dev.paired_views(image)
        assert np.array_equal(center, texture_crops(image)[0])
        assert np.array_equal(full, fullframe_array(image))
        assert np.array_equal(full, repeated)
        assert not np.array_equal(center, full)


def reference_fixture():
    rows = [{'record_id': 'a', 'sha256': 'first'}, {'record_id': 'b', 'sha256': 'second'}]
    cache = dict(binding=np.array('bound'), roles=np.array(['DEVELOPMENT']*2),
        record_ids=np.array(['a', 'b']), image_sha256=np.array(['first', 'second']),
        raw=np.random.default_rng(126).normal(size=(2, 3, 768)).astype(np.float32))
    return rows, cache, pooled(cache['raw'])


@pytest.mark.parametrize('field,value', [('binding', np.array('changed')),
    ('roles', np.array(['TRAIN']*2)), ('record_ids', np.array(['b', 'a'])),
    ('image_sha256', np.array(['second', 'first']))])
def test_reference_cache_cannot_be_reordered_relabelled_or_rebound(field, value):
    rows, cache, clip = reference_fixture()
    assert np.array_equal(dev.reference_raw(cache, rows, 'bound', clip), cache['raw'][:, 0])
    cache[field] = value
    with pytest.raises(ValueError, match='identity/role/order'): dev.reference_raw(cache, rows, 'bound', clip)


def test_reference_features_and_new_vectors_fail_closed_on_drift():
    rows, cache, clip = reference_fixture()
    altered = copy.deepcopy(cache); altered['raw'][0, 1, 0] += 1
    with pytest.raises(ValueError, match='aggregate'): dev.reference_raw(altered, rows, 'bound', clip)
    raw = np.ones((3, 768), dtype=np.float32)
    assert dev.verify_encoded(raw, raw[0], 1e-5) == (0., 0.)
    with pytest.raises(ValueError, match='parity'): dev.verify_encoded(raw, raw[0]+.1, 1e-5)
    raw[2, 0] += .1
    with pytest.raises(ValueError, match='parity'): dev.verify_encoded(raw, raw[0], 1e-5)
    raw[2, 0] = float('nan')
    with pytest.raises(ValueError, match='Invalid'): dev.verify_encoded(raw, raw[0], 1e-5)
