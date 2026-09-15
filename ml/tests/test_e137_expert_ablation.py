import json
import numpy as np
import pytest
from experiments import e137_expert_ablation as e


def test_omission_preserves_named_coordinates_and_rejects_bad_blocks():
    blocks = {name: np.full((3, i + 1), i) for i, name in enumerate(e.EXPERTS)}
    for i, omitted in enumerate(e.EXPERTS):
        result = e.without(blocks, omitted)
        assert result.shape == (3, 10 - i - 1)
        assert set(result.ravel()) == set(range(4)) - {i}
    with pytest.raises(ValueError): e.without(blocks, 'unknown')
    with pytest.raises(ValueError): e.without(blocks | {'dear': np.full((3, 3), np.nan)}, 'dear')
    with pytest.raises(ValueError): e.without(blocks | {'clip': np.ones((2, 2))}, 'dino')


def test_frozen_input_mutation_is_rejected(tmp_path, monkeypatch):
    contract = tmp_path / 'contract.json'; evidence = tmp_path / 'evidence'; evidence.mkdir()
    dependency = tmp_path / 'dependency'; dependency.write_text('first')
    contract.write_text(json.dumps({'inputs': {str(dependency): e.digest(dependency)}}))
    (evidence / 'e137_expert_ablation_contract.json').write_text(json.dumps({'contract_sha256': e.digest(contract)}))
    monkeypatch.setattr(e, 'CONTRACT', contract); monkeypatch.setattr(e, 'EVIDENCE', evidence)
    e.validate(); dependency.write_text('changed')
    with pytest.raises(ValueError, match='Frozen ablation input'): e.validate()


@pytest.fixture
def small_run(tmp_path, monkeypatch):
    root = tmp_path / 'e137'; root.mkdir(); base = tmp_path / 'e131'; base.mkdir()
    evidence = tmp_path / 'evidence'; evidence.mkdir(); ml = tmp_path / 'repo/ml'; ml.mkdir(parents=True)
    contract = root / 'contract.json'; contract.write_text('{}')
    conditions = ['clean', 'assigned_transport', 'q75', 'social_q75']; folds = np.repeat(np.arange(3), 2)
    rows = [dict(parent_id=str(i), label=i % 2, source=f'group{i//2}') for i in range(6)]
    prior = dict(rows=rows, outer_fold={str(i): int(folds[i]) for i in range(6)},
                 components={str(i): f'g{i//2}' for i in range(6)})
    (base / 'contract.json').write_text(json.dumps(prior)); binding = e.digest(base / 'contract.json')
    features = {k: np.column_stack([np.repeat(np.arange(6), 4),
        np.repeat(np.arange(6) % 2 * 6 - 3, 4) + np.tile(np.arange(4) * .1 * (j + 1), 6)])
        for j, k in enumerate(('dino', 'clip', 'dear', 'center_control', 'full_frame'))}
    transform = dict(center=np.zeros(2), scale=np.ones(2), mean=np.zeros(2),
                     components=np.eye(2), latent_scale=np.ones(2))
    parameters = np.arange(9) * .01
    for fold in range(3):
        shared = {name + '_' + k: v for name in e.EXPERTS[:-1] for k, v in transform.items()}
        np.savez(base / f'fold{fold}_shared.npz', **shared, contract_sha256=binding)
        for branch in e.BRANCHES:
            np.savez(base / f'fold{fold}_{branch}.npz', **transform, parameters=parameters, contract_sha256=binding)
    baseline = {b: e.holdout_linear.predict(np.column_stack(
        [features[k] for k in ('dino', 'clip', 'dear', b)]), parameters).reshape(6, 4) for b in e.BRANCHES}
    np.savez(base / 'scores.npz', **baseline, parents=np.array([str(i) for i in range(6)]),
             conditions=conditions, outer_fold=folds, global_role='TRAIN', usage='INTERNAL_HELD_OUT', contract_sha256=binding)
    for name, value in [('ROOT', root), ('CONTRACT', contract), ('EVIDENCE', evidence), ('ML_ROOT', ml)]:
        monkeypatch.setattr(e, name, value)
    monkeypatch.setattr(e.previous, 'ROOT', base); monkeypatch.setattr(e.previous, 'CONTRACT', base / 'contract.json')
    monkeypatch.setattr(e.previous, 'load_features', lambda rows: features)
    monkeypatch.setattr(e, 'validate', lambda: dict(max_seconds=60, conditions=conditions, limits='synthetic'))
    monkeypatch.setattr(e, 'resource_check', lambda *a: None)
    return root, base, folds


def test_actual_24_fits_exclude_outer_fold_and_lock_all_predictions_first(small_run, monkeypatch):
    root, base, folds = small_run
    fits = []; actual_fit = e.holdout_linear.fit_head
    def fit(x, *args, **kwargs):
        fits.append(set(x[:, 0].astype(int).tolist()))
        assert x.shape[1] == 6
        return actual_fit(x, *args, **kwargs)
    monkeypatch.setattr(e.holdout_linear, 'fit_head', fit)
    actual_metrics = e.previous.metrics
    def metrics(*args, **kwargs):
        lock = json.loads((root / 'locked_scores.json').read_text())
        assert len(lock['fits']) == 24
        assert lock['context_omitted_geometry_max_error'] == 0
        return actual_metrics(*args, **kwargs)
    monkeypatch.setattr(e.previous, 'metrics', metrics)
    report = e.fit()
    for i, selected in enumerate(fits): assert selected == set(np.where(folds != i // 8)[0])
    assert len(fits) == 24 and report['promotion_allowed'] is False
    assert report['parents'] == 6 and len(report['passes_internal_individual_nonregression']) == 8
    lock = json.loads((root / 'locked_scores.json').read_text())
    assert all(v == 0 for v in lock['baseline_replays'].values())
    assert all(v['saved_score_max_error'] == 0 for v in lock['fits'].values())
    with pytest.raises(FileExistsError): e.fit()


def test_wrong_comparator_role_stops_before_fitting(small_run, monkeypatch):
    root, base, _ = small_run
    with np.load(base / 'scores.npz', allow_pickle=False) as a: arrays = {k: a[k] for k in a.files}
    arrays['global_role'] = 'TEST'; np.savez(base / 'scores.npz', **arrays)
    def forbidden(*args, **kwargs): raise AssertionError('Must not fit with a wrong-role comparator')
    monkeypatch.setattr(e.holdout_linear, 'fit_head', forbidden)
    with pytest.raises(ValueError, match='role identity'): e.fit()
    assert not (root / 'locked_scores.json').exists()
