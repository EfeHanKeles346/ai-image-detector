import itertools
import json
import numpy as np
from experiments import e146_nested_calibration as exp


def test_fit_cal_eval_pipeline_never_fits_on_other_folds(tmp_path, monkeypatch):
    root = tmp_path / 'run'
    root.mkdir()
    evidence = tmp_path / 'evidence'
    evidence.mkdir()
    contract = root / 'contract.json'
    contract.write_text('{}')
    rows = [dict(parent_id=str(i), label=i % 2, source=f'source{i // 6}', role='TRAIN') for i in range(18)]
    prior = dict(rows=rows, outer_fold={str(i): i // 6 for i in range(18)},
                 components={str(i): f'group{i // 6}' for i in range(18)})
    prior_path = tmp_path / 'prior.json'
    prior_path.write_text(json.dumps(prior))
    assignments = [dict(FIT=f, CAL=c, EVAL=e) for f, c, e in itertools.permutations(range(3))]
    conditions = ['clean', 'assigned_transport', 'q75', 'social_q75']
    monkeypatch.setattr(exp, 'ROOT', root)
    monkeypatch.setattr(exp, 'EVIDENCE', evidence)
    monkeypatch.setattr(exp, 'CONTRACT', contract)
    monkeypatch.setattr(exp.previous, 'CONTRACT', prior_path)
    monkeypatch.setattr(exp, 'validate', lambda: dict(max_seconds=30, conditions=conditions, assignments=assignments, limits='synthetic'))
    monkeypatch.setattr(exp, 'resource_check', lambda _: None)
    monkeypatch.setattr(exp, 'FEATURES', [(k, 1) for k in ('dino', 'clip', 'dear', 'full_frame')])
    features = np.repeat(np.array([[i // 6, .9 if i % 2 else .1] for i in range(18)], dtype=np.float32), 4, axis=0)
    monkeypatch.setattr(exp.previous, 'load_features', lambda _: {k: features.copy() for k in ('dino', 'clip', 'dear', 'full_frame', 'center_control')})
    seen_folds = []

    def fit_map(values, width):
        assert len(values) == 24 and width == 1
        assert len(set(values[:, 0])) == 1
        seen_folds.append(int(values[0, 0]))
        return dict(center=np.zeros(2), scale=np.ones(2), mean=np.zeros(2), components=np.array([[0., 1.]]), latent_scale=np.ones(1))

    def fit_head(x, y, weights, check):
        assert x.shape == (24, 4) and np.array_equal(x[:, 0] > .5, y == 1)
        assert np.isclose(weights.sum(), 1)
        return np.array([2., 2., 2., 2., -4.]), dict(success=True)

    def checked_components(*args, **kwargs):
        # This function is first called for EVAL: both locks must already exist.
        assert (root / 'locked_scores.json').exists()
        assert (root / 'locked_calibration.json').exists()
        return []

    monkeypatch.setattr(exp.holdout_linear, 'fit_map', fit_map)
    monkeypatch.setattr(exp.holdout_linear, 'fit_head', fit_head)
    monkeypatch.setattr(exp, 'component_metrics', checked_components)
    result = exp.fit()
    assert seen_folds == [0] * 4 + [1] * 4 + [2] * 4
    assert result['accepted_CAL_assignments'] == 6
    assert result['passed_EVAL_assignments'] == 6
    with np.load(root / 'scores.npz') as saved:
        for fold in range(3):
            assert np.isnan(saved['scores'][fold, saved['folds'] == fold]).all()
            assert np.isfinite(saved['scores'][fold, saved['folds'] != fold]).all()
