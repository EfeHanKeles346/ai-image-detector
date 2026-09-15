import json
from pathlib import Path
import numpy as np
from PIL import Image
import pytest
from experiments import e132_patch_learning as e


def test_small_real_fit_locks_all_fold_maps_before_metrics(tmp_path, monkeypatch):
    data = tmp_path/'data'; root = data/'e132'; root.mkdir(parents=True)
    evidence = tmp_path/'evidence'; evidence.mkdir()
    ml = tmp_path/'repo/ml'; ml.mkdir(parents=True)
    contract = root/'contract.json'; contract.write_text('{}')
    old = data/'e130'; old.mkdir(); (old/'contract.json').write_text('{}')
    rng = np.random.default_rng(132); tokens = {}; rows = []; folds = [0, 0, 1, 1]
    for i in range(4):
        mask = np.zeros((512, 512), dtype=np.uint8); mask[128:256, 128:256] = 255
        path = tmp_path/f'mask{i}.png'; Image.fromarray(mask).save(path)
        rows.append({'index': i, 'source': f'sensor{i//2}', 'files': {'mask': {'path': str(path)}}})
        for condition in e.CONDITIONS:
            for variant in e.VARIANTS:
                value = rng.normal(0, .1, (32, 32, 384)).astype(np.float32)
                if variant == 'ai_composite': value[8:16, 8:16, 0] += 3
                tokens[f'{i:03d}_{variant}_{condition}'] = value
    np.savez(old/'tokens.npz', **tokens, contract_sha256=e.digest(old/'contract.json'))
    refs = {c: {'accepted_parent_ranking_baselines': {'radial_center_auc': .5},
                'ai_composite': {'metrics': {'pixel_auc': {'mean': .5}}}} for c in e.CONDITIONS}
    (evidence/'e130_patch_drift_audit.json').write_text(json.dumps({'summary': refs}))
    for name, value in [('ROOT', root), ('CONTRACT', contract), ('DATA_ROOT', data), ('EVIDENCE', evidence), ('ML_ROOT', ml)]:
        monkeypatch.setattr(e, name, value)
    monkeypatch.setattr(e, 'validate', lambda: dict(rows=rows, folds=folds, fold_count=2, max_seconds=60, limits='synthetic test'))
    monkeypatch.setattr(e, 'resource_check', lambda *args: None)
    train_calls = []; original_train = e.patch_learning.training_arrays
    def train(selected, *args, **kwargs):
        train_calls.append([r['index'] for r in selected])
        return original_train(selected, *args, **kwargs)
    monkeypatch.setattr(e.patch_learning, 'training_arrays', train)
    original_metric = e.spatial_evaluation.evaluate_triplet
    def metric(*args, **kwargs):
        assert (root/'locked_scores.json').exists()
        with np.load(root/'scores.npz') as saved:
            assert len(saved.files) == 25
        return original_metric(*args, **kwargs)
    monkeypatch.setattr(e.spatial_evaluation, 'evaluate_triplet', metric)
    result = e.fit()
    assert train_calls == [[2, 3], [0, 1]]
    assert result['promotion_allowed'] is False
    assert result['summary']['original']['ai_composite']['mean_pixel_auc'] > .99
    assert result['summary']['original']['authentic']['mean_flagged_area'] < .01
    with pytest.raises(FileExistsError):
        e.fit()


def test_e132_lifecycle_requires_both_restorations(tmp_path, monkeypatch):
    tools = Path(__file__).resolve().parents[1]/'tools'
    monkeypatch.syspath_prepend(str(tools))
    monkeypatch.setenv('PIXELPROOF_DATA_ROOT', str(tmp_path))
    import run_registered_research as runner
    for stage in ('e130_pipeline', 'e131_pipeline'):
        folder = tmp_path/stage; folder.mkdir()
        (folder/'status.json').write_text(json.dumps({'state': 'complete', 'E92_restored': True}))
    runner.patch_learning_ready(tmp_path)
    (tmp_path/'e131_pipeline/status.json').write_text(json.dumps({'state': 'complete', 'E92_restored': False}))
    with pytest.raises(RuntimeError):
        runner.patch_learning_ready(tmp_path)
