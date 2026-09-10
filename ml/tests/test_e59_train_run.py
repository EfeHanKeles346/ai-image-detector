import fcntl

import pytest

from experiments import e59_train_run as r


def test_pending_stages_require_all_nine_fits(tmp_path, monkeypatch):
    monkeypatch.setattr(r, 'ROOT', tmp_path)
    monkeypatch.setattr(r, 'CONTRACT', tmp_path / 'contract.json')
    monkeypatch.setattr(r, 'RESULT', tmp_path / 'result.json')
    monkeypatch.setattr(r, 'require_features', lambda: None)
    assert r.pending() == ['freeze', 'fit', 'report']
    r.CONTRACT.touch()
    (tmp_path / 'training').mkdir()
    for arm in r.ARMS:
        for fold in range(3):
            (tmp_path / 'training' / f'{arm}_fold{fold}.json').touch()
    assert r.pending() == ['report']
    r.RESULT.touch()
    assert r.pending() == []


def test_training_cannot_overlap_extraction_lock(tmp_path, monkeypatch):
    work = tmp_path / 'ml/work'
    work.mkdir(parents=True)
    monkeypatch.setattr(r, '__file__', str(tmp_path / 'ml/experiments/e59_train_run.py'))
    monkeypatch.setattr(r, 'safe', lambda deadline: None)
    def forbidden():
        raise AssertionError('must not inspect/start stages while extraction holds lock')
    monkeypatch.setattr(r, 'pending', forbidden)
    with (work / 'e59_features.lock').open('a') as held:
        fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):
            r.run(1)
