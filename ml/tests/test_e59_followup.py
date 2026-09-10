from types import SimpleNamespace

import pytest

from experiments import e59_followup as f


def test_guard_identity_and_exit(monkeypatch):
    monkeypatch.setattr(f.subprocess, 'run', lambda *a, **k: SimpleNamespace(returncode=0, stdout='python -m experiments.e59_features run --minutes 60'))
    assert f.guard_present(123)
    monkeypatch.setattr(f.subprocess, 'run', lambda *a, **k: SimpleNamespace(returncode=0, stdout='python unrelated.py'))
    with pytest.raises(RuntimeError):
        f.guard_present(123)
    monkeypatch.setattr(f.subprocess, 'run', lambda *a, **k: SimpleNamespace(returncode=1, stdout=''))
    assert not f.guard_present(123)
    with pytest.raises(ValueError):
        f.guard_present(-1)


def test_handoff_respects_remaining_total_time():
    assert f.remaining_minutes(3600, 600) == 50
    assert f.remaining_minutes(3600, 3500) == 1
    with pytest.raises(TimeoutError):
        f.remaining_minutes(3600, 3541)
