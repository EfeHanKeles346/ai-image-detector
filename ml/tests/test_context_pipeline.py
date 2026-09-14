"""The unattended runner must stop dependent work after a child fails."""
import importlib.util
from pathlib import Path
import pytest


def test_child_failure_is_recorded_locally_and_raised(tmp_path, monkeypatch):
    path = Path(__file__).parents[1]/'tools/run_e112_e113.py'
    monkeypatch.setenv('PIXELPROOF_DATA_ROOT', str(tmp_path))
    spec = importlib.util.spec_from_file_location('context_pipeline_test', path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    monkeypatch.setattr(module, 'RUN', tmp_path)
    class FailedChild:
        pid = 123
        def wait(self): return 7
    monkeypatch.setattr(module.subprocess, 'Popen', lambda *a, **kw: FailedChild())
    with pytest.raises(RuntimeError, match='exited 7'):
        module.run_stage('experiments.fake_stage', 'fit')
    assert (tmp_path/'status.json').is_file()
    assert (tmp_path/'fake_stage_fit.log').is_file()
