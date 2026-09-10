import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace


def test_wb_metadata_predicate_without_installing_rawpy_in_ml(monkeypatch):
    monkeypatch.setitem(sys.modules,'rawpy',SimpleNamespace())
    path=Path(__file__).resolve().parents[1]/'experiments/e57_raw_probe.py'
    spec=importlib.util.spec_from_file_location('isolated_probe_test',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    assert module.valid_wb([1.,2.,3.,0.])
    assert not module.valid_wb([0.,1.,0.,0.])
    assert not module.valid_wb([1.,2.,float('nan'),0.])
    assert not module.valid_wb([1.,2.,3.])
