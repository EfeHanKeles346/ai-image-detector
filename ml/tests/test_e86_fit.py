import json
import pytest
from experiments import e86_fit as m


def test_fit_evidence_cannot_overwrite_rejected_predecessor(monkeypatch,tmp_path):
    old=tmp_path/'e83_fit.json';old.write_text('immutable E83')
    monkeypatch.setattr(m,'EVIDENCE',tmp_path);monkeypatch.setattr(m,'REPORT',tmp_path/'fit.json')
    m.write_report({'state':'E86 synthetic result'})
    assert old.read_text()=='immutable E83'
    assert json.loads((tmp_path/'e86_fit.json').read_text())=={'state':'E86 synthetic result'}
    with pytest.raises(FileExistsError):m.write_report({'state':'replacement'})
