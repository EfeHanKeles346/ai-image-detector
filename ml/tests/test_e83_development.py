from pathlib import Path
import pytest
from experiments import e83_development as m


def test_failed_train_never_opens_pixels_or_contract(monkeypatch,tmp_path):
    monkeypatch.setattr(m,'validate_fit',lambda:None)
    monkeypatch.setattr(m,'read',lambda path:{'dev_scoring_permitted':False})
    monkeypatch.setattr(m,'digest',lambda path:'same')
    monkeypatch.setattr(m,'write_once',lambda *args:pytest.fail('write after failed TRAIN'))
    with pytest.raises(ValueError,match='TRAIN guard failed'):m.freeze()


def test_experiment_artifacts_and_model_are_separate():
    assert m.ROOT.name=='e83'
    assert m.FIT_CONTRACT.parent==m.ROOT and m.FIT_REPORT.parent==m.ROOT
    assert m.CANDIDATE.parent==m.ROOT and m.CONTRACT.parent==m.ROOT
    assert m.SCORES.parent==m.ROOT and m.REPORT.parent==m.ROOT
    assert m.model.__name__=='experiments.e83_model'
    assert m.validate_fit.__module__=='experiments.e83_fit'
    source=Path(m.__file__).read_text()
    assert "EVIDENCE/'e83_development.json'" in source
    assert "EVIDENCE/'e83_dev_scores.json'" in source
    assert "EVIDENCE/'e83_dev_contract.json'" in source
    assert "EVIDENCE/'e80_" not in source
