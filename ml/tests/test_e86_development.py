import pytest
from experiments import e86_development as m


def test_failed_train_stops_before_consumed_feature_cache(monkeypatch):
    monkeypatch.setattr(m,'validate_fit',lambda:None)
    monkeypatch.setattr(m,'read',lambda path:{'dev_scoring_permitted':False})
    monkeypatch.setattr(m,'digest',lambda path:'same')
    monkeypatch.setattr(m,'validate_cache',lambda:pytest.fail('cache opened after failed TRAIN'))
    with pytest.raises(ValueError,match='TRAIN/runtime'):m.freeze()
