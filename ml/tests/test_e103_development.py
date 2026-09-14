import pytest
from experiments import e103_development as dev


def test_failed_training_denies_development_cache(monkeypatch, tmp_path):
    report = tmp_path/'fit.json'; report.write_text('{"dev_scoring_permitted": false}')
    monkeypatch.setattr(dev, 'FIT_REPORT', report)
    monkeypatch.setattr(dev, 'EVIDENCE', tmp_path)
    (tmp_path/'e103_fit.json').write_bytes(report.read_bytes())
    monkeypatch.setattr(dev, 'validate_fit', lambda: None)
    def forbidden():
        pytest.fail('No DEV cache access after failed TRAIN')
    monkeypatch.setattr(dev, 'validate_cache', forbidden)
    with pytest.raises(ValueError, match='TRAIN/runtime pass'):
        dev.freeze()
