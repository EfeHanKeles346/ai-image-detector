import pytest
from experiments import e148_processing_inventory as audit


def test_only_unique_local_train_rows_are_eligible(tmp_path, monkeypatch):
    monkeypatch.setattr(audit, 'DATA_ROOT', tmp_path)
    row = dict(parent_id='test', role='TRAIN', label=0, sha256='a'*64, path=str(tmp_path/'body.png'))
    audit.validate_rows([row])
    for change in [{'role': 'DEV'}, {'label': True}, {'path': str(tmp_path.parent/'body.png')}]:
        with pytest.raises(ValueError):
            audit.validate_rows([row | change])
    with pytest.raises(ValueError):
        audit.validate_rows([row, row])
