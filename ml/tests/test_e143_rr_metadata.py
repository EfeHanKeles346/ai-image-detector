import pytest
from experiments.e143_rr_metadata import bound_rows


def test_rr_members_must_bind_to_official_train_bodies(tmp_path):
    a=dict(parent_id='rr:1',source='rr:topic',label=1,role='train',sha256='a'*64,bytes=12,
           path=str(tmp_path/'e42/rr_train/ai/a.png'))
    b=dict(relative_path='rr_train/ai/a.png',source='topic',label=1,sha256='a'*64,bytes=12,
           upstream_member='RRDataset_original_train_val/train/ai/a.png')
    assert bound_rows([a],[b],tmp_path)==[a]
    for change in [{'sha256':'b'*64},{'role':'FINAL'}, {'path':str(tmp_path/'test/a.png')}]:
        with pytest.raises(ValueError):bound_rows([{**a,**change}],[b],tmp_path)
    with pytest.raises(ValueError):bound_rows([a],[{**b,'upstream_member':'RRDataset_original_train_val/val/ai/a.png'}],tmp_path)
    with pytest.raises(ValueError):bound_rows([a],[b,b],tmp_path)
