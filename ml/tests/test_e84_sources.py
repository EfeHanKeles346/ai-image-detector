import hashlib
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from experiments import e84_sources as m


def test_parquet_reader_converts_only_selected_train_cells(tmp_path,monkeypatch):
    bodies=[f'body{i}'.encode() for i in range(11)];path=tmp_path/'images.parquet'
    pq.write_table(pa.table({'image':[{'bytes':b} for b in bodies],'secret_label':list(range(11))}),path,row_group_size=4)
    actual=[];original=m.legacy.raw_image
    def inspect(value):actual.append(value['bytes']);return original(value)
    monkeypatch.setattr(m.legacy,'raw_image',inspect)
    result=dict(m.selected_cells(path,'image',{1,7,10}))
    assert result=={i:bodies[i] for i in [1,7,10]}
    assert actual==[bodies[i] for i in [1,7,10]]
    with pytest.raises(ValueError,match='absent'):list(m.selected_cells(path,'image',{15}))


def test_no_standardized_body_or_protected_role_fallback():
    body=b'original';row={'role':'train','sha256':hashlib.sha256(body).hexdigest()}
    m.verify_body(body,row)
    for raw,r in [(b'reencoded',row),(body,row|{'role':'DEVELOPMENT'})]:
        with pytest.raises(ValueError,match='TRAIN role'):m.verify_body(raw,r)
