import numpy as np
import pytest
from experiments.e150_source_pixel_features import check_chunk, geometry_audit
from pixelproof.source_pixel_views import CONDITIONS


def test_geometry_keeps_every_parent_and_reports_unsupported_views():
    rows=[dict(parent_id=str(i),width=500,height=183,header_orientation=6) for i in range(100)]
    report=geometry_audit(rows)
    assert all(x['parents']==100 for x in report.values())
    assert report['clean']['unsupported']==0
    assert report['assigned_transport']['unsupported']>0


def test_resume_rejects_wrong_parent_body_binding_conditions_and_incomplete_features():
    rows=[dict(parent_id='a',sha256='a'*64)]
    value=np.zeros((1,4,2,12,25),dtype=np.float32);value[...,12]=1
    saved=dict(binding=np.asarray('binding'),parents=np.asarray(['a']),source_sha256=np.asarray(['a'*64]),
               conditions=np.asarray(CONDITIONS),features=value.reshape(1,4,2,300))
    assert check_chunk(saved,rows,'binding').shape==(1,4,2,300)
    for change in [dict(binding=np.asarray('bad')),dict(parents=np.asarray(['b'])),
                   dict(source_sha256=np.asarray(['b'*64])),dict(conditions=np.asarray(CONDITIONS[::-1])),
                   dict(features=np.zeros((1,4,2,300),dtype=np.float32)),
                   dict(features=np.full((1,4,2,300),np.nan,dtype=np.float32))]:
        with pytest.raises(ValueError):check_chunk(saved|change,rows,'binding')
