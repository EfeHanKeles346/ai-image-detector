import pytest
from PIL import Image
from io import BytesIO
from experiments.e65_diagnostic import rates, paired_report, social_q75_bytes, AI_CUT, REAL_CUT


def test_boundary_decisions_partition_every_real_observation():
    r=rates([{'score':s} for s in [0,REAL_CUT-1e-10,REAL_CUT,AI_CUT-1e-10,AI_CUT,1]])
    assert (r['automatic_real'],r['uncertain'],r['false_ai']) == (2,2,2)
    assert sum(r[k] for k in ['automatic_real','uncertain','false_ai']) == r['views']


def test_transport_requires_complete_unique_pairs_and_counts_both_directions():
    rows=[dict(parent_id=p,source='WIFD',condition=c,score=s) for p,c,s in
          [('a','original',0),('a','social_q75',1),('b','original',1),('b','social_q75',0)]]
    assert paired_report(rows)['WIFD']['q75_new_false_ai']==1
    assert paired_report(rows)['WIFD']['q75_rescued_real']==1
    with pytest.raises(ValueError,match='incomplete'): paired_report(rows[:-1])
    with pytest.raises(ValueError,match='duplicate'): paired_report(rows+[rows[0]])


def test_q75_orients_then_caps_long_side(tmp_path):
    path=tmp_path/'oriented.jpg'
    exif=Image.Exif();exif[274]=6
    Image.new('RGB',(1600,800),(123,70,20)).save(path,exif=exif)
    raw=social_q75_bytes(path)
    with Image.open(BytesIO(raw)) as im:
        assert im.format=='JPEG'
        assert im.size==(540,1080)
        assert im.getexif().get(274,1)==1
    assert social_q75_bytes(path)==raw
