import pytest
from experiments.e66_sidd_dev import select_noisy,propagate_scene_failures


def member(kind,instance='0001_001_S6_00100_00060_3200_L',frame='010'):
    return {'filename':f'SIDD_Small_sRGB_Only/Data/{instance}/{kind}_SRGB_{frame}.PNG','bytes':100,'directory':False}


def test_sidd_companions_form_one_camera_observation_and_scene_group():
    rows=select_noisy([member('NOISY'),member('GT')])
    assert len(rows)==1 and rows[0]['parent_id']=='SIDD:0001'
    assert rows[0]['scene_group']=='SIDD:scene:001' and rows[0]['iso']==100
    assert not rows[0]['training_allowed']
    with pytest.raises(ValueError,match='companion'):select_noisy([member('NOISY')])
    with pytest.raises(ValueError,match='companion'):select_noisy([member('NOISY'),member('GT',frame='011')])


def test_zip_paths_and_duplicate_observations_rejected():
    with pytest.raises(ValueError,match='unsafe'):select_noisy([member('NOISY')|{'filename':'../escape.PNG'}])
    with pytest.raises(ValueError,match='duplicate'):select_noisy([member('NOISY'),member('NOISY'),member('GT')])


def test_failure_propagates_across_cameras_and_settings_of_same_scene():
    rows=[{'parent_id':p,'scene_group':s} for p,s in [('camera1','scene1'),('camera2','scene1'),('different','scene2')]]
    assert propagate_scene_failures(rows,{'camera1'})=={'camera1','camera2'}
