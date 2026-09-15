import numpy as np
import pytest
from pixelproof.mask_location import corner_mask, accepted_indices, CORNERS


@pytest.mark.parametrize('index',range(16))
def test_translation_preserves_shape_area_and_corner(index):
    mask=np.zeros((512,512),dtype=np.uint8);mask[180:280,140:340]=255;mask[200:230,160:190]=0
    original=mask.copy();shifted,g=corner_mask(mask,index)
    x,y,right,bottom=g['new_bbox']
    np.testing.assert_array_equal(shifted[y:bottom,x:right],mask[180:280,140:340])
    np.testing.assert_array_equal(mask,original)
    assert np.count_nonzero(shifted)==np.count_nonzero(mask)
    assert g['corner']==CORNERS[index%4]
    assert (x==16) if index%2==0 else (right==496)
    assert (y==16) if index%4<2 else (bottom==496)


@pytest.mark.parametrize('bad',[-1,16,True,1.0])
def test_invalid_index_rejected(bad):
    mask=np.zeros((512,512),dtype=np.uint8);mask[50:100,50:100]=255
    with pytest.raises(ValueError):corner_mask(mask,bad)


def test_no_clipping_or_nonbinary_admission():
    mask=np.zeros((512,512),dtype=np.uint8);mask[1:500,1:500]=255
    with pytest.raises(ValueError):corner_mask(mask,0)
    mask=np.zeros((512,512),dtype=np.uint8);mask[100:200,100:200]=1
    with pytest.raises(ValueError):corner_mask(mask,0)


def cases():
    checks=dict(safety_check_negative=True,nonblank_output=True,masked_pixels_changed=True,background_exact=True)
    return [dict(index=i,seed=119000+i,branch='fp16_sdpa',passed=True,checks=checks.copy()) for i in range(16)]


def test_failures_retained_without_replacement():
    rows=cases();rows[3]['passed']=False;rows[3]['checks']['safety_check_negative']=False
    rows[9]=dict(index=9,seed=119009,branch='fp16_sdpa',passed=False,first_nonfinite_stage='unet',rendered_images=0,safety_classification_performed=False)
    assert accepted_indices(rows)==[i for i in range(16) if i not in (3,9)]
    with pytest.raises(ValueError):accepted_indices(rows[:15])
    rows[3]['checks']['safety_check_negative']=True
    with pytest.raises(ValueError):accepted_indices(rows)
