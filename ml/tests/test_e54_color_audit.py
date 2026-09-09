import numpy as np

from experiments.e54_color_audit import colour_stats


def test_monochrome_is_channel_agreement_not_brightness():
    for value in (0,128,255):
        assert colour_stats(np.full((224,224,3),value,dtype=np.uint8))['near_monochrome_global_crop']


def test_coloured_crop_is_not_monochrome():
    rgb=np.zeros((224,224,3),dtype=np.uint8);rgb[:,:,0]=255
    facts=colour_stats(rgb)
    assert not facts['near_monochrome_global_crop'] and facts['mean_channel_range_255']==255
