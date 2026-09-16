import io
import numpy as np
import pytest
from PIL import Image
from pixelproof.source_pixel_residual import center_patch, residual_pairs, paired_patch_features, extract


def test_center_crop_preserves_exact_pixels_without_resize(monkeypatch):
    rgb=np.random.default_rng(149).integers(0,256,(183,501,3),dtype=np.uint8)
    def forbidden(*args,**kwargs):
        raise AssertionError('Source crop must not resize')
    monkeypatch.setattr(Image.Image,'resize',forbidden)
    np.testing.assert_array_equal(center_patch(Image.fromarray(rgb)),rgb[27:155,186:314])
    with pytest.raises(ValueError):center_patch(Image.new('RGB',(127,300)))


def test_exif_orientation_precedes_center_crop():
    rgb=np.random.default_rng(2).integers(0,256,(183,201,3),dtype=np.uint8)
    image=Image.fromarray(rgb);image.getexif()[274]=6
    oriented=np.rot90(rgb,k=3)
    np.testing.assert_array_equal(center_patch(image),oriented[36:164,27:155])


def test_flat_histograms_and_paired_response_are_exact():
    patch=np.full((128,128,3),127,dtype=np.uint8)
    expected=np.zeros((12,25),dtype=np.float32);expected[:,12]=1
    np.testing.assert_array_equal(residual_pairs(patch),expected.ravel())
    pair=paired_patch_features(patch)
    np.testing.assert_array_equal(pair[0],pair[1])


def test_joint_histogram_matches_independent_scalar_counts_without_uint_wrap():
    rng=np.random.default_rng(6);patch=rng.integers(0,256,(128,128,3),dtype=np.uint8)
    got=residual_pairs(patch).reshape(12,25)
    offset=0
    for channel in range(3):
        for order in (1,2):
            for axis in (0,1):
                residual=np.diff(patch[:,:,channel].astype(np.int64),n=order,axis=axis)
                counts=np.zeros(25)
                for y in range(residual.shape[0]-(axis==0)):
                    for x in range(residual.shape[1]-(axis==1)):
                        a=max(-2,min(2,round(int(residual[y,x])/4)))+2
                        b=max(-2,min(2,round(int(residual[y+(axis==0),x+(axis==1)])/4)))+2
                        counts[a*5+b]+=1
                np.testing.assert_allclose(got[offset],counts/counts.sum(),rtol=0,atol=1e-7)
                offset+=1
    np.testing.assert_allclose(got.sum(axis=1),1,atol=1e-6)


def test_control_uses_only_the_selected_patch_and_extract_replays():
    rgb=np.random.default_rng(4).integers(0,256,(180,180,3),dtype=np.uint8)
    patch=center_patch(Image.fromarray(rgb));before=rgb.copy();before[:20]=0
    np.testing.assert_array_equal(paired_patch_features(patch),paired_patch_features(center_patch(Image.fromarray(before))))
    raw=io.BytesIO();Image.fromarray(rgb).save(raw,format='PNG')
    p,a=extract(raw.getvalue());q,b=extract(raw.getvalue())
    np.testing.assert_array_equal(p,q);np.testing.assert_array_equal(a,b)
    assert np.abs(a[1]-a[0]).mean()>0
    with pytest.raises(ValueError):residual_pairs(p.astype(np.float32))


def test_grayscale_and_alpha_match_existing_rgb_conversion():
    for mode,color in [('L',100),('RGBA',(30,60,90,0))]:
        image=Image.new(mode,(128,128),color)
        np.testing.assert_array_equal(center_patch(image),np.asarray(image.convert('RGB')))
