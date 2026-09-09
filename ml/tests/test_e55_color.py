import numpy as np
import pytest
from PIL import Image

from experiments.e55_color import grayscale_crops,mixture_weights


def test_grayscale_preserves_geometry_uint8_and_is_class_independent():
    crops=np.random.default_rng(55).integers(0,256,(3,224,224,3),dtype=np.uint8)
    original=crops.copy();gray=grayscale_crops(crops)
    assert gray.shape==crops.shape and gray.dtype==np.uint8
    np.testing.assert_array_equal(crops,original)
    np.testing.assert_array_equal(gray[...,0],gray[...,1])
    np.testing.assert_array_equal(gray[...,1],gray[...,2])
    np.testing.assert_array_equal(gray[0],np.asarray(Image.fromarray(crops[0]).convert('L').convert('RGB')))
    np.testing.assert_array_equal(grayscale_crops(gray),gray)


def test_rejects_non_contract_input():
    with pytest.raises(ValueError):grayscale_crops(np.zeros((3,224,224,3),dtype=np.float32))
    with pytest.raises(ValueError):grayscale_crops(np.zeros((3,225,224,3),dtype=np.uint8))


def test_mixture_preserves_parent_class_source_and_total_mass():
    y=np.repeat([0,0,0,1,1],3);s=np.repeat(['a','a','b','c','d'],3);p=np.repeat(list('vwxyz'),3)
    w=mixture_weights(y,s,p,9)
    assert w.sum()==pytest.approx(9)
    assert w[:len(y)].sum()==pytest.approx(7.2)
    assert w[len(y):].sum()==pytest.approx(1.8)
    yy=np.tile(y,2);ss=np.tile(s,2);pp=np.tile(p,2)
    for label in (0,1):assert w[yy==label].sum()==pytest.approx(4.5)
    for source in set(s):assert w[ss==source].sum()==pytest.approx(2.25)
    for parent in set(p):
        m=p==parent
        assert w[len(y):][m].sum()/w[:len(y)][m].sum()==pytest.approx(.25)


def test_invalid_mass_and_split_rejected():
    for mass,fraction in ((0,.2),(1,0),(1,1),(1,-.2)):
        with pytest.raises(ValueError):mixture_weights([0,1],['a','b'],['p','q'],mass,fraction)
