from io import BytesIO
import numpy as np
import pytest
from PIL import Image, ImageOps
from pixelproof import source_pixel_views as new, source_pixel_residual as residual
from experiments import e42_features as old
from experiments.e65_diagnostic import social_q75_bytes


def test_hash_and_assigned_pixels_match_all_historical_transports():
    rng=np.random.default_rng(150)
    image=Image.fromarray(rng.integers(0,256,(1610,2110,3),dtype=np.uint8))
    parents={}
    for i in range(100):parents.setdefault(old.assigned_transport(str(i)),str(i))
    assert set(parents)==set(new.TRANSPORTS)
    for name,parent in parents.items():
        assert new.assigned_transport(parent)==name
        with old.transport_image(image,name) as expected, new.assigned_image(image,parent) as actual:
            np.testing.assert_array_equal(np.asarray(actual),np.asarray(expected))
            assert new.geometry(image.size,parent)[1]==actual.size


def test_clean_and_social_replay_and_q75_is_before_semantic_cap():
    image=Image.fromarray(np.random.default_rng(2).integers(0,256,(1500,2200,3),dtype=np.uint8))
    exif=Image.Exif();exif[274]=6
    buf=BytesIO();image.save(buf,format='JPEG',exif=exif);raw=buf.getvalue()
    actual,sizes=new.extract(raw,'test')
    np.testing.assert_array_equal(actual[0],residual.extract(raw)[1])
    with Image.open(BytesIO(social_q75_bytes(BytesIO(raw)))) as social:
        np.testing.assert_array_equal(actual[3],residual.paired_patch_features(residual.center_patch(social)))
    with Image.open(BytesIO(raw)) as im:full=ImageOps.exif_transpose(im).convert('RGB')
    q=BytesIO();full.save(q,format='JPEG',quality=75,subsampling=2,optimize=False)
    np.testing.assert_array_equal(actual[2],residual.extract(q.getvalue())[1])
    assert tuple(sizes[2])==full.size and max(sizes[2])>2048


def test_undersized_transport_fails_instead_of_resizing_or_dropping():
    parent=next(str(i) for i in range(100) if new.assigned_transport(str(i))=='resize_jpeg')
    raw=BytesIO();Image.new('RGB',(500,183)).save(raw,format='PNG')
    assert min(new.geometry((500,183),parent)[1])<128
    with pytest.raises(ValueError,match='Every condition'):new.extract(raw.getvalue(),parent)
