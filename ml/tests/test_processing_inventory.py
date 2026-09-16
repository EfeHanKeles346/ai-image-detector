import io
import numpy as np
import pytest
from PIL import Image, JpegImagePlugin, PngImagePlugin
from pixelproof.processing_inventory import processing_history, inspect_geometry, summarize


def test_native_boolean_does_not_certify_original_pixels():
    for native in (True, False, None):
        assert processing_history(dict(source='test', native=native)) == 'admitted_source_body_upstream_history_unverified'
    assert processing_history(dict(source='test', condition='publisher-224-resize')) == 'publisher_224_resize'
    assert processing_history(dict(source='test', provenance='e32_fixed_replay')) == 'local_224_jpeg90_replay'
    assert processing_history(dict(source='SID:Sony', original_sha256='a'*64, decode={'height': 3000})) == 'local_full_resolution_RAW_render'


@pytest.mark.parametrize('format,plugin', [('JPEG', JpegImagePlugin.JpegImageFile), ('PNG', PngImagePlugin.PngImageFile)])
def test_lazy_geometry_does_not_decode_or_scan_png_exif(monkeypatch, format, plugin):
    raw = io.BytesIO()
    exif = Image.Exif(); exif[274] = 6
    Image.fromarray(np.zeros((27, 41, 3), dtype=np.uint8)).save(raw, format=format, exif=exif)
    def forbidden(*args, **kwargs):
        raise AssertionError('Pixel decode or EXIF scan is forbidden')
    monkeypatch.setattr(plugin, 'load', forbidden)
    # JPEG's lazy parser may read its already-loaded EXIF for DPI. PNG getexif
    # can seek after IDAT and decode pixels, so forbid that path specifically.
    if format == 'PNG':
        monkeypatch.setattr(Image.Image, 'getexif', forbidden)
    old = PngImagePlugin.MAX_TEXT_CHUNK, PngImagePlugin.MAX_TEXT_MEMORY
    raw.seek(0)
    result = inspect_geometry(raw)
    assert (result['width'], result['height'], result['header_orientation']) == (41, 27, 6)
    assert set(result) == {'width', 'height', 'format', 'mode', 'header_orientation', 'exif_parse_error'}
    assert (PngImagePlugin.MAX_TEXT_CHUNK, PngImagePlugin.MAX_TEXT_MEMORY) == old


def test_header_limits_restored_on_failure():
    old = PngImagePlugin.MAX_TEXT_CHUNK, PngImagePlugin.MAX_TEXT_MEMORY
    with pytest.raises(OSError):
        inspect_geometry(io.BytesIO(b'not an image'))
    assert (PngImagePlugin.MAX_TEXT_CHUNK, PngImagePlugin.MAX_TEXT_MEMORY) == old


def test_capacity_counts_preserve_failures_and_known_derivatives():
    common = dict(legacy_native_flag=False, processing_history='publisher_224_resize',
                  exif_parse_error=False, declared_geometry_matches=True, format='JPEG')
    records = [dict(common, width=224, height=224),
               dict(common, processing_history='admitted_source_body_upstream_history_unverified', width=4000, height=3000),
               dict(legacy_native_flag=True, processing_history='admitted_source_body_upstream_history_unverified', header_error='OSError')]
    report = summarize(records)
    assert report['parents'] == 3 and report['header_errors'] == 1
    assert report['source_pixel_crop_capacity'] == {'224': 2, '256': 1, '512': 1, '1024': 1}
    assert report['known_224_derivatives'] == 1
    assert report['larger_than_existing_2048_cap'] == 1
    assert report['processing_geometry_conflicts'] == 0
    records[0]['width'] = 300
    assert summarize(records)['processing_geometry_conflicts'] == 1
