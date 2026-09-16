import io
import pytest
from PIL import Image, PngImagePlugin
from pixelproof.header_metadata import inspect_header, summarize_value, TEXT_LIMIT


def test_png_declarations_are_read_without_loading_pixels(monkeypatch):
    data = io.BytesIO(); info = PngImagePlugin.PngInfo()
    info.add_text('parameters', 'untrusted generation settings')
    Image.new('RGB', (32,32)).save(data, format='PNG', pnginfo=info)
    def forbidden(*args, **kwargs): raise AssertionError('Pixel decode is forbidden')
    monkeypatch.setattr(Image.Image, 'load', forbidden)
    result = inspect_header(data.getvalue())
    assert result['fields']['parameters']['text'] == 'untrusted generation settings'
    assert not result['complete_container_metadata'] and not result['pixel_decode']


def test_camera_model_is_not_a_generator_field():
    data=io.BytesIO(); exif=Image.Exif(); exif[272]='Camera model'; exif[305]='Editor'
    Image.new('RGB',(32,32)).save(data,format='JPEG',exif=exif)
    result=inspect_header(data.getvalue())
    assert result['fields']['exif_camera_model']['text']=='Camera model'
    assert 'model' not in result['fields'] and 'generator' not in result['fields']


def test_long_or_empty_metadata_does_not_become_an_unbounded_text_record():
    assert summarize_value('') is None and summarize_value(42) is None
    result=summarize_value('a'*(TEXT_LIMIT+1))
    assert result['text'] is None and result['text_omitted']
    assert len(result['sha256'])==64


def test_header_failure_restores_global_png_limits():
    before=(PngImagePlugin.MAX_TEXT_CHUNK,PngImagePlugin.MAX_TEXT_MEMORY)
    with pytest.raises(OSError): inspect_header(b'not a photo')
    assert (PngImagePlugin.MAX_TEXT_CHUNK,PngImagePlugin.MAX_TEXT_MEMORY)==before
