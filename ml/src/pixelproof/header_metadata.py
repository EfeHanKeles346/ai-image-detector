"""Bounded, lazy Pillow header inspection; metadata is untrusted declaration only."""
import hashlib
import io
from PIL import Image, PngImagePlugin

TEXT_LIMIT = 65536
INFO_KEYS = {'prompt', 'negative_prompt', 'parameters', 'workflow', 'generator',
             'model', 'model_name', 'software', 'comment', 'description', 'author'}
EXIF_KEYS = {270: 'exif_description', 271: 'exif_make', 272: 'exif_camera_model',
             305: 'exif_software', 37510: 'exif_user_comment'}


def summarize_value(value):
    if isinstance(value, str):
        raw = value.encode('utf-8', errors='replace')
        text = value if len(raw) <= TEXT_LIMIT else None
    elif isinstance(value, bytes):
        raw = value
        text = raw.decode('utf-8', errors='replace') if len(raw) <= TEXT_LIMIT else None
    else:
        return None
    if not raw:
        return None
    return dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest(), text=text,
                text_omitted=len(raw) > TEXT_LIMIT)


def inspect_header(raw):
    # Prevent compressed text bombs; no load(), verify(), getexif() or pixel conversion.
    # PNG getexif() can load pixels while looking for EXIF after IDAT, so parse only
    # an EXIF payload already present in the lazy header info dictionary.
    old_chunk, old_memory = PngImagePlugin.MAX_TEXT_CHUNK, PngImagePlugin.MAX_TEXT_MEMORY
    try:
        PngImagePlugin.MAX_TEXT_CHUNK = TEXT_LIMIT
        PngImagePlugin.MAX_TEXT_MEMORY = 4 * TEXT_LIMIT
        with Image.open(io.BytesIO(raw)) as im:
            if im.format not in ('PNG', 'JPEG', 'WEBP'):
                raise ValueError('Unsupported registered photo container')
            fields = {}
            for key, value in im.info.items():
                name = str(key).casefold()
                if name in INFO_KEYS:
                    item = summarize_value(value)
                    if item: fields[name] = item
            exif_error = None
            if im.info.get('exif'):
                try:
                    exif = Image.Exif()
                    exif.load(im.info['exif'])
                    for key, name in EXIF_KEYS.items():
                        item = summarize_value(exif.get(key))
                        if item: fields[name] = item
                except (ValueError, TypeError, OSError, SyntaxError) as exc:
                    exif_error = type(exc).__name__
            return dict(format=im.format, fields=fields, exif_parse_error=exif_error,
                        pixel_decode=False, complete_container_metadata=False)
    finally:
        PngImagePlugin.MAX_TEXT_CHUNK, PngImagePlugin.MAX_TEXT_MEMORY = old_chunk, old_memory
