"""Separate recorded processing history from source-file geometry and cohort flags."""
from collections import Counter
from PIL import Image, PngImagePlugin


def processing_history(row):
    # E54 native is an expansion-cohort flag, not an authenticity/processing certificate.
    if row.get('condition') == 'publisher-224-resize':
        return 'publisher_224_resize'
    if row.get('provenance') == 'e32_fixed_replay':
        return 'local_224_jpeg90_replay'
    if row.get('original_sha256') and row.get('decode') and row['source'].startswith('SID:'):
        return 'local_full_resolution_RAW_render'
    return 'admitted_source_body_upstream_history_unverified'


def inspect_geometry(stream):
    """Lazy header only; do not call load, getexif, verify or pixel conversions."""
    old = PngImagePlugin.MAX_TEXT_CHUNK, PngImagePlugin.MAX_TEXT_MEMORY
    try:
        PngImagePlugin.MAX_TEXT_CHUNK = 65536
        PngImagePlugin.MAX_TEXT_MEMORY = 4 * 65536
        with Image.open(stream) as im:
            width, height = im.size
            if min(width, height) <= 0 or width * height > 100000000:
                raise ValueError('Outside registered geometry budget')
            orientation = None
            exif_error = False
            if im.info.get('exif'):
                try:
                    exif = Image.Exif()
                    exif.load(im.info['exif'])
                    value = exif.get(274)
                    if type(value) is int and 1 <= value <= 8:
                        orientation = value
                except (TypeError, ValueError, OSError, SyntaxError):
                    exif_error = True
            return dict(width=width, height=height, format=im.format, mode=im.mode,
                        header_orientation=orientation, exif_parse_error=exif_error)
    finally:
        PngImagePlugin.MAX_TEXT_CHUNK, PngImagePlugin.MAX_TEXT_MEMORY = old


def summarize(records):
    valid = [r for r in records if 'header_error' not in r]
    def counts(key):
        return dict(sorted(Counter(str(r.get(key)) for r in records).items()))
    return dict(parents=len(records), header_errors=len(records)-len(valid),
        processing_history=counts('processing_history'), legacy_native_flag=counts('legacy_native_flag'),
        formats=dict(sorted(Counter(r['format'] for r in valid).items())),
        geometry_mismatches=sum(not r['declared_geometry_matches'] for r in valid),
        known_224_derivatives=sum(r['processing_history'] in ('publisher_224_resize', 'local_224_jpeg90_replay') for r in records),
        source_pixel_crop_capacity={str(side): sum(min(r['width'], r['height']) >= side for r in valid) for side in (224, 256, 512, 1024)},
        processing_geometry_conflicts=sum(
            r['processing_history'] in ('publisher_224_resize', 'local_224_jpeg90_replay') and
            ((r['width'], r['height']) != (224, 224) or
             (r['processing_history'] == 'local_224_jpeg90_replay' and r['format'] != 'JPEG')) for r in valid),
        larger_than_existing_2048_cap=sum(max(r['width'], r['height']) > 2048 for r in valid),
        exif_parse_errors=sum(r['exif_parse_error'] for r in valid))
