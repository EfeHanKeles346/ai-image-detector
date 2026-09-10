import pytest
from experiments.e56_fivek_metadata import parse_index, license_for


def test_parse_metadata_without_loading_image():
    html = '<table><tr><td><a href="input/a0001-test.dng">DNG</a></td><td>Subject: nature<br>Light: sun<br>Location: outdoors<br>Time: day<br>EXIF: NIKON D70 30mm</td></tr></table>'
    rows = parse_index(html)
    assert rows[0]['id'] == 1
    assert rows[0]['subject'] == 'nature'
    assert rows[0]['exif_text'] == 'NIKON D70 30mm'
    with pytest.raises(ValueError): parse_index(html+html)
    with pytest.raises(ValueError): parse_index(html.replace('input/a0001-test.dng', 'https://example.com/a0001-test.dng'))


def test_license_requires_one_exact_basename():
    assert license_for('a.dng', {'Adobe': {'a.dng'}, 'AdobeMIT': set()}) == 'Adobe'
    with pytest.raises(ValueError): license_for('a.dng', {'Adobe': {'aa.dng'}})
    with pytest.raises(ValueError): license_for('a.dng', {'Adobe': {'a.dng'}, 'AdobeMIT': {'a.dng'}})
