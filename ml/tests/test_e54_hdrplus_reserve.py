import base64
import hashlib

import pytest

from experiments.e54_hdrplus_reserve import select, verify_bytes


def test_excludes_synthetic_and_preserves_fixed_count():
    rows = [{'name': f'20171106/results_20171023/A001_20160101_123456_{i:03d}/final.jpg', 'size': 10} for i in range(100)]
    rows.append({'name': '20171106/results_20171023/synthetic_boxes_day/final.jpg', 'size': 1})
    chosen, total, synthetic = select(rows)
    assert len(chosen) == 100 and total == 101 and len(synthetic) == 1
    assert all('synthetic' not in r['name'] for r in chosen)


def test_publisher_checksum_detects_changed_image():
    raw = b'example-image'
    row = {'bytes': len(raw), 'md5Hash': base64.b64encode(hashlib.md5(raw).digest()).decode()}
    verify_bytes(raw, row)
    with pytest.raises(ValueError):
        verify_bytes(b'changed-image', row)


def test_unknown_capture_fails_before_selection():
    with pytest.raises(ValueError, match='unknown capture'):
        select([{'name': '20171106/results_20171023/ambiguous/final.jpg', 'size': 1}])
