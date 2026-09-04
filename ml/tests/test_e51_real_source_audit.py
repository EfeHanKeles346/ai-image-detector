from __future__ import annotations

from types import SimpleNamespace

import pytest

from experiments.e51_real_source_audit import summarize_scmi30


def _rows(count: int, device_count: int = 30):
    rows = []
    for index in range(count):
        branch = "Random" if index < 5_287 else "Similar"
        device = index % device_count + 1
        rows.append(SimpleNamespace(
            name=f"SCMI30-IITRPR/{branch}/D{device:02d}_Phone/image_{index}.jpg",
            total_bytes=100,
        ))
    return rows


def test_scmi30_requires_exact_image_and_device_inventory():
    metadata = SimpleNamespace(current_version_number=2, last_updated="date", license_name="NC")
    found = summarize_scmi30(metadata, _rows(9_937))
    assert found["image_files"] == 9_937
    assert found["device_ids"] == 30
    assert found["branches"] == {"Random": 5_287, "Similar": 4_650}


def test_scmi30_rejects_device_inventory_drift():
    metadata = SimpleNamespace(current_version_number=2, last_updated="date", license_name="NC")
    with pytest.raises(ValueError, match="device inventory"):
        summarize_scmi30(metadata, _rows(9_937, device_count=29))
