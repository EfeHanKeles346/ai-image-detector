from __future__ import annotations

from experiments.e32_r1b_role_manifest import append_records


def test_role_extension_preserves_old_prefix_and_splits_complete_devices() -> None:
    old = [{"record_id": "old", "role": "TRAIN"}]
    rows = []
    for device in range(1, 11):
        for index in range(4):
            rows.append(
                {
                    "eligible": True,
                    "device": f"iPhone14_{device}",
                    "source_key": f"real/{device}/{index}.jpg",
                    "parent_group": f"p:{device}:{index}",
                    "sha256": f"{device:02d}{index:02d}".ljust(64, "0"),
                    "decoded_format": "MPO",
                    "width": 10,
                    "height": 10,
                }
            )
    combined, summary = append_records(old, rows)
    assert combined[0] == old[0]
    assert summary["role_group_counts"] == {"TRAIN": 8, "CALIBRATION": 2}
    by_device = {}
    for row in combined[1:]:
        by_device.setdefault(row["device"], set()).add(row["role"])
    assert all(len(roles) == 1 for roles in by_device.values())
