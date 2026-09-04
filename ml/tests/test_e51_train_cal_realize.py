from __future__ import annotations

from collections import Counter

from io import BytesIO
from PIL import Image

from experiments.e51_train_cal_realize import _decode, _phash_image, select_clean_scimd


def _rows() -> list[dict]:
    return [
        {"identity": f"{device}-{index}", "device": device, "rank": f"{index:02d}",
         "sha256": f"sha-{device}-{index}", "dhash": f"dh-{device}-{index}"}
        for device in ("a", "b") for index in range(4)
    ]


def test_clean_scimd_selection_is_device_balanced():
    selected, reasons = select_clean_scimd(_rows(), set(), set(), target_per_device=2)
    assert reasons == {}
    assert Counter(row["device"] for row in selected) == {"a": 2, "b": 2}


def test_clean_scimd_selection_uses_prebound_headroom():
    rows = _rows()
    protected = {rows[0]["sha256"]}
    selected, reasons = select_clean_scimd(rows, protected, set(), target_per_device=3)
    assert rows[0]["identity"] in reasons
    assert len(selected) == 6
    assert rows[0]["identity"] not in {row["identity"] for row in selected}


def test_clean_scimd_rejects_insufficient_device_headroom():
    rows = _rows()
    try:
        select_clean_scimd(rows, {row["sha256"] for row in rows if row["device"] == "a"}, set(),
                           target_per_device=1)
    except ValueError as error:
        assert "insufficient clean SCIMD-17" in str(error)
    else:
        raise AssertionError("depleted device should fail closed")


def test_decode_records_complementary_dhash_conventions():
    image = Image.new("RGB", (16, 16))
    for x in range(16):
        for y in range(16):
            image.putpixel((x, y), (x * 15, y * 9, 0))
    output = BytesIO()
    image.save(output, format="PNG")
    facts, _ = _decode(output.getvalue(), "synthetic")
    assert int(facts["dhash"], 16) ^ int(facts["legacy_dhash"], 16) == (1 << 64) - 1


def test_phash_separates_flat_dhash_collision_candidates():
    left = Image.new("RGB", (32, 32), "black")
    right = Image.new("RGB", (32, 32), "black")
    for index in range(32):
        left.putpixel((index, index), (255, 255, 255))
        right.putpixel((31 - index, index), (255, 255, 255))
    assert (_phash_image(left) ^ _phash_image(right)).bit_count() > 4
