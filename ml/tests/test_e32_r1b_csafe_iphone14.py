from __future__ import annotations

import pytest

from experiments.e32_r1b_csafe_iphone14 import parse_member


def test_iphone14_member_contract_extracts_device_content_and_lens() -> None:
    assert parse_member("iPhone14/iPhone14_10/natural/telephoto/IMG_1234.JPG") == {
        "device": "iPhone14_10",
        "content_type": "natural",
        "lens": "telephoto",
        "filename": "IMG_1234.JPG",
    }


@pytest.mark.parametrize(
    "name",
    [
        "iPhone14/iPhone14_11/natural/wide/IMG_1.JPG",
        "iPhone14/iPhone14_1/unknown/wide/IMG_1.JPG",
        "iPhone14/iPhone14_1/natural/macro/IMG_1.JPG",
        "iPhone14/iPhone14_1/natural/wide/IMG_1.PNG",
        "../iPhone14/iPhone14_1/natural/wide/IMG_1.JPG",
    ],
)
def test_iphone14_member_contract_rejects_unknown_hierarchy(name: str) -> None:
    with pytest.raises(ValueError, match="unexpected iPhone14"):
        parse_member(name)
