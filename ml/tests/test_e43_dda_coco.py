import zipfile

import pytest

from experiments.e43_dda_coco import (
    COCO_BYTES,
    COCO_ETAG,
    EXPECTED_VARIANT_COUNTS,
    real_ids_from_infos,
    synthetic_ids_from_names,
    validate_coco_headers,
)


def _info(name: str, size: int = 10) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name)
    info.file_size = size
    return info


def test_coco_headers_bind_size_and_etag():
    result = validate_coco_headers({
        "content-length": str(COCO_BYTES),
        "etag": COCO_ETAG,
        "last-modified": "Wed, 11 Jul 2018 05:08:47 GMT",
    })
    assert result["bytes"] == COCO_BYTES


def test_coco_real_schema_rejects_wrong_path():
    infos = [_info(f"val2017/{index:012d}.jpg") for index in range(5_000)]
    assert len(real_ids_from_infos(infos)) == 5_000
    infos[-1] = _info("train2017/000000004999.jpg")
    with pytest.raises(ValueError, match="unexpected COCO member"):
        real_ids_from_infos(infos)


def test_synthetic_structure_binds_every_variant_count():
    names = []
    for variant, count in EXPECTED_VARIANT_COUNTS.items():
        names.extend(f"DDA-COCO/{variant}/val2017/{index:012d}.jpg" for index in range(count))
    grouped = synthetic_ids_from_names(names)
    assert {variant: len(ids) for variant, ids in grouped.items()} == EXPECTED_VARIANT_COUNTS


def test_synthetic_structure_rejects_unknown_variant():
    with pytest.raises(ValueError, match="unexpected DDA-COCO member"):
        synthetic_ids_from_names(["DDA-COCO/unknown/val2017/000000000001.jpg"])
