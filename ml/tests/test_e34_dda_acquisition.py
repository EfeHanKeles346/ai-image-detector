from __future__ import annotations

from copy import deepcopy
import zipfile

import pytest

from experiments.e34_dda_acquisition import (
    EXPECTED_BYTES,
    FILENAME,
    REPO_ID,
    REVISION,
    inspect_zip,
    validate_repository,
)


def _repository() -> dict:
    return {
        "id": REPO_ID,
        "sha": REVISION,
        "cardData": {"license": "apache-2.0"},
        "siblings": [{"rfilename": FILENAME}],
    }


def test_dda_repository_binds_revision_licence_and_archive() -> None:
    result = validate_repository(_repository())
    assert result["bytes"] == EXPECTED_BYTES
    assert result["role"] == "paired_train_cal_candidate"


def test_dda_repository_rejects_revision_drift() -> None:
    payload = deepcopy(_repository())
    payload["sha"] = "0" * 40
    with pytest.raises(ValueError, match="revision"):
        validate_repository(payload)


def _zip_info(name: str, size: int = 10) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name)
    info.file_size = size
    info.compress_size = size
    return info


def test_dda_zip_inventory_counts_images() -> None:
    result = inspect_zip([_zip_info("DDA/real/1.jpg"), _zip_info("DDA/fake/1.png")])
    assert result["image_count"] == 2
    assert result["by_top"] == {"DDA": 2}


def test_dda_zip_inventory_rejects_traversal() -> None:
    with pytest.raises(ValueError, match="unsafe"):
        inspect_zip([_zip_info("DDA/../escape.jpg")])
