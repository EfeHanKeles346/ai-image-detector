from __future__ import annotations

from pathlib import Path
import zipfile

import pytest

from experiments.e51_scimd17_transfer import (
    ARCHIVE_BYTES,
    ARCHIVE_MD5,
    ARCHIVE_NAME,
    CONTENT_URL,
    ZENODO_DOI,
    ZENODO_LICENSE,
    ZENODO_RECORD,
    ZENODO_VERSION,
    summarize_archive,
    validate_record,
)


def _record() -> dict:
    return {
        "id": ZENODO_RECORD,
        "doi": ZENODO_DOI,
        "metadata": {"version": ZENODO_VERSION, "license": {"id": ZENODO_LICENSE}},
        "files": [{"key": ARCHIVE_NAME, "size": ARCHIVE_BYTES,
                   "checksum": f"md5:{ARCHIVE_MD5}", "links": {"self": CONTENT_URL}}],
    }


def test_scimd17_record_requires_exact_identity_bytes_and_license():
    assert validate_record(_record())["bytes"] == ARCHIVE_BYTES
    for mutation in ("id", "license", "bytes", "url"):
        payload = _record()
        if mutation == "id":
            payload["id"] += 1
        elif mutation == "license":
            payload["metadata"]["license"]["id"] = "unknown"
        elif mutation == "bytes":
            payload["files"][0]["size"] += 1
        else:
            payload["files"][0]["links"]["self"] += "?changed=1"
        with pytest.raises(ValueError, match="SCIMD-17"):
            validate_record(payload)


def test_scimd17_archive_summary_is_model_blind(tmp_path: Path):
    path = tmp_path / "sample.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("device/a.jpg", b"not-decoded-here")
        archive.writestr("README.txt", b"terms")
    summary = summarize_archive(path)
    assert summary["image_files"] == 1
    assert summary["suffixes"] == {".jpg": 1, ".txt": 1}


def test_scimd17_archive_rejects_traversal(tmp_path: Path):
    path = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("../escape.jpg", b"bad")
    with pytest.raises(ValueError, match="unsafe SCIMD-17"):
        summarize_archive(path)
