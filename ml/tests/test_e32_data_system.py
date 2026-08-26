from __future__ import annotations

import importlib.util
import hashlib
import json
import sys
from pathlib import Path

import pytest


MODULE_PATH = Path(__file__).parents[1] / "experiments/e32_data_system.py"
SPEC = importlib.util.spec_from_file_location("e32_data_system", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
e32 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = e32
SPEC.loader.exec_module(e32)


def test_registry_has_three_unique_real_sources_and_explicit_licences():
    sources = e32.registry()
    assert set(sources) == {"vision-base-native", "forchheim-fodb", "csafe-mcsidb-s21"}
    assert all(source["license"] for source in sources.values())
    assert all(source["assigned_role"] == "train_or_calibration_candidate" for source in sources.values())


def test_vision_parser_keeps_only_native_images_and_device_identity():
    raw = b"\n".join(
        [
            b"https://lesc.dinfo.unifi.it/VISION/dataset/D01_Samsung_X/images/flat/a.jpg",
            b"https://lesc.dinfo.unifi.it/VISION/dataset/D01_Samsung_X/images/nat/a.jpg",
            b"https://lesc.dinfo.unifi.it/VISION/dataset/D01_Samsung_X/images/natWA/a.jpg",
            b"https://lesc.dinfo.unifi.it/VISION/dataset/D02_Apple_Y/images/nat/b.jpeg",
            b"https://lesc.dinfo.unifi.it/VISION/dataset/D02_Apple_Y/videos/nat/c.mp4",
        ]
    )
    assets = e32.parse_vision_native_urls(raw)
    assert [asset["source_key"] for asset in assets] == [
        "D01_Samsung_X/images/nat/a.jpg",
        "D02_Apple_Y/images/nat/b.jpeg",
    ]
    assert [asset["device"] for asset in assets] == ["D01", "D02"]


@pytest.mark.parametrize("value", ["../escape.zip", "/absolute.zip", "safe/../../escape.zip", ""])
def test_destination_cannot_escape_external_e32_root(tmp_path, monkeypatch, value):
    monkeypatch.setattr(e32, "OUTPUT_ROOT", tmp_path / "e32")
    with pytest.raises(ValueError):
        e32._safe_destination(value)


def test_completed_file_is_reused_without_network(tmp_path, monkeypatch):
    monkeypatch.setattr(e32, "OUTPUT_ROOT", tmp_path / "e32")
    destination = e32._safe_destination("real/source/archive.zip")
    destination.parent.mkdir(parents=True)
    destination.write_bytes(b"done")
    result = e32._stream_download("https://invalid.example/archive.zip", destination, 4)
    assert result["state"] == "already_complete"
    assert result["bytes"] == 4

    result_without_declared_size = e32._stream_download(
        "https://invalid.example/archive.zip", destination
    )
    assert result_without_declared_size["state"] == "already_complete"
    assert result_without_declared_size["bytes"] == 4


def test_new_download_uses_quiet_tls_curl_and_atomic_partial(tmp_path, monkeypatch):
    monkeypatch.setattr(e32, "OUTPUT_ROOT", tmp_path / "e32")
    monkeypatch.setattr(e32, "MIN_FREE_BYTES", 0)
    seen = {}

    def fake_run(command, check):
        seen["command"] = command
        output_index = command.index("--output") + 1
        Path(command[output_index]).write_bytes(b"image")

    monkeypatch.setattr(e32.subprocess, "run", fake_run)
    destination = e32._safe_destination("real/source/image.jpg")
    result = e32._stream_download("https://example.test/image.jpg", destination, 5)
    assert result["state"] == "downloaded"
    assert destination.read_bytes() == b"image"
    assert "--silent" in seen["command"]
    assert "--show-error" in seen["command"]
    assert seen["command"][-1] == "https://example.test/image.jpg"


def test_frozen_selection_loader_rejects_unexpected_state(tmp_path, monkeypatch):
    path = tmp_path / "selection.json"
    path.write_text(json.dumps({"state": "downloaded"}))
    monkeypatch.setattr(e32, "DETAILED_SELECTION", path)
    with pytest.raises(ValueError, match="unexpected"):
        e32._load_selection()


def test_range_plan_is_contiguous_and_exact():
    plan = e32._range_plan(20, 101, 4)
    assert plan == [(20, 40), (41, 61), (62, 82), (83, 100)]
    assert sum(end - start + 1 for start, end in plan) == 81
    assert e32._parse_content_range("bytes 20-40/101") == (20, 40, 101)


def test_range_assembly_preserves_prefix_and_verifies_full_md5(tmp_path):
    prefix = tmp_path / "archive.zip.partial"
    first = tmp_path / "range-1.partial"
    second = tmp_path / "range-2.partial"
    destination = tmp_path / "archive.zip"
    prefix.write_bytes(b"prefix-")
    first.write_bytes(b"middle-")
    second.write_bytes(b"end")
    expected = b"prefix-middle-end"

    result = e32._assemble_ranges(
        prefix,
        [first, second],
        destination,
        len(expected),
        hashlib.md5(expected, usedforsecurity=False).hexdigest(),
    )

    assert destination.read_bytes() == expected
    assert result["bytes"] == len(expected)
    assert prefix.read_bytes() == b"prefix-"


def test_range_assembly_keeps_prefix_when_md5_fails(tmp_path):
    prefix = tmp_path / "archive.zip.partial"
    part = tmp_path / "range.partial"
    destination = tmp_path / "archive.zip"
    prefix.write_bytes(b"safe-prefix")
    part.write_bytes(b"tail")

    with pytest.raises(ValueError, match="MD5"):
        e32._assemble_ranges(prefix, [part], destination, 15, "0" * 32)

    assert prefix.read_bytes() == b"safe-prefix"
    assert not destination.exists()
