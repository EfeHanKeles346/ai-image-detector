from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image


MODULE_PATH = Path(__file__).parents[1] / "experiments/e32_source_realization.py"
EXPERIMENTS = str(MODULE_PATH.parent)
if EXPERIMENTS not in sys.path:
    sys.path.insert(0, EXPERIMENTS)
SPEC = importlib.util.spec_from_file_location("e32_source_realization", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
e32 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = e32
SPEC.loader.exec_module(e32)


def _png(path: Path, color: tuple[int, int, int], size: tuple[int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", size, color)
    for y in range(size[1]):
        for x in range(size[0]):
            image.putpixel(
                (x, y),
                (
                    (color[0] + x * (color[0] + 3) + y) % 256,
                    (color[1] + y * (color[1] + 5) + x) % 256,
                    (color[2] + (x + y) * (color[2] + 7)) % 256,
                ),
            )
    image.save(path, format="PNG")


def _isolate(tmp_path, monkeypatch):
    output = tmp_path / "e32"
    evidence = tmp_path / "evidence"
    monkeypatch.setattr(e32, "OUTPUT_ROOT", output)
    monkeypatch.setattr(e32, "AUDIT_ROOT", output / "audits")
    monkeypatch.setattr(e32, "EVIDENCE_ROOT", evidence)
    monkeypatch.setattr(e32.real_acquisition, "OUTPUT_ROOT", output)
    monkeypatch.setattr(e32.ai_acquisition.real_acquisition, "OUTPUT_ROOT", output)
    monkeypatch.setattr(e32, "_protected_hashes", lambda: (set(), set(), 0))
    monkeypatch.setattr(e32.ai_acquisition, "_require_smoke_gate", lambda: None)
    return output, evidence


def test_ai_audit_decodes_payload_not_declared_extension(tmp_path, monkeypatch):
    output, evidence = _isolate(tmp_path, monkeypatch)
    assets = []
    for variation in range(4):
        stem = f"animal_00001_{variation}"
        image_key = f"data/animal/dataset_{variation}/{stem}.jxl"
        text_key = f"data/animal/dataset_{variation}/{stem}.txt"
        image_path = output / "ai" / "qwen-image-2512" / image_key
        text_path = output / "ai" / "qwen-image-2512" / text_key
        _png(image_path, (variation * 40, 10, 20), (32, 32))
        text_path.write_text("same prompt\n")
        for key, path in ((image_key, image_path), (text_key, text_path)):
            assets.append(
                {
                    "path": key,
                    "bytes": path.stat().st_size,
                    "category": "animal",
                    "prompt_group": "animal:animal_00001",
                }
            )
    selection = {
        "state": "selection_frozen_decoder_smoke_required_before_bulk",
        "sources": [
            {
                "source_id": "qwen-image-2512",
                "family": "fixture",
                "revision": "a" * 40,
                "license_tag": "license:test",
                "expected_width": 32,
                "expected_height": 32,
                "selected_images": 4,
                "selected_prompt_groups": 1,
                "assets": assets,
            }
        ],
    }
    selection_path = tmp_path / "ai_selection.json"
    selection_path.write_text(json.dumps(selection))
    monkeypatch.setattr(e32.ai_acquisition, "DETAILED_SELECTION", selection_path)

    report = e32.audit_ai_source("qwen-image-2512")

    assert report["state"] == "source_realization_passed_candidate_only"
    assert report["format_counts"] == {"PNG": 4}
    assert report["extension_matches_decoded_format"] == 0
    assert report["realized_prompt_groups"] == 1
    compact = json.loads((evidence / "e32_qwen-image-2512_realization.json").read_text())
    assert "records" not in compact
    assert "failures" not in compact
    assert compact["failure_reason_counts"] == {}
    detailed = output / compact["detailed_report_external_path"]
    assert hashlib.sha256(detailed.read_bytes()).hexdigest() == compact["detailed_report_sha256"]


def test_ai_audit_rejects_missing_member_without_assigning_role(tmp_path, monkeypatch):
    output, _ = _isolate(tmp_path, monkeypatch)
    selection = {
        "state": "selection_frozen_decoder_smoke_required_before_bulk",
        "sources": [
            {
                "source_id": "flux2-klein-9b",
                "family": "fixture",
                "revision": "b" * 40,
                "license_tag": "license:test",
                "expected_width": 16,
                "expected_height": 16,
                "selected_images": 4,
                "selected_prompt_groups": 1,
                "assets": [
                    {
                        "path": f"data/animal/dataset_{variation}/animal_00001_{variation}.{suffix}",
                        "bytes": 1,
                        "category": "animal",
                        "prompt_group": "animal:animal_00001",
                    }
                    for variation in range(4)
                    for suffix in ("jxl", "txt")
                ],
            }
        ],
    }
    selection_path = tmp_path / "ai_selection.json"
    selection_path.write_text(json.dumps(selection))
    monkeypatch.setattr(e32.ai_acquisition, "DETAILED_SELECTION", selection_path)

    report = e32.audit_ai_source("flux2-klein-9b")

    assert report["state"] == "source_realization_rejected_no_role_assignment"
    assert report["realized_images"] == 0
    assert any(item["reason"] == "missing_file" for item in report["failures"])
    assert "role" not in report


def test_vision_audit_records_device_and_exif_summary(tmp_path, monkeypatch):
    output, _ = _isolate(tmp_path, monkeypatch)
    source_key = "D01_Phone/images/nat/photo.jpg"
    image_path = output / "real" / "vision" / source_key
    _png(image_path, (12, 34, 56), (24, 20))
    selection = {
        "state": "selection_frozen_no_image_bytes_claimed",
        "sources": [
            {
                "source_id": "vision-base-native",
                "parent_count": 1,
                "assets": [
                    {
                        "source_key": source_key,
                        "device": "D01",
                        "camera_pipeline": "D01_Phone",
                    }
                ],
            }
        ],
    }
    selection_path = tmp_path / "real_selection.json"
    selection_path.write_text(json.dumps(selection))
    monkeypatch.setattr(e32.real_acquisition, "DETAILED_SELECTION", selection_path)

    report = e32.audit_vision()

    assert report["state"] == "source_realization_passed_candidate_only"
    assert report["device_counts"] == {"D01": 1}
    assert report["format_counts"] == {"PNG": 1}
    assert report["records"][0]["exif_present"] is False


def test_raw_image_accepts_bytes_and_arrow_style_mapping():
    buffer = io.BytesIO()
    Image.new("RGB", (12, 10), (1, 2, 3)).save(buffer, format="PNG")
    raw = buffer.getvalue()
    assert e32._raw_image(raw) == raw
    assert e32._raw_image({"bytes": raw, "path": None}) == raw
    assert e32._raw_image({"path": "missing"}) is None
    record = e32._image_record_raw(raw)
    assert record["decoded_format"] == "PNG"
    assert (record["width"], record["height"]) == (12, 10)


def test_prompt_decoder_is_utf8_first_with_one_explicit_windows_1252_fallback():
    utf8 = "İstanbul — café".encode("utf-8")
    windows_1252 = "A painter’s studio — façade".encode("windows-1252")

    assert e32._decode_prompt(utf8) == ("İstanbul — café", "utf-8")
    assert e32._decode_prompt(windows_1252) == (
        "A painter’s studio — façade",
        "windows-1252",
    )


def test_prompt_decoder_rejects_bytes_undefined_in_windows_1252():
    try:
        e32._decode_prompt(b"prompt\x81")
    except UnicodeDecodeError:
        pass
    else:
        raise AssertionError("undefined Windows-1252 byte must not be silently replaced")


def test_pool_parquet_audit_reads_selected_image_bytes(tmp_path, monkeypatch):
    folder = tmp_path / "nano"
    folder.mkdir()
    buffer = io.BytesIO()
    Image.new("RGB", (14, 13), (20, 30, 40)).save(buffer, format="PNG")
    raw = buffer.getvalue()
    pq.write_table(pa.table({"image": [raw, raw + b"unused"]}), folder / "part.parquet")
    monkeypatch.setattr(e32, "_pool_spec", lambda source_id: {"dirname": "nano"})
    source = {
        "source_id": "nano-banana-local",
        "records": [
            {
                "source_key": "part.parquet:0",
                "shard": "part.parquet",
                "row_index": 0,
                "parent_group": "nano:0",
                "declared_width": 14,
                "declared_height": 13,
                "declared_mode": "RGB",
                "declared_format": "PNG",
            }
        ],
    }
    records, failures = e32._audit_pool_parquet(tmp_path, source, image_column="image")
    assert failures == []
    assert len(records) == 1
    assert records[0]["sha256"] == hashlib.sha256(raw).hexdigest()


def test_passed_peer_scan_ignores_exfat_appledouble_receipts(tmp_path, monkeypatch):
    audit_root = tmp_path / "audits"
    audit_root.mkdir()
    (audit_root / "._peer.json").write_bytes(b"\x00\x05\x16\xb0appledouble")
    (audit_root / "peer.json").write_text(
        json.dumps(
            {
                "source_id": "peer",
                "state": "source_realization_passed_candidate_only",
                "records": [{"sha256": "exact", "dhash": "perceptual"}],
            }
        )
    )
    monkeypatch.setattr(e32, "AUDIT_ROOT", audit_root)
    exact, perceptual, legacy_dhash, reports = e32._passed_peer_hashes("current")
    assert exact == {"exact"}
    assert perceptual == set()
    assert legacy_dhash == {"perceptual"}
    assert reports == 1


def test_dhash_collision_needs_close_phash_to_be_duplicate():
    distinct = [
        {"source_key": "a", "dhash": "0f" * 8, "phash": "00" * 8},
        {"source_key": "b", "dhash": "0f" * 8, "phash": "ff" * 8},
    ]
    assert e32._confirmed_perceptual_duplicates(distinct) == []
    close = [
        {"source_key": "a", "dhash": "0f" * 8, "phash": "00" * 8},
        {"source_key": "b", "dhash": "0f" * 8, "phash": "0000000000000003"},
    ]
    assert e32._confirmed_perceptual_duplicates(close) == [["a", "b"]]
