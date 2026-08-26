"""E32/C1-C2 — realize frozen source bytes as role-free, audited candidates.

This gate never assigns TRAIN or CALIBRATION. It verifies that a frozen acquisition arrived
unchanged, decodes every selected image from its bytes, measures container/geometry metadata,
checks prompt-group integrity for AI sources, and rejects exact/perceptual overlap with protected
E30 manifests or previously passed E32 source audits.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence

import pyarrow.parquet as pq
import numpy as np
from PIL import Image

import e32_ai_pool_selection as pool_selection
import e32_data_system as real_acquisition
import e32_gap_acquisition as ai_acquisition
import e32_gpt_acquisition as gpt_acquisition
from pixelproof.project_paths import ML_ROOT


REPO_ROOT = ML_ROOT.parent
OUTPUT_ROOT = real_acquisition.OUTPUT_ROOT
AUDIT_ROOT = OUTPUT_ROOT / "audits"
EVIDENCE_ROOT = REPO_ROOT / "evidence"
CHUNK_BYTES = 8 * 1024**2
PHASH_DUPLICATE_MAX_DISTANCE = 5


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def _write_atomic(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_bytes(raw)
    temporary.replace(path)


def _dhash(image: Image.Image) -> str:
    grayscale = image.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
    pixels = list(grayscale.get_flattened_data())
    bits = 0
    for row in range(8):
        for column in range(8):
            bits = (bits << 1) | int(
                pixels[row * 9 + column] > pixels[row * 9 + column + 1]
            )
    return f"{bits:016x}"


def _protected_hashes(repo_root: Path = REPO_ROOT) -> tuple[set[str], set[str], int]:
    exact: set[str] = set()
    perceptual: set[str] = set()
    manifests = sorted((repo_root / "ml" / "data" / "e30").rglob("*manifest.json"))
    for path in manifests:
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        for row in payload.get("records", []):
            if row.get("sha256"):
                exact.add(str(row["sha256"]))
            if row.get("dhash"):
                perceptual.add(str(row["dhash"]))
    return exact, perceptual, len(manifests)


def _passed_peer_hashes(
    source_id: str,
) -> tuple[set[str], set[tuple[str, str]], set[str], int]:
    exact: set[str] = set()
    perceptual: set[tuple[str, str]] = set()
    legacy_dhash: set[str] = set()
    reports = 0
    if not AUDIT_ROOT.exists():
        return exact, perceptual, legacy_dhash, reports
    for path in sorted(AUDIT_ROOT.glob("*.json")):
        if path.name.startswith("._"):
            continue
        try:
            payload = json.loads(path.read_text())
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        if payload.get("source_id") == source_id:
            continue
        if payload.get("state") != "source_realization_passed_candidate_only":
            continue
        reports += 1
        for row in payload.get("records", []):
            if row.get("sha256"):
                exact.add(str(row["sha256"]))
            if row.get("dhash") and row.get("phash"):
                perceptual.add((str(row["dhash"]), str(row["phash"])))
            elif row.get("dhash"):
                legacy_dhash.add(str(row["dhash"]))
    return exact, perceptual, legacy_dhash, reports


def _phash(image: Image.Image) -> str:
    size = 32
    low = 8
    values = np.asarray(
        image.convert("L").resize((size, size), Image.Resampling.LANCZOS), dtype=np.float64
    )
    positions = np.arange(size, dtype=np.float64)
    frequencies = positions[:, None]
    transform = np.cos(np.pi * (2 * positions + 1) * frequencies / (2 * size))
    transform[0] /= np.sqrt(2.0)
    coefficients = transform @ values @ transform.T
    block = coefficients[:low, :low].reshape(-1)
    median = float(np.median(block[1:]))
    bits = 0
    for value in block:
        bits = (bits << 1) | int(value > median)
    return f"{bits:016x}"


def _hamming(left: str, right: str) -> int:
    return (int(left, 16) ^ int(right, 16)).bit_count()


def _image_record_raw(raw: bytes) -> dict[str, Any]:
    with Image.open(io.BytesIO(raw)) as opened:
        opened.load()
        decoded_format = str(opened.format or "UNKNOWN").upper()
        mode = opened.mode
        width, height = opened.size
        exif = opened.getexif()
        orientation = exif.get(274) if exif else None
        perceptual = _dhash(opened)
        phash = _phash(opened)
    return {
        "bytes": len(raw),
        "sha256": _sha256_bytes(raw),
        "dhash": perceptual,
        "phash": phash,
        "decoded_format": decoded_format,
        "mode": mode,
        "width": width,
        "height": height,
        "orientation": orientation,
        "exif_present": bool(exif),
        "bytes_per_pixel": round(len(raw) / max(1, width * height), 6),
    }


def _image_record(path: Path) -> dict[str, Any]:
    return _image_record_raw(path.read_bytes())


def _raw_image(value: Any) -> bytes | None:
    if isinstance(value, (bytes, bytearray)):
        return bytes(value)
    if isinstance(value, Mapping) and isinstance(value.get("bytes"), (bytes, bytearray)):
        return bytes(value["bytes"])
    return None


def _decode_prompt(raw: bytes) -> tuple[str, str]:
    """Decode pinned prompt bytes without guessing across an open-ended codec list."""
    try:
        return raw.decode("utf-8"), "utf-8"
    except UnicodeDecodeError:
        return raw.decode("windows-1252"), "windows-1252"


def _duplicates(records: Sequence[Mapping[str, Any]], key: str) -> list[list[str]]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for row in records:
        grouped[str(row[key])].append(str(row["source_key"]))
    return [sorted(paths) for paths in grouped.values() if len(paths) > 1]


def _confirmed_perceptual_duplicates(
    records: Sequence[Mapping[str, Any]],
) -> list[list[str]]:
    by_dhash: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in records:
        by_dhash[str(row["dhash"])].append(row)
    output: list[list[str]] = []
    for candidates in by_dhash.values():
        if len(candidates) < 2:
            continue
        parent = list(range(len(candidates)))

        def find(index: int) -> int:
            while parent[index] != index:
                parent[index] = parent[parent[index]]
                index = parent[index]
            return index

        def union(left: int, right: int) -> None:
            left_root, right_root = find(left), find(right)
            if left_root != right_root:
                parent[right_root] = left_root

        for left in range(len(candidates)):
            for right in range(left + 1, len(candidates)):
                distance = _hamming(
                    str(candidates[left]["phash"]), str(candidates[right]["phash"])
                )
                if distance <= PHASH_DUPLICATE_MAX_DISTANCE:
                    union(left, right)
        groups: dict[int, list[str]] = defaultdict(list)
        for index, row in enumerate(candidates):
            groups[find(index)].append(str(row["source_key"]))
        output.extend(sorted(paths) for paths in groups.values() if len(paths) > 1)
    return sorted(output)


def _selection_source(payload: Mapping[str, Any], source_id: str) -> Mapping[str, Any]:
    try:
        return next(source for source in payload["sources"] if source["source_id"] == source_id)
    except StopIteration as error:
        raise KeyError(f"source {source_id!r} is absent from the frozen selection") from error


def _finalize(
    *,
    source_id: str,
    kind: str,
    selection_raw: bytes,
    expected_images: int,
    records: list[dict[str, Any]],
    failures: list[dict[str, str]],
    extra: Mapping[str, Any],
) -> dict[str, Any]:
    protected_exact, protected_dhash, protected_manifests = _protected_hashes()
    peer_exact, peer_perceptual, peer_legacy_dhash, peer_reports = _passed_peer_hashes(source_id)
    within_exact = _duplicates(records, "sha256")
    within_dhash_candidates = _duplicates(records, "dhash")
    within_perceptual = _confirmed_perceptual_duplicates(records)
    protected_exact_hits = sorted(
        row["source_key"] for row in records if row["sha256"] in protected_exact
    )
    protected_dhash_hits = sorted(
        row["source_key"] for row in records if row["dhash"] in protected_dhash
    )
    peer_exact_hits = sorted(row["source_key"] for row in records if row["sha256"] in peer_exact)
    peer_dhash_hits = sorted(
        row["source_key"]
        for row in records
        if (row["dhash"], row["phash"]) in peer_perceptual
        or row["dhash"] in peer_legacy_dhash
    )
    if within_exact:
        failures.append(
            {"reason": "within_source_exact_duplicates", "count": str(len(within_exact))}
        )
    if within_perceptual:
        failures.append(
            {
                "reason": "within_source_confirmed_perceptual_duplicates",
                "count": str(len(within_perceptual)),
            }
        )
    for reason, hits in (
        ("protected_exact_overlap", protected_exact_hits),
        ("protected_dhash_overlap", protected_dhash_hits),
        ("passed_peer_exact_overlap", peer_exact_hits),
        ("passed_peer_dhash_overlap", peer_dhash_hits),
    ):
        if hits:
            failures.append({"reason": reason, "count": str(len(hits))})
    if len(records) != expected_images:
        failures.append(
            {
                "reason": "realized_count_mismatch",
                "count": f"{len(records)}/{expected_images}",
            }
        )

    state = (
        "source_realization_rejected_no_role_assignment"
        if failures
        else "source_realization_passed_candidate_only"
    )
    report = {
        "schema_version": 2,
        "experiment": "E32/C1-C2-source-realization",
        "state": state,
        "source_id": source_id,
        "kind": kind,
        "selection_sha256": _sha256_bytes(selection_raw),
        "expected_images": expected_images,
        "realized_images": len(records),
        "counts": {
            "failures": len(failures),
            "unique_sha256": len({row["sha256"] for row in records}),
            "unique_dhash": len({row["dhash"] for row in records}),
            "unique_phash": len({row["phash"] for row in records}),
            "within_source_exact_duplicate_groups": len(within_exact),
            "within_source_dhash_candidate_groups": len(within_dhash_candidates),
            "within_source_confirmed_perceptual_duplicate_groups": len(within_perceptual),
            "protected_exact_overlaps": len(protected_exact_hits),
            "protected_dhash_overlaps": len(protected_dhash_hits),
            "passed_peer_exact_overlaps": len(peer_exact_hits),
            "passed_peer_dhash_overlaps": len(peer_dhash_hits),
        },
        "format_counts": dict(sorted(Counter(row["decoded_format"] for row in records).items())),
        "mode_counts": dict(sorted(Counter(row["mode"] for row in records).items())),
        "exif_present": sum(bool(row["exif_present"]) for row in records),
        "protected_scope": {
            "e30_manifest_files": protected_manifests,
            "e30_exact_hashes": len(protected_exact),
            "e30_dhashes": len(protected_dhash),
            "passed_peer_audit_files": peer_reports,
        },
        "duplicate_groups": {
            "exact": within_exact,
            "dhash_candidates": within_dhash_candidates,
            "perceptual_confirmed": within_perceptual,
        },
        "overlap_source_keys": {
            "protected_exact": protected_exact_hits,
            "protected_dhash": protected_dhash_hits,
            "passed_peer_exact": peer_exact_hits,
            "passed_peer_dhash": peer_dhash_hits,
        },
        "failures": failures,
        **dict(extra),
        "records": records,
        "boundary": (
            "A passing source is only an eligible candidate. This report assigns no TRAIN, "
            "CALIBRATION, DEVELOPMENT or LOCKED FINAL role."
        ),
    }
    detailed_raw = _json_bytes(report)
    detailed_path = AUDIT_ROOT / f"{source_id}.json"
    _write_atomic(detailed_path, detailed_raw)
    compact = {
        key: value
        for key, value in report.items()
        if key not in {"records", "duplicate_groups", "overlap_source_keys", "failures"}
    }
    compact.update(
        {
            "failure_reason_counts": dict(
                sorted(Counter(item["reason"] for item in failures).items())
            ),
            "failure_examples": failures[:20],
            "detailed_report_sha256": _sha256_bytes(detailed_raw),
            "detailed_report_bytes": len(detailed_raw),
            "detailed_report_external_path": detailed_path.relative_to(OUTPUT_ROOT).as_posix(),
        }
    )
    _write_atomic(EVIDENCE_ROOT / f"e32_{source_id}_realization.json", _json_bytes(compact))
    return report


def audit_vision() -> dict[str, Any]:
    selection_raw = real_acquisition.DETAILED_SELECTION.read_bytes()
    payload = json.loads(selection_raw)
    if payload.get("state") != "selection_frozen_no_image_bytes_claimed":
        raise ValueError("unexpected REAL selection state")
    source = _selection_source(payload, "vision-base-native")
    records: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for asset in source["assets"]:
        source_key = str(asset["source_key"])
        path = real_acquisition._safe_destination(f"real/vision/{source_key}")
        partial = path.with_suffix(path.suffix + ".partial")
        if partial.exists():
            failures.append({"source_key": source_key, "reason": "partial_file_present"})
        if not path.is_file():
            failures.append({"source_key": source_key, "reason": "missing_file"})
            continue
        try:
            record = _image_record(path)
        except Exception as error:
            failures.append(
                {"source_key": source_key, "reason": f"decode_failure:{type(error).__name__}"}
            )
            continue
        record.update(
            {
                "source_key": source_key,
                "device": asset["device"],
                "camera_pipeline": asset["camera_pipeline"],
                "declared_extension": PurePosixPath(source_key).suffix.lower(),
            }
        )
        records.append(record)
    return _finalize(
        source_id="vision-base-native",
        kind="real",
        selection_raw=selection_raw,
        expected_images=int(source["parent_count"]),
        records=records,
        failures=failures,
        extra={
            "device_counts": dict(sorted(Counter(row["device"] for row in records).items())),
            "camera_pipeline_counts": dict(
                sorted(Counter(row["camera_pipeline"] for row in records).items())
            ),
        },
    )


def audit_ai_source(source_id: str) -> dict[str, Any]:
    ai_acquisition._require_smoke_gate()
    selection_raw = ai_acquisition.DETAILED_SELECTION.read_bytes()
    payload = json.loads(selection_raw)
    source = _selection_source(payload, source_id)
    records: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    prompt_groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for asset in source["assets"]:
        prompt_groups[str(asset["prompt_group"])].append(asset)
    if len(prompt_groups) != int(source["selected_prompt_groups"]):
        failures.append(
            {
                "reason": "prompt_group_count_mismatch",
                "count": f"{len(prompt_groups)}/{source['selected_prompt_groups']}",
            }
        )
    for group, assets in sorted(prompt_groups.items()):
        images = [asset for asset in assets if str(asset["path"]).endswith(".jxl")]
        sidecars = [asset for asset in assets if str(asset["path"]).endswith(".txt")]
        if len(images) != 4 or len(sidecars) != 4:
            failures.append({"source_key": group, "reason": "incomplete_prompt_group_contract"})
            continue
        prompt_hashes = set()
        sidecars_by_stem = {PurePosixPath(str(asset["path"])).stem: asset for asset in sidecars}
        for asset in images:
            source_key = str(asset["path"])
            image_path = ai_acquisition._destination(source_id, source_key)
            sidecar = sidecars_by_stem.get(PurePosixPath(source_key).stem)
            if sidecar is None:
                failures.append({"source_key": source_key, "reason": "missing_prompt_contract"})
                continue
            text_key = str(sidecar["path"])
            text_path = ai_acquisition._destination(source_id, text_key)
            for expected, path in ((asset, image_path), (sidecar, text_path)):
                partial = path.with_suffix(path.suffix + ".partial")
                if partial.exists():
                    failures.append(
                        {"source_key": str(expected["path"]), "reason": "partial_file_present"}
                    )
                if not path.is_file():
                    failures.append(
                        {"source_key": str(expected["path"]), "reason": "missing_file"}
                    )
                elif path.stat().st_size != int(expected["bytes"]):
                    failures.append(
                        {"source_key": str(expected["path"]), "reason": "byte_count_mismatch"}
                    )
            if not image_path.is_file() or not text_path.is_file():
                continue
            try:
                prompt_raw = text_path.read_bytes()
                prompt = prompt_raw.decode("utf-8").strip()
            except (OSError, UnicodeDecodeError) as error:
                failures.append(
                    {
                        "source_key": text_key,
                        "reason": f"prompt_decode_failure:{type(error).__name__}",
                    }
                )
                continue
            if not prompt:
                failures.append({"source_key": text_key, "reason": "empty_prompt"})
                continue
            prompt_sha256 = _sha256_bytes(prompt.encode())
            prompt_hashes.add(prompt_sha256)
            try:
                record = _image_record(image_path)
            except Exception as error:
                failures.append(
                    {"source_key": source_key, "reason": f"decode_failure:{type(error).__name__}"}
                )
                continue
            if (record["width"], record["height"]) != (
                int(source["expected_width"]),
                int(source["expected_height"]),
            ):
                failures.append({"source_key": source_key, "reason": "unexpected_dimensions"})
            record.update(
                {
                    "source_key": source_key,
                    "category": asset["category"],
                    "prompt_group": group,
                    "prompt_sha256": prompt_sha256,
                    "prompt_bytes": len(prompt_raw),
                    "declared_extension": PurePosixPath(source_key).suffix.lower(),
                }
            )
            records.append(record)
        if len(prompt_hashes) != 1:
            failures.append({"source_key": group, "reason": "prompt_changed_within_group"})
    return _finalize(
        source_id=source_id,
        kind="ai",
        selection_raw=selection_raw,
        expected_images=int(source["selected_images"]),
        records=records,
        failures=failures,
        extra={
            "family": source["family"],
            "revision": source["revision"],
            "license_tag": source["license_tag"],
            "expected_prompt_groups": int(source["selected_prompt_groups"]),
            "realized_prompt_groups": len({row["prompt_group"] for row in records}),
            "category_counts": dict(
                sorted(Counter(row["category"] for row in records).items())
            ),
            "extension_matches_decoded_format": sum(
                row["declared_extension"].lstrip(".").lower()
                == row["decoded_format"].lower()
                for row in records
            ),
        },
    )


POOL_SOURCE_IDS = (
    "nano-banana-local",
    "nano-banana-pro-ash-local",
    "communityforensics-ai-local",
    "gpt-image-1",
)


def audit_fodb() -> dict[str, Any]:
    receipt_path = OUTPUT_ROOT / "fodb_orig_extraction.json"
    compact_path = EVIDENCE_ROOT / "e32_fodb_orig_extraction.json"
    receipt_raw = receipt_path.read_bytes()
    compact = json.loads(compact_path.read_text())
    if _sha256_bytes(receipt_raw) != compact.get("detailed_report_sha256"):
        raise ValueError("FODB extraction receipt binding changed")
    receipt = json.loads(receipt_raw)
    if receipt.get("state") != "orig_extraction_complete_role_free":
        raise ValueError("FODB extraction receipt has unexpected state")
    records: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for selected in receipt["records"]:
        source_key = str(selected["source_key"])
        path = OUTPUT_ROOT / source_key
        partial = path.with_suffix(path.suffix + ".partial")
        if partial.exists():
            failures.append({"source_key": source_key, "reason": "partial_file_present"})
        if not path.is_file():
            failures.append({"source_key": source_key, "reason": "missing_file"})
            continue
        if path.stat().st_size != int(selected["bytes"]):
            failures.append({"source_key": source_key, "reason": "byte_count_mismatch"})
        try:
            record = _image_record(path)
        except Exception as error:
            failures.append(
                {"source_key": source_key, "reason": f"decode_failure:{type(error).__name__}"}
            )
            continue
        if record["sha256"] != str(selected["sha256"]):
            failures.append({"source_key": source_key, "reason": "extraction_sha256_mismatch"})
        record.update(
            {
                "source_key": source_key,
                "parent_group": f"fodb:{selected['camera_pipeline']}:{PurePosixPath(source_key).stem}",
                "camera_pipeline": selected["camera_pipeline"],
                "device": selected["device"],
                "scene_group": selected["scene_group"],
                "native_social_state": "orig",
            }
        )
        records.append(record)
    return _finalize(
        source_id="forchheim-fodb",
        kind="real",
        selection_raw=receipt_raw,
        expected_images=int(receipt["parent_count"]),
        records=records,
        failures=failures,
        extra={
            "camera_pipeline_counts": dict(
                sorted(Counter(str(row["camera_pipeline"]) for row in records).items())
            ),
            "device_counts": dict(sorted(Counter(str(row["device"]) for row in records).items())),
            "scene_group_count": len({str(row["scene_group"]) for row in records}),
            "native_social_state_counts": dict(
                sorted(Counter(str(row["native_social_state"]) for row in records).items())
            ),
        },
    )


def audit_csafe() -> dict[str, Any]:
    receipt_path = OUTPUT_ROOT / "csafe_natural_extraction.json"
    compact_path = EVIDENCE_ROOT / "e32_csafe_natural_extraction.json"
    receipt_raw = receipt_path.read_bytes()
    compact = json.loads(compact_path.read_text())
    if _sha256_bytes(receipt_raw) != compact.get("detailed_report_sha256"):
        raise ValueError("CSAFE extraction receipt binding changed")
    receipt = json.loads(receipt_raw)
    if receipt.get("state") != "natural_extraction_complete_role_free":
        raise ValueError("CSAFE extraction receipt has unexpected state")
    records: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for selected in receipt["records"]:
        source_key = str(selected["source_key"])
        path = OUTPUT_ROOT / source_key
        partial = path.with_suffix(path.suffix + ".partial")
        if partial.exists():
            failures.append({"source_key": source_key, "reason": "partial_file_present"})
        if not path.is_file():
            failures.append({"source_key": source_key, "reason": "missing_file"})
            continue
        if path.stat().st_size != int(selected["bytes"]):
            failures.append({"source_key": source_key, "reason": "byte_count_mismatch"})
        try:
            record = _image_record(path)
        except Exception as error:
            failures.append(
                {"source_key": source_key, "reason": f"decode_failure:{type(error).__name__}"}
            )
            continue
        if record["sha256"] != str(selected["sha256"]):
            failures.append({"source_key": source_key, "reason": "extraction_sha256_mismatch"})
        record.update(
            {
                "source_key": source_key,
                "parent_group": selected["parent_group"],
                "camera_pipeline": selected["camera_pipeline"],
                "device": selected["device"],
                "lens": selected["lens"],
                "content_type": "natural",
            }
        )
        records.append(record)
    return _finalize(
        source_id="csafe-mcsidb-s21",
        kind="real",
        selection_raw=receipt_raw,
        expected_images=int(receipt["parent_count"]),
        records=records,
        failures=failures,
        extra={
            "camera_pipeline_counts": dict(
                sorted(Counter(str(row["camera_pipeline"]) for row in records).items())
            ),
            "device_counts": dict(sorted(Counter(str(row["device"]) for row in records).items())),
            "lens_counts": dict(sorted(Counter(str(row["lens"]) for row in records).items())),
            "content_type_counts": dict(
                sorted(Counter(str(row["content_type"]) for row in records).items())
            ),
        },
    )


def _pool_spec(source_id: str) -> Mapping[str, Any]:
    try:
        return next(
            source for source in pool_selection.registry()["sources"] if source["id"] == source_id
        )
    except StopIteration as error:
        raise KeyError(f"missing AI-pool registry source {source_id}") from error


def _verify_pool_fingerprint(root: Path, source: Mapping[str, Any]) -> None:
    source_id = str(source["source_id"])
    spec = _pool_spec(source_id)
    if source_id == "nano-banana-local":
        _, fingerprint = pool_selection._parquet_locator_rows(
            root / str(spec["dirname"]),
            ("id", "format", "mode", "width", "height", "uploadtime"),
        )
    elif source_id == "communityforensics-ai-local":
        _, fingerprint = pool_selection._parquet_locator_rows(
            root / str(spec["dirname"]),
            (
                "image_name",
                "model_name",
                "subset",
                "split",
                "label",
                "architecture",
                "prompt",
            ),
        )
    elif source_id == "nano-banana-pro-ash-local":
        fingerprint = pool_selection._freeze_nbp(root, spec)["source_fingerprint"]
    else:
        return
    if fingerprint != source.get("source_fingerprint"):
        raise ValueError(f"{source_id} local source fingerprint changed after selection freeze")


def _audit_pool_parquet(
    root: Path,
    source: Mapping[str, Any],
    *,
    image_column: str,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    spec = _pool_spec(str(source["source_id"]))
    folder = root / str(spec["dirname"])
    wanted_by_shard: dict[str, dict[int, Mapping[str, Any]]] = defaultdict(dict)
    for selected in source["records"]:
        wanted_by_shard[str(selected["shard"])][int(selected["row_index"])] = selected
    records: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for shard, wanted in sorted(wanted_by_shard.items()):
        path = folder / shard
        if not path.is_file():
            failures.append({"source_key": shard, "reason": "missing_parquet_shard"})
            continue
        parquet = pq.ParquetFile(path)
        if image_column not in parquet.schema_arrow.names:
            failures.append({"source_key": shard, "reason": "missing_image_column"})
            continue
        row_index = 0
        found: set[int] = set()
        for batch in parquet.iter_batches(columns=[image_column], batch_size=64):
            for row in batch.to_pylist():
                selected = wanted.get(row_index)
                if selected is not None:
                    found.add(row_index)
                    raw = _raw_image(row[image_column])
                    if raw is None:
                        failures.append(
                            {"source_key": selected["source_key"], "reason": "missing_image_bytes"}
                        )
                        row_index += 1
                        continue
                    try:
                        record = _image_record_raw(raw)
                    except Exception as error:
                        failures.append(
                            {
                                "source_key": selected["source_key"],
                                "reason": f"decode_failure:{type(error).__name__}",
                            }
                        )
                        row_index += 1
                        continue
                    if source["source_id"] == "nano-banana-local" and (
                        record["width"] != int(selected["declared_width"])
                        or record["height"] != int(selected["declared_height"])
                        or record["mode"] != str(selected["declared_mode"])
                        or record["decoded_format"] != str(selected["declared_format"]).upper()
                    ):
                        failures.append(
                            {
                                "source_key": selected["source_key"],
                                "reason": "declared_metadata_mismatch",
                            }
                        )
                    record.update(
                        {
                            "source_key": selected["source_key"],
                            "parent_group": selected["parent_group"],
                            "model_name": selected.get("model_name"),
                            "architecture": selected.get("architecture"),
                            "prompt_sha256": selected.get("prompt_sha256"),
                            "storage": "local_parquet",
                        }
                    )
                    records.append(record)
                row_index += 1
        for missing in sorted(set(wanted) - found):
            failures.append(
                {"source_key": wanted[missing]["source_key"], "reason": "missing_selected_row"}
            )
    return records, failures


def _audit_pool_nbp(
    root: Path, source: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    folder = root / str(_pool_spec(str(source["source_id"]))["dirname"])
    records: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for selected in source["records"]:
        path = folder / str(selected["source_key"])
        if not path.is_file():
            failures.append({"source_key": selected["source_key"], "reason": "missing_file"})
            continue
        if path.stat().st_size != int(selected["expected_bytes"]):
            failures.append({"source_key": selected["source_key"], "reason": "byte_count_mismatch"})
        try:
            record = _image_record(path)
        except Exception as error:
            failures.append(
                {
                    "source_key": selected["source_key"],
                    "reason": f"decode_failure:{type(error).__name__}",
                }
            )
            continue
        record.update(
            {
                "source_key": selected["source_key"],
                "parent_group": selected["parent_group"],
                "storage": "local_loose",
            }
        )
        records.append(record)
    return records, failures


def _audit_pool_gpt(
    root: Path, source: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    records: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for selected in source["records"]:
        image_key = str(selected["source_key"])
        prompt_key = str(selected["prompt_key"])
        image_path = gpt_acquisition._available_asset(
            root, image_key, int(selected["expected_image_bytes"])
        )
        prompt_path = gpt_acquisition._available_asset(
            root, prompt_key, int(selected["expected_prompt_bytes"])
        )
        for key in (image_key, prompt_key):
            partial = gpt_acquisition._e32_asset(key).with_suffix(
                gpt_acquisition._e32_asset(key).suffix + ".partial"
            )
            if partial.exists():
                failures.append({"source_key": key, "reason": "partial_file_present"})
        if image_path is None:
            failures.append({"source_key": image_key, "reason": "missing_or_wrong_size"})
        if prompt_path is None:
            failures.append({"source_key": prompt_key, "reason": "missing_or_wrong_size"})
        if image_path is None or prompt_path is None:
            continue
        try:
            prompt_raw = prompt_path.read_bytes()
            prompt_text, prompt_encoding = _decode_prompt(prompt_raw)
            prompt = prompt_text.strip()
        except (OSError, UnicodeDecodeError) as error:
            failures.append(
                {
                    "source_key": prompt_key,
                    "reason": f"prompt_decode_failure:{type(error).__name__}",
                }
            )
            continue
        if not prompt:
            failures.append({"source_key": prompt_key, "reason": "empty_prompt"})
            continue
        try:
            record = _image_record(image_path)
        except Exception as error:
            failures.append(
                {"source_key": image_key, "reason": f"decode_failure:{type(error).__name__}"}
            )
            continue
        record.update(
            {
                "source_key": image_key,
                "parent_group": selected["parent_group"],
                "prompt_sha256": _sha256_bytes(prompt.encode()),
                "prompt_bytes_sha256": _sha256_bytes(prompt_raw),
                "prompt_bytes": len(prompt_raw),
                "prompt_encoding": prompt_encoding,
                "storage": selected["storage"],
            }
        )
        records.append(record)
    return records, failures


def audit_pool_source(root: Path, source_id: str) -> dict[str, Any]:
    payload = gpt_acquisition._load_selection()
    selection_raw = pool_selection.DETAILED_SELECTION.read_bytes()
    source = _selection_source(payload, source_id)
    _verify_pool_fingerprint(root, source)
    if source_id == "nano-banana-local":
        records, failures = _audit_pool_parquet(root, source, image_column="image")
    elif source_id == "communityforensics-ai-local":
        records, failures = _audit_pool_parquet(root, source, image_column="image_data")
    elif source_id == "nano-banana-pro-ash-local":
        records, failures = _audit_pool_nbp(root, source)
    elif source_id == "gpt-image-1":
        records, failures = _audit_pool_gpt(root, source)
    else:
        raise KeyError(source_id)
    return _finalize(
        source_id=source_id,
        kind="ai",
        selection_raw=selection_raw,
        expected_images=int(source["selected"]),
        records=records,
        failures=failures,
        extra={
            "family": source["family"],
            "revision": source["revision"],
            "pool_record_selection_sha256": payload["selection_sha256"],
            "model_identity_counts": dict(
                sorted(
                    Counter(
                        str(row["model_name"]) for row in records if row.get("model_name")
                    ).items()
                )
            ),
            "prompt_encoding_counts": dict(
                sorted(
                    Counter(
                        str(row["prompt_encoding"])
                        for row in records
                        if row.get("prompt_encoding")
                    ).items()
                )
            ),
        },
    )


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("audit-vision")
    subparsers.add_parser("audit-fodb")
    subparsers.add_parser("audit-csafe")
    ai_parser = subparsers.add_parser("audit-ai")
    ai_parser.add_argument(
        "--source", choices=("qwen-image-2512", "flux2-klein-9b"), required=True
    )
    pool_parser = subparsers.add_parser("audit-pool-ai")
    pool_parser.add_argument("--root", type=Path, required=True)
    pool_parser.add_argument("--source", choices=POOL_SOURCE_IDS, required=True)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "audit-vision":
        report = audit_vision()
    elif args.command == "audit-fodb":
        report = audit_fodb()
    elif args.command == "audit-csafe":
        report = audit_csafe()
    elif args.command == "audit-ai":
        report = audit_ai_source(args.source)
    else:
        report = audit_pool_source(args.root, args.source)
    summary = {key: value for key, value in report.items() if key != "records"}
    print(json.dumps(summary, indent=2, sort_keys=True))
    return int(report["state"] != "source_realization_passed_candidate_only")


if __name__ == "__main__":
    raise SystemExit(main())
