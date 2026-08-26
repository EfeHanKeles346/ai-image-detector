"""E32/C1-C2 — realize frozen source bytes as role-free, audited candidates.

This gate never assigns TRAIN or CALIBRATION. It verifies that a frozen acquisition arrived
unchanged, decodes every selected image from its bytes, measures container/geometry metadata,
checks prompt-group integrity for AI sources, and rejects exact/perceptual overlap with protected
E30 manifests or previously passed E32 source audits.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence

from PIL import Image

import e32_data_system as real_acquisition
import e32_gap_acquisition as ai_acquisition
from pixelproof.project_paths import ML_ROOT


REPO_ROOT = ML_ROOT.parent
OUTPUT_ROOT = real_acquisition.OUTPUT_ROOT
AUDIT_ROOT = OUTPUT_ROOT / "audits"
EVIDENCE_ROOT = REPO_ROOT / "evidence"
CHUNK_BYTES = 8 * 1024**2


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


def _passed_peer_hashes(source_id: str) -> tuple[set[str], set[str], int]:
    exact: set[str] = set()
    perceptual: set[str] = set()
    reports = 0
    if not AUDIT_ROOT.exists():
        return exact, perceptual, reports
    for path in sorted(AUDIT_ROOT.glob("*.json")):
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if payload.get("source_id") == source_id:
            continue
        if payload.get("state") != "source_realization_passed_candidate_only":
            continue
        reports += 1
        for row in payload.get("records", []):
            if row.get("sha256"):
                exact.add(str(row["sha256"]))
            if row.get("dhash"):
                perceptual.add(str(row["dhash"]))
    return exact, perceptual, reports


def _image_record(path: Path) -> dict[str, Any]:
    raw_bytes = path.stat().st_size
    raw_sha256 = _sha256_file(path)
    with Image.open(path) as opened:
        opened.load()
        decoded_format = str(opened.format or "UNKNOWN").upper()
        mode = opened.mode
        width, height = opened.size
        exif = opened.getexif()
        orientation = exif.get(274) if exif else None
        perceptual = _dhash(opened)
    return {
        "bytes": raw_bytes,
        "sha256": raw_sha256,
        "dhash": perceptual,
        "decoded_format": decoded_format,
        "mode": mode,
        "width": width,
        "height": height,
        "orientation": orientation,
        "exif_present": bool(exif),
        "bytes_per_pixel": round(raw_bytes / max(1, width * height), 6),
    }


def _duplicates(records: Sequence[Mapping[str, Any]], key: str) -> list[list[str]]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for row in records:
        grouped[str(row[key])].append(str(row["source_key"]))
    return [sorted(paths) for paths in grouped.values() if len(paths) > 1]


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
    peer_exact, peer_dhash, peer_reports = _passed_peer_hashes(source_id)
    within_exact = _duplicates(records, "sha256")
    within_dhash = _duplicates(records, "dhash")
    protected_exact_hits = sorted(
        row["source_key"] for row in records if row["sha256"] in protected_exact
    )
    protected_dhash_hits = sorted(
        row["source_key"] for row in records if row["dhash"] in protected_dhash
    )
    peer_exact_hits = sorted(row["source_key"] for row in records if row["sha256"] in peer_exact)
    peer_dhash_hits = sorted(row["source_key"] for row in records if row["dhash"] in peer_dhash)
    if within_exact:
        failures.append({"reason": "within_source_exact_duplicates", "count": str(len(within_exact))})
    if within_dhash:
        failures.append({"reason": "within_source_dhash_duplicates", "count": str(len(within_dhash))})
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
        "schema_version": 1,
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
            "within_source_exact_duplicate_groups": len(within_exact),
            "within_source_dhash_duplicate_groups": len(within_dhash),
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
        "duplicate_groups": {"exact": within_exact, "dhash": within_dhash},
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
                    {"source_key": text_key, "reason": f"prompt_decode_failure:{type(error).__name__}"}
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


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("audit-vision")
    ai_parser = subparsers.add_parser("audit-ai")
    ai_parser.add_argument(
        "--source", choices=("qwen-image-2512", "flux2-klein-9b"), required=True
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    report = audit_vision() if args.command == "audit-vision" else audit_ai_source(args.source)
    print(json.dumps({key: value for key, value in report.items() if key != "records"}, indent=2, sort_keys=True))
    return int(report["state"] != "source_realization_passed_candidate_only")


if __name__ == "__main__":
    raise SystemExit(main())
