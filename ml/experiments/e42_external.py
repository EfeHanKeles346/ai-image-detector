"""Acquire the parent-grouped B-Free viral stress set without model access."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import hashlib
import json
import re
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

import requests
from PIL import Image

from pixelproof.data_contract import dhash_image
from pixelproof.project_paths import DATA_ROOT, ML_ROOT


REPO_ROOT = ML_ROOT.parent
BFREE_ROOT = ML_ROOT / "external" / "B-Free"
REGISTRY = BFREE_ROOT / "viral_images_dataset" / "BFree_viral_images.csv"
REGISTRY_SHA256 = "3c727c4f8990ca91e129c97842fbf3c997b25fa6430fc316ffccc756f2373fc8"
BFREE_REVISION = "c6a9f898782fb466b29af01f21960b67415afb0e"
OUTPUT_ROOT = DATA_ROOT / "e42_external" / "bfree_viral"
DETAIL = OUTPUT_ROOT / "acquisition_manifest.json"
EVIDENCE = REPO_ROOT / "evidence" / "e42_bfree_acquisition.json"
MANIFEST = OUTPUT_ROOT / "unscored_manifest.json"
MANIFEST_EVIDENCE = REPO_ROOT / "evidence" / "e42_bfree_manifest.json"
CONTRACT = REPO_ROOT / "evidence" / "e42_external_contract.json"
EXPECTED_ROWS = 1_111
EXPECTED_BY_LABEL = {"REAL": 361, "FAKE": 750}
EXPECTED_SOURCES_BY_LABEL = {"REAL": 17, "FAKE": 17}
LABEL_MAP = {"REAL": 0, "FAKE": 1}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
MD5_RE = re.compile(r"^[0-9a-f]{32}$")


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def _write_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_bytes(_json_bytes(value))
    temporary.replace(path)


def _digest(path: Path, algorithm: str = "sha256") -> str:
    digest = hashlib.new(algorithm, usedforsecurity=False)
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024**2):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_relative_path(value: str, label: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or len(path.parts) != 3:
        raise ValueError(f"unsafe B-Free filename: {value!r}")
    if path.parts[0] != label or path.suffix.lower() not in IMAGE_SUFFIXES:
        raise ValueError(f"B-Free filename/label mismatch: {value!r}")
    return path


def verified_path(row: Mapping[str, Any], root: Path = OUTPUT_ROOT) -> Path:
    relative = _safe_relative_path(str(row["relative_path"]), str(row["label_name"]))
    return root / "images" / Path(*relative.parts)


def load_registry(path: Path = REGISTRY) -> list[dict[str, Any]]:
    if _digest(path) != REGISTRY_SHA256:
        raise ValueError("B-Free viral registry changed")
    with path.open(newline="", encoding="utf-8") as stream:
        records = list(csv.DictReader(stream))
    required = {"filename", "label", "source_id", "date", "days_since_1st_post", "w", "h", "md5", "url"}
    if not records or set(records[0]) != required:
        raise ValueError("B-Free viral registry schema changed")

    rows: list[dict[str, Any]] = []
    seen_paths: set[str] = set()
    source_labels: dict[str, str] = {}
    for raw in records:
        label = str(raw["label"])
        if label not in LABEL_MAP:
            raise ValueError(f"unknown B-Free label: {label!r}")
        relative = _safe_relative_path(str(raw["filename"]), label)
        if str(relative) in seen_paths:
            raise ValueError(f"duplicate B-Free filename: {relative}")
        seen_paths.add(str(relative))
        source_id = str(raw["source_id"])
        prior = source_labels.setdefault(source_id, label)
        if prior != label:
            raise ValueError(f"B-Free source event crosses labels: {source_id}")
        md5 = str(raw["md5"]).lower()
        if not MD5_RE.fullmatch(md5):
            raise ValueError(f"invalid B-Free MD5: {md5!r}")
        url = str(raw["url"])
        if not url.startswith(("https://", "http://")):
            raise ValueError(f"invalid B-Free URL: {url!r}")
        width, height = int(float(raw["w"])), int(float(raw["h"]))
        if width < 1 or height < 1:
            raise ValueError(f"invalid B-Free geometry: {width}x{height}")
        rows.append({
            "record_id": f"bfree-viral:{relative}",
            "relative_path": str(relative),
            "label_name": label,
            "label": LABEL_MAP[label],
            "source_id": source_id,
            "date": str(raw["date"]),
            "days_since_first_post": float(raw["days_since_1st_post"]),
            "expected_width": width,
            "expected_height": height,
            "expected_md5": md5,
            "url": url,
        })

    by_label = {label: sum(row["label_name"] == label for row in rows) for label in LABEL_MAP}
    sources = {
        label: len({row["source_id"] for row in rows if row["label_name"] == label})
        for label in LABEL_MAP
    }
    if len(rows) != EXPECTED_ROWS or by_label != EXPECTED_BY_LABEL or sources != EXPECTED_SOURCES_BY_LABEL:
        raise ValueError(f"B-Free registry population changed: rows={len(rows)}, labels={by_label}, sources={sources}")
    return rows


def _download_one(row: Mapping[str, Any], root: Path, timeout: float) -> dict[str, Any]:
    destination = root / "images" / str(row["relative_path"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if _digest(destination, "md5") != row["expected_md5"]:
            return {**row, "status": "existing_md5_mismatch"}
        transfer = "already_complete"
    else:
        partial = destination.with_suffix(destination.suffix + ".partial")
        try:
            with requests.get(
                str(row["url"]),
                stream=True,
                timeout=(10, timeout),
                headers={"User-Agent": "PixelProof-research/1.0"},
            ) as response:
                response.raise_for_status()
                digest = hashlib.md5(usedforsecurity=False)
                with partial.open("wb") as stream:
                    for chunk in response.iter_content(1024**2):
                        if chunk:
                            stream.write(chunk)
                            digest.update(chunk)
            if digest.hexdigest() != row["expected_md5"]:
                partial.unlink(missing_ok=True)
                return {**row, "status": "downloaded_md5_mismatch"}
            partial.replace(destination)
            transfer = "downloaded"
        except requests.RequestException as error:
            partial.unlink(missing_ok=True)
            return {**row, "status": "request_failed", "error": type(error).__name__}

    try:
        with Image.open(destination) as image:
            image.verify()
        with Image.open(destination) as image:
            width, height = image.size
            image.convert("RGB").load()
    except Exception as error:  # Pillow exposes multiple decoder exception types.
        return {**row, "status": "decode_failed", "error": type(error).__name__}
    if (width, height) != (row["expected_width"], row["expected_height"]):
        return {**row, "status": "geometry_mismatch", "actual_width": width, "actual_height": height}
    return {
        **row,
        "status": "verified",
        "transfer": transfer,
        "bytes": destination.stat().st_size,
        "actual_width": width,
        "actual_height": height,
        "sha256": _digest(destination),
        "local_path": str(destination),
    }


def summarize(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    verified = [row for row in rows if row["status"] == "verified"]
    failures: dict[str, int] = {}
    for row in rows:
        if row["status"] != "verified":
            failures[str(row["status"])] = failures.get(str(row["status"]), 0) + 1
    return {
        "declared_rows": len(rows),
        "verified_rows": len(verified),
        "coverage": len(verified) / len(rows),
        "verified_by_label": {
            label: sum(row["label_name"] == label for row in verified) for label in LABEL_MAP
        },
        "covered_sources_by_label": {
            label: len({row["source_id"] for row in verified if row["label_name"] == label})
            for label in LABEL_MAP
        },
        "verified_bytes": sum(int(row.get("bytes", 0)) for row in verified),
        "failure_statuses": failures,
    }


def excluded_event_ids(
    rows: Sequence[Mapping[str, Any]], prior_exact: set[str], prior_dhash: set[str]
) -> tuple[set[str], dict[str, list[str]]]:
    excluded = {
        str(row["source_id"])
        for row in rows
        if str(row["sha256"]) in prior_exact or str(row["dhash"]) in prior_dhash
    }
    cross_event: dict[str, list[str]] = {}
    for field in ("sha256", "dhash"):
        by_hash: dict[str, set[str]] = {}
        for row in rows:
            by_hash.setdefault(str(row[field]), set()).add(str(row["source_id"]))
        for value, sources in by_hash.items():
            if len(sources) > 1:
                key = f"{field}:{value}"
                cross_event[key] = sorted(sources)
                excluded.update(sources)
    return excluded, cross_event


def _prior_hashes() -> tuple[set[str], set[str], int]:
    exact: set[str] = set()
    perceptual: set[str] = set()
    paths = list(sorted((DATA_ROOT / "e32" / "audits").glob("*.json")))
    paths.extend([
        DATA_ROOT / "e33_rrdataset" / "r1c_cal_manifest.json",
        DATA_ROOT / "e36" / "cal_manifest.json",
        DATA_ROOT / "e36" / "final_manifest.json",
        DATA_ROOT / "e39" / "final_manifest.json",
    ])
    seen = 0
    for path in paths:
        if not path.is_file() or path.name.startswith("._"):
            continue
        payload = json.loads(path.read_text())
        records = payload.get("records", payload.get("rows", []))
        for row in records:
            if row.get("sha256"):
                exact.add(str(row["sha256"]))
            if row.get("dhash"):
                perceptual.add(str(row["dhash"]))
        seen += 1
    return exact, perceptual, seen


def freeze_manifest() -> dict[str, Any]:
    if MANIFEST.exists() or MANIFEST_EVIDENCE.exists():
        raise FileExistsError("B-Free unscored manifest already exists; no silent remanifest")
    contract = json.loads(CONTRACT.read_text())
    acquisition = json.loads(DETAIL.read_text())
    if contract.get("state") != "frozen_before_external_bytes":
        raise ValueError("E42 external contract changed")
    if (
        acquisition.get("state") != "acquisition_complete_with_declared_coverage"
        or acquisition.get("candidate_sha256") != contract["e41"]["artifact_sha256"]
        or hashlib.sha256(_json_bytes(acquisition)).hexdigest()
        != json.loads(EVIDENCE.read_text()).get("detailed_manifest_sha256")
    ):
        raise ValueError("B-Free acquisition binding changed")

    audited: list[dict[str, Any]] = []
    for row in acquisition["rows"]:
        if row["status"] != "verified":
            continue
        path = verified_path(row)
        if _digest(path) != row["sha256"] or _digest(path, "md5") != row["expected_md5"]:
            raise ValueError(f"B-Free verified bytes changed: {row['relative_path']}")
        with Image.open(path) as image:
            image.load()
            dhash = dhash_image(image.convert("RGB"))
        audited.append({
            "record_id": row["record_id"],
            "parent_id": f"bfree-viral:{row['source_id']}",
            "source_id": row["source_id"],
            "path": str(path),
            "relative_path": row["relative_path"],
            "label": int(row["label"]),
            "label_name": row["label_name"],
            "condition": "web_propagated_version",
            "days_since_first_post": row["days_since_first_post"],
            "sha256": row["sha256"],
            "dhash": dhash,
            "bytes": row["bytes"],
            "width": row["actual_width"],
            "height": row["actual_height"],
        })
    prior_exact, prior_dhash, prior_files = _prior_hashes()
    if prior_files == 0:
        raise ValueError(
            "no protected prior manifests found; set PIXELPROOF_DATA_ROOT to the established dataset root"
        )
    excluded, cross_event = excluded_event_ids(audited, prior_exact, prior_dhash)
    selected = [row for row in audited if row["source_id"] not in excluded]
    source_counts = {
        label: len({row["source_id"] for row in selected if row["label_name"] == label})
        for label in LABEL_MAP
    }
    if min(source_counts.values()) < 15:
        raise ValueError(f"B-Free decontamination leaves too few parent events: {source_counts}")
    payload = {
        "schema_version": 1,
        "experiment": "E41/B-Free-viral-external-stress",
        "state": "wild_stress_manifest_frozen_unscored",
        "candidate_sha256": contract["e41"]["artifact_sha256"],
        "threshold": contract["e41"]["threshold"],
        "acquisition_manifest_sha256": hashlib.sha256(_json_bytes(acquisition)).hexdigest(),
        "counts": {
            "rows": len(selected),
            "real_rows": sum(row["label"] == 0 for row in selected),
            "ai_rows": sum(row["label"] == 1 for row in selected),
            "source_events": sum(source_counts.values()),
        },
        "source_events_by_label": source_counts,
        "prior_manifest_files": prior_files,
        "prior_exact_hashes": len(prior_exact),
        "prior_dhashes": len(prior_dhash),
        "excluded_source_events": sorted(excluded),
        "cross_event_duplicate_groups": cross_event,
        "rows": selected,
        "boundary": "Decoded, parent-grouped and decontaminated before the first E41 score. Model access remains forbidden.",
    }
    raw = _json_bytes(payload)
    _write_atomic(MANIFEST, payload)
    compact = {
        "schema_version": 1,
        "state": payload["state"],
        "candidate_sha256": payload["candidate_sha256"],
        "counts": payload["counts"],
        "source_events_by_label": source_counts,
        "excluded_source_events": sorted(excluded),
        "cross_event_duplicate_group_count": len(cross_event),
        "prior_manifest_files": prior_files,
        "detailed_manifest_bytes": len(raw),
        "detailed_manifest_sha256": hashlib.sha256(raw).hexdigest(),
    }
    _write_atomic(MANIFEST_EVIDENCE, compact)
    return compact


def acquire(workers: int = 16, timeout: float = 60) -> dict[str, Any]:
    contract = json.loads(CONTRACT.read_text())
    if contract.get("state") != "frozen_before_external_bytes" or contract.get("e41", {}).get("threshold") != 0.6195540428161622:
        raise ValueError("E42 external contract changed")
    rows = load_registry()
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_download_one, row, OUTPUT_ROOT, timeout): row for row in rows}
        for index, future in enumerate(as_completed(futures), 1):
            results.append(future.result())
            if index % 50 == 0 or index == len(rows):
                print(f"B-Free viral {index}/{len(rows)}", flush=True)
    results.sort(key=lambda row: row["relative_path"])
    summary = summarize(results)
    detailed = {
        "schema_version": 1,
        "experiment": "E41/B-Free-viral-external-stress",
        "state": "acquisition_complete_with_declared_coverage",
        "candidate_sha256": contract["e41"]["artifact_sha256"],
        "registry_sha256": REGISTRY_SHA256,
        "bfree_revision": BFREE_REVISION,
        "summary": summary,
        "rows": results,
        "boundary": "No model was loaded. Failed URLs and changed bytes remain explicit coverage failures.",
    }
    raw = _json_bytes(detailed)
    _write_atomic(DETAIL, detailed)
    compact = {
        "schema_version": 1,
        "experiment": detailed["experiment"],
        "state": detailed["state"],
        "candidate_sha256": detailed["candidate_sha256"],
        "registry_sha256": REGISTRY_SHA256,
        "bfree_revision": BFREE_REVISION,
        "summary": summary,
        "detailed_manifest_bytes": len(raw),
        "detailed_manifest_sha256": hashlib.sha256(raw).hexdigest(),
    }
    _write_atomic(EVIDENCE, compact)
    return compact


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("validate-registry", "acquire", "freeze-manifest"))
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--timeout", type=float, default=60)
    args = parser.parse_args()
    if args.command == "validate-registry":
        rows = load_registry()
        print(json.dumps({"rows": len(rows), "summary": summarize([{**row, "status": "verified"} for row in rows])}, indent=2))
    elif args.command == "acquire":
        print(json.dumps(acquire(args.workers, args.timeout), indent=2))
    else:
        print(json.dumps(freeze_manifest(), indent=2))


if __name__ == "__main__":
    main()
