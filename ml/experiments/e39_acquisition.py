"""Freeze and acquire the source-disjoint E39 FINAL without model access."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence

import requests

from pixelproof.project_paths import DATA_ROOT, ML_ROOT


ROOT = DATA_ROOT / "e39"
SOURCE_CONTRACT = ML_ROOT.parent / "evidence" / "e39_source_contract.json"
SOURCE_SELECTION = ROOT / "source_selection.json"
PREFLIGHT_EVIDENCE = ML_ROOT.parent / "evidence" / "e39_source_preflight.json"
DECISION_CONTRACT = ROOT / "e39_threshold_candidate.json"
DECISION_CONTRACT_SHA256 = "7d49792911b9b24acca2ad58d08d5ecc14bded5ee85174218d08d1a4d3712cef"
FLOREVIEW_CATALOG_URL = "https://lesc.dinfo.unifi.it/FloreView/FloreView_Dataset.txt"
FLOREVIEW_CATALOG_SHA256 = "90d8408c47d8290cd31db2c6cde6ae2aa17b8c5b3b65191cec3f70969e7b186b"
FLOREVIEW_README_URL = "https://lesc.dinfo.unifi.it/FloreView/readme.md"
FLOREVIEW_README_SHA256 = "bc690a1337d3ee7be7c7984bae7b3f87ec9fbb53327ffc8a39ae6e9b754270b1"
REAL_DEVICES = (
    "D14_Apple_iPhone13mini",
    "D27_DOOGEE_S96Pro",
    "D34_Google_Pixel5",
    "D43_OnePlus_8T",
)
REAL_PER_DEVICE = 40
HF_REPO = "sha6th/AIGenImages2026"
HF_REVISION = "d634f663049678ef33ba66f785b12496b250c0c5"
HF_API = f"https://huggingface.co/api/datasets/{HF_REPO}/revision/{HF_REVISION}"
HF_TREE = f"https://huggingface.co/api/datasets/{HF_REPO}/tree/{HF_REVISION}?recursive=true&expand=true"
AI_ARCHIVE = "aigenimages2026.tar.gz"
AI_ARCHIVE_BYTES = 11_138_511_098
AI_ARCHIVE_SHA256 = "67c6042712f783aebfdb29f8a8903dfc94fc7ac54fee5c154eaf6b880d0ec498"
AI_XET_HASH = "6ff1c1e7bc2c29f34a7c0ad0ab75852a11dc186930f7e377fbf536771164533a"
AI_GENERATORS = (
    "Reve Image 1.0",
    "HiDream I1 Dev",
    "Ideogram 3",
    "Midjourney v7",
    "Adobe Firefly Image 5",
    "Z Image Turbo",
    "Gemini 3 Pro Image",
)
AI_PER_GENERATOR = 40
MIN_FREE_BYTES = 100 * 1024**3


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def _write_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_bytes(_json_bytes(value))
    temporary.replace(path)


def _request_bytes(url: str) -> bytes:
    response = requests.get(url, timeout=(30, 180))
    response.raise_for_status()
    return response.content


def _real_sort_key(url: str) -> tuple[int, int, int, str]:
    name = PurePosixPath(url).name
    match = re.search(r"_L(\d+)S(\d+)C(\d+)\.(?:jpe?g)$", name, re.IGNORECASE)
    if match is None:
        raise ValueError(f"unexpected FloreView natural-image name: {name}")
    location, subject, capture = map(int, match.groups())
    return capture, subject, location, url.lower()


def select_real_urls(catalog: bytes) -> dict[str, list[dict[str, Any]]]:
    if _sha256(catalog) != FLOREVIEW_CATALOG_SHA256:
        raise ValueError("FloreView catalog changed")
    lines = [line.strip() for line in catalog.decode().splitlines() if line.strip()]
    selected: dict[str, list[dict[str, Any]]] = {}
    for device in REAL_DEVICES:
        prefix = f"https://lesc.dinfo.unifi.it/FloreView/Dataset/{device}/Nat/jpeg-h264/"
        eligible = [line for line in lines if line.startswith(prefix) and line.lower().endswith((".jpg", ".jpeg"))]
        eligible.sort(key=_real_sort_key)
        if len(eligible) < REAL_PER_DEVICE:
            raise ValueError(f"FloreView device has fewer than 40 native natural images: {device}")
        selected[device] = [
            {"url": url, "remote_path": url.removeprefix("https://lesc.dinfo.unifi.it/FloreView/Dataset/")}
            for url in eligible[:REAL_PER_DEVICE]
        ]
    return selected


def validate_huggingface(metadata: Mapping[str, Any], tree: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if metadata.get("id") != HF_REPO or metadata.get("sha") != HF_REVISION:
        raise ValueError("AIGenImages2026 identity/revision changed")
    if metadata.get("cardData", {}).get("license") != "cc-by-4.0":
        raise ValueError("AIGenImages2026 licence changed")
    archive = next((item for item in tree if item.get("path") == AI_ARCHIVE), None)
    if archive is None:
        raise ValueError("AIGenImages2026 archive disappeared")
    lfs = archive.get("lfs") or {}
    if (
        int(archive.get("size", -1)) != AI_ARCHIVE_BYTES
        or lfs.get("oid") != AI_ARCHIVE_SHA256
        or archive.get("xetHash") != AI_XET_HASH
    ):
        raise ValueError("AIGenImages2026 archive contract changed")
    return {
        "repo_id": HF_REPO,
        "revision": HF_REVISION,
        "license": "CC BY 4.0",
        "archive": AI_ARCHIVE,
        "bytes": AI_ARCHIVE_BYTES,
        "sha256": AI_ARCHIVE_SHA256,
        "xet_hash": AI_XET_HASH,
        "url": f"https://huggingface.co/datasets/{HF_REPO}/resolve/{HF_REVISION}/{AI_ARCHIVE}?download=true",
        "generators": {name: AI_PER_GENERATOR for name in AI_GENERATORS},
    }


def freeze() -> dict[str, Any]:
    if SOURCE_SELECTION.exists() or PREFLIGHT_EVIDENCE.exists():
        raise FileExistsError("E39 source selection already exists; no silent refreeze")
    contract = json.loads(SOURCE_CONTRACT.read_text())
    if contract.get("state") != "source_contract_frozen_zero_image_bytes" or contract.get("counts", {}).get("total") != 440:
        raise ValueError("E39 source contract changed")
    if not DECISION_CONTRACT.is_file() or _digest(DECISION_CONTRACT) != DECISION_CONTRACT_SHA256:
        raise ValueError("E39 decision contract changed")
    readme = _request_bytes(FLOREVIEW_README_URL)
    if _sha256(readme) != FLOREVIEW_README_SHA256 or b"Attribution-ShareAlike 4.0" not in readme:
        raise ValueError("FloreView licence receipt changed")
    catalog = _request_bytes(FLOREVIEW_CATALOG_URL)
    real = select_real_urls(catalog)
    metadata_response = requests.get(HF_API, timeout=(30, 180))
    metadata_response.raise_for_status()
    tree_response = requests.get(HF_TREE, timeout=(30, 180))
    tree_response.raise_for_status()
    ai = validate_huggingface(metadata_response.json(), tree_response.json())
    detailed = {
        "schema_version": 1,
        "experiment": "E39/new-native-modern-final",
        "state": "source_selection_frozen_zero_image_bytes",
        "decision_contract_sha256": DECISION_CONTRACT_SHA256,
        "source_contract_sha256": _digest(SOURCE_CONTRACT),
        "real": real,
        "ai": ai,
        "counts": {"real": 160, "ai": 280, "total": 440},
        "boundary": "Metadata-only freeze. No E39 FINAL image byte downloaded and no model loaded.",
    }
    raw = _json_bytes(detailed)
    _write_atomic(SOURCE_SELECTION, detailed)
    evidence = {
        "schema_version": 1,
        "state": detailed["state"],
        "counts": detailed["counts"],
        "real_devices": list(REAL_DEVICES),
        "real_rows": sum(len(rows) for rows in real.values()),
        "ai_generators": list(AI_GENERATORS),
        "ai_archive_bytes": AI_ARCHIVE_BYTES,
        "detailed_selection_bytes": len(raw),
        "detailed_selection_sha256": _sha256(raw),
        "new_image_bytes_downloaded_by_freeze": 0,
    }
    _write_atomic(PREFLIGHT_EVIDENCE, evidence)
    return evidence


def _curl(item: Mapping[str, Any], destination: Path) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    expected_bytes = item.get("bytes")
    expected_sha = item.get("sha256")
    if destination.exists():
        if expected_bytes is not None and destination.stat().st_size != int(expected_bytes):
            raise ValueError(f"completed file size changed: {destination}")
        if expected_sha is not None and _digest(destination) != str(expected_sha):
            raise ValueError(f"completed file hash changed: {destination}")
        return "already_complete"
    partial = destination.with_suffix(destination.suffix + ".partial")
    command = [
        "/usr/bin/curl", "--fail", "--location", "--silent", "--show-error",
        "--connect-timeout", "30", "--retry", "12", "--retry-delay", "3",
        "--retry-all-errors", "--speed-limit", "1024", "--speed-time", "180",
        "--output", str(partial),
    ]
    if partial.exists() and partial.stat().st_size:
        command.extend(["--continue-at", "-"])
    command.append(str(item["url"]))
    subprocess.run(command, check=True)
    if expected_bytes is not None and partial.stat().st_size != int(expected_bytes):
        raise ValueError(f"download size mismatch: {destination}")
    if expected_sha is not None and _digest(partial) != str(expected_sha):
        raise ValueError(f"download SHA-256 mismatch: {destination}")
    partial.replace(destination)
    return "downloaded"


def download_real(workers: int = 8) -> dict[str, Any]:
    selection = json.loads(SOURCE_SELECTION.read_text())
    if selection.get("state") != "source_selection_frozen_zero_image_bytes":
        raise ValueError("E39 source selection is not frozen")
    if shutil.disk_usage(ROOT).free < MIN_FREE_BYTES + 2 * 1024**3:
        raise OSError("E39 free-space floor is not satisfied")
    rows = [(device, row) for device, group in selection["real"].items() for row in group]
    states: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_curl, row, ROOT / "final" / "real" / device / PurePosixPath(row["remote_path"]).name): (device, row)
            for device, row in rows
        }
        for index, future in enumerate(as_completed(futures), 1):
            device, row = futures[future]
            states[f"{device}/{PurePosixPath(row['remote_path']).name}"] = future.result()
            if index % 20 == 0:
                print(f"E39 REAL {index}/{len(rows)}", flush=True)
    receipt = {"schema_version": 1, "state": "real_download_complete_unscored", "rows": len(rows), "states": states}
    _write_atomic(ROOT / "real_download_receipt.json", receipt)
    return receipt


def download_ai() -> dict[str, Any]:
    selection = json.loads(SOURCE_SELECTION.read_text())
    if selection.get("state") != "source_selection_frozen_zero_image_bytes":
        raise ValueError("E39 source selection is not frozen")
    if shutil.disk_usage(ROOT).free < MIN_FREE_BYTES + 30 * 1024**3:
        raise OSError("E39 free-space floor is not satisfied")
    item = selection["ai"]
    destination = ROOT / "archives" / AI_ARCHIVE
    state = _curl(item, destination)
    receipt = {
        "schema_version": 1,
        "state": "ai_archive_download_complete_verified_unscored",
        "archive": AI_ARCHIVE,
        "bytes": AI_ARCHIVE_BYTES,
        "sha256": AI_ARCHIVE_SHA256,
        "download_state": state,
    }
    _write_atomic(ROOT / "ai_download_receipt.json", receipt)
    return receipt


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("freeze")
    real = sub.add_parser("download-real")
    real.add_argument("--workers", type=int, default=8)
    sub.add_parser("download-ai")
    args = parser.parse_args(argv)
    if args.command == "freeze":
        result = freeze()
    elif args.command == "download-real":
        result = download_real(args.workers)
    else:
        result = download_ai()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
