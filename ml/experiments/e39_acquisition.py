"""Freeze and acquire the source-disjoint E39 FINAL without model access."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import hashlib
import io
import json
import re
import shutil
import subprocess
import tarfile
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence

import requests
from PIL import Image

from pixelproof.data_contract import dhash_image
from pixelproof.project_paths import DATA_ROOT, ML_ROOT


ROOT = DATA_ROOT / "e39"
SOURCE_CONTRACT = ML_ROOT.parent / "evidence" / "e39_source_contract.json"
SOURCE_SELECTION = ROOT / "source_selection.json"
PREFLIGHT_EVIDENCE = ML_ROOT.parent / "evidence" / "e39_source_preflight.json"
AI_INVENTORY = ROOT / "ai_inventory.json"
AI_INVENTORY_EVIDENCE = ML_ROOT.parent / "evidence" / "e39_ai_inventory.json"
FINAL_MANIFEST = ROOT / "final_manifest.json"
FINAL_MANIFEST_EVIDENCE = ML_ROOT.parent / "evidence" / "e39_final_manifest.json"
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
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
MAX_TAR_MEMBER_BYTES = 100 * 1024**2
MAX_TAR_TOTAL_BYTES = 30 * 1024**3
AI_PREFIXES = {
    "Reve Image 1.0": "fal-ai_reve_text-to-image_",
    "HiDream I1 Dev": "fal-ai_hidream-i1-dev_",
    "Ideogram 3": "fal-ai_ideogram_v3_",
    "Midjourney v7": "image_midjourneyv7_",
    "Adobe Firefly Image 5": "Firefly_",
    "Z Image Turbo": "fal-ai_z-image_turbo_",
    "Gemini 3 Pro Image": "fal-ai_gemini-3-pro-image-preview_",
}


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


def _generator_for_member(name: str) -> str | None:
    path = PurePosixPath(name)
    if "1_fake" not in path.parts or path.suffix.lower() not in IMAGE_SUFFIXES:
        return None
    basename = path.name
    return next((generator for generator, prefix in AI_PREFIXES.items() if basename.startswith(prefix)), None)


def select_ai_members(members: Sequence[Mapping[str, Any]]) -> tuple[dict[str, list[dict[str, Any]]], dict[str, int]]:
    eligible: dict[str, list[dict[str, Any]]] = {generator: [] for generator in AI_GENERATORS}
    for member in members:
        name = str(member["member"])
        generator = _generator_for_member(name)
        if generator is None:
            continue
        split = "train" if "/train/" in name else "test" if "/val/" in name else "unknown"
        rank_key = _sha256(f"{generator}\0{name}".encode())
        eligible[generator].append({**member, "source": generator, "upstream_split": split, "rank_key": rank_key})
    selected: dict[str, list[dict[str, Any]]] = {}
    for generator, rows in eligible.items():
        rows.sort(key=lambda row: (row["rank_key"], row["member"]))
        if len(rows) < AI_PER_GENERATOR:
            raise ValueError(f"AIGenImages2026 has fewer than 40 eligible rows for {generator}")
        selected[generator] = [
            {**row, "local_name": f"{index:03d}_{row['rank_key'][:12]}{PurePosixPath(row['member']).suffix.lower()}"}
            for index, row in enumerate(rows[:AI_PER_GENERATOR])
        ]
    return selected, {generator: len(rows) for generator, rows in eligible.items()}


def inventory_ai() -> dict[str, Any]:
    if AI_INVENTORY.exists() or AI_INVENTORY_EVIDENCE.exists():
        raise FileExistsError("E39 AI inventory already exists; no silent reinventory")
    receipt = json.loads((ROOT / "ai_download_receipt.json").read_text())
    archive_path = ROOT / "archives" / AI_ARCHIVE
    if (
        receipt.get("state") != "ai_archive_download_complete_verified_unscored"
        or not archive_path.is_file()
        or archive_path.stat().st_size != AI_ARCHIVE_BYTES
        or _digest(archive_path) != AI_ARCHIVE_SHA256
    ):
        raise ValueError("E39 AI archive binding changed before inventory")
    members: list[dict[str, Any]] = []
    metadata_raw: bytes | None = None
    total_members = 0
    regular_files = 0
    total_regular_bytes = 0
    with tarfile.open(archive_path, "r:gz") as bundle:
        for member in bundle:
            total_members += 1
            path = PurePosixPath(member.name)
            if path.is_absolute() or ".." in path.parts or member.issym() or member.islnk():
                raise ValueError(f"unsafe E39 tar member: {member.name}")
            if member.isdir():
                continue
            if not member.isfile() or member.size < 0 or member.size > MAX_TAR_MEMBER_BYTES:
                raise ValueError(f"unsupported E39 tar member: {member.name}")
            regular_files += 1
            total_regular_bytes += member.size
            if total_regular_bytes > MAX_TAR_TOTAL_BYTES:
                raise ValueError("E39 tar expansion exceeds the frozen safety ceiling")
            members.append({"member": member.name, "bytes": member.size})
            if path.name == "fake_metadata.csv":
                stream = bundle.extractfile(member)
                if stream is None:
                    raise ValueError("E39 fake metadata cannot be read")
                metadata_raw = stream.read()
    if metadata_raw is None:
        raise ValueError("E39 archive lacks fake_metadata.csv")
    selected, eligible_counts = select_ai_members(members)
    metadata_path = ROOT / "metadata" / "fake_metadata.csv"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_part = metadata_path.with_suffix(".csv.part")
    metadata_part.write_bytes(metadata_raw)
    metadata_part.replace(metadata_path)
    payload = {
        "schema_version": 1,
        "experiment": "E39/new-native-modern-final",
        "state": "ai_inventory_passed_selected_unextracted_unscored",
        "archive_sha256": AI_ARCHIVE_SHA256,
        "total_members": total_members,
        "regular_files": regular_files,
        "total_regular_bytes": total_regular_bytes,
        "eligible_counts": eligible_counts,
        "selected_counts": {generator: len(rows) for generator, rows in selected.items()},
        "selected": selected,
        "metadata_bytes": len(metadata_raw),
        "metadata_sha256": _sha256(metadata_raw),
        "boundary": "Archive safety and deterministic member selection only; no image extracted, decoded or scored.",
    }
    raw = _json_bytes(payload)
    _write_atomic(AI_INVENTORY, payload)
    evidence = {
        "schema_version": 1,
        "state": payload["state"],
        "archive_sha256": AI_ARCHIVE_SHA256,
        "total_members": total_members,
        "regular_files": regular_files,
        "total_regular_bytes": total_regular_bytes,
        "eligible_counts": eligible_counts,
        "selected_counts": payload["selected_counts"],
        "metadata_sha256": payload["metadata_sha256"],
        "detailed_inventory_bytes": len(raw),
        "detailed_inventory_sha256": _sha256(raw),
    }
    _write_atomic(AI_INVENTORY_EVIDENCE, evidence)
    return evidence


def _audit_image(path: Path) -> dict[str, Any]:
    with Image.open(path) as image:
        image.load()
        width, height = image.size
        decoded_format = image.format
        exif_present = bool(image.getexif())
        dhash = dhash_image(image.convert("RGB"))
    if min(width, height) < 336:
        raise ValueError(f"E39 image is below the native/clean input floor: {path}")
    return {
        "bytes": path.stat().st_size,
        "sha256": _digest(path),
        "dhash": dhash,
        "width": width,
        "height": height,
        "megapixels": width * height / 1_000_000,
        "decoded_format": decoded_format,
        "exif_present": exif_present,
    }


def _prior_hashes() -> tuple[set[str], set[str]]:
    exact: set[str] = set()
    perceptual: set[str] = set()
    for path in sorted((DATA_ROOT / "e32" / "audits").glob("*.json")):
        if path.name.startswith("._"):
            continue
        payload = json.loads(path.read_text())
        for row in payload.get("records", []):
            if row.get("sha256"):
                exact.add(str(row["sha256"]))
            if row.get("dhash"):
                perceptual.add(str(row["dhash"]))
    for path in (DATA_ROOT / "e36" / "cal_manifest.json", DATA_ROOT / "e36" / "final_manifest.json"):
        payload = json.loads(path.read_text())
        for row in payload.get("rows", []):
            exact.add(str(row["sha256"]))
            perceptual.add(str(row["dhash"]))
    return exact, perceptual


def _metadata_prompts(path: Path) -> dict[str, str]:
    rows = list(csv.DictReader(io.StringIO(path.read_text(errors="replace"))))
    if not rows:
        return {}
    filename_keys = ("filename", "file_name", "image", "image_name", "path")
    prompt_keys = ("prompt", "text", "caption")
    output: dict[str, str] = {}
    for row in rows:
        filename = next((str(row[key]) for key in filename_keys if row.get(key)), "")
        prompt = next((str(row[key]) for key in prompt_keys if row.get(key)), "")
        if filename and prompt:
            output[PurePosixPath(filename).name] = prompt
    return output


def manifest_final() -> dict[str, Any]:
    if FINAL_MANIFEST.exists() or FINAL_MANIFEST_EVIDENCE.exists():
        raise FileExistsError("E39 FINAL manifest already exists; no silent remanifest")
    selection = json.loads(SOURCE_SELECTION.read_text())
    inventory = json.loads(AI_INVENTORY.read_text())
    if selection.get("state") != "source_selection_frozen_zero_image_bytes":
        raise ValueError("E39 source selection changed")
    if inventory.get("state") != "ai_inventory_passed_selected_unextracted_unscored":
        raise ValueError("E39 AI inventory has not passed")
    if json.loads((ROOT / "real_download_receipt.json").read_text()).get("rows") != 160:
        raise ValueError("E39 REAL download is incomplete")

    real_rows = []
    for device, rows in selection["real"].items():
        for selected in rows:
            path = ROOT / "final" / "real" / device / PurePosixPath(selected["remote_path"]).name
            audit = _audit_image(path)
            if audit["megapixels"] < 2.0 or not audit["exif_present"]:
                raise ValueError(f"E39 REAL row is not a native high-resolution camera parent: {path}")
            real_rows.append({
                "role": "locked_final",
                "label": 0,
                "source": device,
                "condition": "camera_native_outdoor",
                "parent_id": f"floreview:{selected['remote_path']}",
                "upstream_url": selected["url"],
                "path": str(path.relative_to(ROOT)),
                **audit,
            })

    selected_by_member = {
        row["member"]: row
        for rows in inventory["selected"].values()
        for row in rows
    }
    ai_root = ROOT / "final" / "ai"
    expected_paths = {
        member: ai_root / re.sub(r"[^a-z0-9]+", "-", row["source"].lower()).strip("-") / row["local_name"]
        for member, row in selected_by_member.items()
    }
    missing = {member for member, path in expected_paths.items() if not path.exists()}
    if missing:
        archive_path = ROOT / "archives" / AI_ARCHIVE
        with tarfile.open(archive_path, "r|gz") as bundle:
            for member in bundle:
                if member.name not in missing:
                    continue
                destination = expected_paths[member.name]
                destination.parent.mkdir(parents=True, exist_ok=True)
                partial = destination.with_suffix(destination.suffix + ".partial")
                if partial.exists():
                    raise FileExistsError(f"partial E39 extraction requires audit: {partial}")
                source = bundle.extractfile(member)
                if source is None:
                    raise ValueError(f"cannot extract E39 AI member: {member.name}")
                with partial.open("xb") as output:
                    shutil.copyfileobj(source, output, length=8 * 1024**2)
                if partial.stat().st_size != int(selected_by_member[member.name]["bytes"]):
                    raise ValueError(f"E39 AI extraction size mismatch: {member.name}")
                partial.replace(destination)
                missing.remove(member.name)
                if len(missing) % 40 == 0:
                    print(f"E39 AI extracted {len(expected_paths) - len(missing)}/{len(expected_paths)}", flush=True)
                if not missing:
                    break
    if missing:
        raise ValueError(f"E39 AI selected members missing after extraction: {len(missing)}")

    prompts = _metadata_prompts(ROOT / "metadata" / "fake_metadata.csv")
    ai_rows = []
    for member, selected in sorted(selected_by_member.items(), key=lambda item: (item[1]["source"], item[1]["rank_key"])):
        path = expected_paths[member]
        audit = _audit_image(path)
        row = {
            "role": "locked_final",
            "label": 1,
            "source": selected["source"],
            "condition": "clean_generator_output",
            "parent_id": f"aigenimages2026:{member}",
            "upstream_member": member,
            "upstream_split": selected["upstream_split"],
            "path": str(path.relative_to(ROOT)),
            **audit,
        }
        prompt = prompts.get(PurePosixPath(member).name)
        if prompt:
            row["prompt"] = prompt
        ai_rows.append(row)

    rows = real_rows + ai_rows
    if len(real_rows) != 160 or len(ai_rows) != 280 or len(rows) != 440:
        raise ValueError("E39 FINAL count contract changed")
    if len({row["sha256"] for row in rows}) != len(rows):
        raise ValueError("E39 FINAL contains exact duplicate bytes")
    prior_exact, prior_dhash = _prior_hashes()
    if any(row["sha256"] in prior_exact for row in rows):
        raise ValueError("E39 FINAL exact-overlaps an earlier role")
    if any(row["dhash"] in prior_dhash for row in rows):
        raise ValueError("E39 FINAL perceptually overlaps an earlier role")
    by_dhash: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_dhash.setdefault(row["dhash"], []).append(row)
    if any(len(group) > 1 for group in by_dhash.values()):
        raise ValueError("E39 FINAL contains within-role perceptual duplicate parents")

    payload = {
        "schema_version": 1,
        "experiment": "E39/new-native-modern-final",
        "state": "final_manifest_frozen_unscored",
        "decision_contract_sha256": DECISION_CONTRACT_SHA256,
        "source_selection_sha256": _digest(SOURCE_SELECTION),
        "ai_inventory_sha256": _digest(AI_INVENTORY),
        "counts": {"real": 160, "ai": 280, "total": 440},
        "real_source_counts": {source: sum(row["source"] == source for row in real_rows) for source in REAL_DEVICES},
        "ai_family_counts": {source: sum(row["source"] == source for row in ai_rows) for source in AI_GENERATORS},
        "prompt_rows": sum("prompt" in row for row in ai_rows),
        "prior_exact_overlap_count": 0,
        "prior_dhash_overlap_count": 0,
        "within_final_duplicate_dhash_count": 0,
        "rows": rows,
        "boundary": "Frozen, decoded and decontaminated before first E39 prediction; one score only.",
    }
    raw = _json_bytes(payload)
    _write_atomic(FINAL_MANIFEST, payload)
    evidence = {
        "schema_version": 1,
        "state": payload["state"],
        "decision_contract_sha256": DECISION_CONTRACT_SHA256,
        "counts": payload["counts"],
        "real_source_counts": payload["real_source_counts"],
        "ai_family_counts": payload["ai_family_counts"],
        "prompt_rows": payload["prompt_rows"],
        "prior_exact_overlap_count": 0,
        "prior_dhash_overlap_count": 0,
        "within_final_duplicate_dhash_count": 0,
        "detailed_manifest_bytes": len(raw),
        "detailed_manifest_sha256": _sha256(raw),
    }
    _write_atomic(FINAL_MANIFEST_EVIDENCE, evidence)
    return evidence


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("freeze")
    real = sub.add_parser("download-real")
    real.add_argument("--workers", type=int, default=8)
    sub.add_parser("download-ai")
    sub.add_parser("inventory-ai")
    sub.add_parser("manifest-final")
    args = parser.parse_args(argv)
    if args.command == "freeze":
        result = freeze()
    elif args.command == "download-real":
        result = download_real(args.workers)
    elif args.command == "download-ai":
        result = download_ai()
    elif args.command == "inventory-ai":
        result = inventory_ai()
    else:
        result = manifest_final()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
