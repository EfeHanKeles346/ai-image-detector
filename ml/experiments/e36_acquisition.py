"""Freeze, acquire and realize E36 CAL while keeping FINAL physically sealed."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import re
import shutil
import subprocess
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence
from urllib.parse import quote

import requests
from PIL import Image

from pixelproof.data_contract import dhash_image
from pixelproof.project_paths import DATA_ROOT, ML_ROOT


ZENODO_ID = 18_136_670
ZENODO_API = f"https://zenodo.org/api/records/{ZENODO_ID}"
ZENODO_VERSION = "1.0.0"
ZENODO_FILES = {
    "device_001.zip": (406_035_087, "4411a326bb766517424ddebe89651d68"),
    "device_002.zip": (404_883_776, "fcdcd3fc506fe1fbef4c505cd40ed978"),
    "device_003.zip": (680_518_609, "adf9a0327e1102cfa0c1c1c6096dd79d"),
    "device_004.zip": (438_910_996, "ed52168813bccbcee22d9077c9ea1b8d"),
    "device_005.zip": (270_303_786, "39006d62ad3da2b3c4afb283de306ba2"),
    "device_006.zip": (386_910_344, "29cedc4ea0efd3ae5c37f529848ee1b6"),
    "device_007.zip": (474_343_058, "663f58da8699c2627b00cda74899e0cd"),
    "device_008.zip": (738_676_982, "815433c4711837438ebdcf335a081448"),
    "device_009.zip": (290_864_762, "4b34e343384530a580c339739ea38cb8"),
}
CAL_REAL = ("device_001.zip", "device_002.zip", "device_003.zip", "device_005.zip", "device_009.zip")
FINAL_REAL = ("device_004.zip", "device_006.zip", "device_007.zip", "device_008.zip")

HF_REPO = "Qwen/Qwen-Image-Bench"
HF_REVISION = "d2493deb153b020cf169c7e3f57d15e4dd697038"
HF_API = f"https://huggingface.co/api/datasets/{HF_REPO}/revision/{HF_REVISION}?blobs=true"
CAL_AI = (
    "gpt-image-2",
    "nano-banana-2.0",
    "Seedream-5.0",
    "Qwen-Image-2.0-pro",
    "FLUX.2_max",
    "GLM-Image",
)
FINAL_AI = (
    "GPT-Image-1.5",
    "nano-banana-pro",
    "Imagen-4.0-Ultra",
    "HunyuanImage-3.0",
    "FLUX.2-pro",
    "Seedream-4.5",
)
CAL_AI_RANGE = range(101, 201)
FINAL_AI_RANGE = range(1, 41)

ROOT = DATA_ROOT / "e36"
SELECTION = ROOT / "acquisition_selection.json"
CAL_MANIFEST = ROOT / "cal_manifest.json"
EVIDENCE_ROOT = ML_ROOT.parent / "evidence"
EVIDENCE = EVIDENCE_ROOT / "e36_acquisition.json"
ROLE_AMENDMENT = EVIDENCE_ROOT / "e36_qwen_role_amendment.json"
CAL_EVIDENCE = EVIDENCE_ROOT / "e36_cal_manifest.json"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
MIN_FREE_BYTES = 100 * 1024**3
MAX_MEMBER_BYTES = 100 * 1024**2
MAX_ARCHIVE_EXPANSION = 12 * 1024**3


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def _write_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_bytes(_json_bytes(value))
    temporary.replace(path)


def _digest(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm, usedforsecurity=False)
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024**2):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def validate_zenodo(payload: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    metadata = payload.get("metadata", {})
    if int(payload.get("id", -1)) != ZENODO_ID or metadata.get("version") != ZENODO_VERSION:
        raise ValueError("E36 Zenodo identity/version changed")
    if metadata.get("license", {}).get("id") != "cc-by-4.0":
        raise ValueError("E36 REAL licence is not CC BY 4.0")
    published = {item.get("key"): item for item in payload.get("files", [])}
    selected = {}
    for name, (expected_bytes, expected_md5) in ZENODO_FILES.items():
        item = published.get(name)
        if item is None:
            raise ValueError(f"E36 REAL archive disappeared: {name}")
        found = (int(item.get("size", -1)), str(item.get("checksum", "")))
        if found != (expected_bytes, f"md5:{expected_md5}"):
            raise ValueError(f"E36 REAL archive contract changed: {name}")
        url = str(item.get("links", {}).get("self", ""))
        if not url.startswith(f"https://zenodo.org/api/records/{ZENODO_ID}/") or not url.endswith("/content"):
            raise ValueError(f"E36 REAL URL is not pinned Zenodo content: {name}")
        role = "cal" if name in CAL_REAL else "locked_final"
        selected[name] = {"bytes": expected_bytes, "md5": expected_md5, "url": url, "role": role}
    return selected


def _selected_ai_rows(payload: Mapping[str, Any], families: Sequence[str], indices: range, role: str) -> list[dict[str, Any]]:
    siblings = payload.get("siblings", [])
    output = []
    for family in families:
        pattern = re.compile(rf"^images/{re.escape(family)}/(\d{{6}})_[^/]+\.(?:jpg|jpeg|png)$", re.IGNORECASE)
        found = []
        for item in siblings:
            path = str(item.get("rfilename", ""))
            match = pattern.fullmatch(path)
            if match is None or int(match.group(1)) not in indices:
                continue
            lfs = item.get("lfs") or {}
            sha = str(lfs.get("sha256", ""))
            size = int(item.get("size", -1))
            if len(sha) != 64 or size <= 0:
                raise ValueError(f"E36 AI blob lacks size/SHA-256: {path}")
            found.append({
                "role": role,
                "family": family,
                "prompt_id": int(match.group(1)),
                "path": path,
                "bytes": size,
                "sha256": sha,
                "url": f"https://huggingface.co/datasets/{HF_REPO}/resolve/{HF_REVISION}/{quote(path, safe='/')}?download=true",
            })
        if len(found) != len(indices):
            raise ValueError(f"E36 AI selection count changed for {family}: {len(found)}")
        output.extend(sorted(found, key=lambda row: row["prompt_id"]))
    return output


def validate_huggingface(payload: Mapping[str, Any]) -> dict[str, Any]:
    if payload.get("id") != HF_REPO or payload.get("sha") != HF_REVISION:
        raise ValueError("E36 AI repository identity/revision changed")
    if payload.get("cardData", {}).get("license") != "apache-2.0":
        raise ValueError("E36 AI licence is not Apache-2.0")
    if set(CAL_AI) & set(FINAL_AI):
        raise AssertionError("CAL and FINAL AI families overlap")
    return {
        "cal": _selected_ai_rows(payload, CAL_AI, CAL_AI_RANGE, "cal"),
        "locked_final": _selected_ai_rows(payload, FINAL_AI, FINAL_AI_RANGE, "locked_final"),
    }


def freeze() -> dict[str, Any]:
    if SELECTION.exists() or EVIDENCE.exists() or ROLE_AMENDMENT.exists():
        raise FileExistsError("E36 selection already exists; no silent refreeze")
    real_response = requests.get(ZENODO_API, timeout=(20, 120))
    real_response.raise_for_status()
    real = validate_zenodo(real_response.json())
    ai_response = requests.get(HF_API, timeout=(20, 180))
    ai_response.raise_for_status()
    ai = validate_huggingface(ai_response.json())
    detailed = {
        "schema_version": 1,
        "experiment": "E36/balanced-clean-cal-final",
        "state": "selection_frozen_no_image_bytes_claimed",
        "real": {"record_id": ZENODO_ID, "version": ZENODO_VERSION, "license": "CC BY 4.0", "archives": real},
        "ai": {"repo_id": HF_REPO, "revision": HF_REVISION, "license": "Apache-2.0", "rows": ai},
        "boundaries": [
            "Only CAL archives/rows may be downloaded before a frozen threshold receipt exists.",
            "FINAL device archives and generator families are physically and logically disjoint from CAL.",
            "Threshold selection must satisfy real false-positive and AI-recall subgroup gates together.",
            "No image byte was downloaded by this metadata freeze.",
        ],
    }
    raw = _json_bytes(detailed)
    _write_atomic(SELECTION, detailed)
    compact = {
        "schema_version": 1,
        "experiment": detailed["experiment"],
        "state": detailed["state"],
        "real_cal_archives": len(CAL_REAL),
        "real_final_archives": len(FINAL_REAL),
        "real_cal_archive_bytes": sum(real[name]["bytes"] for name in CAL_REAL),
        "real_final_archive_bytes": sum(real[name]["bytes"] for name in FINAL_REAL),
        "ai_cal_families": list(CAL_AI),
        "ai_cal_rows": len(ai["cal"]),
        "ai_cal_bytes": sum(row["bytes"] for row in ai["cal"]),
        "ai_final_families": list(FINAL_AI),
        "ai_final_rows": len(ai["locked_final"]),
        "ai_final_bytes": sum(row["bytes"] for row in ai["locked_final"]),
        "detailed_selection_bytes": len(raw),
        "detailed_selection_sha256": _sha256(raw),
        "new_image_bytes_downloaded_by_freeze": 0,
    }
    _write_atomic(EVIDENCE, compact)
    _write_atomic(ROLE_AMENDMENT, {
        "schema_version": 1,
        "state": "old_qwen_scout_superseded_unscored_before_e36_bytes",
        "old_selection": "evidence/e30_qwen_sealed_selection.json",
        "reason": "E36 uses family-disjoint CAL and FINAL cells with adequate per-family counts.",
        "invariant": "No old scout row was scored; CAL uses prompt ids 101-200 only.",
    })
    return compact


def _selection() -> dict[str, Any]:
    payload = json.loads(SELECTION.read_text())
    if payload.get("state") != "selection_frozen_no_image_bytes_claimed":
        raise ValueError("E36 selection is not frozen")
    return payload


def _curl_verified(item: Mapping[str, Any], destination: Path, algorithm: str) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    expected_bytes = int(item["bytes"])
    expected_digest = str(item[algorithm])
    if destination.exists():
        if destination.stat().st_size != expected_bytes or _digest(destination, algorithm) != expected_digest:
            raise ValueError(f"completed download contract mismatch: {destination}")
        return "already_complete"
    partial = destination.with_suffix(destination.suffix + ".partial")
    current = partial.stat().st_size if partial.exists() else 0
    if current > expected_bytes:
        raise ValueError(f"partial exceeds expected size: {partial}")
    command = [
        "/usr/bin/curl", "--fail", "--location", "--silent", "--show-error",
        "--connect-timeout", "30", "--retry", "8", "--retry-delay", "3",
        "--retry-all-errors", "--speed-limit", "1024", "--speed-time", "120",
        "--output", str(partial),
    ]
    if current:
        command.extend(["--continue-at", "-"])
    command.append(str(item["url"]))
    subprocess.run(command, check=True)
    if partial.stat().st_size != expected_bytes or _digest(partial, algorithm) != expected_digest:
        raise ValueError(f"download verification failed: {destination}")
    partial.replace(destination)
    return "downloaded"


def download_cal(workers: int = 8) -> dict[str, Any]:
    selection = _selection()
    if shutil.disk_usage(ROOT).free < MIN_FREE_BYTES + 8 * 1024**3:
        raise OSError("E36 free-space floor is not satisfied")
    real_states = {}
    real_items = {name: selection["real"]["archives"][name] for name in CAL_REAL}
    if any(item["role"] != "cal" for item in real_items.values()):
        raise ValueError("FINAL REAL archive reached CAL downloader")
    with ThreadPoolExecutor(max_workers=min(workers, len(real_items))) as executor:
        real_futures = {
            executor.submit(_curl_verified, item, ROOT / "archives" / name, "md5"): name
            for name, item in real_items.items()
        }
        for future in as_completed(real_futures):
            name = real_futures[future]
            real_states[name] = future.result()
            print(f"E36 REAL CAL {name} complete", flush=True)
    ai_rows = selection["ai"]["rows"]["cal"]
    if any(row["role"] != "cal" for row in ai_rows):
        raise ValueError("FINAL AI row reached CAL downloader")
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_curl_verified, row, ROOT / "cal" / "ai" / row["path"], "sha256"): row
            for row in ai_rows
        }
        done = 0
        for future in as_completed(futures):
            future.result()
            done += 1
            if done % 50 == 0:
                print(f"E36 AI CAL {done}/{len(ai_rows)}", flush=True)
    receipt = {
        "schema_version": 1,
        "state": "cal_download_complete_verified",
        "selection_sha256": _sha256(SELECTION.read_bytes()),
        "real_archive_states": real_states,
        "real_archive_bytes": sum(selection["real"]["archives"][name]["bytes"] for name in CAL_REAL),
        "ai_rows": len(ai_rows),
        "ai_bytes": sum(row["bytes"] for row in ai_rows),
        "final_bytes_downloaded": 0,
    }
    _write_atomic(ROOT / "cal_download_receipt.json", receipt)
    return receipt


def inspect_zip(infos: Sequence[zipfile.ZipInfo]) -> dict[str, Any]:
    names = set()
    image_rows = []
    expanded = 0
    for info in infos:
        path = PurePosixPath(info.filename)
        if path.is_absolute() or ".." in path.parts or not path.parts:
            raise ValueError(f"unsafe ZIP path: {info.filename!r}")
        if info.filename in names:
            raise ValueError(f"duplicate ZIP member: {info.filename}")
        names.add(info.filename)
        mode = info.external_attr >> 16
        if mode & 0o170000 == 0o120000:
            raise ValueError(f"ZIP symlink is forbidden: {info.filename}")
        if info.is_dir():
            continue
        if info.flag_bits & 0x1 or info.file_size < 0 or info.file_size > MAX_MEMBER_BYTES:
            raise ValueError(f"unsafe ZIP member: {info.filename}")
        expanded += info.file_size
        if path.suffix.lower() in IMAGE_SUFFIXES:
            image_rows.append({
                "name": info.filename,
                "bytes": info.file_size,
                "crc": info.CRC,
                "condition": next((part for part in path.parts if part.startswith("view_")), "unknown"),
            })
    if expanded > MAX_ARCHIVE_EXPANSION or not image_rows:
        raise ValueError("E36 ZIP expansion/image contract failed")
    return {"member_count": len(infos), "image_count": len(image_rows), "expanded_bytes": expanded, "images": image_rows}


def inventory_cal() -> dict[str, Any]:
    selection = _selection()
    archives = {}
    for name in CAL_REAL:
        path = ROOT / "archives" / name
        item = selection["real"]["archives"][name]
        if not path.is_file() or path.stat().st_size != item["bytes"] or _digest(path, "md5") != item["md5"]:
            raise FileNotFoundError(f"verified CAL archive unavailable: {name}")
        with zipfile.ZipFile(path) as bundle:
            summary = inspect_zip(bundle.infolist())
            bad = bundle.testzip()
        if bad is not None:
            raise ValueError(f"CRC failure in {name}: {bad}")
        condition_counts = {}
        for row in summary.pop("images"):
            condition_counts[row["condition"]] = condition_counts.get(row["condition"], 0) + 1
        archives[name] = {**summary, "condition_counts": dict(sorted(condition_counts.items()))}
    result = {
        "schema_version": 1,
        "state": "cal_real_archive_inventory_passed",
        "selection_sha256": _sha256(SELECTION.read_bytes()),
        "archives": archives,
    }
    _write_atomic(ROOT / "cal_real_inventory.json", result)
    return result


def _protected_hashes() -> tuple[set[str], set[str]]:
    exact, perceptual = set(), set()
    for path in sorted((DATA_ROOT / "e32" / "audits").glob("*.json")):
        payload = json.loads(path.read_text())
        if payload.get("state") not in {"realization_passed", "source_realization_passed"}:
            continue
        for row in payload.get("records", []):
            if row.get("sha256"):
                exact.add(str(row["sha256"]))
            if row.get("dhash"):
                perceptual.add(str(row["dhash"]))
    return exact, perceptual


def _audit_image(path: Path) -> dict[str, Any]:
    with Image.open(path) as picture:
        picture.load()
        width, height = picture.size
        decoded_format = picture.format
        dhash = dhash_image(picture.convert("RGB"))
        exif_present = bool(picture.getexif())
    if min(width, height) < 336:
        raise ValueError(f"E36 image is below native input floor: {path}")
    return {
        "bytes": path.stat().st_size,
        "sha256": _digest(path, "sha256"),
        "dhash": dhash,
        "width": width,
        "height": height,
        "decoded_format": decoded_format,
        "exif_present": exif_present,
    }


def extract_and_manifest_cal() -> dict[str, Any]:
    if CAL_MANIFEST.exists() or CAL_EVIDENCE.exists():
        raise FileExistsError("E36 CAL manifest already exists; no silent rerun")
    inventory = json.loads((ROOT / "cal_real_inventory.json").read_text())
    if inventory.get("state") != "cal_real_archive_inventory_passed":
        raise ValueError("E36 CAL inventory has not passed")
    real_root = ROOT / "cal" / "real"
    real_rows = []
    for archive_name in CAL_REAL:
        device = archive_name.removesuffix(".zip")
        archive = ROOT / "archives" / archive_name
        destination_root = real_root / device
        if destination_root.exists():
            raise FileExistsError(f"E36 partial/old extraction requires audit: {destination_root}")
        with zipfile.ZipFile(archive) as bundle:
            members = [
                info for info in bundle.infolist()
                if not info.is_dir()
                and PurePosixPath(info.filename).suffix.lower() in IMAGE_SUFFIXES
                and "view_000" in PurePosixPath(info.filename).parts
            ]
            members = sorted(members, key=lambda info: info.filename)[:100]
            if len(members) < 80:
                raise ValueError(f"E36 CAL device has fewer than 80 normal originals: {device}")
            temporary = real_root / f"{device}.partial"
            if temporary.exists():
                raise FileExistsError(f"partial extraction requires audit: {temporary}")
            temporary.mkdir(parents=True)
            for index, info in enumerate(members):
                suffix = PurePosixPath(info.filename).suffix.lower()
                destination = temporary / f"{index:03d}{suffix}"
                with bundle.open(info) as source, destination.open("xb") as output:
                    shutil.copyfileobj(source, output, length=8 * 1024**2)
                if destination.stat().st_size != info.file_size:
                    raise ValueError(f"E36 extraction size mismatch: {info.filename}")
                audit = _audit_image(destination)
                real_rows.append({
                    "role": "cal",
                    "label": 0,
                    "source": device,
                    "condition": "native_view_000",
                    "parent_id": f"{device}:{info.filename}",
                    "path": str((destination_root / destination.name).relative_to(ROOT)),
                    **audit,
                })
            temporary.replace(destination_root)
    selection = _selection()
    ai_rows = []
    for row in selection["ai"]["rows"]["cal"]:
        path = ROOT / "cal" / "ai" / row["path"]
        audit = _audit_image(path)
        if audit["sha256"] != row["sha256"] or audit["bytes"] != row["bytes"]:
            raise ValueError(f"E36 AI realization contract mismatch: {row['path']}")
        ai_rows.append({
            "role": "cal",
            "label": 1,
            "source": row["family"],
            "condition": "clean",
            "parent_id": f"qwen-bench:{row['prompt_id']}",
            "prompt_id": row["prompt_id"],
            "path": str(path.relative_to(ROOT)),
            **audit,
        })
    rows = real_rows + ai_rows
    exact = [row["sha256"] for row in rows]
    if len(exact) != len(set(exact)):
        raise ValueError("E36 CAL contains exact duplicate bytes")
    protected_exact, protected_dhash = _protected_hashes()
    exact_hits = [row["path"] for row in rows if row["sha256"] in protected_exact]
    if exact_hits:
        raise ValueError(f"E36 CAL exact-overlaps prior realized data: {exact_hits[:3]}")
    by_dhash = {}
    for row in rows:
        by_dhash.setdefault(row["dhash"], []).append(row)
    cross_label_dhash = [
        [item["path"] for item in group]
        for group in by_dhash.values()
        if len({item["label"] for item in group}) > 1
    ]
    if cross_label_dhash:
        raise ValueError("E36 CAL has cross-label perceptual duplicates")
    manifest = {
        "schema_version": 1,
        "experiment": "E36/clean-cal",
        "state": "cal_manifest_frozen_unscored",
        "selection_sha256": _sha256(SELECTION.read_bytes()),
        "counts": {"real": len(real_rows), "ai": len(ai_rows), "total": len(rows)},
        "real_source_counts": {source: sum(row["source"] == source for row in real_rows) for source in sorted({row["source"] for row in real_rows})},
        "ai_family_counts": {source: sum(row["source"] == source for row in ai_rows) for source in sorted({row["source"] for row in ai_rows})},
        "prior_exact_overlap_count": 0,
        "prior_dhash_match_count_diagnostic": sum(row["dhash"] in protected_dhash for row in rows),
        "rows": rows,
        "boundary": "Unscored CAL only; FINAL files remain undownloaded and no threshold exists.",
    }
    raw = _json_bytes(manifest)
    _write_atomic(CAL_MANIFEST, manifest)
    compact = {
        "schema_version": 1,
        "state": manifest["state"],
        "counts": manifest["counts"],
        "real_source_counts": manifest["real_source_counts"],
        "ai_family_counts": manifest["ai_family_counts"],
        "prior_exact_overlap_count": 0,
        "prior_dhash_match_count_diagnostic": manifest["prior_dhash_match_count_diagnostic"],
        "detailed_manifest_bytes": len(raw),
        "detailed_manifest_sha256": _sha256(raw),
        "final_bytes_downloaded": 0,
    }
    _write_atomic(CAL_EVIDENCE, compact)
    return compact


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("freeze")
    download_parser = sub.add_parser("download-cal")
    download_parser.add_argument("--workers", type=int, default=8)
    sub.add_parser("inventory-cal")
    sub.add_parser("manifest-cal")
    args = parser.parse_args(argv)
    if args.command == "freeze":
        result = freeze()
    elif args.command == "download-cal":
        result = download_cal(args.workers)
    elif args.command == "inventory-cal":
        result = inventory_cal()
    else:
        result = extract_and_manifest_cal()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
