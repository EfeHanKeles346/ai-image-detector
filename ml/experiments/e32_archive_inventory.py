"""Safely inventory E32 ZIP archives and extract only frozen FODB original parents."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping
from zipfile import ZipFile, ZipInfo

import e32_data_system as acquisition
from pixelproof.project_paths import ML_ROOT


REPO_ROOT = ML_ROOT.parent
OUTPUT_ROOT = acquisition.OUTPUT_ROOT
EVIDENCE_ROOT = REPO_ROOT / "evidence"
CHUNK_BYTES = 8 * 1024**2
MAX_MEMBER_BYTES = 512 * 1024**2
MAX_EXPANSION_RATIO = 4.0
FODB_MEMBER = re.compile(
    r"^(?P<pipeline>D\d+_[^/]+)/(?P<transport>[^/]+)/"
    r"(?P<device>D\d+)_img_(?P<name_transport>[^_]+)_(?P<scene>\d+)\.(?P<suffix>jpe?g)$",
    re.IGNORECASE,
)


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def _sha256(raw: bytes) -> str:
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


def _validate_member(info: ZipInfo) -> PurePosixPath:
    name = info.filename
    if not name or "\\" in name or "\x00" in name:
        raise ValueError(f"unsafe ZIP member name: {name!r}")
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"unsafe ZIP member path: {name!r}")
    mode = info.external_attr >> 16
    if stat.S_ISLNK(mode):
        raise ValueError(f"ZIP symlink is forbidden: {name}")
    if info.flag_bits & 0x1:
        raise ValueError(f"encrypted ZIP member is forbidden: {name}")
    if info.file_size < 0 or info.compress_size < 0 or info.file_size > MAX_MEMBER_BYTES:
        raise ValueError(f"implausible ZIP member size: {name}")
    return path


def _inventory_zip(path: Path, declared_bytes: int) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size != declared_bytes:
        raise ValueError(f"archive size mismatch: {path}")
    members: list[dict[str, Any]] = []
    names: set[str] = set()
    with ZipFile(path) as archive:
        bad = archive.testzip()
        if bad is not None:
            raise ValueError(f"ZIP CRC failure: {bad}")
        for info in archive.infolist():
            member = _validate_member(info)
            if info.filename in names:
                raise ValueError(f"duplicate ZIP member: {info.filename}")
            names.add(info.filename)
            if info.is_dir():
                continue
            members.append(
                {
                    "name": member.as_posix(),
                    "bytes": info.file_size,
                    "compressed_bytes": info.compress_size,
                    "crc32": f"{info.CRC:08x}",
                }
            )
    uncompressed = sum(item["bytes"] for item in members)
    compressed = sum(item["compressed_bytes"] for item in members)
    if uncompressed > max(1, compressed) * MAX_EXPANSION_RATIO:
        raise ValueError(f"ZIP expansion ratio exceeds {MAX_EXPANSION_RATIO}:1")
    return {
        "archive": path.name,
        "archive_bytes": declared_bytes,
        "archive_sha256": _sha256_file(path),
        "file_members": len(members),
        "uncompressed_bytes": uncompressed,
        "compressed_member_bytes": compressed,
        "suffix_counts": dict(sorted(Counter(PurePosixPath(item["name"]).suffix.lower() for item in members).items())),
        "root_counts": dict(sorted(Counter(PurePosixPath(item["name"]).parts[0] for item in members).items())),
        "second_level_counts": dict(sorted(Counter(PurePosixPath(item["name"]).parts[1] for item in members if len(PurePosixPath(item["name"]).parts) > 1).items())),
        "members": members,
    }


def _source(source_id: str) -> Mapping[str, Any]:
    return acquisition.registry()[source_id]


def inventory_fodb() -> dict[str, Any]:
    source = _source("forchheim-fodb")
    inventories = []
    seen_roots: set[str] = set()
    parents: dict[tuple[str, str], dict[str, Any]] = {}
    for expected in source["archives"]:
        path = OUTPUT_ROOT / "real" / "fodb" / "archives" / str(expected["name"])
        inventory = _inventory_zip(path, int(expected["bytes"]))
        roots = set(inventory["root_counts"])
        if roots & seen_roots:
            raise ValueError(f"FODB device root repeated across archives: {sorted(roots & seen_roots)}")
        seen_roots.update(roots)
        for member in inventory["members"]:
            match = FODB_MEMBER.fullmatch(str(member["name"]))
            if match is None:
                raise ValueError(f"unexpected FODB member contract: {member['name']}")
            fields = match.groupdict()
            if fields["transport"].lower() != fields["name_transport"].lower():
                raise ValueError(f"FODB path/filename transport mismatch: {member['name']}")
            if not fields["pipeline"].startswith(fields["device"] + "_"):
                raise ValueError(f"FODB device mismatch: {member['name']}")
            parent_key = (fields["pipeline"], fields["scene"])
            parent = parents.setdefault(
                parent_key,
                {
                    "camera_pipeline": fields["pipeline"],
                    "device": fields["device"],
                    "scene_group": f"fodb-scene:{fields['scene']}",
                    "scene_index": fields["scene"],
                    "members": {},
                },
            )
            transport = fields["transport"].lower()
            if transport in parent["members"]:
                raise ValueError(f"duplicate FODB transport for {parent_key}: {transport}")
            parent["members"][transport] = {
                "archive": path.name,
                "member": member["name"],
                "bytes": member["bytes"],
                "crc32": member["crc32"],
            }
        inventories.append({key: value for key, value in inventory.items() if key != "members"})
    expected_transports = {"orig", "facebook", "instagram", "telegram", "twitter", "whatsapp"}
    for parent_key, parent in parents.items():
        if set(parent["members"]) != expected_transports:
            raise ValueError(f"incomplete FODB parent {parent_key}: {sorted(parent['members'])}")
    if len(parents) != int(source["expected_parent_count"]):
        raise ValueError(f"expected {source['expected_parent_count']} FODB parents, found {len(parents)}")
    if len(seen_roots) != int(source["expected_devices"]):
        raise ValueError(f"expected {source['expected_devices']} FODB roots, found {len(seen_roots)}")
    detailed = {
        "schema_version": 1,
        "experiment": "E32/C1-fodb-archive-inventory",
        "state": "archive_inventory_passed_orig_parents_unextracted",
        "source_id": source["id"],
        "license": source["license"],
        "archives": inventories,
        "archive_count": len(inventories),
        "device_count": len(seen_roots),
        "parent_count": len(parents),
        "scene_group_count": len({parent["scene_group"] for parent in parents.values()}),
        "transports": sorted(expected_transports),
        "parents": sorted(parents.values(), key=lambda row: (row["camera_pipeline"], row["scene_index"])),
        "boundary": "Only orig members may be extracted as role-free candidates; social members remain parent-linked transport evidence.",
    }
    detailed_path = OUTPUT_ROOT / "fodb_archive_inventory.json"
    raw = _json_bytes(detailed)
    _write_atomic(detailed_path, raw)
    compact = {key: value for key, value in detailed.items() if key != "parents"}
    compact.update(
        {
            "detailed_report_sha256": _sha256(raw),
            "detailed_report_bytes": len(raw),
            "detailed_report_external_path": detailed_path.relative_to(OUTPUT_ROOT).as_posix(),
        }
    )
    _write_atomic(EVIDENCE_ROOT / "e32_fodb_archive_inventory.json", _json_bytes(compact))
    return compact


def extract_fodb_orig() -> dict[str, Any]:
    detailed_path = OUTPUT_ROOT / "fodb_archive_inventory.json"
    compact_path = EVIDENCE_ROOT / "e32_fodb_archive_inventory.json"
    raw = detailed_path.read_bytes()
    compact = json.loads(compact_path.read_text())
    if _sha256(raw) != compact["detailed_report_sha256"]:
        raise ValueError("FODB inventory binding changed")
    inventory = json.loads(raw)
    if inventory.get("state") != "archive_inventory_passed_orig_parents_unextracted":
        raise ValueError("FODB inventory has unexpected state")
    by_archive: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for parent in inventory["parents"]:
        by_archive[parent["members"]["orig"]["archive"]].append(parent)
    records = []
    for archive_name, parents in sorted(by_archive.items()):
        archive_path = OUTPUT_ROOT / "real" / "fodb" / "archives" / archive_name
        with ZipFile(archive_path) as archive:
            for parent in parents:
                member = parent["members"]["orig"]
                destination = acquisition._safe_destination(
                    f"real/fodb/orig/{parent['camera_pipeline']}/{PurePosixPath(member['member']).name}"
                )
                destination.parent.mkdir(parents=True, exist_ok=True)
                expected_bytes = int(member["bytes"])
                if destination.exists():
                    if destination.stat().st_size != expected_bytes:
                        raise ValueError(f"existing FODB parent size mismatch: {destination}")
                    state = "already_complete"
                else:
                    partial = destination.with_suffix(destination.suffix + ".partial")
                    if partial.exists():
                        raise ValueError(f"stale FODB extraction partial requires review: {partial}")
                    with archive.open(str(member["member"])) as source, partial.open("xb") as target:
                        while chunk := source.read(CHUNK_BYTES):
                            target.write(chunk)
                    if partial.stat().st_size != expected_bytes:
                        raise ValueError(f"extracted FODB parent size mismatch: {member['member']}")
                    os.replace(partial, destination)
                    state = "extracted"
                records.append(
                    {
                        "source_key": destination.relative_to(OUTPUT_ROOT).as_posix(),
                        "camera_pipeline": parent["camera_pipeline"],
                        "device": parent["device"],
                        "scene_group": parent["scene_group"],
                        "bytes": expected_bytes,
                        "sha256": _sha256_file(destination),
                        "state": state,
                    }
                )
    receipt = {
        "schema_version": 1,
        "experiment": "E32/C1-fodb-orig-extraction",
        "state": "orig_extraction_complete_role_free",
        "inventory_sha256": compact["detailed_report_sha256"],
        "parent_count": len(records),
        "bytes": sum(row["bytes"] for row in records),
        "records": records,
        "boundary": "Extracted originals are role-free candidates; social derivatives were not extracted.",
    }
    receipt_path = OUTPUT_ROOT / "fodb_orig_extraction.json"
    receipt_raw = _json_bytes(receipt)
    _write_atomic(receipt_path, receipt_raw)
    result = {key: value for key, value in receipt.items() if key != "records"}
    result.update(
        {
            "detailed_report_sha256": _sha256(receipt_raw),
            "detailed_report_bytes": len(receipt_raw),
            "detailed_report_external_path": receipt_path.relative_to(OUTPUT_ROOT).as_posix(),
        }
    )
    _write_atomic(EVIDENCE_ROOT / "e32_fodb_orig_extraction.json", _json_bytes(result))
    return result


def inventory_csafe() -> dict[str, Any]:
    source = _source("csafe-mcsidb-s21")
    expected = source["archive"]
    path = OUTPUT_ROOT / "real" / "csafe" / "archives" / str(expected["name"])
    md5 = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as handle:
        while chunk := handle.read(CHUNK_BYTES):
            md5.update(chunk)
    if md5.hexdigest() != expected["md5"]:
        raise ValueError("CSAFE archive MD5 changed before inventory")
    inventory = _inventory_zip(path, int(expected["bytes"]))
    detailed = {
        "schema_version": 1,
        "experiment": "E32/C1-csafe-archive-inventory",
        "state": "archive_inventory_frozen_internal_rows_unselected",
        "source_id": source["id"],
        "license": source["license"],
        "published_md5": expected["md5"],
        **inventory,
        "boundary": "Inventory does not select, extract or assign a role to any internal row.",
    }
    detailed_path = OUTPUT_ROOT / "csafe_archive_inventory.json"
    raw = _json_bytes(detailed)
    _write_atomic(detailed_path, raw)
    compact = {key: value for key, value in detailed.items() if key != "members"}
    compact.update(
        {
            "detailed_report_sha256": _sha256(raw),
            "detailed_report_bytes": len(raw),
            "detailed_report_external_path": detailed_path.relative_to(OUTPUT_ROOT).as_posix(),
        }
    )
    _write_atomic(EVIDENCE_ROOT / "e32_csafe_archive_inventory.json", _json_bytes(compact))
    return compact


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("inventory-fodb", "extract-fodb-orig", "inventory-csafe"))
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    result = {
        "inventory-fodb": inventory_fodb,
        "extract-fodb-orig": extract_fodb_orig,
        "inventory-csafe": inventory_csafe,
    }[args.command]()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
