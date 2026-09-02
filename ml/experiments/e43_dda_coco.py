"""Acquire and audit the official COCO real companion for E43 DDA-COCO."""

from __future__ import annotations

import argparse
from collections import defaultdict
from io import BytesIO
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
from typing import Any, Iterable, Mapping, Sequence
import zipfile

import requests
from PIL import Image

from pixelproof.data_contract import dhash_image
from pixelproof.project_paths import DATA_ROOT, ML_ROOT


ROOT = DATA_ROOT / "e43_dda_coco"
COCO_URL = "https://s3.amazonaws.com/images.cocodataset.org/zips/val2017.zip"
COCO_BYTES = 815_585_330
COCO_ETAG = '"d366be60d3dc737327160d62453e3973-98"'
COCO_ARCHIVE = ROOT / "archives" / "val2017.zip"
COCO_RECEIPT = ROOT / "real_download_receipt.json"
COCO_INVENTORY = ROOT / "real_inventory.json"
STRUCTURE_REPORT = ROOT / "paired_structure.json"
STRUCTURE_EVIDENCE = ML_ROOT.parent / "evidence" / "e43_dda_coco_structure.json"
MANIFEST = ROOT / "unscored_manifest.json"
MANIFEST_EVIDENCE = ML_ROOT.parent / "evidence" / "e43_dda_coco_manifest.json"
DDA_ARCHIVE = DATA_ROOT / "e34_dda_coco" / "archives" / "DDA-COCO.zip"
DDA_SHA256 = "8cd600779aaecef21605b07bff9ab3963a7fb9b9614a3d9a0588cd4a5e099c24"
MIN_FREE_BYTES = 100 * 1024**3
REAL_RE = re.compile(r"^val2017/(\d{12})\.jpg$")
SYNTHETIC_RE = re.compile(r"^DDA-COCO/([^/]+)/val2017/(\d{12})\.jpg$")
VARIANTS = (
    "FLUX.1",
    "sd-vae-ft-ema",
    "sd-vae-ft-mse",
    "sdxl-vae",
    "stable-diffusion-2-1",
    "stable-diffusion-3.5-large",
)
EXPECTED_VARIANT_COUNTS = {
    "FLUX.1": 4_971,
    "sd-vae-ft-ema": 5_000,
    "sd-vae-ft-mse": 5_000,
    "sdxl-vae": 5_000,
    "stable-diffusion-2-1": 5_000,
    "stable-diffusion-3.5-large": 4_998,
}
CANDIDATE = DATA_ROOT / "e43" / "e43_small_predev.joblib"
CANDIDATE_SHA256 = "a3aec445926bcc8707b3775f01d2cdd9491ba8495ad8a8ec306840556ca47390"
THRESHOLD = 0.8712875247001649


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024**2):
            digest.update(chunk)
    return digest.hexdigest()


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_bytes(_json_bytes(value))
    temporary.replace(path)


def validate_coco_headers(headers: Mapping[str, str]) -> dict[str, Any]:
    if int(headers.get("content-length", -1)) != COCO_BYTES:
        raise ValueError("COCO val2017 size changed")
    if headers.get("etag") != COCO_ETAG:
        raise ValueError("COCO val2017 ETag changed")
    return {"bytes": COCO_BYTES, "etag": COCO_ETAG, "last_modified": headers.get("last-modified")}


def download_real() -> dict[str, Any]:
    if COCO_RECEIPT.exists():
        raise FileExistsError("COCO val2017 receipt already exists; no silent replacement")
    response = requests.head(COCO_URL, timeout=(20, 120))
    response.raise_for_status()
    source = validate_coco_headers({key.lower(): value for key, value in response.headers.items()})
    COCO_ARCHIVE.parent.mkdir(parents=True, exist_ok=True)
    partial = COCO_ARCHIVE.with_suffix(".zip.partial")
    current = partial.stat().st_size if partial.exists() else 0
    if current > COCO_BYTES:
        raise ValueError("COCO val2017 partial exceeds expected size")
    remaining = COCO_BYTES - current
    if shutil.disk_usage(ROOT).free < remaining + MIN_FREE_BYTES:
        raise OSError("insufficient free space for COCO val2017")
    command = [
        "/usr/bin/curl", "--fail", "--location", "--silent", "--show-error",
        "--connect-timeout", "30", "--retry", "8", "--retry-delay", "5",
        "--retry-all-errors", "--speed-limit", "1024", "--speed-time", "120",
        "--output", str(partial),
    ]
    if current:
        command.extend(["--continue-at", "-"])
    command.append(COCO_URL)
    subprocess.run(command, check=True)
    if partial.stat().st_size != COCO_BYTES:
        raise ValueError("COCO val2017 download size mismatch")
    sha256 = _digest(partial)
    partial.replace(COCO_ARCHIVE)
    receipt = {
        "schema_version": 1,
        "state": "coco_val2017_download_complete_sha256_bound",
        "url": COCO_URL,
        **source,
        "sha256": sha256,
        "network_bytes": remaining,
        "model_scores_created": 0,
    }
    _write(COCO_RECEIPT, receipt)
    return receipt


def real_ids_from_infos(infos: Sequence[zipfile.ZipInfo]) -> set[str]:
    ids: set[str] = set()
    for info in infos:
        if info.is_dir():
            continue
        path = PurePosixPath(info.filename)
        if path.is_absolute() or ".." in path.parts or info.flag_bits & 0x1:
            raise ValueError(f"unsafe COCO ZIP member: {info.filename}")
        match = REAL_RE.fullmatch(info.filename)
        if match is None or info.file_size <= 0 or info.file_size > 50 * 1024**2:
            raise ValueError(f"unexpected COCO member: {info.filename}")
        if match.group(1) in ids:
            raise ValueError("duplicate COCO image ID")
        ids.add(match.group(1))
    if len(ids) != 5_000:
        raise ValueError(f"COCO val2017 count changed: {len(ids)}")
    return ids


def inventory_real() -> dict[str, Any]:
    if COCO_INVENTORY.exists():
        raise FileExistsError("COCO val2017 inventory already exists")
    receipt = json.loads(COCO_RECEIPT.read_text())
    if receipt.get("state") != "coco_val2017_download_complete_sha256_bound":
        raise ValueError("COCO val2017 receipt state changed")
    if COCO_ARCHIVE.stat().st_size != COCO_BYTES or _digest(COCO_ARCHIVE) != receipt["sha256"]:
        raise ValueError("COCO val2017 archive changed after acquisition")
    with zipfile.ZipFile(COCO_ARCHIVE) as bundle:
        infos = bundle.infolist()
        ids = real_ids_from_infos(infos)
        bad = bundle.testzip()
        expanded = sum(info.file_size for info in infos if not info.is_dir())
    if bad is not None:
        raise ValueError(f"COCO val2017 CRC failure: {bad}")
    report = {
        "schema_version": 1,
        "state": "coco_val2017_inventory_passed",
        "archive_sha256": receipt["sha256"],
        "image_count": len(ids),
        "expanded_bytes": expanded,
        "id_list_sha256": hashlib.sha256(("\n".join(sorted(ids)) + "\n").encode()).hexdigest(),
        "zip_crc_passed": True,
        "model_scores_created": 0,
    }
    _write(COCO_INVENTORY, report)
    return report


def synthetic_ids_from_names(names: Iterable[str]) -> dict[str, set[str]]:
    grouped = {variant: set() for variant in VARIANTS}
    for name in names:
        if name.endswith("/"):
            continue
        match = SYNTHETIC_RE.fullmatch(name)
        if match is None or match.group(1) not in grouped:
            raise ValueError(f"unexpected DDA-COCO member: {name}")
        variant, parent = match.groups()
        if parent in grouped[variant]:
            raise ValueError("duplicate DDA-COCO parent within variant")
        grouped[variant].add(parent)
    counts = {variant: len(ids) for variant, ids in grouped.items()}
    if counts != EXPECTED_VARIANT_COUNTS:
        raise ValueError(f"DDA-COCO variant counts changed: {counts}")
    return grouped


def _protected_hashes() -> tuple[set[str], set[str], list[str]]:
    paths = list(sorted((DATA_ROOT / "e32" / "audits").glob("*.json")))
    paths.extend([
        DATA_ROOT / "e33_rrdataset" / "r1c_cal_manifest.json",
        DATA_ROOT / "e36" / "cal_manifest.json",
        DATA_ROOT / "e36" / "final_manifest.json",
        DATA_ROOT / "e39" / "final_manifest.json",
        DATA_ROOT / "e42" / "parent_manifest.json",
        DATA_ROOT / "e42_external" / "bfree_viral" / "unscored_manifest.json",
        DATA_ROOT / "e33_rrdataset" / "e42_rr_unscored_manifest.json",
    ])
    exact: set[str] = set()
    perceptual: set[str] = set()
    consumed: list[str] = []
    for path in paths:
        if not path.is_file() or path.name.startswith("._"):
            continue
        payload = json.loads(path.read_text())
        rows = payload.get("rows", payload.get("records", []))
        for row in rows:
            if row.get("sha256"):
                exact.add(str(row["sha256"]))
            if row.get("dhash"):
                perceptual.add(str(row["dhash"]))
        consumed.append(str(path))
    if not consumed or not exact or not perceptual:
        raise ValueError("no usable protected prior manifests")
    return exact, perceptual, consumed


def audit_parent_rows(
    rows: Sequence[Mapping[str, Any]], prior_exact: set[str], prior_dhash: set[str]
) -> dict[str, Any]:
    by_parent: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    by_exact: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    by_dhash: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_parent[str(row["parent_id"])].append(row)
        by_exact[str(row["sha256"])].append(row)
        by_dhash[str(row["dhash"])].append(row)

    exclusions: dict[str, set[str]] = defaultdict(set)
    protected_exact_rows = []
    protected_dhash_rows = []
    for row in rows:
        parent = str(row["parent_id"])
        if str(row["sha256"]) in prior_exact:
            protected_exact_rows.append(str(row["record_id"]))
            exclusions[parent].add("protected_exact_overlap")
        if str(row["dhash"]) in prior_dhash:
            protected_dhash_rows.append(str(row["record_id"]))
            exclusions[parent].add("protected_dhash_overlap")

    exact_groups = []
    cross_label_exact = []
    for digest, versions in sorted(by_exact.items()):
        parents = sorted({str(row["parent_id"]) for row in versions})
        labels = sorted({int(row["label"]) for row in versions})
        if len(versions) < 2:
            continue
        group = {"sha256": digest, "parents": parents, "labels": labels}
        exact_groups.append(group)
        if len(labels) > 1:
            cross_label_exact.append(group)
            for parent in parents:
                exclusions[parent].add("cross_label_exact_duplicate")
        elif len(parents) > 1:
            for parent in parents[1:]:
                exclusions[parent].add(f"same_label_exact_duplicate_of:{parents[0]}")

    dhash_groups = []
    for digest, versions in sorted(by_dhash.items()):
        parents = sorted({str(row["parent_id"]) for row in versions})
        if len(parents) > 1:
            dhash_groups.append({
                "dhash": digest,
                "parents": parents,
                "labels": sorted({int(row["label"]) for row in versions}),
            })

    malformed = []
    expected_conditions = {"REAL", *VARIANTS}
    for parent, versions in sorted(by_parent.items()):
        conditions = [str(row["condition"]) for row in versions]
        labels = {int(row["label"]) for row in versions}
        if (
            len(versions) != 7
            or len(conditions) != len(set(conditions))
            or set(conditions) != expected_conditions
            or labels != {0, 1}
        ):
            malformed.append(parent)
    return {
        "passed": not malformed,
        "malformed_parents": malformed,
        "protected_exact_overlap_records": sorted(protected_exact_rows),
        "protected_dhash_overlap_records": sorted(protected_dhash_rows),
        "cross_parent_exact_groups": exact_groups,
        "cross_label_exact_groups": cross_label_exact,
        "cross_parent_dhash_diagnostic": dhash_groups,
        "excluded_parent_ids": sorted(exclusions),
        "exclusion_reasons": {
            parent: sorted(reasons) for parent, reasons in sorted(exclusions.items())
        },
    }


def _decoded_row(payload: bytes, parent: str, condition: str, member: str) -> dict[str, Any]:
    with Image.open(BytesIO(payload)) as opened:
        opened.verify()
    with Image.open(BytesIO(payload)) as opened:
        rgb = opened.convert("RGB")
        rgb.load()
        width, height = rgb.size
        perceptual = dhash_image(rgb)
    label = 0 if condition == "REAL" else 1
    return {
        "record_id": f"dda-coco:{condition}:{parent}",
        "parent_id": f"dda-coco:{parent}",
        "member": member,
        "archive": "COCO-val2017" if condition == "REAL" else "DDA-COCO",
        "condition": condition,
        "label": label,
        "bytes": len(payload),
        "width": width,
        "height": height,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "dhash": perceptual,
    }


def freeze_manifest() -> dict[str, Any]:
    if MANIFEST.exists() or MANIFEST_EVIDENCE.exists():
        raise FileExistsError("DDA-COCO unscored manifest already exists")
    structure = json.loads(STRUCTURE_REPORT.read_text())
    if structure.get("state") != "e43_dda_coco_structure_passed_before_decode":
        raise ValueError("DDA-COCO structure binding changed")
    if _digest(CANDIDATE) != CANDIDATE_SHA256:
        raise ValueError("E43-S candidate changed before manifest")
    if _digest(COCO_ARCHIVE) != structure["real_archive_sha256"]:
        raise ValueError("COCO val2017 archive changed before decode")
    if _digest(DDA_ARCHIVE) != structure["dda_archive_sha256"]:
        raise ValueError("DDA-COCO archive changed before decode")

    with zipfile.ZipFile(COCO_ARCHIVE) as real_bundle, zipfile.ZipFile(DDA_ARCHIVE) as synthetic_bundle:
        real_ids = real_ids_from_infos(real_bundle.infolist())
        grouped = synthetic_ids_from_names(info.filename for info in synthetic_bundle.infolist())
        complete = sorted(set.intersection(real_ids, *grouped.values()))
        if len(complete) != int(structure["complete_parent_count"]):
            raise ValueError("DDA-COCO complete parent count changed")
        rows = []
        for index, parent in enumerate(complete, 1):
            real_member = f"val2017/{parent}.jpg"
            rows.append(_decoded_row(real_bundle.read(real_member), parent, "REAL", real_member))
            if index % 1_000 == 0:
                print(f"E43 DDA decode REAL {index}/{len(complete)}", flush=True)
        for variant in VARIANTS:
            for index, parent in enumerate(complete, 1):
                member = f"DDA-COCO/{variant}/val2017/{parent}.jpg"
                rows.append(_decoded_row(synthetic_bundle.read(member), parent, variant, member))
                if index % 1_000 == 0:
                    print(f"E43 DDA decode {variant} {index}/{len(complete)}", flush=True)

    rows.sort(key=lambda row: row["record_id"])
    prior_exact, prior_dhash, prior_manifests = _protected_hashes()
    audit = audit_parent_rows(rows, prior_exact, prior_dhash)
    if not audit["passed"]:
        raise ValueError(f"DDA-COCO structural audit failed: {audit['malformed_parents'][:10]}")
    excluded = set(audit["excluded_parent_ids"])
    selected = [row for row in rows if str(row["parent_id"]) not in excluded]
    if any(str(row["sha256"]) in prior_exact or str(row["dhash"]) in prior_dhash for row in selected):
        raise ValueError("DDA-COCO protected overlap remains after decontamination")
    parent_count = len({str(row["parent_id"]) for row in selected})
    if len(selected) != parent_count * 7 or parent_count < 4_000:
        raise ValueError("DDA-COCO decontamination leaves insufficient complete parents")
    condition_counts = {
        condition: sum(str(row["condition"]) == condition for row in selected)
        for condition in ("REAL", *VARIANTS)
    }
    payload = {
        "schema_version": 1,
        "experiment": "E43/DDA-COCO-open-independent-test",
        "state": "e43_dda_coco_manifest_frozen_unscored",
        "candidate_sha256": CANDIDATE_SHA256,
        "threshold": THRESHOLD,
        "structure_report_sha256": _digest(STRUCTURE_REPORT),
        "counts": {
            "rows": len(selected),
            "parents": parent_count,
            "by_condition": condition_counts,
            "excluded_parents": len(excluded),
            "excluded_rows": len(rows) - len(selected),
            "image_bytes": sum(int(row["bytes"]) for row in selected),
        },
        "protected_manifests": prior_manifests,
        "protected_exact_hashes": len(prior_exact),
        "protected_dhashes": len(prior_dhash),
        "audit": audit,
        "rows": selected,
        "boundary": "All complete seven-view parents decoded and decontaminated before model access; model scores created: 0.",
    }
    raw = _json_bytes(payload)
    _write(MANIFEST, payload)
    compact = {
        "schema_version": 1,
        "state": payload["state"],
        "candidate_sha256": CANDIDATE_SHA256,
        "threshold": THRESHOLD,
        "counts": payload["counts"],
        "audit": {
            "passed": audit["passed"],
            "protected_exact_overlap_records": len(audit["protected_exact_overlap_records"]),
            "protected_dhash_overlap_records": len(audit["protected_dhash_overlap_records"]),
            "cross_parent_exact_groups": len(audit["cross_parent_exact_groups"]),
            "cross_label_exact_groups": len(audit["cross_label_exact_groups"]),
            "cross_parent_dhash_diagnostic": len(audit["cross_parent_dhash_diagnostic"]),
        },
        "protected_manifest_count": len(prior_manifests),
        "detailed_manifest_bytes": len(raw),
        "detailed_manifest_sha256": hashlib.sha256(raw).hexdigest(),
        "model_scores_created": 0,
    }
    _write(MANIFEST_EVIDENCE, compact)
    return compact


def audit_structure() -> dict[str, Any]:
    if STRUCTURE_REPORT.exists() or STRUCTURE_EVIDENCE.exists():
        raise FileExistsError("DDA-COCO paired structure already exists")
    inventory = json.loads(COCO_INVENTORY.read_text())
    if inventory.get("state") != "coco_val2017_inventory_passed":
        raise ValueError("COCO val2017 inventory state changed")
    if DDA_ARCHIVE.stat().st_size != 4_301_452_066 or _digest(DDA_ARCHIVE) != DDA_SHA256:
        raise ValueError("DDA-COCO archive changed")
    with zipfile.ZipFile(COCO_ARCHIVE) as real_bundle:
        real_ids = real_ids_from_infos(real_bundle.infolist())
    with zipfile.ZipFile(DDA_ARCHIVE) as synthetic_bundle:
        grouped = synthetic_ids_from_names(info.filename for info in synthetic_bundle.infolist())
    if any(not ids <= real_ids for ids in grouped.values()):
        raise ValueError("DDA-COCO contains a parent absent from official COCO val2017")
    complete = set.intersection(real_ids, *grouped.values())
    report = {
        "schema_version": 1,
        "state": "e43_dda_coco_structure_passed_before_decode",
        "real_archive_sha256": inventory["archive_sha256"],
        "dda_archive_sha256": DDA_SHA256,
        "real_count": len(real_ids),
        "variant_counts": {variant: len(grouped[variant]) for variant in VARIANTS},
        "complete_parent_count": len(complete),
        "complete_row_count": len(complete) * (1 + len(VARIANTS)),
        "complete_parent_ids_sha256": hashlib.sha256(
            ("\n".join(sorted(complete)) + "\n").encode()
        ).hexdigest(),
        "model_scores_created": 0,
        "next": "Decode/hash complete seven-view parent groups and audit protected overlap before scoring.",
    }
    _write(STRUCTURE_REPORT, report)
    _write(STRUCTURE_EVIDENCE, report)
    return report


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("download-real", "inventory-real", "audit-structure", "freeze-manifest")
    )
    args = parser.parse_args(argv)
    actions = {
        "download-real": download_real,
        "inventory-real": inventory_real,
        "audit-structure": audit_structure,
        "freeze-manifest": freeze_manifest,
    }
    print(json.dumps(actions[args.command](), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
