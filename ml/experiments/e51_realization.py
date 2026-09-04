"""Freeze and realize E51 role manifests before any detector access."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence
import zipfile

from experiments.e51_scimd17_transfer import ARCHIVE_MD5, ARCHIVE_NAME
from experiments.e51_scimd17_transfer import PAYLOAD as SCIMD_ARCHIVE
from experiments.e51_transfer import ROUTE_CONTRACT_SHA256
from pixelproof.project_paths import DATA_ROOT, ML_ROOT


NAMESPACE = "E51_REALIZATION_V1"
SCIMD_ARCHIVE_SHA256 = "ef1fe3e77e21d44c4cb29f4b44ad69bc3b83bfd3038531605c5ee315a2cb0201"
SCIMD_DEVICES = 17
SCIMD_TARGET_PER_DEVICE = 100
SCIMD_RESERVE_PER_DEVICE = 120
AI_CAL_PER_SOURCE = 20
AI_CAL_MIN_SOURCE_ROWS = 40
EXPECTED_AI_CAL_SOURCES = 18

ROOT = DATA_ROOT / "e51"
ROUTE = ROOT / "route" / "contract_untransferred.json"
E42_MANIFEST = DATA_ROOT / "e42" / "parent_manifest.json"
CONTRACT = ROOT / "realization_contract_unscored.json"
EVIDENCE = ML_ROOT.parent / "evidence" / "e51_realization_contract.json"
RECEIPTS = (
    ROOT / "receipts" / "ieee_spcup_download_unscored.json",
    ROOT / "receipts" / "datapoint_shards_download_unscored.json",
    ROOT / "receipts" / "scmi30_cal_download_unscored.json",
    ROOT / "receipts" / "scimd17_archive_download_unscored.json",
)


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def _write_atomic(path: Path, value: Any) -> bytes:
    raw = _json_bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_bytes(raw)
    temporary.replace(path)
    return raw


def _digest(path: Path, algorithm: str = "sha256") -> str:
    digest = hashlib.new(algorithm, usedforsecurity=False)
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def _rank(cell: str, identity: str) -> str:
    return hashlib.sha256(f"{NAMESPACE}|{cell}|{identity}".encode()).hexdigest()


def select_ai_cal(
    rows: Sequence[Mapping[str, Any]],
    *,
    per_source: int = AI_CAL_PER_SOURCE,
    minimum_source_rows: int = AI_CAL_MIN_SOURCE_ROWS,
) -> list[dict[str, Any]]:
    by_source: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if str(row.get("role", "")).lower() == "train" and int(row.get("label", -1)) == 1:
            by_source[str(row["source"])].append(row)
    selected = []
    for source, candidates in sorted(by_source.items()):
        if len(candidates) < minimum_source_rows:
            continue
        ordered = sorted(
            candidates,
            key=lambda row: (_rank("ai-cal", str(row["parent_id"])), str(row["parent_id"])),
        )
        selected.extend({**row, "e51_role": "CAL", "e51_origin_role": "TRAIN",
                         "e51_rank": _rank("ai-cal", str(row["parent_id"]))}
                        for row in ordered[:per_source])
    return sorted(selected, key=lambda row: (str(row["source"]), str(row["e51_rank"])))


def select_scimd_reserve(
    infos: Sequence[zipfile.ZipInfo],
    *,
    reserve_per_device: int = SCIMD_RESERVE_PER_DEVICE,
) -> list[dict[str, Any]]:
    by_device: dict[str, list[zipfile.ZipInfo]] = defaultdict(list)
    for info in infos:
        member = PurePosixPath(info.filename)
        if info.is_dir():
            continue
        if (
            member.is_absolute()
            or ".." in member.parts
            or len(member.parts) != 3
            or member.parts[0] != "SCIMD-17"
            or member.suffix.lower() not in {".jpg", ".jpeg"}
        ):
            raise ValueError(f"unexpected SCIMD-17 image member: {info.filename}")
        by_device[member.parts[1]].append(info)
    selected = []
    for device, candidates in sorted(by_device.items()):
        ordered = sorted(
            candidates,
            key=lambda info: (_rank(f"scimd:{device}", info.filename), info.filename),
        )
        if len(ordered) < reserve_per_device:
            raise ValueError(f"insufficient SCIMD-17 reserve for {device}")
        for info in ordered[:reserve_per_device]:
            selected.append({
                "identity": f"zenodo:17317613:{info.filename}",
                "member": info.filename,
                "device": device,
                "expected_bytes": info.file_size,
                "expected_crc32": f"{info.CRC:08x}",
                "rank": _rank(f"scimd:{device}", info.filename),
                "label": 0,
                "role": "TRAIN",
                "source": "SCIMD-17",
                "restriction": "224x224 resized REAL hard negative",
            })
    return sorted(selected, key=lambda row: (str(row["device"]), str(row["rank"])))


def bind() -> dict[str, Any]:
    if CONTRACT.exists() or EVIDENCE.exists():
        raise FileExistsError("E51 realization contract already exists")
    route_raw = ROUTE.read_bytes()
    if hashlib.sha256(route_raw).hexdigest() != ROUTE_CONTRACT_SHA256:
        raise ValueError("E51 route contract changed before realization bind")
    route = json.loads(route_raw)
    if route.get("model_scores_created") != 0:
        raise ValueError("E51 route is no longer model-blind")
    receipt_hashes = {}
    for path in RECEIPTS:
        raw = path.read_bytes()
        receipt = json.loads(raw)
        if receipt.get("model_scores_created") != 0:
            raise ValueError(f"E51 acquisition receipt contains scores: {path.name}")
        receipt_hashes[path.name] = hashlib.sha256(raw).hexdigest()
    if _digest(SCIMD_ARCHIVE, "sha256") != SCIMD_ARCHIVE_SHA256 or _digest(
        SCIMD_ARCHIVE, "md5"
    ) != ARCHIVE_MD5:
        raise ValueError("SCIMD-17 archive changed before realization bind")
    with zipfile.ZipFile(SCIMD_ARCHIVE) as archive:
        scimd_rows = select_scimd_reserve(archive.infolist())
    if len(scimd_rows) != SCIMD_DEVICES * SCIMD_RESERVE_PER_DEVICE:
        raise ValueError("SCIMD-17 device reserve count changed")
    e42_raw = E42_MANIFEST.read_bytes()
    e42 = json.loads(e42_raw)
    ai_cal = select_ai_cal(e42.get("rows") or [])
    if (
        len(ai_cal) != EXPECTED_AI_CAL_SOURCES * AI_CAL_PER_SOURCE
        or set(Counter(str(row["source"]) for row in ai_cal).values()) != {AI_CAL_PER_SOURCE}
    ):
        raise ValueError("E51 internal AI CAL source balance changed")
    payload = {
        "schema_version": 1,
        "state": "e51_realization_frozen_before_new_image_decode_unscored",
        "namespace": NAMESPACE,
        "route_contract_sha256": ROUTE_CONTRACT_SHA256,
        "acquisition_receipt_sha256": dict(sorted(receipt_hashes.items())),
        "historical_train_manifest": {
            "path": str(E42_MANIFEST),
            "sha256": hashlib.sha256(e42_raw).hexdigest(),
            "policy": "reuse only original TRAIN rows; held-out AI-CAL identities are excluded from fit",
        },
        "internal_ai_cal": {
            "selection": "namespace SHA-256 rank within every historical AI TRAIN source with >=40 rows",
            "per_source": AI_CAL_PER_SOURCE,
            "sources": EXPECTED_AI_CAL_SOURCES,
            "rows": ai_cal,
        },
        "scimd17_train": {
            "archive_name": ARCHIVE_NAME,
            "archive_sha256": SCIMD_ARCHIVE_SHA256,
            "target_per_device": SCIMD_TARGET_PER_DEVICE,
            "reserve_per_device": SCIMD_RESERVE_PER_DEVICE,
            "devices": SCIMD_DEVICES,
            "rows": scimd_rows,
        },
        "paired_transport": {
            "conditions": ["original", "q75"],
            "q75": "EXIF-transposed RGB JPEG; quality=75, subsampling=2, optimize=false",
            "parent_grouping": "original and q75 share one parent_id and cannot cross roles",
        },
        "role_boundary": {
            "TRAIN": "E42 historical TRAIN minus internal AI-CAL, plus 100 clean SCIMD rows/device",
            "CAL": "1,200 SCMI30 REAL plus 360 held-out historical TRAIN AI; original/Q75 paired",
            "DEVELOPMENT": "all 2,640 IEEE REAL plus 800 Datapoint AI; original/Q75 paired",
            "FINAL": "none; E52 remains unopened",
        },
        "forbidden": [
            "Datapoint in TRAIN or CAL",
            "SCMI30 outside CAL",
            "IEEE outside DEVELOPMENT",
            "SCIMD-17 outside TRAIN",
            "held-out internal AI-CAL identities in fit",
            "score-dependent replacement or threshold before manifests freeze",
        ],
        "new_image_bodies_decoded": 0,
        "model_scores_created": 0,
    }
    raw = _write_atomic(CONTRACT, payload)
    evidence = {
        "schema_version": 1,
        "state": payload["state"],
        "namespace": NAMESPACE,
        "contract_bytes": len(raw),
        "contract_sha256": hashlib.sha256(raw).hexdigest(),
        "internal_ai_cal_rows": len(ai_cal),
        "internal_ai_cal_sources": EXPECTED_AI_CAL_SOURCES,
        "scimd17_reserve_rows": len(scimd_rows),
        "scimd17_target_rows": SCIMD_DEVICES * SCIMD_TARGET_PER_DEVICE,
        "new_image_bodies_decoded": 0,
        "model_scores_created": 0,
    }
    _write_atomic(EVIDENCE, evidence)
    return evidence


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("bind",))
    parser.parse_args(argv)
    print(json.dumps(bind(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
