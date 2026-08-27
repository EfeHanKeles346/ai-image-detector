"""Safely realize and freeze the preregistered E42 train/development parents."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import shutil
import tarfile
from typing import Any, Iterable, Mapping, Sequence

from PIL import Image, ImageOps

from experiments.e33_rrdataset import _safe_member_name
from pixelproof.e32_candidate import image_paths
from pixelproof.project_paths import DATA_ROOT, ML_ROOT


REPO_ROOT = ML_ROOT.parent
E32_ROOT = DATA_ROOT / "e32"
E33_ROOT = DATA_ROOT / "e33_rrdataset"
E36_ROOT = DATA_ROOT / "e36"
E39_ROOT = DATA_ROOT / "e39"
E42_ROOT = DATA_ROOT / "e42"
RR_ARCHIVE = E33_ROOT / "archives" / "RRDataset_original_train_val.tar.gz"
RR_INVENTORY = E33_ROOT / "cal_archive_inventory.json"
RR_INVENTORY_SHA256 = "a8ed63d83c360f1044b53fc6edaf3078cc50122e11b5a5f14c072a95dc427ef2"
RR_ROOT = E42_ROOT / "rr_train"
RR_RECEIPT = E42_ROOT / "rr_train_receipt.json"
MANIFEST = E42_ROOT / "parent_manifest.json"
EVIDENCE = REPO_ROOT / "evidence" / "e42_data_manifest.json"
CONTRACT = REPO_ROOT / "evidence" / "e42_fixed_contract.json"
CONTRACT_SHA256 = "dfb61d51a9b5188b5d3e8d7a7e406430f3c7447d8cf74d657e50e1d1156341c2"

E32_RECEIPT = E32_ROOT / "r1b_input_receipt.json"
E36_CAL = E36_ROOT / "cal_manifest.json"
E36_FINAL = E36_ROOT / "final_manifest.json"
E39_MANIFEST = E39_ROOT / "final_manifest.json"
IPN = E32_ROOT / "r1b_ipn_realization.json"
OWNER_DEFAULT = Path.home() / "Desktop" / "PixelProof Workspace" / "Samples" / "fotoğraf galeri"
OWNER_IDENTITY = "390e3c210ee61d70252d7e4714b8640463f44d57760942d25a1bdf7eab5aac09"
OWNER_COUNT = 210
RESERVE = {
    "name": "WhatsApp Image 2026-08-25 at 17.14.51.jpeg",
    "bytes": 206_418,
    "sha256": "e04755bfa5ef63da5536dc10395bd3f1faf2f79a6e304bc4eeba87a5e4ec57e3",
}
REPLAY_SALT = "E40_REPLAY_V1"
RR_SOURCES = {
    "Culture_&_Religion": "culture_and_religion",
    "Medical_&_Public_Health": "medical_and_public_health",
    "Natural_Disasters_&_Accidents": "natural_disasters_and_accidents",
    "Political_&_Social_Events": "political_and_social_events",
    "War_&_Conflict_Scenes": "war_and_conflict",
    "normal": "everyday_life",
    "production": "labor_and_production",
}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024**2):
            digest.update(chunk)
    return digest.hexdigest()


def _write_atomic(path: Path, value: Any) -> bytes:
    raw = _json_bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_bytes(raw)
    temporary.replace(path)
    return raw


def _dhash(image: Image.Image) -> str:
    grey = ImageOps.exif_transpose(image).convert("L").resize((9, 8), Image.Resampling.LANCZOS)
    pixels = list(grey.getdata())
    value = 0
    for y in range(8):
        for x in range(8):
            value = (value << 1) | int(pixels[y * 9 + x] > pixels[y * 9 + x + 1])
    return f"{value:016x}"


def rr_source(filename: str, class_name: str) -> str:
    if class_name == "real":
        if not re.fullmatch(r"real_\d+\.(?:jpg|jpeg|png)", filename, flags=re.IGNORECASE):
            raise ValueError(f"undeclared RR REAL filename: {filename}")
        return "rrdataset_real_pool"
    match = re.fullmatch(r"(.+)_\d+\.(?:jpg|jpeg|png)", filename, flags=re.IGNORECASE)
    if not match or match.group(1) not in RR_SOURCES:
        raise ValueError(f"undeclared RR AI filename: {filename}")
    return RR_SOURCES[match.group(1)]


def fixed_replay(rows: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    groups: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["role"] == "TRAIN":
            groups[(str(row["label"]), str(row["source_id"]))].append(row)
    selected = []
    for key in sorted(groups):
        count = round(len(groups[key]) * 0.05)
        ranked = sorted(
            groups[key],
            key=lambda row: hashlib.sha256(
                f"{REPLAY_SALT}|{row['record_id']}".encode()
            ).digest(),
        )
        selected.extend(ranked[:count])
    return sorted(selected, key=lambda row: str(row["record_id"]))


def _image_facts(path: Path, expected_sha256: str | None = None) -> dict[str, Any]:
    sha256 = _sha256_file(path)
    if expected_sha256 is not None and sha256 != expected_sha256:
        raise ValueError(f"input SHA-256 changed: {path}")
    with Image.open(path) as image:
        image.load()
        width, height = image.size
        decoded_format = image.format or path.suffix.lstrip(".").upper()
        dhash = _dhash(image)
    if width <= 0 or height <= 0:
        raise ValueError(f"invalid decoded geometry: {path}")
    return {
        "bytes": path.stat().st_size,
        "decoded_format": decoded_format,
        "dhash": dhash,
        "height": height,
        "sha256": sha256,
        "width": width,
    }


def extract_rr_train() -> dict[str, Any]:
    if RR_ROOT.exists() or RR_RECEIPT.exists():
        raise FileExistsError("E42 RR train extraction already exists")
    if _sha256_file(RR_INVENTORY) != RR_INVENTORY_SHA256:
        raise ValueError("RR archive inventory changed")
    inventory = json.loads(RR_INVENTORY.read_text())
    if inventory.get("by_split_class", {}).get("train/real") != 1250 or inventory.get(
        "by_split_class", {}
    ).get("train/ai") != 1250:
        raise ValueError("RR train population changed")
    if not RR_ARCHIVE.is_file() or RR_ARCHIVE.stat().st_size != 2_163_176_547:
        raise FileNotFoundError("verified RR original train/validation archive is absent")
    temporary = RR_ROOT.with_name(RR_ROOT.name + ".partial")
    if temporary.exists():
        raise FileExistsError(f"partial RR extraction requires audit: {temporary}")
    temporary.mkdir(parents=True)
    rows = []
    try:
        with tarfile.open(RR_ARCHIVE, mode="r:gz") as bundle:
            for member in bundle:
                path = _safe_member_name(member.name, "RRDataset_original_train_val")
                if not member.isfile() or len(path.parts) != 4 or path.parts[1] != "train":
                    continue
                class_name = path.parts[2]
                if class_name not in {"real", "ai"} or path.suffix.lower() not in IMAGE_SUFFIXES:
                    continue
                source = rr_source(path.name, class_name)
                destination = temporary / class_name / path.name
                destination.parent.mkdir(parents=True, exist_ok=True)
                extracted = bundle.extractfile(member)
                if extracted is None:
                    raise ValueError(f"cannot read RR member: {member.name}")
                with extracted, destination.open("xb") as output:
                    shutil.copyfileobj(extracted, output, length=8 * 1024**2)
                if destination.stat().st_size != member.size:
                    raise ValueError(f"RR extracted size changed: {member.name}")
                facts = _image_facts(destination)
                rows.append({
                    **facts,
                    "class_name": class_name,
                    "label": 1 if class_name == "ai" else 0,
                    "relative_path": f"rr_train/{class_name}/{path.name}",
                    "source": source,
                    "upstream_member": member.name,
                })
        counts = Counter((row["class_name"] for row in rows))
        if counts != {"ai": 1250, "real": 1250} or len(rows) != 2500:
            raise ValueError(f"RR extraction count mismatch: {counts}")
        temporary.replace(RR_ROOT)
    except Exception:
        raise
    rows.sort(key=lambda row: row["relative_path"])
    receipt = {
        "schema_version": 1,
        "experiment": "E42/RRDataset-train-extraction",
        "state": "rr_train_extracted_decoded_and_hash_bound",
        "inventory_sha256": RR_INVENTORY_SHA256,
        "counts": {"total": len(rows), **dict(sorted(counts.items()))},
        "source_counts": dict(sorted(Counter(row["source"] for row in rows).items())),
        "image_bytes": sum(int(row["bytes"]) for row in rows),
        "rows": rows,
        "boundary": "Only official RR train members; validation and test remain excluded.",
    }
    raw = _write_atomic(RR_RECEIPT, receipt)
    return {key: value for key, value in receipt.items() if key != "rows"} | {
        "receipt_sha256": hashlib.sha256(raw).hexdigest(),
        "receipt_bytes": len(raw),
    }


def _resolved(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _row(
    *, path: Path, parent_id: str, label: int, source: str, role: str,
    provenance: str, expected_sha256: str | None = None,
) -> dict[str, Any]:
    return {
        **_image_facts(path, expected_sha256),
        "label": label,
        "parent_id": parent_id,
        "path": str(path),
        "provenance": provenance,
        "role": role,
        "source": source,
    }


def _owner_rows(folder: Path) -> list[dict[str, Any]]:
    discovered = image_paths([str(folder)])
    reserve = [
        path for path in discovered
        if path.name == RESERVE["name"] and path.stat().st_size == RESERVE["bytes"]
        and _sha256_file(path) == RESERVE["sha256"]
    ]
    if len(discovered) == 211 and len(reserve) == 1:
        selected = [path for path in discovered if path != reserve[0]]
    elif len(discovered) == 210 and not reserve:
        selected = discovered
    else:
        raise ValueError("owner gallery membership changed")
    identity = [
        {"name": path.name, "bytes": path.stat().st_size, "sha256": _sha256_file(path)}
        for path in selected
    ]
    if len(selected) != OWNER_COUNT or hashlib.sha256(_json_bytes(identity)).hexdigest() != OWNER_IDENTITY:
        raise ValueError("owner gallery identity changed")
    return [
        _row(
            path=path, parent_id=f"owner:{item['sha256']}", label=0,
            source="owner_gallery", role="development", provenance="owner_gallery",
            expected_sha256=item["sha256"],
        )
        for path, item in zip(selected, identity, strict=True)
    ]


def _validate_contract() -> None:
    if CONTRACT_SHA256 == "TO_BE_PINNED" or _sha256_file(CONTRACT) != CONTRACT_SHA256:
        raise ValueError("E42 fixed contract changed or is not pinned in code")
    contract = json.loads(CONTRACT.read_text())
    if contract.get("state") != "fixed_before_e42_rr_train_extraction_or_features":
        raise ValueError("E42 fixed contract state changed")


def freeze_manifest(owner_gallery: Path = OWNER_DEFAULT) -> dict[str, Any]:
    if MANIFEST.exists() or EVIDENCE.exists():
        raise FileExistsError("E42 parent manifest already exists")
    _validate_contract()
    rr_raw = RR_RECEIPT.read_bytes()
    rr = json.loads(rr_raw)
    if rr.get("state") != "rr_train_extracted_decoded_and_hash_bound" or rr.get("counts", {}).get("total") != 2500:
        raise ValueError("RR train receipt is not complete")
    rows: list[dict[str, Any]] = []

    e32 = json.loads(E32_RECEIPT.read_text())
    replay = fixed_replay(e32["records"])
    if len(replay) != 1067:
        raise ValueError(f"E32 replay changed: {len(replay)}")
    for item in replay:
        rows.append(_row(
            path=E32_ROOT / "model_inputs" / "r0_global_jpeg90" / item["input_path"],
            parent_id=f"e32:{item['record_id']}", label=1 if item["label"] == "ai" else 0,
            source=f"e32:{item['source_id']}", role="train", provenance="e32_fixed_replay",
            expected_sha256=item["input_sha256"],
        ))

    for manifest_path, root, role, provenance in (
        (E36_CAL, E36_ROOT, "train", "e36_cal_consumed"),
        (E36_FINAL, E36_ROOT, "development", "e36_former_final_consumed"),
        (E39_MANIFEST, E39_ROOT, "development", "e39_consumed"),
    ):
        payload = json.loads(manifest_path.read_text())
        prefix = "e39" if manifest_path == E39_MANIFEST else "e36"
        for item in payload["rows"]:
            rows.append(_row(
                path=_resolved(root, str(item["path"])),
                parent_id=f"{prefix}:{item['parent_id']}", label=int(item["label"]),
                source=f"{prefix}:{item['source']}", role=role, provenance=provenance,
                expected_sha256=item["sha256"],
            ))

    for item in rr["rows"]:
        rows.append(_row(
            path=E42_ROOT / item["relative_path"], parent_id=f"rr:{item['sha256']}",
            label=int(item["label"]), source=f"rr:{item['source']}", role="train",
            provenance="rrdataset_official_train", expected_sha256=item["sha256"],
        ))

    ipn = json.loads(IPN.read_text())
    for item in ipn["records"]:
        rows.append(_row(
            path=E32_ROOT / item["source_key"], parent_id=f"ipn:{item['file_id']}", label=0,
            source=f"ipn:{item['device']}", role="development", provenance="ipn_consumed",
            expected_sha256=item["sha256"],
        ))
    rows.extend(_owner_rows(owner_gallery))
    rows.sort(key=lambda row: row["parent_id"])

    if len(rows) != 6888 or len({row["parent_id"] for row in rows}) != len(rows):
        raise ValueError("E42 parent population changed")
    role_counts = Counter(row["role"] for row in rows)
    if role_counts != {"train": 4638, "development": 2250}:
        raise ValueError(f"E42 role counts changed: {role_counts}")
    exact: dict[str, set[str]] = defaultdict(set)
    perceptual: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        exact[row["sha256"]].add(row["role"])
        perceptual[row["dhash"]].add(row["role"])
    exact_cross = sorted(key for key, roles in exact.items() if len(roles) > 1)
    dhash_cross = sorted(key for key, roles in perceptual.items() if len(roles) > 1)
    if exact_cross or dhash_cross:
        raise ValueError(
            f"E42 train/development contamination: exact={len(exact_cross)}, dhash={len(dhash_cross)}"
        )
    manifest = {
        "schema_version": 1,
        "experiment": "E42/texture-intermediate-source-heldout-recovery",
        "state": "e42_parent_manifest_frozen_before_features",
        "contract_sha256": CONTRACT_SHA256,
        "rr_receipt_sha256": hashlib.sha256(rr_raw).hexdigest(),
        "counts": {
            "total": len(rows),
            "roles": dict(sorted(role_counts.items())),
            "labels": dict(sorted(Counter(str(row["label"]) for row in rows).items())),
            "role_labels": dict(sorted(Counter(f"{row['role']}/{row['label']}" for row in rows).items())),
            "sources": len({row["source"] for row in rows}),
        },
        "decontamination": {
            "cross_role_exact_sha256_groups": exact_cross,
            "cross_role_exact_dhash_groups": dhash_cross,
        },
        "rows": rows,
        "boundary": "Consumed train/development only; B-Free and RR test are absent and unopened.",
    }
    raw = _write_atomic(MANIFEST, manifest)
    compact = {
        key: value for key, value in manifest.items() if key not in {"rows", "decontamination"}
    } | {
        "manifest_bytes": len(raw),
        "manifest_sha256": hashlib.sha256(raw).hexdigest(),
        "decontamination": manifest["decontamination"],
        "source_counts": dict(sorted(Counter(row["source"] for row in rows).items())),
    }
    _write_atomic(EVIDENCE, compact)
    return compact


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("extract-rr-train", "freeze-manifest"))
    parser.add_argument("--owner-gallery", type=Path, default=OWNER_DEFAULT)
    args = parser.parse_args(argv)
    result = extract_rr_train() if args.command == "extract-rr-train" else freeze_manifest(args.owner_gallery)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
