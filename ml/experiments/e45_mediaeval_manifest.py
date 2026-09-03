"""Decode, hash and decontaminate MediaEval ITW-SM before model access."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from io import BytesIO
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
from typing import Any, Iterable, Mapping, Sequence
import zipfile

from PIL import Image

from pixelproof.data_contract import dhash_image
from pixelproof.project_paths import DATA_ROOT, ML_ROOT


ROOT = DATA_ROOT / "e45_mediaeval_itwsm"
ARCHIVE = ROOT / "archives" / "itw-sm-sid-val.zip"
ARCHIVE_SHA256 = "18f1806e1cef6bc9f7ed6e49b61379a6cb4bac63cb4f3ed4f9fffffdf177b6e3"
INVENTORY = ROOT / "unscored_inventory.json"
MANIFEST = ROOT / "unscored_manifest.json"
EVIDENCE = ML_ROOT.parent / "evidence" / "e45_mediaeval_manifest.json"
CORRUPT_MEMBERS = {"ITW-SM/1_fake/x_618.jpg"}
EXPECTED_USABLE = 9_999
MAX_MEMBER_BYTES = 100 * 1024**2
MAX_PIXELS = 50_000_000
PLATFORM_NAMES = {
    "facebook": "Facebook",
    "instagram": "Instagram",
    "linkedin": "LinkedIn",
    "x": "X",
}
NAME_RE = re.compile(r"^(Facebook|Instagram|Linkedin|X)_real_\d+\.jpg$", re.IGNORECASE)
FAKE_RE = re.compile(r"^(facebook|instagram|linkedin|x)_\d+\.jpg$", re.IGNORECASE)


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def _write_atomic(path: Path, value: Any) -> bytes:
    raw = _json_bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_bytes(raw)
    temporary.replace(path)
    return raw


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024**2):
            digest.update(chunk)
    return digest.hexdigest()


def parse_member(name: str) -> tuple[int, str]:
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or len(path.parts) != 3:
        raise ValueError(f"unexpected E45 member path: {name!r}")
    root, class_dir, filename = path.parts
    if root != "ITW-SM" or class_dir not in {"0_real", "1_fake"}:
        raise ValueError(f"unexpected E45 member class: {name!r}")
    matcher = NAME_RE.fullmatch(filename) if class_dir == "0_real" else FAKE_RE.fullmatch(filename)
    if matcher is None:
        raise ValueError(f"unexpected E45 filename: {name!r}")
    prefix = filename.split("_", 1)[0].lower()
    return (0 if class_dir == "0_real" else 1), PLATFORM_NAMES[prefix]


def _protected_paths() -> list[Path]:
    paths = list(sorted((DATA_ROOT / "e32" / "audits").glob("*.json")))
    paths.extend([
        DATA_ROOT / "e32" / "c3_role_manifest.json",
        DATA_ROOT / "e32" / "r1b_role_manifest.json",
        DATA_ROOT / "e33_rrdataset" / "r1c_cal_manifest.json",
        DATA_ROOT / "e33_rrdataset" / "e42_rr_unscored_manifest.json",
        DATA_ROOT / "e36" / "cal_manifest.json",
        DATA_ROOT / "e36" / "final_manifest.json",
        DATA_ROOT / "e39" / "final_manifest.json",
        DATA_ROOT / "e42" / "parent_manifest.json",
        DATA_ROOT / "e42_external" / "bfree_viral" / "unscored_manifest.json",
        DATA_ROOT / "e43_dda_coco" / "unscored_manifest.json",
        DATA_ROOT / "e44" / "fusion_contract.json",
        DATA_ROOT / "e44" / "successor_contract.json",
    ])
    return [path for path in paths if path.is_file() and not path.name.startswith("._")]


def _row_collections(payload: Mapping[str, Any]) -> Iterable[Mapping[str, Any]]:
    for key in ("rows", "records", "e35_rows"):
        value = payload.get(key)
        if isinstance(value, list):
            yield from (row for row in value if isinstance(row, Mapping))


def protected_hashes() -> tuple[set[str], set[str], list[dict[str, Any]]]:
    exact: set[str] = set()
    perceptual: set[str] = set()
    sources: list[dict[str, Any]] = []
    for path in _protected_paths():
        raw = path.read_bytes()
        payload = json.loads(raw)
        for row in _row_collections(payload):
            if row.get("sha256"):
                exact.add(str(row["sha256"]))
            if row.get("dhash"):
                perceptual.add(str(row["dhash"]))
        sources.append({
            "path": str(path),
            "sha256": hashlib.sha256(raw).hexdigest(),
        })
    if not sources or not exact or not perceptual:
        raise ValueError("E45 protected-role hashes are unavailable")
    return exact, perceptual, sources


def audit_rows(
    rows: Sequence[Mapping[str, Any]], prior_exact: set[str], prior_dhash: set[str]
) -> dict[str, Any]:
    by_exact: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    by_dhash: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    exclusions: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        record_id = str(row["record_id"])
        exact = str(row["sha256"])
        perceptual = str(row["dhash"])
        by_exact[exact].append(row)
        by_dhash[perceptual].append(row)
        if exact in prior_exact:
            exclusions[record_id].add("protected_exact_overlap")
        if perceptual in prior_dhash:
            exclusions[record_id].add("protected_dhash_overlap")

    exact_groups = []
    cross_label_exact = []
    for digest, versions in sorted(by_exact.items()):
        if len(versions) < 2:
            continue
        ids = sorted(str(row["record_id"]) for row in versions)
        labels = sorted({int(row["label"]) for row in versions})
        group = {"sha256": digest, "record_ids": ids, "labels": labels}
        exact_groups.append(group)
        if len(labels) > 1:
            cross_label_exact.append(group)
            for record_id in ids:
                exclusions[record_id].add("cross_label_exact_duplicate")
        else:
            for record_id in ids[1:]:
                exclusions[record_id].add(f"same_label_exact_duplicate_of:{ids[0]}")

    dhash_groups = []
    for digest, versions in sorted(by_dhash.items()):
        if len(versions) > 1:
            dhash_groups.append({
                "dhash": digest,
                "record_ids": sorted(str(row["record_id"]) for row in versions),
                "labels": sorted({int(row["label"]) for row in versions}),
            })
    return {
        "excluded_record_ids": sorted(exclusions),
        "exclusion_reasons": {
            record_id: sorted(reasons) for record_id, reasons in sorted(exclusions.items())
        },
        "cross_record_exact_groups": exact_groups,
        "cross_label_exact_groups": cross_label_exact,
        "cross_record_dhash_diagnostic": dhash_groups,
    }


def _decode(payload: bytes, member: str) -> dict[str, Any]:
    if not payload or len(payload) > MAX_MEMBER_BYTES:
        raise ValueError(f"E45 member byte size is unsafe: {member}")
    with Image.open(BytesIO(payload)) as opened:
        opened.verify()
    with Image.open(BytesIO(payload)) as opened:
        width, height = opened.size
        if width <= 0 or height <= 0 or width * height > MAX_PIXELS:
            raise ValueError(f"E45 image geometry is unsafe: {member} {width}x{height}")
        decoded_format = str(opened.format)
        rgb = opened.convert("RGB")
        rgb.load()
        perceptual = dhash_image(rgb)
    return {
        "bytes": len(payload),
        "width": width,
        "height": height,
        "decoded_format": decoded_format,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "dhash": perceptual,
    }


def build_manifest() -> dict[str, Any]:
    if MANIFEST.exists() or EVIDENCE.exists():
        raise FileExistsError("E45 manifest already exists; no silent replacement")
    inventory_raw = INVENTORY.read_bytes()
    inventory = json.loads(inventory_raw)
    if inventory.get("archive_sha256") != ARCHIVE_SHA256:
        raise ValueError("E45 inventory archive identity changed")
    failed = {str(row["member"]) for row in inventory.get("crc_failures", [])}
    if failed != CORRUPT_MEMBERS:
        raise ValueError(f"E45 CRC exclusion set changed: {sorted(failed)}")
    if ARCHIVE.stat().st_size != 3_553_693_205 or _file_sha256(ARCHIVE) != ARCHIVE_SHA256:
        raise ValueError("E45 archive identity changed before decode")

    rows: list[dict[str, Any]] = []
    with zipfile.ZipFile(ARCHIVE) as bundle:
        for info in bundle.infolist():
            if info.is_dir() or info.filename in CORRUPT_MEMBERS:
                continue
            label, platform = parse_member(info.filename)
            decoded = _decode(bundle.read(info), info.filename)
            rows.append({
                "record_id": f"e45:{info.filename}",
                "parent_id": f"e45:{info.filename}",
                "member": info.filename,
                "label": label,
                "platform": platform,
                **decoded,
            })
    if len(rows) != EXPECTED_USABLE:
        raise ValueError(f"E45 decoded row count changed: {len(rows)}")

    prior_exact, prior_dhash, protected = protected_hashes()
    audit = audit_rows(rows, prior_exact, prior_dhash)
    excluded = set(audit["excluded_record_ids"])
    selected = [row for row in rows if str(row["record_id"]) not in excluded]
    if any(str(row["sha256"]) in prior_exact or str(row["dhash"]) in prior_dhash for row in selected):
        raise ValueError("E45 protected overlap survived exclusion")

    label_platform = Counter((int(row["label"]), str(row["platform"])) for row in selected)
    payload = {
        "schema_version": 1,
        "state": "e45_mediaeval_decontaminated_manifest_frozen_unscored",
        "experiment": "E45",
        "role": "UNTOUCHED_FINAL",
        "archive_sha256": ARCHIVE_SHA256,
        "inventory_sha256": hashlib.sha256(inventory_raw).hexdigest(),
        "counts": {
            "official_rows": 10_000,
            "crc_usable_rows": EXPECTED_USABLE,
            "selected_rows": len(selected),
            "excluded_rows": len(rows) - len(selected),
            "selected_by_label": dict(sorted(Counter(int(row["label"]) for row in selected).items())),
            "selected_by_label_platform": {
                f"{label}:{platform}": count
                for (label, platform), count in sorted(label_platform.items())
            },
            "official_archive_coverage": len(selected) / 10_000,
        },
        "protected_exact_hashes": len(prior_exact),
        "protected_dhashes": len(prior_dhash),
        "protected_manifests": protected,
        "crc_excluded_members": sorted(CORRUPT_MEMBERS),
        "audit": audit,
        "rows": selected,
        "model_scores_created": 0,
        "boundary": "All retained images decoded and decontaminated before model access; no score exists.",
    }
    raw = _write_atomic(MANIFEST, payload)
    evidence = {
        "schema_version": 1,
        "state": payload["state"],
        "role": payload["role"],
        "archive_sha256": ARCHIVE_SHA256,
        "manifest_bytes": len(raw),
        "manifest_sha256": hashlib.sha256(raw).hexdigest(),
        "counts": payload["counts"],
        "audit_counts": {
            "protected_exact_or_dhash_exclusions": sum(
                1 for reasons in audit["exclusion_reasons"].values()
                if any(reason.startswith("protected_") for reason in reasons)
            ),
            "cross_record_exact_groups": len(audit["cross_record_exact_groups"]),
            "cross_label_exact_groups": len(audit["cross_label_exact_groups"]),
            "cross_record_dhash_diagnostic": len(audit["cross_record_dhash_diagnostic"]),
        },
        "protected_manifest_count": len(protected),
        "model_scores_created": 0,
        "boundary": payload["boundary"],
    }
    _write_atomic(EVIDENCE, evidence)
    return evidence


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("manifest",))
    args = parser.parse_args(argv)
    result = build_manifest() if args.command == "manifest" else None
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
