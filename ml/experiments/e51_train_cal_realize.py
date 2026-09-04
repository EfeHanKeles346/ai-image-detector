"""Realize E51 TRAIN and CAL manifests without loading a detector."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
from io import BytesIO
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
import zipfile

from PIL import Image, ImageOps
import numpy as np
from scipy.fft import dctn

from experiments.e48_manifest import PROTECTED_ROLE_PATHS, _protected_role_hashes
from experiments.e51_realization import (
    CONTRACT,
    E42_MANIFEST,
    SCIMD_ARCHIVE,
    SCIMD_TARGET_PER_DEVICE,
)
from pixelproof.data_contract import dhash_image
from pixelproof.project_paths import DATA_ROOT, ML_ROOT


CONTRACT_SHA256 = "48e078fe5a96a108c102a06aef436fe47a5e937a581656c1fd63df29816adbce"
ROOT = DATA_ROOT / "e51"
SCMI_RECEIPT = ROOT / "receipts" / "scmi30_cal_download_unscored.json"
SCMI_AMENDMENT_RECEIPT = ROOT / "receipts" / "scmi30_cal_amendment_unscored.json"
SCIMD_ROOT = ROOT / "realized" / "scimd17_train"
CAL_Q75_ROOT = ROOT / "realized" / "cal_q75"
TRAIN_MANIFEST = ROOT / "manifests" / "train_parents_unscored.json"
CAL_MANIFEST = ROOT / "manifests" / "cal_paired_unscored.json"
EVIDENCE = ML_ROOT.parent / "evidence" / "e51_train_cal_manifest.json"
MAX_PIXELS = 100_000_000
ALLOWED_FORMATS = {"JPEG", "MPO", "PNG", "WEBP", "TIFF", "BMP"}
EXTRA_PROTECTED = (
    DATA_ROOT / "e48" / "manifest_unscored.json",
    DATA_ROOT / "e49" / "dotting" / "manifest_unscored.json",
    DATA_ROOT / "e49" / "open_components_v2" / "commons_manifest_unscored.json",
    DATA_ROOT / "e49" / "open_components_v2" / "stylegan2_manifest_unscored.json",
    DATA_ROOT / "e49" / "openfake" / "manifest_unscored.json",
    DATA_ROOT / "e49" / "final" / "manifest_unscored.json",
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


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def _decode(raw: bytes, identity: str) -> tuple[dict[str, Any], Image.Image]:
    with Image.open(BytesIO(raw)) as opened:
        opened.verify()
    with Image.open(BytesIO(raw)) as opened:
        opened.load()
        width, height = opened.size
        decoded_format = str(opened.format or "UNKNOWN").upper()
        if decoded_format not in ALLOWED_FORMATS or width <= 0 or height <= 0:
            raise ValueError(f"unsupported E51 image: {identity}")
        if width * height > MAX_PIXELS:
            raise ValueError(f"unsafe E51 image geometry: {identity}")
        rgb = ImageOps.exif_transpose(opened).convert("RGB")
        grey = rgb.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
        pixels = list(grey.get_flattened_data())
        legacy_value = 0
        for y in range(8):
            for x in range(8):
                legacy_value = (legacy_value << 1) | int(
                    pixels[y * 9 + x] > pixels[y * 9 + x + 1]
                )
        facts = {
            "bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "dhash": dhash_image(rgb),
            "legacy_dhash": f"{legacy_value:016x}",
            "format": decoded_format,
            "width": width,
            "height": height,
            "mode": str(opened.mode),
        }
    return facts, rgb


def _q75(rgb: Image.Image) -> bytes:
    output = BytesIO()
    rgb.save(output, format="JPEG", quality=75, subsampling=2, optimize=False)
    return output.getvalue()


def _phash_image(image: Image.Image) -> int:
    pixels = np.asarray(
        image.convert("L").resize((32, 32), Image.Resampling.LANCZOS), dtype=np.float32
    )
    coefficients = dctn(pixels, norm="ortho")[:8, :8].ravel()
    bits = coefficients[1:] > np.median(coefficients[1:])
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


def _phash_path(path: Path) -> int:
    with Image.open(path) as opened:
        opened.load()
        return _phash_image(ImageOps.exif_transpose(opened).convert("RGB"))


def select_clean_scimd(
    rows: Sequence[Mapping[str, Any]],
    protected_exact: set[str],
    protected_dhash: set[str],
    *,
    target_per_device: int = SCIMD_TARGET_PER_DEVICE,
) -> tuple[list[dict[str, Any]], dict[str, list[str]]]:
    reasons: dict[str, list[str]] = defaultdict(list)
    seen_exact: dict[str, str] = {}
    seen_dhash: dict[str, str] = {}
    for row in sorted(rows, key=lambda item: (str(item["device"]), str(item["rank"]))):
        identity = str(row["identity"])
        exact, perceptual = str(row["sha256"]), str(row["dhash"])
        if exact in protected_exact:
            reasons[identity].append("protected_exact_overlap")
        if perceptual in protected_dhash:
            reasons[identity].append("protected_dhash_overlap")
        if exact in seen_exact:
            reasons[identity].append(f"internal_exact_duplicate_of:{seen_exact[exact]}")
        else:
            seen_exact[exact] = identity
        if perceptual in seen_dhash:
            reasons[identity].append(f"internal_dhash_duplicate_of:{seen_dhash[perceptual]}")
        else:
            seen_dhash[perceptual] = identity
    selected = []
    for device in sorted({str(row["device"]) for row in rows}):
        clean = sorted(
            (dict(row) for row in rows if row["device"] == device and row["identity"] not in reasons),
            key=lambda row: (str(row["rank"]), str(row["identity"])),
        )
        if len(clean) < target_per_device:
            raise ValueError(f"insufficient clean SCIMD-17 rows for {device}: {len(clean)}")
        selected.extend(clean[:target_per_device])
    return selected, dict(sorted(reasons.items()))


def _protected() -> tuple[
    set[str], set[str], list[dict[str, str]], dict[str, list[str]],
]:
    exact, perceptual, sources = _protected_role_hashes()
    dhash_paths: dict[str, list[str]] = defaultdict(list)
    for path in tuple(PROTECTED_ROLE_PATHS) + EXTRA_PROTECTED:
        if not path.is_file():
            continue
        raw = path.read_bytes()
        payload = json.loads(raw)
        rows = []
        for key in ("rows", "records", "e35_rows"):
            if isinstance(payload.get(key), list):
                rows.extend(row for row in payload[key] if isinstance(row, Mapping))
        for row in rows:
            if row.get("sha256"):
                exact.add(str(row["sha256"]))
            if row.get("dhash"):
                value = str(row["dhash"])
                perceptual.add(value)
                image_path = Path(str(row.get("path", "")))
                if image_path.is_file():
                    dhash_paths[value].append(str(image_path))
        if path in EXTRA_PROTECTED:
            sources.append({"path": str(path), "sha256": hashlib.sha256(raw).hexdigest()})
    # Historical manifests use both left>right and right>left dHash conventions.
    # Their 64-bit values are complements; protecting both avoids a false non-overlap.
    complements = set()
    for value in perceptual:
        if len(value) == 16:
            try:
                complements.add(f"{int(value, 16) ^ ((1 << 64) - 1):016x}")
            except ValueError:
                continue
    perceptual.update(complements)
    for value, paths in list(dhash_paths.items()):
        try:
            complement = f"{int(value, 16) ^ ((1 << 64) - 1):016x}"
        except ValueError:
            continue
        dhash_paths[complement].extend(paths)
    return exact, perceptual, sources, {
        key: sorted(set(paths)) for key, paths in sorted(dhash_paths.items())
    }


def _realize_scimd(
    contract_rows: Sequence[Mapping[str, Any]], protected_exact: set[str], protected_dhash: set[str],
) -> tuple[list[dict[str, Any]], list[dict[str, str]], dict[str, list[str]]]:
    decoded, failures = [], []
    by_member = {str(row["member"]): row for row in contract_rows}
    with zipfile.ZipFile(SCIMD_ARCHIVE) as archive:
        for member, row in by_member.items():
            try:
                raw = archive.read(member)
                facts, _ = _decode(raw, str(row["identity"]))
                if len(raw) != int(row["expected_bytes"]):
                    raise ValueError("SCIMD-17 member byte count changed")
                if f"{archive.getinfo(member).CRC:08x}" != row["expected_crc32"]:
                    raise ValueError("SCIMD-17 member CRC changed")
                if (facts["width"], facts["height"]) != (224, 224):
                    raise ValueError("SCIMD-17 member is not the declared 224x224 resize")
                decoded.append({**row, **facts})
            except (OSError, ValueError, zipfile.BadZipFile, KeyError) as error:
                failures.append({"identity": str(row["identity"]),
                                 "error": f"{type(error).__name__}: {error}"})
    if len(decoded) + len(failures) != len(contract_rows):
        raise ValueError("SCIMD-17 realization coverage changed")
    selected, reasons = select_clean_scimd(decoded, protected_exact, protected_dhash)
    with zipfile.ZipFile(SCIMD_ARCHIVE) as archive:
        for row in selected:
            raw = archive.read(str(row["member"]))
            suffix = Path(str(row["member"])).suffix.lower()
            destination = SCIMD_ROOT / str(row["device"]) / f"{row['rank']}{suffix}"
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_suffix(destination.suffix + ".part")
            temporary.write_bytes(raw)
            temporary.replace(destination)
            row["path"] = str(destination)
            row["parent_id"] = str(row["identity"])
            row["condition"] = "publisher-224-resize"
    return selected, failures, reasons


def _cal_parent_rows(contract: Mapping[str, Any]) -> list[dict[str, Any]]:
    scmi_raw = SCMI_RECEIPT.read_bytes()
    scmi = json.loads(scmi_raw)
    if scmi.get("model_scores_created") != 0 or len(scmi.get("rows") or []) != 1200:
        raise ValueError("SCMI30 receipt boundary changed")
    source_rows = list(scmi["rows"])
    if SCMI_AMENDMENT_RECEIPT.is_file():
        amendment = json.loads(SCMI_AMENDMENT_RECEIPT.read_text())
        if amendment.get("model_scores_created") != 0 or not amendment.get("selected"):
            raise ValueError("SCMI30 identity amendment boundary changed")
        source_rows = [row for row in source_rows if row["identity"] != amendment["excluded_identity"]]
        source_rows.append(amendment["selected"])
    if len(source_rows) != 1200:
        raise ValueError("SCMI30 amended CAL count changed")
    scmi_rows = []
    for row in source_rows:
        path = Path(str(row["path"]))
        raw = path.read_bytes()
        facts, _ = _decode(raw, str(row["identity"]))
        if facts["sha256"] != row["sha256"] or facts["bytes"] != int(row["bytes"]):
            raise ValueError(f"SCMI30 payload changed: {row['identity']}")
        scmi_rows.append({**row, **facts, "parent_id": str(row["identity"]), "role": "CAL",
                          "condition": "original", "cal_origin": "new-independent-real"})
    ai_rows = []
    for row in contract["internal_ai_cal"]["rows"]:
        path = Path(str(row["path"]))
        raw = path.read_bytes()
        facts, _ = _decode(raw, str(row["parent_id"]))
        if facts["sha256"] != row["sha256"] or facts["legacy_dhash"] != row["dhash"]:
            raise ValueError(f"historical AI-CAL payload changed: {row['parent_id']}")
        ai_rows.append({**row, **facts, "role": "CAL", "condition": "original",
                        "cal_origin": "held-out-historical-train"})
    return sorted(scmi_rows + ai_rows, key=lambda row: (int(row["label"]), str(row["source"]),
                                                         str(row["parent_id"])))


def _pair_cal(parents: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    observations = []
    seen_child_exact: dict[str, str] = {}
    for parent in parents:
        parent_id = str(parent["parent_id"])
        path = Path(str(parent["path"]))
        raw = path.read_bytes()
        _, rgb = _decode(raw, parent_id)
        original = {**parent, "record_id": f"{parent_id}:original", "condition": "original"}
        child_raw = _q75(rgb)
        child_facts, _ = _decode(child_raw, parent_id + ":q75")
        if child_facts["sha256"] in seen_child_exact:
            raise ValueError(f"E51 Q75 exact collision: {parent_id}")
        seen_child_exact[child_facts["sha256"]] = parent_id
        destination = CAL_Q75_ROOT / f"{hashlib.sha256(parent_id.encode()).hexdigest()}.jpg"
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(".jpg.part")
        temporary.write_bytes(child_raw)
        temporary.replace(destination)
        child = {
            **{key: value for key, value in parent.items() if key not in {
                "bytes", "sha256", "dhash", "format", "width", "height", "mode", "path",
            }},
            **child_facts,
            "path": str(destination),
            "record_id": f"{parent_id}:q75",
            "condition": "q75",
        }
        observations.extend((original, child))
    return observations


def realize() -> dict[str, Any]:
    if TRAIN_MANIFEST.exists() or CAL_MANIFEST.exists() or EVIDENCE.exists():
        raise FileExistsError("E51 TRAIN/CAL realization already exists")
    contract_raw = CONTRACT.read_bytes()
    if hashlib.sha256(contract_raw).hexdigest() != CONTRACT_SHA256:
        raise ValueError("E51 realization contract changed")
    contract = json.loads(contract_raw)
    if contract.get("model_scores_created") != 0 or contract.get("new_image_bodies_decoded") != 0:
        raise ValueError("E51 realization contract is no longer pre-score")
    protected_exact, protected_dhash, protected_sources, protected_dhash_paths = _protected()
    scimd, scimd_failures, scimd_reasons = _realize_scimd(
        contract["scimd17_train"]["rows"], protected_exact, protected_dhash
    )
    e42_raw = E42_MANIFEST.read_bytes()
    historical = json.loads(e42_raw).get("rows") or []
    heldout = {str(row["parent_id"]) for row in contract["internal_ai_cal"]["rows"]}
    base_train = [dict(row) for row in historical if str(row.get("role", "")).lower() == "train"
                  and str(row["parent_id"]) not in heldout]
    if heldout & {str(row["parent_id"]) for row in base_train}:
        raise ValueError("held-out AI-CAL leaked into E51 fit")
    train_rows = sorted(base_train + scimd, key=lambda row: (int(row["label"]), str(row["source"]),
                                                             str(row["parent_id"])))
    train_payload = {
        "schema_version": 1,
        "state": "e51_train_parents_decontaminated_frozen_unscored",
        "realization_contract_sha256": CONTRACT_SHA256,
        "counts": {
            "parents": len(train_rows),
            "REAL": sum(int(row["label"]) == 0 for row in train_rows),
            "AI": sum(int(row["label"]) == 1 for row in train_rows),
            "scimd17_selected": len(scimd),
            "scimd17_decode_failures": len(scimd_failures),
            "scimd17_identity_exclusions": len(scimd_reasons),
        },
        "source_counts": dict(sorted(Counter(str(row["source"]) for row in train_rows).items())),
        "scimd17_decode_failures": scimd_failures,
        "scimd17_identity_exclusion_reasons": scimd_reasons,
        "heldout_ai_cal_parent_ids": sorted(heldout),
        "protected_role_manifests": protected_sources,
        "rows": train_rows,
        "model_scores_created": 0,
    }
    train_raw = _write_atomic(TRAIN_MANIFEST, train_payload)
    cal_parents = _cal_parent_rows(contract)
    new_real = [row for row in cal_parents if row["cal_origin"] == "new-independent-real"]
    perceptual_collision_audit = []
    for row in new_real:
        if row["sha256"] in protected_exact:
            raise ValueError(f"SCMI30 CAL exact-overlaps protected evidence: {row['parent_id']}")
        candidates = protected_dhash_paths.get(str(row["dhash"]), [])
        if row["dhash"] in protected_dhash and not candidates:
            raise ValueError(f"SCMI30 CAL dHash collision cannot be verified: {row['parent_id']}")
        if candidates:
            target_phash = _phash_path(Path(str(row["path"])))
            distances = [(target_phash ^ _phash_path(Path(path))).bit_count() for path in candidates]
            minimum = min(distances)
            perceptual_collision_audit.append({
                "parent_id": row["parent_id"], "dhash": row["dhash"],
                "candidate_matches": len(candidates), "minimum_phash_distance": minimum,
                "confirmed_near_duplicate": minimum <= 4,
            })
            if minimum <= 4:
                raise ValueError(f"SCMI30 CAL perceptually overlaps protected evidence: {row['parent_id']}")
    cal_observations = _pair_cal(cal_parents)
    cal_payload = {
        "schema_version": 1,
        "state": "e51_cal_paired_decontaminated_frozen_unscored",
        "realization_contract_sha256": CONTRACT_SHA256,
        "counts": {
            "parents": len(cal_parents),
            "observations": len(cal_observations),
            "REAL_parents": sum(int(row["label"]) == 0 for row in cal_parents),
            "AI_parents": sum(int(row["label"]) == 1 for row in cal_parents),
            "original": sum(row["condition"] == "original" for row in cal_observations),
            "q75": sum(row["condition"] == "q75" for row in cal_observations),
        },
        "source_counts": dict(sorted(Counter(str(row["source"]) for row in cal_parents).items())),
        "perceptual_collision_audit": perceptual_collision_audit,
        "parents": cal_parents,
        "rows": cal_observations,
        "model_scores_created": 0,
        "boundary": "CAL selects candidates/thresholds only; no row may fit or enter DEVELOPMENT/final.",
    }
    cal_raw = _write_atomic(CAL_MANIFEST, cal_payload)
    evidence = {
        "schema_version": 1,
        "state": "e51_train_cal_realized_unscored",
        "realization_contract_sha256": CONTRACT_SHA256,
        "train_counts": train_payload["counts"],
        "cal_counts": cal_payload["counts"],
        "train_manifest_bytes": len(train_raw),
        "train_manifest_sha256": hashlib.sha256(train_raw).hexdigest(),
        "cal_manifest_bytes": len(cal_raw),
        "cal_manifest_sha256": hashlib.sha256(cal_raw).hexdigest(),
        "model_scores_created": 0,
    }
    _write_atomic(EVIDENCE, evidence)
    return evidence


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("realize",))
    parser.parse_args(argv)
    print(json.dumps(realize(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
