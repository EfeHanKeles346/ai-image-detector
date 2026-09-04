"""Realize, decontaminate and pair the frozen E49-C OpenFake reserve without a detector."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
from io import BytesIO
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from PIL import Image, ImageOps

from experiments.e48_manifest import _protected_role_hashes
from experiments.e49_dotting import MANIFEST as DOTTING_MANIFEST, social_q75_bytes
from experiments.e49_open_realize import MANIFEST as STYLEGAN2_MANIFEST
from experiments.e49_openfake import CONTRACT_C, MODEL_KEYS_C
from experiments.e49_openfake_download import (
    ASSET_CONTRACT_SHA256,
    EXPECTED_BYTES,
    EXPECTED_ROWS,
    RECEIPT,
)
from pixelproof.data_contract import dhash_image
from pixelproof.project_paths import DATA_ROOT, ML_ROOT


IDENTITY_CONTRACT_SHA256 = "0abae56af862c9b402ef5ef594a21181cbbb7f72ba7495a491b0389bfdfcd702"
TARGET_PER_MODEL = 160
# Exact largest geometry already frozen by the no-body asset contract (row 43,863).
MAX_PIXELS = 67_633_152
ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP"}

ROOT = DATA_ROOT / "e49" / "openfake"
PAIRED_ROOT = ROOT / "paired" / "social_q75"
MANIFEST = ROOT / "manifest_unscored.json"
EVIDENCE = ML_ROOT.parent / "evidence" / "e49_c_openfake_manifest.json"


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def _write_atomic(path: Path, value: Any) -> bytes:
    raw = _json_bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_bytes(raw)
    temporary.replace(path)
    return raw


def _decode(path: Path, identity: str) -> dict[str, Any]:
    raw = path.read_bytes()
    with Image.open(BytesIO(raw)) as opened:
        opened.verify()
    with Image.open(BytesIO(raw)) as opened:
        opened.load()
        width, height = opened.size
        decoded_format = str(opened.format or "UNKNOWN").upper()
        if decoded_format not in ALLOWED_FORMATS or width <= 0 or height <= 0:
            raise ValueError(f"E49-C unsupported payload: {identity}")
        if width * height > MAX_PIXELS:
            raise ValueError(f"E49-C unsafe geometry: {identity}")
        rgb = ImageOps.exif_transpose(opened).convert("RGB")
        return {
            "bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "dhash": dhash_image(rgb),
            "format": decoded_format,
            "width": width,
            "height": height,
            "mode": str(opened.mode),
        }


def select_clean_candidates(
    rows: Sequence[Mapping[str, Any]],
    prior_exact: set[str],
    prior_dhash: set[str],
    *,
    target_per_model: int = TARGET_PER_MODEL,
) -> tuple[list[dict[str, Any]], dict[str, list[str]]]:
    """Freeze balanced candidates by their prebound ranks after model-blind identity checks."""
    reasons: dict[str, list[str]] = defaultdict(list)
    seen_exact: dict[str, str] = {}
    seen_dhash: dict[str, str] = {}
    for source in sorted(rows, key=lambda row: (str(row["rank"]), str(row["record_id"]))):
        row = dict(source)
        record_id = str(row["record_id"])
        exact, perceptual = str(row["sha256"]), str(row["dhash"])
        if exact in prior_exact:
            reasons[record_id].append("protected_exact_overlap")
        if perceptual in prior_dhash:
            reasons[record_id].append("protected_dhash_overlap")
        if exact in seen_exact:
            reasons[record_id].append(f"internal_exact_duplicate_of:{seen_exact[exact]}")
        else:
            seen_exact[exact] = record_id
        if perceptual in seen_dhash:
            reasons[record_id].append(f"internal_dhash_duplicate_of:{seen_dhash[perceptual]}")
        else:
            seen_dhash[perceptual] = record_id

    selected: list[dict[str, Any]] = []
    for model in MODEL_KEYS_C:
        candidates = sorted(
            (dict(row) for row in rows if row["model"] == model),
            key=lambda row: (str(row["rank"]), str(row["record_id"])),
        )
        clean = [row for row in candidates if str(row["record_id"]) not in reasons]
        if len(clean) < target_per_model:
            raise ValueError(f"E49-C clean target unavailable for {model}: {len(clean)}/{target_per_model}")
        selected.extend(clean[:target_per_model])
    return selected, dict(sorted(reasons.items()))


def _protected() -> tuple[set[str], set[str], list[dict[str, str]]]:
    exact, dhashes, sources = _protected_role_hashes()
    for path in (DOTTING_MANIFEST, STYLEGAN2_MANIFEST):
        if not path.is_file():
            raise FileNotFoundError(f"E49-C protected component manifest missing: {path}")
        raw = path.read_bytes()
        for row in json.loads(raw).get("rows", []):
            if row.get("sha256"):
                exact.add(str(row["sha256"]))
            if row.get("dhash"):
                dhashes.add(str(row["dhash"]))
        sources.append({"path": str(path), "sha256": hashlib.sha256(raw).hexdigest()})
    return exact, dhashes, sources


def _bound_rows() -> tuple[list[dict[str, Any]], bytes]:
    identity_raw = CONTRACT_C.read_bytes()
    if hashlib.sha256(identity_raw).hexdigest() != IDENTITY_CONTRACT_SHA256:
        raise ValueError("E49-C OpenFake identity contract changed")
    receipt_raw = RECEIPT.read_bytes()
    receipt = json.loads(receipt_raw)
    downloaded = receipt.get("rows") or []
    if (
        receipt.get("state") != "e49_c_openfake_downloaded_decoded_unscored"
        or receipt.get("asset_contract_sha256") != ASSET_CONTRACT_SHA256
        or receipt.get("files") != EXPECTED_ROWS
        or receipt.get("bytes") != EXPECTED_BYTES
        or receipt.get("model_scores_created") != 0
        or len(downloaded) != EXPECTED_ROWS
    ):
        raise ValueError("E49-C OpenFake download receipt changed")
    by_record = {str(row["record_id"]): row for row in downloaded}
    identities = json.loads(identity_raw).get("rows") or []
    if len(identities) != EXPECTED_ROWS or set(by_record) != {str(row["record_id"]) for row in identities}:
        raise ValueError("E49-C identity/download coverage changed")
    return [{**row, **by_record[str(row["record_id"])]} for row in identities], receipt_raw


def freeze_manifest() -> dict[str, Any]:
    if MANIFEST.exists() or EVIDENCE.exists():
        raise FileExistsError("E49-C OpenFake manifest already exists")
    bound, receipt_raw = _bound_rows()
    decoded: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for row in bound:
        record_id = str(row["record_id"])
        path = Path(str(row["path"]))
        try:
            found = _decode(path, record_id)
            if (
                found["bytes"] != int(row["bytes"])
                or found["sha256"] != row["sha256"]
                or found["format"] != row["format"]
                or (found["width"], found["height"]) != (int(row["width"]), int(row["height"]))
            ):
                raise ValueError("payload differs from exact download receipt")
            decoded.append({**row, **found})
        except (OSError, ValueError) as error:
            failures.append({"record_id": record_id, "error": f"{type(error).__name__}: {error}"})
    if len(decoded) + len(failures) != EXPECTED_ROWS:
        raise ValueError("E49-C reserve decode coverage changed")

    prior_exact, prior_dhash, protected_sources = _protected()
    _, reasons = select_clean_candidates(decoded, prior_exact, prior_dhash)

    observations: list[dict[str, Any]] = []
    parents: list[dict[str, Any]] = []
    child_exact: dict[str, str] = {}
    child_dhash: dict[str, str] = {}
    final_selected: list[dict[str, Any]] = []
    # A child collision rejects its parent; continue in the same prebound rank order into reserve.
    for model, source in MODEL_KEYS_C.items():
        candidates = sorted(
            (row for row in decoded if row["model"] == model and str(row["record_id"]) not in reasons),
            key=lambda row: (str(row["rank"]), str(row["record_id"])),
        )
        accepted = 0
        for row in candidates:
            record_id = str(row["record_id"])
            child_raw = social_q75_bytes(Path(str(row["path"])))
            with Image.open(BytesIO(child_raw)) as opened:
                opened.load()
                child = {
                    "bytes": len(child_raw), "sha256": hashlib.sha256(child_raw).hexdigest(),
                    "dhash": dhash_image(opened.convert("RGB")), "format": str(opened.format).upper(),
                    "width": opened.width, "height": opened.height, "mode": str(opened.mode),
                }
            child_id = record_id + ":social_q75"
            child_reasons = []
            if child["sha256"] in prior_exact:
                child_reasons.append("social_protected_exact_overlap")
            if child["dhash"] in prior_dhash:
                child_reasons.append("social_protected_dhash_overlap")
            if child["sha256"] in child_exact:
                child_reasons.append(f"social_internal_exact_duplicate_of:{child_exact[child['sha256']]}")
            if child["dhash"] in child_dhash:
                child_reasons.append(f"social_internal_dhash_duplicate_of:{child_dhash[child['dhash']]}")
            if child_reasons:
                reasons[record_id] = child_reasons
                continue
            child_exact[child["sha256"]] = child_id
            child_dhash[child["dhash"]] = child_id
            destination = PAIRED_ROOT / model / f"{int(row['row_index']):06d}.jpg"
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_suffix(".jpg.part")
            temporary.write_bytes(child_raw)
            temporary.replace(destination)
            parent = {
                "parent_id": record_id, "label": 1, "source": source, "model": model,
                "rank": row["rank"], "row_index": int(row["row_index"]),
                "release_date": row["release_date"], "type": row["type"],
            }
            parents.append(parent)
            final_selected.append(row)
            observations.extend([
                {**row, "record_id": record_id + ":publisher_original", "parent_id": record_id,
                 "condition": "publisher_original", "source": source, "status": "unscored"},
                {**parent, **child, "record_id": child_id, "condition": "social_q75",
                 "path": str(destination), "status": "unscored"},
            ])
            accepted += 1
            if accepted == TARGET_PER_MODEL:
                break
        if accepted != TARGET_PER_MODEL:
            raise ValueError(f"E49-C paired clean target unavailable for {model}: {accepted}/{TARGET_PER_MODEL}")

    by_model = Counter(row["model"] for row in parents)
    if by_model != Counter({model: TARGET_PER_MODEL for model in MODEL_KEYS_C}):
        raise ValueError("E49-C final OpenFake parents are not model-balanced")
    formats = Counter(str(row["format"]) for row in final_selected)
    geometry = {
        model: {
            "min_width": min(int(row["width"]) for row in final_selected if row["model"] == model),
            "max_width": max(int(row["width"]) for row in final_selected if row["model"] == model),
            "min_height": min(int(row["height"]) for row in final_selected if row["model"] == model),
            "max_height": max(int(row["height"]) for row in final_selected if row["model"] == model),
        }
        for model in MODEL_KEYS_C
    }
    payload = {
        "schema_version": 1,
        "state": "e49_c_openfake_decontaminated_paired_frozen_unscored",
        "role": "FINAL_AI_COMPONENT_PENDING_COMPLETE_E49",
        "identity_contract_sha256": IDENTITY_CONTRACT_SHA256,
        "download_receipt_sha256": hashlib.sha256(receipt_raw).hexdigest(),
        "counts": {
            "candidates": len(bound), "decoded": len(decoded), "decode_failures": len(failures),
            "identity_exclusions": len(reasons), "parents": len(parents),
            "observations": len(observations), "by_model": dict(sorted(by_model.items())),
            "selected_formats": dict(sorted(formats.items())),
        },
        "geometry_by_model": geometry,
        "decode_failures": failures,
        "identity_exclusion_reasons": dict(sorted(reasons.items())),
        "protected_role_manifests": protected_sources,
        "parents": sorted(parents, key=lambda row: (row["source"], row["rank"])),
        "rows": sorted(observations, key=lambda row: (row["condition"], row["source"], row["rank"])),
        "model_scores_created": 0,
        "boundary": "OpenFake decode/decontamination and deterministic Q75 pairing only; no detector access.",
    }
    raw = _write_atomic(MANIFEST, payload)
    evidence = {
        "schema_version": 1, "state": payload["state"], "role": payload["role"],
        "counts": payload["counts"], "geometry_by_model": geometry,
        "identity_contract_sha256": IDENTITY_CONTRACT_SHA256,
        "download_receipt_sha256": payload["download_receipt_sha256"],
        "protected_role_manifest_count": len(protected_sources),
        "manifest_bytes": len(raw), "manifest_sha256": hashlib.sha256(raw).hexdigest(),
        "model_scores_created": 0,
    }
    _write_atomic(EVIDENCE, evidence)
    return evidence


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze",))
    parser.parse_args(argv)
    print(json.dumps(freeze_manifest(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
