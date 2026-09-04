"""Decontaminate and pair the frozen E49 Commons REAL reserve without loading a model."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
from io import BytesIO
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from PIL import Image

from experiments.e48_manifest import _protected_role_hashes
from experiments.e49_acquisition import COMMONS_CATEGORIES, COMMONS_TARGET_PER_DEVICE
from experiments.e49_commons_download import CONTRACT_SHA256, EXPECTED_BYTES, EXPECTED_FILES, RECEIPT
from experiments.e49_dotting import MANIFEST as DOTTING_MANIFEST, social_q75_bytes
from experiments.e49_open_realize import MANIFEST as STYLEGAN2_MANIFEST
from experiments.e49_openfake_realize import MANIFEST as OPENFAKE_MANIFEST
from pixelproof.data_contract import dhash_image
from pixelproof.project_paths import DATA_ROOT, ML_ROOT


ROOT = DATA_ROOT / "e49" / "open_components_v2"
PAIRED_ROOT = ROOT / "paired" / "commons_social_q75"
MANIFEST = ROOT / "commons_manifest_unscored.json"
EVIDENCE = ML_ROOT.parent / "evidence" / "e49_commons_manifest.json"


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def _write_atomic(path: Path, value: Any) -> bytes:
    raw = _json_bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_bytes(raw)
    temporary.replace(path)
    return raw


def _decode_q75(raw: bytes) -> dict[str, Any]:
    with Image.open(BytesIO(raw)) as opened:
        opened.load()
        return {
            "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest(),
            "dhash": dhash_image(opened.convert("RGB")), "format": str(opened.format).upper(),
            "width": opened.width, "height": opened.height, "mode": str(opened.mode),
        }


def _normalized(value: Any) -> str:
    return "".join(character for character in str(value).lower() if character.isalnum())


def device_evidence(row: Mapping[str, Any]) -> str:
    """Classify EXIF device evidence without rejecting category-only originals."""
    source = str(row["source"])
    make = _normalized(row.get("exif_make", ""))
    model = _normalized(row.get("exif_model", ""))
    if not make and not model:
        return "category_only_missing_exif"
    valid = {
        "Canon EOS R5": model == "canoneosr5" and make.startswith("canon"),
        "Google Pixel 7 Pro": model == "pixel7pro" and make == "google",
        "Google Pixel 8 Pro": (
            (model == "pixel8pro" and make == "google")
            or (model == "google" and make.startswith("pixel8pro"))
        ),
        "Nikon Z 8": model in {"nikonz8", "z8"} and make.startswith("nikon"),
        "Samsung Galaxy S23 Ultra": model in {"galaxys23ultra", "samsunggalaxys23ultra"}
                                    and make.startswith("samsung"),
        "Sony ILCE-7M4": model == "ilce7m4" and make == "sony",
        "iPhone 13 Pro": make == "apple" and (model == "iphone13pro" or model.startswith("iphone142")),
        "iPhone 14 Pro": make == "apple" and (model == "iphone14pro" or model.startswith("iphone152")),
        "iPhone 15 Pro": make == "apple" and model == "iphone15pro",
        "iPhone 15 Pro Max": make == "apple" and model == "iphone15promax",
    }
    if source not in valid:
        raise ValueError(f"E49 Commons unexpected source: {source}")
    return "exif_device_match" if valid[source] else "exif_device_mismatch"


def audit_originals(
    rows: Sequence[Mapping[str, Any]], prior_exact: set[str], prior_dhash: set[str],
) -> dict[str, list[str]]:
    """Return deterministic protected/internal exclusions for received originals."""
    reasons: dict[str, list[str]] = defaultdict(list)
    seen_exact: dict[str, str] = {}
    seen_dhash: dict[str, str] = {}
    for row in sorted(rows, key=lambda item: (str(item["rank"]), str(item["identity"]))):
        identity = str(row["identity"])
        exact, perceptual = str(row["sha256"]), str(row["dhash"])
        if device_evidence(row) == "exif_device_mismatch":
            reasons[identity].append("exif_device_mismatch")
        if exact in prior_exact:
            reasons[identity].append("protected_exact_overlap")
        if perceptual in prior_dhash:
            reasons[identity].append("protected_dhash_overlap")
        if exact in seen_exact:
            reasons[identity].append(f"internal_exact_duplicate_of:{seen_exact[exact]}")
        else:
            seen_exact[exact] = identity
        if perceptual in seen_dhash:
            reasons[identity].append(f"internal_dhash_duplicate_of:{seen_dhash[perceptual]}")
        else:
            seen_dhash[perceptual] = identity
    return dict(sorted(reasons.items()))


def _protected() -> tuple[set[str], set[str], list[dict[str, str]]]:
    exact, dhashes, sources = _protected_role_hashes()
    for path in (DOTTING_MANIFEST, STYLEGAN2_MANIFEST, OPENFAKE_MANIFEST):
        if not path.is_file():
            raise FileNotFoundError(f"E49 Commons protected component manifest missing: {path}")
        raw = path.read_bytes()
        for row in json.loads(raw).get("rows", []):
            if row.get("sha256"):
                exact.add(str(row["sha256"]))
            if row.get("dhash"):
                dhashes.add(str(row["dhash"]))
        sources.append({"path": str(path), "sha256": hashlib.sha256(raw).hexdigest()})
    return exact, dhashes, sources


def _receipt_rows() -> tuple[list[dict[str, Any]], bytes]:
    raw = RECEIPT.read_bytes()
    payload = json.loads(raw)
    rows = payload.get("rows") or []
    if (
        payload.get("state") != "e49_commons_downloaded_decoded_unscored"
        or payload.get("source_contract_sha256") != CONTRACT_SHA256
        or payload.get("files") != EXPECTED_FILES
        or payload.get("bytes") != EXPECTED_BYTES
        or payload.get("model_scores_created") != 0
        or len(rows) != EXPECTED_FILES
    ):
        raise ValueError("E49 Commons download receipt changed")
    return rows, raw


def freeze_manifest() -> dict[str, Any]:
    if MANIFEST.exists() or EVIDENCE.exists():
        raise FileExistsError("E49 Commons manifest already exists")
    rows, receipt_raw = _receipt_rows()
    prior_exact, prior_dhash, protected_sources = _protected()
    reasons = audit_originals(rows, prior_exact, prior_dhash)
    parents: list[dict[str, Any]] = []
    observations: list[dict[str, Any]] = []
    child_exact: dict[str, str] = {}
    child_dhash: dict[str, str] = {}
    for category in COMMONS_CATEGORIES:
        source = category.removeprefix("Category:Taken with ")
        candidates = sorted(
            (row for row in rows if row["source"] == source and str(row["identity"]) not in reasons),
            key=lambda row: (str(row["rank"]), str(row["identity"])),
        )
        accepted = 0
        for row in candidates:
            parent_id = str(row["identity"])
            path = Path(str(row["path"]))
            child_raw = social_q75_bytes(path)
            child = _decode_q75(child_raw)
            child_id = parent_id + ":social_q75"
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
                reasons[parent_id] = child_reasons
                continue
            child_exact[child["sha256"]] = child_id
            child_dhash[child["dhash"]] = child_id
            destination = PAIRED_ROOT / source.lower().replace(" ", "-") / f"{row['pageid']}.jpg"
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_suffix(".jpg.part")
            temporary.write_bytes(child_raw)
            temporary.replace(destination)
            parent = {
                "parent_id": parent_id, "label": 0, "source": source,
                "rank": row["rank"], "pageid": int(row["pageid"]), "uploader": row["uploader"],
                "device_evidence": device_evidence(row),
            }
            parents.append(parent)
            observations.extend([
                {**row, "record_id": parent_id + ":publisher_original", "parent_id": parent_id,
                 "condition": "publisher_original", "status": "unscored"},
                {**parent, **child, "record_id": child_id, "condition": "social_q75",
                 "path": str(destination), "status": "unscored"},
            ])
            accepted += 1
            if accepted == COMMONS_TARGET_PER_DEVICE:
                break
        if accepted != COMMONS_TARGET_PER_DEVICE:
            raise ValueError(f"E49 Commons clean target unavailable for {source}: {accepted}")
    by_source = Counter(row["source"] for row in parents)
    expected_counts = Counter({category.removeprefix("Category:Taken with "): COMMONS_TARGET_PER_DEVICE
                               for category in COMMONS_CATEGORIES})
    if by_source != expected_counts or len(parents) != 1_000 or len(observations) != 2_000:
        raise ValueError("E49 Commons final REAL target changed")
    payload = {
        "schema_version": 1, "state": "e49_commons_decontaminated_paired_frozen_unscored",
        "role": "FINAL_REAL_COMPONENT_PENDING_COMPLETE_E49",
        "download_receipt_sha256": hashlib.sha256(receipt_raw).hexdigest(),
        "counts": {
            "candidates": len(rows), "identity_exclusions": len(reasons), "parents": len(parents),
            "observations": len(observations), "by_source": dict(sorted(by_source.items())),
            "original_exif_make_present": sum(bool(row["exif_make"]) for row in rows),
            "original_exif_model_present": sum(bool(row["exif_model"]) for row in rows),
            "selected_device_evidence": dict(sorted(Counter(
                row["device_evidence"] for row in parents
            ).items())),
        },
        "identity_exclusion_reasons": dict(sorted(reasons.items())),
        "protected_role_manifests": protected_sources,
        "parents": sorted(parents, key=lambda row: (row["source"], row["rank"])),
        "rows": sorted(observations, key=lambda row: (row["condition"], row["source"], row["rank"])),
        "model_scores_created": 0,
        "boundary": "Commons identity audit and fixed Q75 pairing only; no detector or metric access.",
    }
    raw = _write_atomic(MANIFEST, payload)
    evidence = {
        "schema_version": 1, "state": payload["state"], "role": payload["role"],
        "download_receipt_sha256": payload["download_receipt_sha256"], "counts": payload["counts"],
        "protected_role_manifest_count": len(protected_sources), "manifest_bytes": len(raw),
        "manifest_sha256": hashlib.sha256(raw).hexdigest(), "model_scores_created": 0,
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
