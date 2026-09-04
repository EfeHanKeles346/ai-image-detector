"""Realize the local StyleGAN2 component of E49 without loading a detector."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
from io import BytesIO
import json
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence

from PIL import Image, ImageOps
import pyarrow.parquet as pq

from experiments.e48_manifest import _protected_role_hashes
from experiments.e49_acquisition import (
    AIGC_GENERATOR_CODE,
    AIGC_LOCAL_ROOT,
    AIGC_RESERVE,
    AIGC_REVISION,
    AIGC_TARGET,
    OPEN_COMPONENTS_V2_CONTRACT,
)
from experiments.e49_dotting import MANIFEST as DOTTING_MANIFEST
from pixelproof.data_contract import dhash_image
from pixelproof.project_paths import DATA_ROOT, ML_ROOT


ROOT = DATA_ROOT / "e49" / "open_components_v2"
STYLEGAN_ROOT = ROOT / "stylegan2_reserve"
MANIFEST = ROOT / "stylegan2_manifest_unscored.json"
EVIDENCE = ML_ROOT.parent / "evidence" / "e49_stylegan2_manifest.json"
OPEN_CONTRACT_SHA256 = "1d4e184c27cb87cf832045a23b6966f382673c3bcd8342a900c07130bd9182aa"
MAX_PIXELS = 50_000_000
ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP"}


def _write(path: Path, value: Any) -> bytes:
    raw = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_bytes(raw)
    temporary.replace(path)
    return raw


def _decode(raw: bytes, identity: str) -> dict[str, Any]:
    with Image.open(BytesIO(raw)) as opened:
        opened.verify()
    with Image.open(BytesIO(raw)) as opened:
        width, height = opened.size
        if width <= 0 or height <= 0 or width * height > MAX_PIXELS:
            raise ValueError(f"E49 StyleGAN2 unsafe geometry: {identity}")
        opened.load()
        decoded_format = str(opened.format or "UNKNOWN").upper()
        if decoded_format not in ALLOWED_FORMATS:
            raise ValueError(f"E49 StyleGAN2 unsupported format: {identity}")
        rgb = ImageOps.exif_transpose(opened).convert("RGB")
        return {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest(),
                "dhash": dhash_image(rgb), "format": decoded_format,
                "width": width, "height": height, "mode": str(opened.mode)}


def select_clean(
    rows: Sequence[Mapping[str, Any]], prior_exact: set[str], prior_dhash: set[str],
    *, target: int = AIGC_TARGET,
) -> tuple[list[dict[str, Any]], dict[str, list[str]]]:
    """Select by prebound rank after score-blind protected/internal identity audit."""
    selected: list[dict[str, Any]] = []
    reasons: dict[str, list[str]] = {}
    seen_exact: dict[str, str] = {}
    seen_dhash: dict[str, str] = {}
    for source in sorted(rows, key=lambda row: (str(row["rank"]), str(row["identity"]))):
        row = dict(source)
        identity = str(row["identity"])
        row_reasons = []
        if row["sha256"] in prior_exact:
            row_reasons.append("protected_exact_overlap")
        if row["dhash"] in prior_dhash:
            row_reasons.append("protected_dhash_overlap")
        if row["sha256"] in seen_exact:
            row_reasons.append(f"internal_exact_duplicate_of:{seen_exact[row['sha256']]}")
        else:
            seen_exact[row["sha256"]] = identity
        if row["dhash"] in seen_dhash:
            row_reasons.append(f"internal_dhash_duplicate_of:{seen_dhash[row['dhash']]}")
        else:
            seen_dhash[row["dhash"]] = identity
        if row_reasons:
            reasons[identity] = row_reasons
        elif len(selected) < target:
            selected.append(row)
    if len(selected) != target:
        raise ValueError(f"E49 StyleGAN2 clean target unavailable: {len(selected)}/{target}")
    return selected, dict(sorted(reasons.items()))


def _contract_rows() -> list[dict[str, Any]]:
    raw = OPEN_COMPONENTS_V2_CONTRACT.read_bytes()
    if hashlib.sha256(raw).hexdigest() != OPEN_CONTRACT_SHA256:
        raise ValueError("E49 open-components V2 contract changed")
    payload = json.loads(raw)
    rows = payload.get("aigc", {}).get("rows", [])
    if (payload.get("state") != "e49_open_components_v2_frozen_untransferred_unscored"
            or payload.get("model_scores_created") != 0 or payload.get("new_image_bytes_downloaded") != 0
            or payload.get("aigc", {}).get("revision") != AIGC_REVISION or len(rows) != AIGC_RESERVE):
        raise ValueError("E49 StyleGAN2 source boundary changed")
    return rows


def _protected() -> tuple[set[str], set[str], list[dict[str, str]]]:
    exact, dhashes, sources = _protected_role_hashes()
    if not DOTTING_MANIFEST.is_file():
        raise FileNotFoundError("E49-D1 diagnostic manifest missing from protected roles")
    raw = DOTTING_MANIFEST.read_bytes()
    for row in json.loads(raw).get("rows", []):
        if row.get("sha256"):
            exact.add(str(row["sha256"]))
        if row.get("dhash"):
            dhashes.add(str(row["dhash"]))
    sources.append({"path": str(DOTTING_MANIFEST), "sha256": hashlib.sha256(raw).hexdigest()})
    return exact, dhashes, sources


def _row_group_for(pf: pq.ParquetFile, index: int) -> tuple[int, int]:
    start = 0
    for group in range(pf.num_row_groups):
        count = pf.metadata.row_group(group).num_rows
        if start <= index < start + count:
            return group, index - start
        start += count
    raise IndexError(f"E49 StyleGAN2 row outside shard: {index}")


def extract_rows(rows: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    by_shard: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_shard[str(row["shard"])].append(row)
    decoded: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    STYLEGAN_ROOT.mkdir(parents=True, exist_ok=True)
    for shard, shard_rows in sorted(by_shard.items()):
        path = AIGC_LOCAL_ROOT / "data" / shard
        pf = pq.ParquetFile(path)
        locations: dict[int, list[tuple[Mapping[str, Any], int]]] = defaultdict(list)
        for row in shard_rows:
            group, offset = _row_group_for(pf, int(row["row_index"]))
            locations[group].append((row, offset))
        for group, targets in sorted(locations.items()):
            table = pf.read_row_group(group, columns=["image", "label", "generator"])
            for row, offset in targets:
                identity = str(row["identity"])
                try:
                    value = table.slice(offset, 1).to_pylist()[0]
                    if int(value["label"]) != 1 or int(value["generator"]) != AIGC_GENERATOR_CODE:
                        raise ValueError("label/generator differs from frozen coordinate")
                    image = value.get("image") or {}
                    raw = image.get("bytes")
                    source_path = str(image.get("path") or "")
                    suffix = PurePosixPath(source_path).suffix.lower()
                    if not isinstance(raw, bytes) or suffix not in {".jpg", ".jpeg", ".png", ".webp"}:
                        raise ValueError("image payload/path is invalid")
                    found = _decode(raw, identity)
                    destination = STYLEGAN_ROOT / f"{Path(shard).stem}-{int(row['row_index']):06d}{suffix}"
                    temporary = destination.with_suffix(destination.suffix + ".part")
                    temporary.write_bytes(raw)
                    temporary.replace(destination)
                    decoded.append({**row, **found, "source_image_path": source_path,
                                    "path": str(destination), "status": "unscored"})
                except (OSError, ValueError, IndexError) as error:
                    failures.append({"identity": identity, "error": f"{type(error).__name__}: {error}"})
    return decoded, failures


def realize_stylegan2() -> dict[str, Any]:
    if MANIFEST.exists() or EVIDENCE.exists():
        raise FileExistsError("E49 StyleGAN2 manifest already exists")
    contract_rows = _contract_rows()
    decoded, failures = extract_rows(contract_rows)
    exact, dhashes, protected_sources = _protected()
    selected, reasons = select_clean(decoded, exact, dhashes)
    payload = {
        "schema_version": 1, "state": "e49_stylegan2_frozen_unscored",
        "role": "FINAL_AI_COMPONENT_PENDING_COMPLETE_E49",
        "open_components_v2_sha256": OPEN_CONTRACT_SHA256,
        "counts": {"reserve": len(contract_rows), "decoded": len(decoded),
                   "decode_failures": len(failures), "identity_exclusions": len(reasons),
                   "selected": len(selected)},
        "decode_failures": failures, "identity_exclusion_reasons": reasons,
        "protected_role_manifests": protected_sources,
        "rows": selected, "model_scores_created": 0,
        "boundary": "Local extraction/decontamination only; no detector access or E49 metric.",
    }
    raw = _write(MANIFEST, payload)
    evidence = {
        "schema_version": 1, "state": payload["state"], "role": payload["role"],
        "counts": payload["counts"], "open_components_v2_sha256": OPEN_CONTRACT_SHA256,
        "protected_role_manifest_count": len(protected_sources), "manifest_bytes": len(raw),
        "manifest_sha256": hashlib.sha256(raw).hexdigest(), "model_scores_created": 0,
    }
    _write(EVIDENCE, evidence)
    return evidence


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("realize-stylegan2",))
    parser.parse_args(argv)
    print(json.dumps(realize_stylegan2(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
