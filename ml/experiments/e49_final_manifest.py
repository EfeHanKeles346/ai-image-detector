"""Assemble the complete paired E49-C final manifest before any detector access."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
from io import BytesIO
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from PIL import Image

from experiments.e48_manifest import _protected_role_hashes
from experiments.e49_commons_realize import MANIFEST as COMMONS_MANIFEST
from experiments.e49_dotting import MANIFEST as DOTTING_MANIFEST, social_q75_bytes
from experiments.e49_evaluation import SOURCE_COUNTS, validate_paired_final
from experiments.e49_open_realize import MANIFEST as STYLEGAN2_MANIFEST
from experiments.e49_openfake_realize import MANIFEST as OPENFAKE_MANIFEST
from pixelproof.data_contract import dhash_image
from pixelproof.project_paths import DATA_ROOT, ML_ROOT


ROOT = DATA_ROOT / "e49" / "final"
STYLEGAN_Q75_ROOT = ROOT / "paired" / "stylegan2_social_q75"
MANIFEST = ROOT / "manifest_unscored.json"
EVIDENCE = ML_ROOT.parent / "evidence" / "e49_final_manifest.json"


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
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_component(path: Path, state: str, parents: int, observations: int) -> tuple[dict[str, Any], bytes]:
    raw = path.read_bytes()
    payload = json.loads(raw)
    counts = payload.get("counts") or {}
    if (
        payload.get("state") != state or payload.get("model_scores_created") != 0
        or counts.get("parents") != parents or counts.get("observations") != observations
        or len(payload.get("rows") or []) != observations
    ):
        raise ValueError(f"E49 final component boundary changed: {path}")
    return payload, raw


def stylegan_pairs(
    rows: Sequence[Mapping[str, Any]], prior_exact: set[str], prior_dhash: set[str],
) -> list[dict[str, Any]]:
    """Create deterministic StyleGAN2 Q75 children; a collision fails instead of replacing a parent."""
    observations: list[dict[str, Any]] = []
    child_exact: dict[str, str] = {}
    child_dhash: dict[str, str] = {}
    for row in sorted(rows, key=lambda item: (str(item["rank"]), str(item["identity"]))):
        parent_id = "stylegan2:" + str(row["identity"])
        path = Path(str(row["path"]))
        if _digest(path) != row["sha256"]:
            raise ValueError(f"E49 StyleGAN2 original changed: {parent_id}")
        child_raw = social_q75_bytes(path)
        with Image.open(BytesIO(child_raw)) as opened:
            opened.load()
            child = {
                "bytes": len(child_raw), "sha256": hashlib.sha256(child_raw).hexdigest(),
                "dhash": dhash_image(opened.convert("RGB")), "format": str(opened.format).upper(),
                "width": opened.width, "height": opened.height, "mode": str(opened.mode),
            }
        child_id = parent_id + ":social_q75"
        if child["sha256"] in prior_exact or child["dhash"] in prior_dhash:
            raise ValueError(f"E49 StyleGAN2 social child overlaps protected evidence: {parent_id}")
        if child["sha256"] in child_exact or child["dhash"] in child_dhash:
            raise ValueError(f"E49 StyleGAN2 social child duplicates another child: {parent_id}")
        child_exact[child["sha256"]] = child_id
        child_dhash[child["dhash"]] = child_id
        destination = STYLEGAN_Q75_ROOT / f"{hashlib.sha256(parent_id.encode()).hexdigest()}.jpg"
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(".jpg.part")
        temporary.write_bytes(child_raw)
        temporary.replace(destination)
        common = {"parent_id": parent_id, "label": 1, "source": "StyleGAN2", "rank": row["rank"]}
        observations.extend([
            {**row, **common, "record_id": parent_id + ":publisher_original",
             "condition": "publisher_original", "status": "unscored"},
            {**common, **child, "record_id": child_id, "condition": "social_q75",
             "path": str(destination), "status": "unscored"},
        ])
    return observations


def validate_payloads(rows: Sequence[Mapping[str, Any]]) -> None:
    """Reproduce every final observation's stored SHA and decoded dimensions."""
    for row in rows:
        path = Path(str(row.get("path", "")))
        if not path.is_file() or _digest(path) != row.get("sha256"):
            raise ValueError(f"E49 final payload changed: {row.get('record_id')}")
        with Image.open(path) as opened:
            if (opened.width, opened.height) != (int(row["width"]), int(row["height"])):
                raise ValueError(f"E49 final geometry changed: {row.get('record_id')}")


def assemble() -> dict[str, Any]:
    if MANIFEST.exists() or EVIDENCE.exists():
        raise FileExistsError("E49 final manifest already exists")
    commons, commons_raw = _load_component(
        COMMONS_MANIFEST, "e49_commons_decontaminated_paired_frozen_unscored", 1_000, 2_000,
    )
    openfake, openfake_raw = _load_component(
        OPENFAKE_MANIFEST, "e49_c_openfake_decontaminated_paired_frozen_unscored", 800, 1_600,
    )
    style_raw = STYLEGAN2_MANIFEST.read_bytes()
    style = json.loads(style_raw)
    style_rows = style.get("rows") or []
    if (
        style.get("state") != "e49_stylegan2_frozen_unscored"
        or style.get("model_scores_created") != 0 or len(style_rows) != 200
    ):
        raise ValueError("E49 StyleGAN2 component boundary changed")
    prior_exact, prior_dhash, protected_sources = _protected_role_hashes()
    for path, payload in ((DOTTING_MANIFEST, json.loads(DOTTING_MANIFEST.read_text())),
                          (COMMONS_MANIFEST, commons), (OPENFAKE_MANIFEST, openfake)):
        raw = path.read_bytes()
        for row in payload.get("rows", []):
            if row.get("sha256"):
                prior_exact.add(str(row["sha256"]))
            if row.get("dhash"):
                prior_dhash.add(str(row["dhash"]))
        protected_sources.append({"path": str(path), "sha256": hashlib.sha256(raw).hexdigest()})
    style_observations = stylegan_pairs(style_rows, prior_exact, prior_dhash)
    rows = list(commons["rows"]) + list(openfake["rows"]) + style_observations
    rows.sort(key=lambda row: (str(row["condition"]), int(row["label"]), str(row["source"]), str(row["rank"])))
    validate_paired_final(rows)
    validate_payloads(rows)
    formats = {
        condition: dict(sorted(Counter(
            str(row["format"]) for row in rows if row["condition"] == condition
        ).items())) for condition in ("publisher_original", "social_q75")
    }
    original_geometry = {
        source: {
            "min_width": min(int(row["width"]) for row in rows
                             if row["condition"] == "publisher_original" and row["source"] == source),
            "max_width": max(int(row["width"]) for row in rows
                             if row["condition"] == "publisher_original" and row["source"] == source),
            "min_height": min(int(row["height"]) for row in rows
                              if row["condition"] == "publisher_original" and row["source"] == source),
            "max_height": max(int(row["height"]) for row in rows
                              if row["condition"] == "publisher_original" and row["source"] == source),
        }
        for source in SOURCE_COUNTS
    }
    component_hashes = {
        "commons": hashlib.sha256(commons_raw).hexdigest(),
        "openfake": hashlib.sha256(openfake_raw).hexdigest(),
        "stylegan2": hashlib.sha256(style_raw).hexdigest(),
    }
    payload = {
        "schema_version": 1, "state": "e49_c_comprehensive_final_frozen_unscored",
        "role": "MODULE_1_ONE_SHOT_COMPREHENSIVE_FINAL",
        "component_manifest_sha256": component_hashes,
        "counts": {"parents": 2_000, "observations": 4_000,
                   "by_source": SOURCE_COUNTS, "conditions": 2},
        "format_audit": formats, "publisher_original_geometry_by_source": original_geometry,
        "protected_role_manifests": protected_sources,
        "rows": rows, "model_scores_created": 0, "metrics_opened": 0,
        "boundary": "Complete paired E49-C final frozen before E43-S access; no row/source repair after scoring.",
    }
    raw = _write_atomic(MANIFEST, payload)
    evidence = {
        "schema_version": 1, "state": payload["state"], "role": payload["role"],
        "component_manifest_sha256": component_hashes, "counts": payload["counts"],
        "format_audit": formats, "publisher_original_geometry_by_source": original_geometry,
        "protected_role_manifest_count": len(protected_sources), "manifest_bytes": len(raw),
        "manifest_sha256": hashlib.sha256(raw).hexdigest(), "model_scores_created": 0,
        "metrics_opened": 0,
    }
    _write_atomic(EVIDENCE, evidence)
    return evidence


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("assemble",))
    parser.parse_args(argv)
    print(json.dumps(assemble(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
