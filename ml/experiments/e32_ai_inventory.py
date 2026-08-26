"""E32/C2a — read-only inventory of modern AI holdings on the external disk."""

from __future__ import annotations

import argparse
import json
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

import pyarrow.parquet as pq

from pixelproof.project_paths import ML_ROOT


REGISTRY_PATH = ML_ROOT / "e32_ai_sources.json"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
SIDECAR_SUFFIXES = {".txt", ".json"}
APPLEDOUBLE = "._"


def registry() -> list[Mapping[str, Any]]:
    payload = json.loads(REGISTRY_PATH.read_text())
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported E32 AI registry schema")
    sources = payload.get("sources", [])
    ids = [source["id"] for source in sources]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate E32 AI source id")
    return sources


def real_files(folder: Path) -> list[Path]:
    return sorted(
        path
        for path in folder.rglob("*")
        if path.is_file()
        and not path.name.startswith(APPLEDOUBLE)
        and ".cache" not in path.parts
    )


def _parquet_inventory(files: list[Path]) -> dict[str, Any]:
    parquets = [path for path in files if path.suffix.lower() == ".parquet"]
    rows = 0
    schemas: Counter[tuple[str, ...]] = Counter()
    for path in parquets:
        parquet = pq.ParquetFile(path)
        rows += parquet.metadata.num_rows
        schemas[tuple(parquet.schema_arrow.names)] += 1
    return {
        "parquet_files": len(parquets),
        "row_count": rows,
        "schemas": [
            {"columns": list(columns), "file_count": count}
            for columns, count in sorted(schemas.items())
        ],
    }


def _loose_inventory(files: list[Path]) -> dict[str, Any]:
    images = [path for path in files if path.suffix.lower() in IMAGE_SUFFIXES]
    sidecars = [path for path in files if path.suffix.lower() in SIDECAR_SUFFIXES]
    image_stems = {path.stem for path in images}
    sidecar_stems = {path.stem for path in sidecars}
    return {
        "loose_image_count": len(images),
        "sidecar_count": len(sidecars),
        "image_with_matching_sidecar_count": len(image_stems & sidecar_stems),
        "image_formats": dict(Counter(path.suffix.lower() for path in images)),
    }


def _zip_inventory(files: list[Path]) -> dict[str, Any]:
    archives = [path for path in files if path.suffix.lower() == ".zip"]
    image_members = 0
    archive_rows = []
    for path in archives:
        with zipfile.ZipFile(path) as archive:
            members = [
                name
                for name in archive.namelist()
                if not Path(name).name.startswith(APPLEDOUBLE)
                and Path(name).suffix.lower() in IMAGE_SUFFIXES
            ]
        image_members += len(members)
        archive_rows.append(
            {
                "name": path.name,
                "bytes": path.stat().st_size,
                "image_members": len(members),
            }
        )
    return {"zip_files": len(archives), "zip_image_members": image_members, "archives": archive_rows}


def audit_source(root: Path, spec: Mapping[str, Any]) -> dict[str, Any]:
    folder = root / spec["dirname"]
    base = {
        key: spec[key]
        for key in (
            "id",
            "repo_id",
            "revision",
            "family",
            "license",
            "provenance",
            "admission_state",
            "counts_toward_verified_modern_families",
        )
    }
    if not folder.is_dir():
        return {**base, "state": "missing"}
    files = real_files(folder)
    result = {
        **base,
        "state": "present",
        "physical_file_count": len(files),
        "physical_bytes": sum(path.stat().st_size for path in files),
        "readme_present": (folder / "README.md").is_file(),
    }
    storage = spec["storage"]
    if storage == "parquet":
        result.update(_parquet_inventory(files))
    elif storage.startswith("loose_images"):
        result.update(_loose_inventory(files))
    elif storage == "zip":
        result.update(_zip_inventory(files))
    else:
        raise ValueError(f"unsupported storage type {storage!r}")
    return result


def build_report(root: Path) -> dict[str, Any]:
    if not root.is_dir():
        raise FileNotFoundError(f"dataset root is unavailable: {root}")
    sources = [audit_source(root, spec) for spec in registry()]
    verified_families = sorted(
        {
            source["family"]
            for source in sources
            if source["state"] == "present"
            and source["counts_toward_verified_modern_families"]
        }
    )
    return {
        "schema_version": 1,
        "experiment": "E32/C2a",
        "audit_mode": "complete_physical_metadata_no_image_decode",
        "source_root_name": root.name,
        "label_invariant": {"real": 0, "ai": 1},
        "sources": sources,
        "verified_admissible_modern_families": verified_families,
        "verified_admissible_modern_family_count": len(verified_families),
        "required_modern_family_count": 5,
        "family_gap": max(0, 5 - len(verified_families)),
        "decision": (
            "research_and_freeze_at_least_two_additional_licensed_families"
            if len(verified_families) < 5
            else "proceed_to_byte_and_duplicate_audit"
        ),
        "boundaries": [
            "A repository or folder name is not sufficient generator provenance.",
            "A missing dataset licence is not inferred from a model or code licence.",
            "Protected test-only sources cannot repair the training-family gap.",
            "Counts are inventory evidence, not decoded, decontaminated or selected parents.",
        ],
    }


def write_report(report: Mapping[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".part")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    temporary.replace(output)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_report(args.root)
    if args.output:
        write_report(report, args.output)
        print(f"wrote {args.output}")
    else:
        print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
