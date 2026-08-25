"""E31/B2 — freeze and realize a source-aware TRAIN-v2 contract.

``freeze`` reads metadata only and commits exact source rows before image bytes are opened.
``realize`` later verifies that frozen selection, hashes every selected image against protected
evaluation content, and writes one deterministic native 128 px tile per row under ignored data.
The external source root is always read-only.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

from pixelproof.build_tile_dataset import pick_tile
from pixelproof.data_contract import dhash_image


SEED = 20260825
CALIBRATION_FOLD = 0
N_FOLDS = 5
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class SourceContract:
    source_id: str
    dirname: str
    image_col: str
    label_col: str | None
    label_map: Mapping[int, str] | None
    generator_col: str | None
    implicit_label: str | None
    selection: str
    target: int


SOURCES = (
    SourceContract(
        "communityforensics-small",
        "OwensLab__CommunityForensics-Small",
        "image_data",
        "label",
        {0: "real", 1: "ai"},
        "model_name",
        None,
        "cf_ai_per_generator",
        8,
    ),
    SourceContract(
        "communityforensics-small",
        "OwensLab__CommunityForensics-Small",
        "image_data",
        "label",
        {0: "real", 1: "ai"},
        "model_name",
        None,
        "cf_real_total",
        2400,
    ),
    SourceContract(
        "ai-vs-real-balanced",
        "theminji__AI-vs-Real-balanced",
        "image",
        "label",
        {0: "ai", 1: "real"},
        None,
        None,
        "balanced_ai_total",
        2000,
    ),
    SourceContract(
        "ai-vs-real-balanced",
        "theminji__AI-vs-Real-balanced",
        "image",
        "label",
        {0: "ai", 1: "real"},
        None,
        None,
        "balanced_real_total",
        3250,
    ),
    SourceContract(
        "flux-1-dev",
        "ash12321__flux-1-dev-generated-10k",
        "image",
        None,
        None,
        None,
        "ai",
        "implicit_total",
        500,
    ),
    SourceContract(
        "nano-banana",
        "bitmind__nano-banana",
        "image",
        None,
        None,
        None,
        "ai",
        "implicit_total",
        500,
    ),
    SourceContract(
        "nano-banana-pro",
        "kaupane__nano-banana-pro-gen",
        "image",
        None,
        None,
        None,
        "ai",
        "implicit_total",
        250,
    ),
)


@dataclass(frozen=True)
class Candidate:
    source_id: str
    dirname: str
    shard: str
    row_index: int
    image_col: str
    label: str
    generator: str | None
    group_id: str

    @property
    def key(self) -> str:
        return f"{self.source_id}:{self.shard}:{self.row_index}"


def stable_digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def real_parquets(folder: Path) -> list[Path]:
    return sorted(
        path for path in folder.rglob("*.parquet")
        if path.is_file() and not path.name.startswith("._") and ".cache" not in path.parts
    )


def project_label(contract: SourceContract, raw: Any) -> str | None:
    if contract.label_map is None:
        return contract.implicit_label
    try:
        return contract.label_map.get(int(raw))
    except (TypeError, ValueError):
        return None


def source_fingerprint(root: Path, contract: SourceContract) -> str:
    rows = []
    folder = root / contract.dirname
    for path in real_parquets(folder):
        parquet = pq.ParquetFile(path)
        rows.append(
            {
                "path": str(path.relative_to(root)),
                "bytes": path.stat().st_size,
                "rows": parquet.metadata.num_rows,
                "columns": parquet.schema_arrow.names,
            }
        )
    if not rows:
        raise FileNotFoundError(f"no real Parquet shards under {folder}")
    return hashlib.sha256(
        json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def iter_candidates(root: Path, contract: SourceContract) -> Iterable[Candidate]:
    folder = root / contract.dirname
    for path in real_parquets(folder):
        parquet = pq.ParquetFile(path)
        names = parquet.schema_arrow.names
        required = [contract.image_col]
        if contract.label_col:
            required.append(contract.label_col)
        if contract.generator_col:
            required.append(contract.generator_col)
        missing = sorted(set(required) - set(names))
        if missing:
            raise ValueError(f"{path.name} is missing columns {missing}")
        metadata_columns = [name for name in (contract.label_col, contract.generator_col) if name]
        row_index = 0
        if metadata_columns:
            batches = parquet.iter_batches(columns=metadata_columns, batch_size=65_536)
            for batch in batches:
                values = batch.to_pydict()
                for offset in range(batch.num_rows):
                    raw = values[contract.label_col][offset] if contract.label_col else None
                    label = project_label(contract, raw)
                    generator = (
                        str(values[contract.generator_col][offset])
                        if contract.generator_col and label == "ai"
                        else None
                    )
                    if label == "ai" and generator:
                        group_id = f"{contract.source_id}:generator:{generator}"
                    else:
                        group_id = f"{contract.source_id}:shard:{path.name}"
                    yield Candidate(
                        contract.source_id,
                        contract.dirname,
                        str(path.relative_to(root)),
                        row_index,
                        contract.image_col,
                        label or "unmapped",
                        generator,
                        group_id,
                    )
                    row_index += 1
        else:
            label = project_label(contract, None)
            for row_index in range(parquet.metadata.num_rows):
                yield Candidate(
                    contract.source_id,
                    contract.dirname,
                    str(path.relative_to(root)),
                    row_index,
                    contract.image_col,
                    label or "unmapped",
                    contract.source_id if label == "ai" else None,
                    f"{contract.source_id}:shard:{path.name}",
                )


def evenly_select(candidates: Sequence[Candidate], target: int) -> list[Candidate]:
    """Round-robin deterministic rows across groups until ``target`` is filled."""
    groups: dict[str, list[Candidate]] = defaultdict(list)
    for candidate in candidates:
        groups[candidate.group_id].append(candidate)
    for values in groups.values():
        values.sort(key=lambda item: stable_digest(f"{SEED}:{item.key}"))
    selected: list[Candidate] = []
    position = 0
    ordered_groups = sorted(groups)
    while len(selected) < target:
        added = False
        for group in ordered_groups:
            values = groups[group]
            if position < len(values):
                selected.append(values[position])
                added = True
                if len(selected) == target:
                    break
        if not added:
            raise ValueError(f"selection requested {target} rows but only {len(selected)} exist")
        position += 1
    return selected


def select_contract_rows(root: Path, contract: SourceContract) -> list[Candidate]:
    candidates = list(iter_candidates(root, contract))
    if any(candidate.label not in {"real", "ai"} for candidate in candidates):
        raise ValueError(f"{contract.source_id} produced an unmapped label")
    if contract.selection == "cf_ai_per_generator":
        by_generator: dict[str, list[Candidate]] = defaultdict(list)
        for candidate in candidates:
            if candidate.label == "ai" and candidate.generator:
                by_generator[candidate.generator].append(candidate)
        if len(by_generator) != 300:
            raise ValueError(f"expected 300 CF AI generators, found {len(by_generator)}")
        selected = []
        for generator in sorted(by_generator):
            values = sorted(
                by_generator[generator], key=lambda item: stable_digest(f"{SEED}:{item.key}")
            )
            if len(values) < contract.target:
                raise ValueError(f"generator {generator!r} has only {len(values)} rows")
            selected.extend(values[: contract.target])
        return selected
    if contract.selection == "cf_real_total":
        return evenly_select([item for item in candidates if item.label == "real"], contract.target)
    if contract.selection == "balanced_ai_total":
        return evenly_select([item for item in candidates if item.label == "ai"], contract.target)
    if contract.selection == "balanced_real_total":
        return evenly_select([item for item in candidates if item.label == "real"], contract.target)
    if contract.selection == "implicit_total":
        return evenly_select(candidates, contract.target)
    raise ValueError(f"unknown selection rule {contract.selection!r}")


def group_fold(group_id: str) -> int:
    return int(stable_digest(f"fold:{SEED}:{group_id}")[:8], 16) % N_FOLDS


def assign_group_folds(candidates: Sequence[Candidate]) -> dict[str, int]:
    """Stratify whole groups over folds within each source.

    Pure modulo hashing left all seven FLUX shards outside CALIBRATION in the first pre-byte dry
    run. Stable rank + round-robin retains determinism and guarantees fold support for sources
    with at least five groups without ever splitting a group.
    """
    by_source: dict[str, set[str]] = defaultdict(set)
    for candidate in candidates:
        by_source[candidate.source_id].add(candidate.group_id)
    output: dict[str, int] = {}
    for source_id, groups in sorted(by_source.items()):
        ordered = sorted(groups, key=lambda group: stable_digest(f"fold:{SEED}:{source_id}:{group}"))
        if len(ordered) < N_FOLDS:
            raise ValueError(f"source {source_id!r} has only {len(ordered)} groups for {N_FOLDS} folds")
        for index, group in enumerate(ordered):
            output[group] = index % N_FOLDS
    return output


def freeze(root: Path) -> dict[str, Any]:
    if not root.is_dir():
        raise FileNotFoundError(f"dataset root is unavailable: {root}")
    selected: dict[str, Candidate] = {}
    fingerprints: dict[str, str] = {}
    for contract in SOURCES:
        fingerprint_key = f"{contract.source_id}:{contract.dirname}"
        fingerprints.setdefault(fingerprint_key, source_fingerprint(root, contract))
        for candidate in select_contract_rows(root, contract):
            if candidate.key in selected:
                raise ValueError(f"duplicate selected source row {candidate.key}")
            selected[candidate.key] = candidate

    fold_by_group = assign_group_folds(list(selected.values()))
    records = []
    for candidate in sorted(selected.values(), key=lambda item: item.key):
        fold = fold_by_group[candidate.group_id]
        role = "calibration" if fold == CALIBRATION_FOLD else "train"
        value = asdict(candidate)
        value.update(
            {
                "record_id": stable_digest(candidate.key)[:24],
                "fold": fold,
                "role": role,
            }
        )
        records.append(value)

    counts = Counter((record["role"], record["label"]) for record in records)
    if set(counts) != {
        ("train", "real"), ("train", "ai"), ("calibration", "real"), ("calibration", "ai")
    }:
        raise ValueError(f"TRAIN/CALIBRATION lacks class support: {dict(counts)}")
    group_roles: dict[str, set[str]] = defaultdict(set)
    for record in records:
        group_roles[record["group_id"]].add(record["role"])
    crossed = [group for group, roles in group_roles.items() if len(roles) != 1]
    if crossed:
        raise ValueError(f"groups crossed roles: {crossed[:3]}")
    source_roles: dict[str, set[str]] = defaultdict(set)
    for record in records:
        source_roles[record["source_id"]].add(record["role"])
    unsupported = [source for source, roles in source_roles.items() if roles != {"train", "calibration"}]
    if unsupported:
        raise ValueError(f"sources lack TRAIN/CALIBRATION support: {unsupported}")

    record_material = json.dumps(records, sort_keys=True, separators=(",", ":"))
    return {
        "schema_version": 1,
        "experiment": "E31/B2",
        "state": "frozen_selection_before_image_bytes",
        "selection_seed": SEED,
        "fold_count": N_FOLDS,
        "calibration_fold": CALIBRATION_FOLD,
        "source_root_name": root.name,
        "source_fingerprints": dict(sorted(fingerprints.items())),
        "selection_rules": [asdict(contract) for contract in SOURCES],
        "selection_sha256": hashlib.sha256(record_material.encode()).hexdigest(),
        "counts": {
            "total": len(records),
            "by_role_label": {
                f"{role}:{label}": count for (role, label), count in sorted(counts.items())
            },
            "by_source_label": {
                f"{source}:{label}": count
                for (source, label), count in sorted(
                    Counter((record["source_id"], record["label"]) for record in records).items()
                )
            },
            "ai_generators": len(
                {record["generator"] for record in records if record["generator"]}
            ),
            "groups": len(group_roles),
        },
        "records": records,
        "boundaries": [
            "No image byte was read to create this selection.",
            "Rows were selected from TRAIN candidates only; no test source is a selectable input.",
            "A group is wholly TRAIN or wholly CALIBRATION; E30 and owner-gallery rows are absent.",
            "Realization must reproduce every source fingerprint and selection SHA before decoding.",
        ],
    }


def write_json_atomic(payload: Mapping[str, Any], output: Path, source_root: Path) -> None:
    try:
        output.resolve().relative_to(source_root.resolve())
    except ValueError:
        pass
    else:
        raise ValueError("refusing to write inside the source dataset root")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".part")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(output)


def _validate_frozen(payload: Mapping[str, Any], root: Path) -> list[dict[str, Any]]:
    if payload.get("state") != "frozen_selection_before_image_bytes":
        raise ValueError("selection is not in the frozen pre-byte state")
    records = list(payload.get("records", []))
    material = json.dumps(records, sort_keys=True, separators=(",", ":"))
    if hashlib.sha256(material.encode()).hexdigest() != payload.get("selection_sha256"):
        raise ValueError("selection SHA-256 mismatch")
    for contract in SOURCES:
        key = f"{contract.source_id}:{contract.dirname}"
        if source_fingerprint(root, contract) != payload["source_fingerprints"][key]:
            raise ValueError(f"source fingerprint changed for {key}")
    return records


def _raw_image(value: Any) -> bytes | None:
    if isinstance(value, (bytes, bytearray)):
        return bytes(value)
    if isinstance(value, Mapping) and isinstance(value.get("bytes"), (bytes, bytearray)):
        return bytes(value["bytes"])
    return None


def _loose_protected(directories: Sequence[Path]) -> tuple[set[str], set[str], Counter[str]]:
    exact: set[str] = set()
    perceptual: set[str] = set()
    counts: Counter[str] = Counter()
    for directory in directories:
        if not directory.exists():
            continue
        for path in sorted(directory.rglob("*")):
            if (
                not path.is_file()
                or path.name.startswith("._")
                or path.suffix.lower() not in IMAGE_SUFFIXES
            ):
                continue
            raw = path.read_bytes()
            try:
                with Image.open(io.BytesIO(raw)) as image:
                    image.load()
                    perceptual.add(dhash_image(image))
            except Exception:
                counts[f"decode_failure:{directory.name}"] += 1
                continue
            exact.add(hashlib.sha256(raw).hexdigest())
            counts[directory.name] += 1
    return exact, perceptual, counts


def _e30_protected(repo_root: Path) -> tuple[set[str], set[str], int]:
    exact: set[str] = set()
    perceptual: set[str] = set()
    manifests = list((repo_root / "ml" / "data" / "e30").rglob("*manifest.json"))
    for path in manifests:
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        for row in payload.get("records", []):
            if row.get("sha256"):
                exact.add(row["sha256"])
            if row.get("dhash"):
                perceptual.add(row["dhash"])
    return exact, perceptual, len(manifests)


def realize(
    root: Path,
    selection_path: Path,
    tile_output: Path,
    protected_directories: Sequence[Path],
) -> dict[str, Any]:
    selection = json.loads(selection_path.read_text())
    records = _validate_frozen(selection, root)
    repo_root = Path(__file__).resolve().parents[2]
    exact, perceptual, e30_manifest_count = _e30_protected(repo_root)
    loose_exact, loose_perceptual, protected_counts = _loose_protected(protected_directories)
    exact |= loose_exact
    perceptual |= loose_perceptual

    wanted_by_shard: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
    for record in records:
        wanted_by_shard[record["shard"]][int(record["row_index"])] = record

    tiles: list[np.ndarray] = []
    labels: list[int] = []
    roles: list[str] = []
    sources: list[str] = []
    generators: list[str] = []
    record_ids: list[str] = []
    realized_rows: list[dict[str, Any]] = []
    exact_overlaps = 0
    dhash_overlaps = 0
    decode_failures = 0
    too_small_or_flat = 0

    for shard in sorted(wanted_by_shard):
        path = root / shard
        wanted = wanted_by_shard[shard]
        first = next(iter(wanted.values()))
        image_col = first["image_col"]
        parquet = pq.ParquetFile(path)
        row_index = 0
        found: set[int] = set()
        for batch in parquet.iter_batches(columns=[image_col], batch_size=64):
            for row in batch.to_pylist():
                record = wanted.get(row_index)
                if record is not None:
                    found.add(row_index)
                    raw = _raw_image(row[image_col])
                    if raw is None:
                        decode_failures += 1
                        row_index += 1
                        continue
                    try:
                        with Image.open(io.BytesIO(raw)) as opened:
                            opened.load()
                            image = opened.convert("RGB")
                    except Exception:
                        decode_failures += 1
                        row_index += 1
                        continue
                    raw_sha = hashlib.sha256(raw).hexdigest()
                    raw_dhash = dhash_image(image)
                    exact_hit = raw_sha in exact
                    dhash_hit = raw_dhash in perceptual
                    exact_overlaps += int(exact_hit)
                    dhash_overlaps += int(dhash_hit)
                    rng_seed = int(stable_digest(f"tile:{SEED}:{record['record_id']}")[:8], 16)
                    tile = pick_tile(image, np.random.RandomState(rng_seed))
                    if tile is None:
                        too_small_or_flat += 1
                        row_index += 1
                        continue
                    tile_sha = hashlib.sha256(tile.tobytes()).hexdigest()
                    realized = dict(record)
                    realized.update(
                        {
                            "raw_sha256": raw_sha,
                            "dhash": raw_dhash,
                            "tile_sha256": tile_sha,
                            "raw_bytes": len(raw),
                            "width": image.width,
                            "height": image.height,
                        }
                    )
                    realized_rows.append(realized)
                    tiles.append(tile)
                    labels.append(int(record["label"] == "ai"))
                    roles.append(record["role"])
                    sources.append(record["source_id"])
                    generators.append(record["generator"] or "")
                    record_ids.append(record["record_id"])
                row_index += 1
        missing = sorted(set(wanted) - found)
        if missing:
            raise ValueError(f"{shard} did not contain selected rows {missing[:3]}")

    if decode_failures or too_small_or_flat:
        raise RuntimeError(
            f"realization incomplete: decode_failures={decode_failures}, "
            f"too_small_or_flat={too_small_or_flat}"
        )
    if exact_overlaps or dhash_overlaps:
        raise RuntimeError(
            f"protected-content overlap: exact={exact_overlaps}, dhash={dhash_overlaps}"
        )
    if len(realized_rows) != len(records):
        raise RuntimeError(f"realized {len(realized_rows)} of {len(records)} selected rows")

    x = np.stack(tiles).astype(np.uint8)
    y = np.asarray(labels, dtype=np.int64)
    tile_output.parent.mkdir(parents=True, exist_ok=True)
    temporary = tile_output.with_suffix(".part.npz")
    np.savez_compressed(
        temporary,
        x=x,
        y=y,
        roles=np.asarray(roles),
        sources=np.asarray(sources),
        generators=np.asarray(generators),
        record_ids=np.asarray(record_ids),
    )
    temporary.replace(tile_output)

    row_material = json.dumps(realized_rows, sort_keys=True, separators=(",", ":"))
    return {
        "schema_version": 1,
        "experiment": "E31/B2",
        "state": "realized_train_v2",
        "selection_sha256": selection["selection_sha256"],
        "realized_manifest_sha256": hashlib.sha256(row_material.encode()).hexdigest(),
        "tile_archive_sha256": hashlib.sha256(tile_output.read_bytes()).hexdigest(),
        "tile_archive_bytes": tile_output.stat().st_size,
        "counts": {
            "selected": len(records),
            "realized": len(realized_rows),
            "real": int((y == 0).sum()),
            "ai": int((y == 1).sum()),
            "train": roles.count("train"),
            "calibration": roles.count("calibration"),
            "decode_failures": decode_failures,
            "too_small_or_flat": too_small_or_flat,
            "protected_exact_overlaps": exact_overlaps,
            "protected_dhash_overlaps": dhash_overlaps,
            "unique_raw_sha256": len({row["raw_sha256"] for row in realized_rows}),
            "unique_tile_sha256": len({row["tile_sha256"] for row in realized_rows}),
        },
        "protected_scope": {
            "e30_manifest_files": e30_manifest_count,
            "e30_and_loose_exact_hashes": len(exact),
            "e30_and_loose_dhashes": len(perceptual),
            "loose_directory_counts": dict(sorted(protected_counts.items())),
        },
        "by_role_label_source": {
            f"{role}:{label}:{source}": count
            for (role, label, source), count in sorted(
                Counter(
                    (row["role"], row["label"], row["source_id"])
                    for row in realized_rows
                ).items()
            )
        },
        "realized_rows": realized_rows,
        "boundaries": [
            "The detailed tile archive remains ignored local data and contains no test image.",
            "Protected hashes are used only for overlap rejection and are not included in evidence.",
            "TRAIN and CALIBRATION groups remain disjoint exactly as frozen before byte access.",
        ],
    }


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    freeze_parser = sub.add_parser("freeze")
    freeze_parser.add_argument("--root", type=Path, required=True)
    freeze_parser.add_argument("--output", type=Path, required=True)
    realize_parser = sub.add_parser("realize")
    realize_parser.add_argument("--root", type=Path, required=True)
    realize_parser.add_argument("--selection", type=Path, required=True)
    realize_parser.add_argument("--tiles", type=Path, required=True)
    realize_parser.add_argument("--output", type=Path, required=True)
    realize_parser.add_argument("--protect", action="append", type=Path, default=[])
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "freeze":
        payload = freeze(args.root)
        write_json_atomic(payload, args.output, args.root)
        print(json.dumps(payload["counts"], indent=2, sort_keys=True))
        print(f"wrote frozen selection {args.output}")
        return 0
    payload = realize(args.root, args.selection, args.tiles, args.protect)
    write_json_atomic(payload, args.output, args.root)
    print(json.dumps(payload["counts"], indent=2, sort_keys=True))
    print(f"wrote realized evidence {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
