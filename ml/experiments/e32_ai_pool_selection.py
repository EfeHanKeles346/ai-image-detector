"""E32/C2c — freeze the exact 15K AI parent selection without decoding images."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict, deque
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Iterable, Mapping, Sequence

import pyarrow.parquet as pq
from huggingface_hub import HfApi

import e32_gap_acquisition as gap
from pixelproof.project_paths import DATA_ROOT, ML_ROOT


REGISTRY_PATH = ML_ROOT / "e32_ai_pool_sources.json"
OUTPUT_ROOT = DATA_ROOT / "e32"
DETAILED_SELECTION = OUTPUT_ROOT / "ai_pool_selection.json"
COMPACT_EVIDENCE = ML_ROOT.parent / "evidence" / "e32_ai_pool_selection.json"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _stable_key(seed: int, source_id: str, key: str) -> str:
    return _sha256(f"{seed}:{source_id}:{key}".encode())


def _write_atomic(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_bytes(raw)
    temporary.replace(path)


def registry() -> dict[str, Any]:
    payload = json.loads(REGISTRY_PATH.read_text())
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported E32 AI-pool registry schema")
    sources = payload.get("sources", [])
    ids = [str(source["id"]) for source in sources]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate E32 AI-pool source id")
    target = int(payload["target_parents"])
    if sum(int(source["target"]) for source in sources) != target:
        raise ValueError("AI-pool allocation does not sum to target")
    ceiling = float(payload["max_source_share"])
    if any(int(source["target"]) / target > ceiling for source in sources):
        raise ValueError("AI-pool source allocation exceeds the frozen share ceiling")
    families = {
        str(source["family"])
        for source in sources
        if source["counts_as_current_family"]
    }
    if len(families) < 5:
        raise ValueError("AI-pool registry has fewer than five verified current families")
    return payload


def _real_parquets(folder: Path) -> list[Path]:
    return sorted(
        path
        for path in folder.rglob("*.parquet")
        if path.is_file() and not path.name.startswith("._") and ".cache" not in path.parts
    )


def _fingerprint(rows: Sequence[Mapping[str, Any]]) -> str:
    material = json.dumps(rows, sort_keys=True, separators=(",", ":"))
    return _sha256(material.encode())


def _parquet_locator_rows(
    folder: Path, columns: Sequence[str]
) -> tuple[list[dict[str, Any]], str]:
    rows: list[dict[str, Any]] = []
    files = []
    for path in _real_parquets(folder):
        parquet = pq.ParquetFile(path)
        missing = sorted(set(columns) - set(parquet.schema_arrow.names))
        if missing:
            raise ValueError(f"{path} lacks required columns {missing}")
        relative = path.relative_to(folder).as_posix()
        files.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "rows": parquet.metadata.num_rows,
            }
        )
        row_index = 0
        for batch in parquet.iter_batches(columns=list(columns), batch_size=4096):
            for values in batch.to_pylist():
                rows.append(
                    {
                        "shard": relative,
                        "row_index": row_index,
                        **values,
                    }
                )
                row_index += 1
    return rows, _fingerprint(files)


def _stable_select(
    records: Sequence[Mapping[str, Any]], *, source_id: str, key_field: str, seed: int, limit: int
) -> list[dict[str, Any]]:
    keys = [str(record[key_field]) for record in records]
    if len(keys) != len(set(keys)):
        raise ValueError(f"{source_id} selection key {key_field!r} is not unique")
    ordered = sorted(
        records,
        key=lambda record: (
            _stable_key(seed, source_id, str(record[key_field])),
            str(record[key_field]),
        ),
    )
    if len(ordered) < limit:
        raise ValueError(f"{source_id} has {len(ordered)} candidates; needs {limit}")
    return [dict(record) for record in ordered[:limit]]


def _model_round_robin(
    records: Sequence[Mapping[str, Any]], *, source_id: str, seed: int, limit: int
) -> list[dict[str, Any]]:
    by_model: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        model = str(record["model_name"])
        if model in {"", "None", "N/A"}:
            raise ValueError(f"{source_id} AI row lacks model identity")
        by_model[model].append(record)
    queues = {
        model: deque(
            sorted(
                rows,
                key=lambda row: _stable_key(
                    seed, source_id, f"{row['shard']}:{row['row_index']}"
                ),
            )
        )
        for model, rows in by_model.items()
    }
    model_order = sorted(queues, key=lambda model: (_stable_key(seed, source_id, model), model))
    selected: list[dict[str, Any]] = []
    while len(selected) < limit:
        changed = False
        for model in model_order:
            if queues[model] and len(selected) < limit:
                selected.append(dict(queues[model].popleft()))
                changed = True
        if not changed:
            break
    if len(selected) != limit:
        raise ValueError(f"{source_id} has only {len(selected)} round-robin candidates")
    return selected


def _freeze_gap(spec: Mapping[str, Any], payload: Mapping[str, Any]) -> dict[str, Any]:
    source = next(item for item in payload["sources"] if item["source_id"] == spec["id"])
    images = [item for item in source["assets"] if str(item["path"]).endswith(".jxl")]
    if len(images) != int(spec["target"]):
        raise ValueError(f"{spec['id']} frozen gap receipt count changed")
    records = [
        {
            "source_key": item["path"],
            "parent_group": item["prompt_group"],
            "category": item["category"],
            "storage": "e32_gap_download",
        }
        for item in images
    ]
    return {
        "source_id": spec["id"],
        "family": spec["family"],
        "counts_as_current_family": spec["counts_as_current_family"],
        "selection_kind": spec["selection_kind"],
        "revision": source["revision"],
        "selected": len(records),
        "records": records,
    }


def _freeze_nano(root: Path, spec: Mapping[str, Any], seed: int) -> dict[str, Any]:
    folder = root / str(spec["dirname"])
    rows, source_fingerprint = _parquet_locator_rows(
        folder, ("id", "format", "mode", "width", "height", "uploadtime")
    )
    selected = _stable_select(
        rows,
        source_id=str(spec["id"]),
        key_field="id",
        seed=seed,
        limit=int(spec["target"]),
    )
    records = [
        {
            "source_key": f"{row['shard']}:{row['row_index']}",
            "shard": row["shard"],
            "row_index": row["row_index"],
            "parent_group": f"nano:{row['id']}",
            "declared_format": row["format"],
            "declared_mode": row["mode"],
            "declared_width": row["width"],
            "declared_height": row["height"],
            "uploadtime": str(row["uploadtime"]),
            "storage": "local_parquet",
        }
        for row in selected
    ]
    return {
        "source_id": spec["id"],
        "family": spec["family"],
        "counts_as_current_family": spec["counts_as_current_family"],
        "selection_kind": spec["selection_kind"],
        "revision": spec["revision"],
        "available": len(rows),
        "source_fingerprint": source_fingerprint,
        "selected": len(records),
        "records": records,
    }


def _freeze_nbp(root: Path, spec: Mapping[str, Any]) -> dict[str, Any]:
    folder = root / str(spec["dirname"])
    paths = sorted(
        path
        for path in folder.rglob("*")
        if path.is_file()
        and not path.name.startswith("._")
        and ".cache" not in path.parts
        and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if len(paths) != int(spec["target"]):
        raise ValueError(f"{spec['id']} expected {spec['target']} images, found {len(paths)}")
    records = [
        {
            "source_key": path.relative_to(folder).as_posix(),
            "parent_group": f"nbp:{path.stem}",
            "expected_bytes": path.stat().st_size,
            "storage": "local_loose",
        }
        for path in paths
    ]
    return {
        "source_id": spec["id"],
        "family": spec["family"],
        "counts_as_current_family": spec["counts_as_current_family"],
        "selection_kind": spec["selection_kind"],
        "revision": spec["revision"],
        "available": len(paths),
        "source_fingerprint": _fingerprint(records),
        "selected": len(records),
        "records": records,
    }


def _freeze_community(root: Path, spec: Mapping[str, Any], seed: int) -> dict[str, Any]:
    folder = root / str(spec["dirname"])
    rows, source_fingerprint = _parquet_locator_rows(
        folder,
        ("image_name", "model_name", "subset", "split", "label", "architecture", "prompt"),
    )
    ai_rows = [row for row in rows if str(row["label"]) == "1"]
    selected = _model_round_robin(
        ai_rows, source_id=str(spec["id"]), seed=seed, limit=int(spec["target"])
    )
    records = [
        {
            "source_key": f"{row['shard']}:{row['row_index']}",
            "shard": row["shard"],
            "row_index": row["row_index"],
            "parent_group": f"community:{row['model_name']}:{row['image_name']}",
            "model_name": row["model_name"],
            "architecture": row["architecture"],
            "prompt_sha256": _sha256(str(row["prompt"] or "").encode()),
            "storage": "local_parquet",
        }
        for row in selected
    ]
    return {
        "source_id": spec["id"],
        "family": spec["family"],
        "counts_as_current_family": spec["counts_as_current_family"],
        "selection_kind": spec["selection_kind"],
        "revision": spec["revision"],
        "available_ai": len(ai_rows),
        "available_model_identities": len({str(row["model_name"]) for row in ai_rows}),
        "selected_model_identities": len({str(row["model_name"]) for row in selected}),
        "max_selected_per_model": max(Counter(str(row["model_name"]) for row in selected).values()),
        "source_fingerprint": source_fingerprint,
        "selected": len(records),
        "records": records,
    }


def _repo_info(repo_id: str, revision: str) -> Any:
    return HfApi().dataset_info(repo_id, revision=revision, files_metadata=True)


def _freeze_gpt(
    root: Path,
    spec: Mapping[str, Any],
    seed: int,
    info_loader: Callable[[str, str], Any] = _repo_info,
) -> dict[str, Any]:
    info = info_loader(str(spec["repo_id"]), str(spec["revision"]))
    if info.sha != spec["revision"]:
        raise ValueError("GPT upstream revision changed")
    if spec["license_tag"] not in set(info.tags or []):
        raise ValueError("GPT dataset licence tag changed")
    siblings = {
        str(item.rfilename): int(item.size)
        for item in info.siblings
        if item.size is not None
    }
    images = {str(PurePosixPath(path).with_suffix("")): path for path in siblings if path.lower().endswith(".png")}
    prompts = {str(PurePosixPath(path).with_suffix("")): path for path in siblings if path.lower().endswith(".txt")}
    paired = sorted(set(images) & set(prompts))
    if len(paired) != int(spec["expected_available_pairs"]):
        raise ValueError(f"GPT expected {spec['expected_available_pairs']} pairs, found {len(paired)}")
    selected = sorted(
        paired,
        key=lambda stem: (_stable_key(seed, str(spec["id"]), stem), stem),
    )[: int(spec["target"])]
    local = root / str(spec["local_dirname"])
    records = []
    local_pairs = 0
    for stem in selected:
        image_key, prompt_key = images[stem], prompts[stem]
        image_local = local.joinpath(*PurePosixPath(image_key).parts)
        prompt_local = local.joinpath(*PurePosixPath(prompt_key).parts)
        present = (
            image_local.is_file()
            and prompt_local.is_file()
            and image_local.stat().st_size == siblings[image_key]
            and prompt_local.stat().st_size == siblings[prompt_key]
        )
        local_pairs += int(present)
        records.append(
            {
                "source_key": image_key,
                "prompt_key": prompt_key,
                "parent_group": f"gpt:{stem}",
                "expected_image_bytes": siblings[image_key],
                "expected_prompt_bytes": siblings[prompt_key],
                "storage": "existing_local" if present else "e32_download_required",
            }
        )
    return {
        "source_id": spec["id"],
        "family": spec["family"],
        "counts_as_current_family": spec["counts_as_current_family"],
        "selection_kind": spec["selection_kind"],
        "repo_id": spec["repo_id"],
        "revision": spec["revision"],
        "license_tag": spec["license_tag"],
        "available_pairs": len(paired),
        "selected": len(records),
        "selected_pairs_already_local": local_pairs,
        "selected_pairs_requiring_download": len(records) - local_pairs,
        "records": records,
    }


def freeze_pool(root: Path, *, info_loader: Callable[[str, str], Any] = _repo_info) -> dict[str, Any]:
    contract = registry()
    gap_raw = gap.DETAILED_SELECTION.read_bytes()
    gap_payload = json.loads(gap_raw)
    sources = []
    for spec in contract["sources"]:
        kind = spec["selection_kind"]
        if kind == "frozen_gap_receipt":
            source = _freeze_gap(spec, gap_payload)
        elif kind == "parquet_stable_hash":
            source = _freeze_nano(root, spec, int(contract["seed"]))
        elif kind == "huggingface_stable_hash_pairs":
            source = _freeze_gpt(root, spec, int(contract["seed"]), info_loader)
        elif kind == "all_loose_images":
            source = _freeze_nbp(root, spec)
        elif kind == "parquet_model_round_robin":
            source = _freeze_community(root, spec, int(contract["seed"]))
        else:
            raise ValueError(f"unsupported selection kind {kind!r}")
        sources.append(source)
    records = [
        {**record, "source_id": source["source_id"], "family": source["family"]}
        for source in sources
        for record in source["records"]
    ]
    if len(records) != int(contract["target_parents"]):
        raise ValueError("realized AI-pool selection does not match target")
    detailed = {
        "schema_version": 1,
        "experiment": "E32/C2c",
        "state": "frozen_ai_parent_selection_before_remaining_bytes_or_decode",
        "seed": contract["seed"],
        "target_parents": contract["target_parents"],
        "max_source_share": contract["max_source_share"],
        "label_invariant": {"real": 0, "ai": 1},
        "gap_selection_sha256": _sha256(gap_raw),
        "selection_sha256": _fingerprint(records),
        "sources": sources,
        "boundaries": [
            "No selected image was decoded or scored by this metadata-only freeze.",
            "Selection is independent of existing local GPT availability.",
            "CommunityForensics contributes only raw label 1 and is model-identity round-robin.",
            "All rows remain role-free candidates until byte and protected-overlap audits pass.",
        ],
    }
    raw = _json_bytes(detailed)
    _write_atomic(DETAILED_SELECTION, raw)
    compact = {
        "schema_version": 1,
        "experiment": detailed["experiment"],
        "state": detailed["state"],
        "selection_sha256": detailed["selection_sha256"],
        "gap_selection_sha256": detailed["gap_selection_sha256"],
        "detailed_selection_sha256": _sha256(raw),
        "detailed_selection_bytes": len(raw),
        "target_parents": detailed["target_parents"],
        "counts_by_source": {source["source_id"]: source["selected"] for source in sources},
        "current_family_count": len(
            {source["family"] for source in sources if source["counts_as_current_family"]}
        ),
        "gpt_selected_pairs_already_local": next(
            source["selected_pairs_already_local"]
            for source in sources
            if source["source_id"] == "gpt-image-1"
        ),
        "gpt_selected_pairs_requiring_download": next(
            source["selected_pairs_requiring_download"]
            for source in sources
            if source["source_id"] == "gpt-image-1"
        ),
        "new_image_bytes_downloaded_by_freeze": 0,
    }
    _write_atomic(COMPACT_EVIDENCE, _json_bytes(compact))
    return compact


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    print(json.dumps(freeze_pool(args.root), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
