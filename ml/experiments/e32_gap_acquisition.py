"""E32/C2b — freeze and acquire two licensed modern-generator gap sources."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Iterable, Mapping, Sequence

from huggingface_hub import HfApi, hf_hub_url
from PIL import Image

import e32_data_system as real_acquisition
from pixelproof.project_paths import DATA_ROOT, ML_ROOT


REGISTRY_PATH = ML_ROOT / "e32_gap_sources.json"
OUTPUT_ROOT = DATA_ROOT / "e32"
DETAILED_SELECTION = OUTPUT_ROOT / "ai_gap_selection.json"
COMPACT_EVIDENCE = ML_ROOT.parent / "evidence" / "e32_ai_gap_selection.json"
SMOKE_EVIDENCE = ML_ROOT.parent / "evidence" / "e32_ai_gap_decoder_smoke.json"
WORKERS = 4


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def _write_atomic(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_bytes(raw)
    temporary.replace(path)


def registry() -> tuple[dict[str, Any], list[Mapping[str, Any]]]:
    payload = json.loads(REGISTRY_PATH.read_text())
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported E32 gap-source registry schema")
    sources = payload.get("sources", [])
    ids = [source["id"] for source in sources]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate E32 gap source id")
    return payload["selection_contract"], sources


def _readme_url(spec: Mapping[str, Any]) -> str:
    return (
        f"https://huggingface.co/datasets/{spec['repo_id']}/raw/"
        f"{spec['revision']}/README.md"
    )


def _category_and_group(source_id: str, path: str) -> tuple[str, str]:
    pure = PurePosixPath(path)
    stem, separator, variation = pure.stem.rpartition("_")
    if not separator or variation not in {"0", "1", "2", "3"}:
        raise ValueError(f"unexpected four-variation filename: {path}")
    if source_id == "flux2-klein-9b" and pure.parts[:2] == ("data_edit", "edited"):
        category = "editing"
    elif pure.parts and pure.parts[0] == "data" and len(pure.parts) >= 2:
        category = pure.parts[1]
    else:
        raise ValueError(f"unexpected source hierarchy: {path}")
    return category, f"{category}:{stem}"


def _round_robin_groups(
    groups_by_category: Mapping[str, Sequence[str]], limit: int
) -> list[str]:
    queues = {category: list(sorted(groups)) for category, groups in groups_by_category.items()}
    selected: list[str] = []
    while len(selected) < limit:
        changed = False
        for category in sorted(queues):
            if queues[category] and len(selected) < limit:
                selected.append(queues[category].pop(0))
                changed = True
        if not changed:
            break
    if len(selected) != limit:
        raise ValueError(f"only {len(selected)} prompt groups are available; need {limit}")
    return selected


def _repo_info(repo_id: str, revision: str) -> Any:
    return HfApi().dataset_info(repo_id, revision=revision, files_metadata=True)


def freeze_source(
    spec: Mapping[str, Any],
    *,
    prompt_group_limit: int,
    info_loader: Callable[[str, str], Any] = _repo_info,
    readme_loader: Callable[[str, str], bytes] | None = None,
) -> dict[str, Any]:
    info = info_loader(str(spec["repo_id"]), str(spec["revision"]))
    if info.sha != spec["revision"]:
        raise ValueError(f"{spec['id']} revision mismatch: {info.sha}")
    if spec["license_tag"] not in set(info.tags or []):
        raise ValueError(f"{spec['id']} licence tag changed")
    if readme_loader is None:
        readme = real_acquisition._get_verified(
            _readme_url(spec), str(spec["readme_sha256"])
        )
    else:
        readme = readme_loader(_readme_url(spec), str(spec["readme_sha256"]))
    if _sha256(readme) != spec["readme_sha256"]:
        raise ValueError(f"{spec['id']} README changed")

    siblings = {
        item.rfilename: int(item.size)
        for item in info.siblings
        if item.size is not None
    }
    included_images = []
    for path, size in siblings.items():
        if not path.lower().endswith(".jxl"):
            continue
        if not any(path.startswith(prefix) for prefix in spec["include_prefixes"]):
            continue
        if any(path.startswith(prefix) for prefix in spec["exclude_prefixes"]):
            continue
        category, group = _category_and_group(str(spec["id"]), path)
        included_images.append(
            {"path": path, "bytes": size, "category": category, "prompt_group": group}
        )
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    categories: dict[str, list[str]] = defaultdict(list)
    for image in included_images:
        groups[image["prompt_group"]].append(image)
    for group, images in groups.items():
        if len(images) != 4:
            raise ValueError(f"{spec['id']} group {group!r} has {len(images)} images, expected four")
        categories[images[0]["category"]].append(group)
    if len(included_images) != int(spec["reported_images"]):
        raise ValueError(
            f"{spec['id']} expected {spec['reported_images']} generated images, "
            f"found {len(included_images)}"
        )
    if len(groups) != int(spec["reported_prompt_groups"]):
        raise ValueError(
            f"{spec['id']} expected {spec['reported_prompt_groups']} prompt groups, found {len(groups)}"
        )

    selected_groups = set(_round_robin_groups(categories, prompt_group_limit))
    selected_images = sorted(
        (image for image in included_images if image["prompt_group"] in selected_groups),
        key=lambda item: item["path"],
    )
    assets = []
    for image in selected_images:
        text_path = str(PurePosixPath(image["path"]).with_suffix(".txt"))
        if text_path not in siblings:
            raise ValueError(f"{spec['id']} lacks prompt sidecar for {image['path']}")
        for path in (image["path"], text_path):
            assets.append(
                {
                    "path": path,
                    "bytes": siblings[path],
                    "category": image["category"],
                    "prompt_group": image["prompt_group"],
                }
            )
    category_counts = defaultdict(int)
    for group in selected_groups:
        category_counts[group.split(":", 1)[0]] += 1
    return {
        "source_id": spec["id"],
        "repo_id": spec["repo_id"],
        "revision": spec["revision"],
        "family": spec["family"],
        "license_tag": spec["license_tag"],
        "native_format": spec["native_format"],
        "expected_width": int(spec["expected_width"]),
        "expected_height": int(spec["expected_height"]),
        "available_generated_images": len(included_images),
        "available_prompt_groups": len(groups),
        "selected_prompt_groups": len(selected_groups),
        "selected_images": len(selected_images),
        "selected_image_bytes": sum(image["bytes"] for image in selected_images),
        "selected_category_group_counts": dict(sorted(category_counts.items())),
        "assets": assets,
    }


def freeze_gap_sources() -> dict[str, Any]:
    contract, specs = registry()
    sources = [
        freeze_source(spec, prompt_group_limit=int(contract["prompt_groups_per_source"]))
        for spec in specs
    ]
    detailed = {
        "schema_version": 1,
        "experiment": "E32/C2b",
        "state": "selection_frozen_decoder_smoke_required_before_bulk",
        "label_invariant": {"real": 0, "ai": 1},
        "selection_contract": contract,
        "sources": sources,
        "boundaries": [
            "Exactly one selected JPEG XL per source may be downloaded for the decoder smoke.",
            "Bulk acquisition is forbidden until both smoke images decode successfully.",
            "All four images from a selected prompt group stay together in later role assignment.",
            "FLUX editing references are excluded because they are not AI outputs.",
            "These sources are TRAIN/CALIBRATION candidates and do not alter Qwen LOCKED FINAL.",
        ],
    }
    raw = _json_bytes(detailed)
    _write_atomic(DETAILED_SELECTION, raw)
    compact = {
        "schema_version": 1,
        "experiment": "E32/C2b",
        "state": detailed["state"],
        "detailed_selection_sha256": _sha256(raw),
        "detailed_selection_bytes": len(raw),
        "new_image_bytes_downloaded_by_freeze": 0,
        "sources": [
            {key: value for key, value in source.items() if key != "assets"}
            for source in sources
        ],
    }
    _write_atomic(COMPACT_EVIDENCE, _json_bytes(compact))
    return compact


def _load_selection() -> dict[str, Any]:
    payload = json.loads(DETAILED_SELECTION.read_text())
    if payload.get("state") != "selection_frozen_decoder_smoke_required_before_bulk":
        raise ValueError("unexpected AI gap selection state")
    return payload


def verify_smokes(payload: Mapping[str, Any]) -> dict[str, Any]:
    selection_raw = DETAILED_SELECTION.read_bytes()
    rows = []
    for source in payload["sources"]:
        asset = next(item for item in source["assets"] if item["path"].endswith(".jxl"))
        path = _destination(str(source["source_id"]), str(asset["path"]))
        if not path.is_file():
            raise FileNotFoundError(f"decoder smoke image is missing: {path}")
        raw = path.read_bytes()
        if len(raw) != int(asset["bytes"]):
            raise ValueError(f"decoder smoke size mismatch: {path}")
        with Image.open(path) as image:
            image.load()
            width, height = image.size
            decoded_format = str(image.format or "UNKNOWN").upper()
            mode = image.mode
        expected_size = (int(source["expected_width"]), int(source["expected_height"]))
        if (width, height) != expected_size:
            raise ValueError(
                f"{source['source_id']} smoke expected {expected_size}, found {(width, height)}"
            )
        rows.append(
            {
                "source_id": source["source_id"],
                "source_key": asset["path"],
                "bytes": len(raw),
                "sha256": _sha256(raw),
                "declared_extension": PurePosixPath(asset["path"]).suffix.lower(),
                "decoded_format": decoded_format,
                "extension_matches_decoded_format": (
                    PurePosixPath(asset["path"]).suffix.lower().lstrip(".")
                    == decoded_format.lower()
                ),
                "width": width,
                "height": height,
                "mode": mode,
            }
        )
    report = {
        "schema_version": 1,
        "experiment": "E32/C2b",
        "state": "decoder_smoke_passed",
        "selection_sha256": _sha256(selection_raw),
        "sources": rows,
        "finding": "Both .jxl source paths decode as PNG payloads; extension metadata is wrong.",
        "bulk_boundary": (
            "Bulk may proceed, but decoded format must be measured from bytes and both classes "
            "must share the chosen model-input normalization."
        ),
    }
    _write_atomic(SMOKE_EVIDENCE, _json_bytes(report))
    return report


def _require_smoke_gate() -> None:
    if not SMOKE_EVIDENCE.is_file():
        raise PermissionError("decoder smoke evidence is missing; bulk acquisition is forbidden")
    report = json.loads(SMOKE_EVIDENCE.read_text())
    if report.get("state") != "decoder_smoke_passed":
        raise PermissionError("decoder smoke did not pass; bulk acquisition is forbidden")
    if report.get("selection_sha256") != _sha256(DETAILED_SELECTION.read_bytes()):
        raise PermissionError("decoder smoke belongs to a different AI gap selection")


def _selected_source(payload: Mapping[str, Any], source_id: str) -> Mapping[str, Any]:
    try:
        return next(source for source in payload["sources"] if source["source_id"] == source_id)
    except StopIteration as error:
        raise KeyError(source_id) from error


def _destination(source_id: str, path: str) -> Path:
    pure = PurePosixPath(path)
    if pure.is_absolute() or ".." in pure.parts:
        raise ValueError(f"unsafe source path: {path}")
    return real_acquisition._safe_destination(f"ai/{source_id}/{pure.as_posix()}")


def download_assets(source: Mapping[str, Any], *, smoke: bool) -> dict[str, Any]:
    if not smoke:
        _require_smoke_gate()
    assets = list(source["assets"])
    if smoke:
        image = next(asset for asset in assets if asset["path"].endswith(".jxl"))
        sidecar_path = str(PurePosixPath(image["path"]).with_suffix(".txt"))
        assets = [image, next(asset for asset in assets if asset["path"] == sidecar_path)]

    def fetch(asset: Mapping[str, Any]) -> dict[str, Any]:
        url = hf_hub_url(
            str(source["repo_id"]),
            str(asset["path"]),
            revision=str(source["revision"]),
            repo_type="dataset",
        )
        result = real_acquisition._stream_download(
            url, _destination(str(source["source_id"]), str(asset["path"])), int(asset["bytes"])
        )
        return {**result, "source_key": asset["path"]}

    results = []
    for start in range(0, len(assets), 100):
        batch = assets[start : start + 100]
        with ThreadPoolExecutor(max_workers=WORKERS) as executor:
            futures = [executor.submit(fetch, asset) for asset in batch]
            results.extend(future.result() for future in as_completed(futures))
        print(f"{source['source_id']} {min(start + len(batch), len(assets))}/{len(assets)} assets", flush=True)
    return {
        "source_id": source["source_id"],
        "mode": "decoder_smoke_assets" if smoke else "frozen_bulk_assets",
        "files": len(results),
        "bytes": sum(result["bytes"] for result in results),
        "paths": sorted(result["path"] for result in results) if smoke else None,
    }


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("freeze")
    subparsers.add_parser("verify-smoke")
    download = subparsers.add_parser("download")
    download.add_argument("--source", choices=("qwen-image-2512", "flux2-klein-9b"), required=True)
    download.add_argument("--smoke", action="store_true")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "freeze":
        result = freeze_gap_sources()
    elif args.command == "verify-smoke":
        result = verify_smokes(_load_selection())
    else:
        source = _selected_source(_load_selection(), args.source)
        result = download_assets(source, smoke=bool(args.smoke))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
