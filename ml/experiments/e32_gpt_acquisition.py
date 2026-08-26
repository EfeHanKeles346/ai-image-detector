"""E32/C2c — acquire only missing GPT Image 1 pairs from the frozen 15K receipt."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping

from huggingface_hub import hf_hub_url
from PIL import Image

import e32_ai_pool_selection as pool
import e32_data_system as acquisition
from pixelproof.project_paths import DATA_ROOT, ML_ROOT


SMOKE_EVIDENCE = ML_ROOT.parent / "evidence" / "e32_gpt_decoder_smoke.json"
WORKERS = 4


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _write_atomic(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_bytes(raw)
    temporary.replace(path)


def _load_selection() -> dict[str, Any]:
    payload = json.loads(pool.DETAILED_SELECTION.read_text())
    if payload.get("state") != "frozen_ai_parent_selection_before_remaining_bytes_or_decode":
        raise ValueError("unexpected E32 AI-pool selection state")
    records = [
        {**record, "source_id": source["source_id"], "family": source["family"]}
        for source in payload["sources"]
        for record in source["records"]
    ]
    if pool._fingerprint(records) != payload.get("selection_sha256"):
        raise ValueError("E32 AI-pool record selection SHA mismatch")
    return payload


def _gpt_source(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    try:
        return next(source for source in payload["sources"] if source["source_id"] == "gpt-image-1")
    except StopIteration as error:
        raise KeyError("frozen AI pool lacks GPT Image 1") from error


def _gpt_spec() -> Mapping[str, Any]:
    return next(source for source in pool.registry()["sources"] if source["id"] == "gpt-image-1")


def _safe_join(root: Path, source_key: str) -> Path:
    pure = PurePosixPath(source_key)
    if pure.is_absolute() or ".." in pure.parts or not pure.parts:
        raise ValueError(f"unsafe GPT source key: {source_key}")
    destination = root.joinpath(*pure.parts)
    destination.resolve().relative_to(root.resolve())
    return destination


def _local_asset(root: Path, source_key: str) -> Path:
    return _safe_join(root / str(_gpt_spec()["local_dirname"]), source_key)


def _e32_asset(source_key: str) -> Path:
    return acquisition._safe_destination(f"ai/gpt-image-1/{source_key}")


def _available_asset(root: Path, source_key: str, expected_bytes: int) -> Path | None:
    for path in (_local_asset(root, source_key), _e32_asset(source_key)):
        if path.is_file() and path.stat().st_size == expected_bytes:
            return path
    return None


def _assets(record: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "kind": "image",
            "source_key": record["source_key"],
            "expected_bytes": int(record["expected_image_bytes"]),
        },
        {
            "kind": "prompt",
            "source_key": record["prompt_key"],
            "expected_bytes": int(record["expected_prompt_bytes"]),
        },
    ]


def _fetch(root: Path, source: Mapping[str, Any], asset: Mapping[str, Any]) -> dict[str, Any]:
    available = _available_asset(root, str(asset["source_key"]), int(asset["expected_bytes"]))
    if available is not None:
        return {
            "kind": asset["kind"],
            "source_key": asset["source_key"],
            "bytes": int(asset["expected_bytes"]),
            "state": "existing_local_or_e32",
            "path": str(available),
        }
    url = hf_hub_url(
        str(source["repo_id"]),
        str(asset["source_key"]),
        revision=str(source["revision"]),
        repo_type="dataset",
    )
    result = acquisition._stream_download(
        url, _e32_asset(str(asset["source_key"])), int(asset["expected_bytes"])
    )
    return {
        "kind": asset["kind"],
        "source_key": asset["source_key"],
        "bytes": result["bytes"],
        "state": result["state"],
        "path": str(_e32_asset(str(asset["source_key"]))),
    }


def smoke(root: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    source = _gpt_source(payload)
    record = next(item for item in source["records"] if item["storage"] == "e32_download_required")
    results = [_fetch(root, source, asset) for asset in _assets(record)]
    paths = {result["kind"]: Path(result["path"]) for result in results}
    prompt_raw = paths["prompt"].read_bytes()
    prompt = prompt_raw.decode("utf-8").strip()
    if not prompt:
        raise ValueError("GPT decoder-smoke prompt is empty")
    with Image.open(paths["image"]) as image:
        image.load()
        decoded_format = str(image.format or "UNKNOWN").upper()
        mode = image.mode
        width, height = image.size
    if width <= 0 or height <= 0:
        raise ValueError("GPT decoder-smoke image has invalid dimensions")
    report = {
        "schema_version": 1,
        "experiment": "E32/C2c",
        "state": "gpt_decoder_smoke_passed",
        "selection_sha256": payload["selection_sha256"],
        "source_revision": source["revision"],
        "source_key": record["source_key"],
        "image_bytes": paths["image"].stat().st_size,
        "image_sha256": _sha256(paths["image"].read_bytes()),
        "prompt_bytes": len(prompt_raw),
        "prompt_sha256": _sha256(prompt.encode()),
        "decoded_format": decoded_format,
        "mode": mode,
        "width": width,
        "height": height,
        "bulk_boundary": "Bulk requires this exact selection SHA and passed decoder/prompt smoke.",
    }
    _write_atomic(SMOKE_EVIDENCE, _json_bytes(report))
    return report


def _require_smoke(payload: Mapping[str, Any]) -> None:
    if not SMOKE_EVIDENCE.is_file():
        raise PermissionError("GPT decoder smoke evidence is missing")
    report = json.loads(SMOKE_EVIDENCE.read_text())
    if report.get("state") != "gpt_decoder_smoke_passed":
        raise PermissionError("GPT decoder smoke did not pass")
    if report.get("selection_sha256") != payload.get("selection_sha256"):
        raise PermissionError("GPT decoder smoke belongs to a different AI-pool selection")


def download(root: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    _require_smoke(payload)
    source = _gpt_source(payload)
    assets = [asset for record in source["records"] for asset in _assets(record)]
    results = []
    for start in range(0, len(assets), 100):
        batch = assets[start : start + 100]
        with ThreadPoolExecutor(max_workers=WORKERS) as executor:
            futures = [executor.submit(_fetch, root, source, asset) for asset in batch]
            results.extend(future.result() for future in as_completed(futures))
        print(f"gpt-image-1 {min(start + len(batch), len(assets))}/{len(assets)} assets", flush=True)
    states = dict(sorted(Counter(result["state"] for result in results).items()))
    return {
        "source_id": source["source_id"],
        "selected_pairs": len(source["records"]),
        "assets": len(results),
        "bytes": sum(int(result["bytes"]) for result in results),
        "states": states,
    }


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("smoke", "download"):
        command = subparsers.add_parser(name)
        command.add_argument("--root", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    payload = _load_selection()
    result = smoke(args.root, payload) if args.command == "smoke" else download(args.root, payload)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
