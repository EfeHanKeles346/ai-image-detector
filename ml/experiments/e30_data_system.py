"""E30 — role-safe, capped acquisition for current data and OOD testing.

The default command only freezes selections.  Image bytes are downloaded only
with an explicit ``download-*`` subcommand and always remain under ignored
``ml/data``.  Run from ``ml/``::

    PYTHONPATH=src .venv/bin/python experiments/e30_data_system.py freeze-mllm
    PYTHONPATH=src .venv/bin/python experiments/e30_data_system.py download-mllm
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping
from urllib.parse import urlparse

from huggingface_hub import HfApi, hf_hub_url
from huggingface_hub.utils import get_session
from PIL import Image

from pixelproof.data_contract import (
    DataRecord,
    DataRole,
    content_set_sha256,
    dhash_image,
    enforce_byte_ceiling,
    record_to_dict,
    sha256_bytes,
    shortcut_audit,
    validate_records,
)
from pixelproof.project_paths import DATA_ROOT, ML_ROOT


REGISTRY_PATH = ML_ROOT / "e30_sources.json"
OUTPUT_ROOT = DATA_ROOT / "e30"
MLLM_ID = "zr-zhang/MLLM-Generated-Image-Detection-Dataset"
QWEN_ID = "Qwen/Qwen-Image-Bench"
LAION_ID = "laionmobile/laion-mobile"
MLLM_PER_CELL = 20
MLLM_TOTAL_CEILING = 15_000_000
DEVELOPMENT_TOTAL_CEILING = 40_000_000
QWEN_PER_GENERATOR = 5
QWEN_TOTAL_CEILING = 70_000_000
LAION_PER_PIPELINE = 10
LAION_PER_FILE_CEILING = 375_000
LAION_IMAGE_CEILING = 30_000_000
MLLM_REGIMES = {
    "Hybrid Images": "hybrid",
    "Structure Images": "structure",
    "Texture Images": "texture",
}
MLLM_CLASSES = {
    "GPT-Image2-fake": ("ai", "GPT Image 2", None),
    "Nano-Banana2-fake": ("ai", "Nano Banana 2", None),
    "real": ("real", None, "MLLMGenSet matched real"),
}
QWEN_GENERATORS = (
    "gpt-image-2",
    "nano-banana-2.0",
    "Seedream-5.0",
    "Qwen-Image-2.0-pro",
    "FLUX.2_max",
    "FLUX.2-pro",
    "GLM-Image",
    "HunyuanImage-3.0",
)
LAION_PIPELINES = (
    ("apple", "iPhone 11"),
    ("apple", "iPhone 11 Pro"),
    ("apple", "iPhone XS"),
    ("apple", "iPhone XR"),
    ("samsung", "SM-G930F"),
    ("samsung", "SM-G950F"),
    ("samsung", "SM-G935F"),
    ("xiaomi", "Redmi Note 4"),
)


@dataclass(frozen=True)
class RemoteAsset:
    source_id: str
    source_revision: str
    source_key: str
    role: DataRole
    label: str
    group: str
    transport: str
    filename: str
    expected_bytes: int | None
    generator: str | None = None
    camera_pipeline: str | None = None
    content_id: str | None = None
    expected_sha256: str | None = None
    url: str | None = None

    @property
    def record_id(self) -> str:
        return sha256_bytes(f"{self.source_id}:{self.source_key}".encode())[:24]


def registry() -> dict[str, Any]:
    payload = json.loads(REGISTRY_PATH.read_text())
    return {source["id"]: source for source in payload["sources"]}


def _repo_files(repo_id: str, revision: str, directory: str) -> list[dict[str, Any]]:
    output = []
    for item in HfApi().list_repo_tree(
        repo_id,
        path_in_repo=directory,
        recursive=False,
        expand=True,
        revision=revision,
        repo_type="dataset",
    ):
        path = getattr(item, "path", "")
        size = getattr(item, "size", None)
        if path and size:
            output.append({"path": str(path), "size": int(size)})
    return output


def _numeric_stem(path: str) -> tuple[int, str]:
    stem = Path(path).stem.split("_", 1)[0]
    try:
        return int(stem), path
    except ValueError:
        return 10**18, path


def freeze_mllm_assets(file_lister: Callable[[str, str, str], list[dict[str, Any]]] = _repo_files) -> list[RemoteAsset]:
    source = registry()[MLLM_ID]
    candidates: list[dict[str, Any]] = []
    for source_regime, regime in MLLM_REGIMES.items():
        for source_class, (label, generator, camera_pipeline) in MLLM_CLASSES.items():
            directory = f"images/Preprocessing/{source_regime}/{source_class}"
            for item in file_lister(MLLM_ID, source["revision"], directory):
                if Path(item["path"]).suffix.lower() != ".jpg":
                    continue
                candidates.append(
                    {
                        "source_key": item["path"],
                        "source_size": int(item["size"]),
                        "label": label,
                        "generator": generator,
                        "camera_pipeline": camera_pipeline,
                        "regime": regime,
                    }
                )
    # The source names are numeric but lexical sorting puts 100 before 11. Re-freeze each cell
    # numerically so source additions cannot reshuffle the chosen first twenty.
    grouped: dict[tuple[Any, ...], list[Mapping[str, Any]]] = {}
    for item in candidates:
        key = (item["label"], item["generator"], item["camera_pipeline"], item["regime"])
        grouped.setdefault(key, []).append(item)
    selected = []
    for key in sorted(grouped, key=lambda value: tuple(map(str, value))):
        ordered = sorted(grouped[key], key=lambda item: _numeric_stem(item["source_key"]))
        if len(ordered) < MLLM_PER_CELL:
            raise RuntimeError(f"MLLMGenSet cell {key!r} has only {len(ordered)} assets")
        selected.extend(ordered[:MLLM_PER_CELL])
    if len(selected) != 180:
        raise RuntimeError(f"MLLMGenSet selection expected 180 assets, got {len(selected)}")
    enforce_byte_ceiling(
        [int(item["source_size"]) for item in selected], total_ceiling=MLLM_TOTAL_CEILING
    )
    assets = []
    for item in selected:
        suffix = Path(item["source_key"]).suffix.lower()
        assets.append(
            RemoteAsset(
                source_id=MLLM_ID,
                source_revision=source["revision"],
                source_key=item["source_key"],
                role=DataRole.DEVELOPMENT_TEST,
                label=item["label"],
                generator=item["generator"],
                camera_pipeline=item["camera_pipeline"],
                group=f"{item['regime']}:{item['generator'] or item['camera_pipeline']}",
                transport="standardized_jpeg",
                filename=f"{sha256_bytes(item['source_key'].encode())[:16]}{suffix}",
                expected_bytes=int(item["source_size"]),
                content_id=None,
            )
        )
    return assets


def freeze_qwen_assets(file_lister: Callable[[str, str, str], list[dict[str, Any]]] = _repo_files) -> list[RemoteAsset]:
    source = registry()[QWEN_ID]
    assets = []
    for generator in QWEN_GENERATORS:
        directory = f"images/{generator}"
        files = [
            item
            for item in file_lister(QWEN_ID, source["revision"], directory)
            if Path(item["path"]).suffix.lower() == ".png"
        ]
        ordered = sorted(files, key=lambda item: _numeric_stem(item["path"]))
        if len(ordered) < QWEN_PER_GENERATOR:
            raise RuntimeError(f"Qwen generator {generator!r} has only {len(ordered)} PNGs")
        for item in ordered[:QWEN_PER_GENERATOR]:
            prompt_id = Path(item["path"]).stem.split("_", 1)[0]
            assets.append(
                RemoteAsset(
                    source_id=QWEN_ID,
                    source_revision=source["revision"],
                    source_key=item["path"],
                    role=DataRole.LOCKED_FINAL_TEST,
                    label="ai",
                    generator=generator,
                    camera_pipeline=None,
                    group=generator,
                    transport="native_png",
                    filename=f"{sha256_bytes(item['path'].encode())[:16]}.png",
                    expected_bytes=int(item["size"]),
                    content_id=f"qwen-prompt:{prompt_id}",
                )
            )
    if len(assets) != len(QWEN_GENERATORS) * QWEN_PER_GENERATOR:
        raise RuntimeError("Qwen selection count changed")
    enforce_byte_ceiling(
        [int(asset.expected_bytes or 0) for asset in assets],
        total_ceiling=QWEN_TOTAL_CEILING,
    )
    return assets


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def freeze_manifest(assets: Iterable[RemoteAsset], path: Path) -> dict[str, Any]:
    values = list(assets)
    payload = {
        "schema_version": 1,
        "state": "selection_frozen",
        "records": [
            {
                "record_id": asset.record_id,
                "role": asset.role.value,
                "source_id": asset.source_id,
                "source_revision": asset.source_revision,
                "source_key": asset.source_key,
                "label": asset.label,
                "group": asset.group,
                "transport": asset.transport,
                "path": f"images/{asset.filename}",
                "generator": asset.generator,
                "camera_pipeline": asset.camera_pipeline,
                "content_id": asset.content_id,
                "parent_id": None,
                "sha256": None,
                "dhash": None,
                "bytes": asset.expected_bytes,
                "width": None,
                "height": None,
                "image_format": None,
            }
            for asset in values
        ],
        "selection_sha256": sha256_bytes(
            "\n".join(
                f"{asset.source_id}:{asset.source_revision}:{asset.source_key}"
                for asset in values
            ).encode()
        ),
        "declared_bytes": sum(int(asset.expected_bytes or 0) for asset in values),
    }
    _atomic_json(path, payload)
    return payload


def _request_with_retry(method: str, url: str, **kwargs: Any):
    last_error: Exception | None = None
    for attempt in range(5):
        try:
            session = get_session()
            request_kwargs = dict(kwargs)
            stream = bool(request_kwargs.pop("stream", False))
            follow_redirects = bool(
                request_kwargs.pop(
                    "allow_redirects", request_kwargs.pop("follow_redirects", True)
                )
            )
            if stream and hasattr(session, "build_request"):
                request = session.build_request(method, url, timeout=30, **request_kwargs)
                response = session.send(
                    request,
                    stream=True,
                    follow_redirects=follow_redirects,
                )
            else:
                response = session.request(
                    method,
                    url,
                    timeout=30,
                    follow_redirects=follow_redirects,
                    **request_kwargs,
                )
            response.raise_for_status()
            return response
        except Exception as error:
            last_error = error
            if attempt < 4:
                time.sleep(0.5 * (2**attempt))
    raise RuntimeError(f"{method} {url} failed after five attempts: {last_error}")


def download_resumable(
    url: str,
    path: Path,
    *,
    expected_bytes: int | None,
    hard_ceiling: int,
    requester: Callable[..., Any] = _request_with_retry,
) -> bytes:
    """Resume a partial asset when the host honours Range; otherwise restart safely."""
    if path.is_file():
        raw = path.read_bytes()
        if expected_bytes is None or len(raw) == expected_bytes:
            return raw
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".part")
    offset = partial.stat().st_size if partial.is_file() else 0
    headers = {"Range": f"bytes={offset}-"} if offset else {}
    response = requester("GET", url, headers=headers, stream=True)
    append = bool(offset and response.status_code == 206)
    if offset and not append:
        offset = 0
    mode = "ab" if append else "wb"
    total = offset
    chunks = (
        response.iter_content(chunk_size=64 * 1024)
        if hasattr(response, "iter_content")
        else response.iter_bytes(chunk_size=64 * 1024)
    )
    try:
        with partial.open(mode) as handle:
            for chunk in chunks:
                if not chunk:
                    continue
                total += len(chunk)
                if total > hard_ceiling or (expected_bytes is not None and total > expected_bytes):
                    raise RuntimeError(f"asset exceeded its byte contract at {total:,} bytes")
                handle.write(chunk)
                handle.flush()
                os.fsync(handle.fileno())
    finally:
        if hasattr(response, "close"):
            response.close()
    if expected_bytes is not None and total != expected_bytes:
        raise RuntimeError(f"asset is incomplete: expected {expected_bytes:,}, got {total:,}")
    partial.replace(path)
    return path.read_bytes()


def _asset_url(asset: RemoteAsset) -> str:
    if asset.url:
        return asset.url
    return hf_hub_url(
        asset.source_id,
        asset.source_key,
        repo_type="dataset",
        revision=asset.source_revision,
    )


def download_assets(
    assets: Iterable[RemoteAsset], output_dir: Path, *, total_ceiling: int
) -> dict[str, Any]:
    values = list(assets)
    declared = [asset.expected_bytes for asset in values]
    if all(size is not None for size in declared):
        enforce_byte_ceiling(
            [int(size) for size in declared if size is not None], total_ceiling=total_ceiling
        )
    image_dir = output_dir / "images"
    records = []
    total = 0
    for index, asset in enumerate(values, 1):
        remaining = total_ceiling - total
        raw = download_resumable(
            _asset_url(asset),
            image_dir / asset.filename,
            expected_bytes=asset.expected_bytes,
            hard_ceiling=remaining,
        )
        total += len(raw)
        digest = sha256_bytes(raw)
        if asset.expected_sha256 and digest != asset.expected_sha256:
            raise RuntimeError(f"upstream content hash changed for {asset.source_key}")
        with Image.open(io.BytesIO(raw)) as image:
            image.load()
            width, height = image.size
            image_format = str(image.format or "UNKNOWN").upper()
            perceptual = dhash_image(image)
        records.append(
            DataRecord(
                record_id=asset.record_id,
                role=asset.role,
                source_id=asset.source_id,
                source_revision=asset.source_revision,
                source_key=asset.source_key,
                label=asset.label,
                group=asset.group,
                transport=asset.transport,
                path=f"images/{asset.filename}",
                generator=asset.generator,
                camera_pipeline=asset.camera_pipeline,
                content_id=asset.content_id,
                sha256=digest,
                dhash=perceptual,
                bytes=len(raw),
                width=width,
                height=height,
                image_format=image_format,
            )
        )
        print(f"downloaded {index}/{len(values)} ({total:,} bytes)", flush=True)
    validated = validate_records(records, require_hashes=True)
    if len({record.sha256 for record in validated}) != len(validated):
        raise RuntimeError("download contains exact duplicate image bytes")
    manifest = {
        "schema_version": 1,
        "state": "download_verified",
        "image_bytes": total,
        "hard_ceiling_bytes": total_ceiling,
        "content_set_sha256": content_set_sha256(validated),
        "records": [record_to_dict(record) for record in validated],
    }
    if {record.label for record in validated} == {"real", "ai"}:
        manifest["shortcut_audit"] = shortcut_audit(validated)
    _atomic_json(output_dir / "manifest.json", manifest)
    return manifest


def laion_candidates(metadata: Iterable[Mapping[str, str]]) -> dict[tuple[str, str], list[Mapping[str, str]]]:
    grouped: dict[tuple[str, str], list[Mapping[str, str]]] = {key: [] for key in LAION_PIPELINES}
    for row in metadata:
        key = (str(row.get("make", "")).lower(), str(row.get("model", "")))
        if key in grouped and row.get("url") and row.get("content_sha256"):
            grouped[key].append(row)
    for key in grouped:
        grouped[key].sort(key=lambda row: int(row["image_id"]))
        if len(grouped[key]) < LAION_PER_PIPELINE:
            raise RuntimeError(f"LAION pipeline {key!r} has only {len(grouped[key])} rows")
    return grouped


def fetch_laion_metadata(output_dir: Path) -> list[dict[str, str]]:
    source = registry()[LAION_ID]
    source_key = "metadata/eval_sample_metadata.csv"
    path = output_dir / "eval_sample_metadata.csv"
    raw = download_resumable(
        hf_hub_url(LAION_ID, source_key, repo_type="dataset", revision=source["revision"]),
        path,
        expected_bytes=int(source["eval_manifest_bytes"]),
        hard_ceiling=3_000_000,
    )
    return list(csv.DictReader(io.StringIO(raw.decode("utf-8"))))


def freeze_laion_assets(rows: Iterable[Mapping[str, str]], head_requester: Callable[..., Any] = _request_with_retry) -> tuple[list[RemoteAsset], list[dict[str, str]]]:
    source = registry()[LAION_ID]
    grouped = laion_candidates(rows)
    selected = []
    failures = []
    running_bytes = 0
    for make, model in LAION_PIPELINES:
        accepted = 0
        for row in grouped[(make, model)]:
            if accepted >= LAION_PER_PIPELINE:
                break
            url = str(row["url"])
            try:
                response = head_requester("HEAD", url, allow_redirects=True)
                length = int(response.headers.get("content-length", "0"))
                content_type = response.headers.get("content-type", "").lower()
                if not 0 < length <= LAION_PER_FILE_CEILING:
                    raise RuntimeError(f"size {length} outside cap")
                if "image" not in content_type:
                    raise RuntimeError(f"content type {content_type!r} is not image")
                if running_bytes + length > LAION_IMAGE_CEILING:
                    raise RuntimeError("complete LAION image ceiling would be exceeded")
            except Exception as error:
                failures.append({"image_id": row["image_id"], "reason": str(error)[:160]})
                continue
            suffix = Path(urlparse(url).path).suffix.lower()
            if suffix not in {".jpg", ".jpeg", ".png", ".webp"}:
                suffix = ".img"
            source_key = f"eval-url:{row['image_id']}"
            selected.append(
                RemoteAsset(
                    source_id=LAION_ID,
                    source_revision=source["revision"],
                    source_key=source_key,
                    role=DataRole.DEVELOPMENT_TEST,
                    label="real",
                    generator=None,
                    camera_pipeline=f"{make}:{model}:web",
                    group=f"{make}:{model}:web",
                    transport="web_reconstructed",
                    filename=f"{sha256_bytes(source_key.encode())[:16]}{suffix}",
                    expected_bytes=length,
                    expected_sha256=str(row["content_sha256"]),
                    content_id=f"laion:{row['image_id']}",
                    url=url,
                )
            )
            running_bytes += length
            accepted += 1
        if accepted != LAION_PER_PIPELINE:
            raise RuntimeError(f"LAION pipeline {(make, model)!r} filled only {accepted}/10")
    enforce_byte_ceiling(
        [int(asset.expected_bytes or 0) for asset in selected],
        total_ceiling=LAION_IMAGE_CEILING,
        per_file_ceiling=LAION_PER_FILE_CEILING,
    )
    return selected, failures


def _selection_summary(payload: Mapping[str, Any]) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for record in payload["records"]:
        counts[record["group"]] = counts.get(record["group"], 0) + 1
    return {
        "state": payload["state"],
        "count": len(payload["records"]),
        "declared_or_downloaded_bytes": payload.get("declared_bytes", payload.get("image_bytes")),
        "groups": counts,
        "selection_sha256": payload.get("selection_sha256"),
        "content_set_sha256": payload.get("content_set_sha256"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=(
            "freeze-mllm",
            "download-mllm",
            "freeze-qwen",
            "download-qwen",
            "freeze-laion",
            "download-laion",
        ),
    )
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()
    root = args.output_root.resolve()

    if args.command.endswith("mllm"):
        assets = freeze_mllm_assets()
        output = root / "mllm_development"
        frozen = freeze_manifest(assets, output / "selection.json")
        payload = (
            download_assets(assets, output, total_ceiling=MLLM_TOTAL_CEILING)
            if args.command.startswith("download")
            else frozen
        )
    elif args.command.endswith("qwen"):
        assets = freeze_qwen_assets()
        output = root / "qwen_locked_final"
        frozen = freeze_manifest(assets, output / "selection.json")
        payload = (
            download_assets(assets, output, total_ceiling=QWEN_TOTAL_CEILING)
            if args.command.startswith("download")
            else frozen
        )
    else:
        output = root / "laion_mobile_development"
        rows = fetch_laion_metadata(output / "metadata")
        assets, failures = freeze_laion_assets(rows)
        frozen = freeze_manifest(assets, output / "selection.json")
        frozen["preflight_failures"] = failures
        _atomic_json(output / "selection.json", frozen)
        payload = (
            download_assets(assets, output, total_ceiling=LAION_IMAGE_CEILING)
            if args.command.startswith("download")
            else frozen
        )
    print(json.dumps(_selection_summary(payload), indent=2))


if __name__ == "__main__":
    main()
