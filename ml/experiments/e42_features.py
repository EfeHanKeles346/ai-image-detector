"""Extract frozen E42 global+texture intermediate DINOv2 features."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from PIL import Image, ImageFilter, ImageOps

from pixelproof.e32_candidate import (
    DINO_MODEL_ID,
    DINO_REPO_ID,
    DINO_WEIGHT_SHA256,
    standardized_array,
)
from pixelproof.project_paths import DATA_ROOT, ML_ROOT


REPO_ROOT = ML_ROOT.parent
E42_ROOT = DATA_ROOT / "e42"
MANIFEST = E42_ROOT / "parent_manifest.json"
MANIFEST_SHA256 = "15124d93f195d618b00c9cf79bec6151ae26fd4397cd9f5529c41842c4e3e238"
CONTRACT = REPO_ROOT / "evidence" / "e42_fixed_contract.json"
CONTRACT_SHA256 = "859d4ba812ba22678c0a6ec5e299244b999cab5d3b8ad72888d7c40f309ed279"
DDA_CHECKPOINT = DATA_ROOT / "e35_dda_model" / "DDA_ckpt.pth"
DDA_SHA256 = "b27a31d39374803ddeff02bfabb2be76e190b04300490cddfafb24f683f37e3e"
FEATURE_ROOT = E42_ROOT / "features"
TRANSPORTS = ("jpeg", "webp", "resize_jpeg", "mild_blur")
BLOCKS = {"small": (2, 5, 8, 11), "large": (5, 11, 17, 23)}
DIMENSIONS = {"small": 384, "large": 1024}
MODEL_IDS = {"small": DINO_MODEL_ID, "large": "vit_large_patch14_dinov2.lvd142m"}
MAX_LONG_SIDE = 2048
CROP_SIZE = 224
CROPS_PER_VIEW = 3


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024**2):
            digest.update(chunk)
    return digest.hexdigest()


def _write_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_bytes(_json_bytes(value))
    temporary.replace(path)


def assigned_transport(parent_id: str) -> str:
    index = int.from_bytes(hashlib.sha256(f"E42_TRANSPORT|{parent_id}".encode()).digest()[:4], "big")
    return TRANSPORTS[index % len(TRANSPORTS)]


def view_plan(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    planned = []
    for row in rows:
        conditions = (
            ("clean", assigned_transport(str(row["parent_id"])))
            if row["role"] == "train"
            else ("clean", *TRANSPORTS)
        )
        for condition in conditions:
            planned.append({
                "record_id": f"{row['parent_id']}|{condition}",
                "parent_id": row["parent_id"],
                "path": row["path"],
                "sha256": row["sha256"],
                "label": int(row["label"]),
                "source": row["source"],
                "role": row["role"],
                "condition": condition,
            })
    return planned


def _encoded(image: Image.Image, format_name: str, quality: int) -> Image.Image:
    stream = io.BytesIO()
    options = {"quality": quality}
    if format_name == "JPEG":
        options.update({"subsampling": 2, "optimize": False, "progressive": False})
    image.save(stream, format=format_name, **options)
    stream.seek(0)
    with Image.open(stream) as decoded:
        return decoded.convert("RGB").copy()


def transport_image(image: Image.Image, condition: str) -> Image.Image:
    oriented = _cap_long_side(ImageOps.exif_transpose(image).convert("RGB"))
    if condition == "clean":
        return oriented.copy()
    if condition == "jpeg":
        return _encoded(oriented, "JPEG", 55)
    if condition == "webp":
        return _encoded(oriented, "WEBP", 60)
    if condition == "resize_jpeg":
        width = max(1, round(oriented.width * 0.65))
        height = max(1, round(oriented.height * 0.65))
        resized = oriented.resize((width, height), Image.Resampling.LANCZOS)
        return _encoded(resized, "JPEG", 65)
    if condition == "mild_blur":
        return oriented.filter(ImageFilter.GaussianBlur(radius=0.8))
    raise ValueError(f"unknown E42 transport: {condition}")


def _cap_long_side(image: Image.Image) -> Image.Image:
    result = image
    long_side = max(result.size)
    if long_side > MAX_LONG_SIDE:
        scale = MAX_LONG_SIDE / long_side
        result = result.resize(
            (max(1, round(result.width * scale)), max(1, round(result.height * scale))),
            Image.Resampling.LANCZOS,
        )
    return result


def _cap_and_floor(image: Image.Image) -> Image.Image:
    result = _cap_long_side(image)
    short_side = min(result.size)
    if short_side < CROP_SIZE:
        scale = CROP_SIZE / short_side
        result = result.resize(
            (max(CROP_SIZE, round(result.width * scale)), max(CROP_SIZE, round(result.height * scale))),
            Image.Resampling.LANCZOS,
        )
    return result


def _positions(length: int) -> list[int]:
    available = length - CROP_SIZE
    if available <= 0:
        return [0]
    return sorted({round(available * fraction / 4) for fraction in range(5)})


def _texture_score(array: np.ndarray) -> float:
    grey = array.astype(np.float32).mean(axis=2)
    contrast = float(grey.std())
    gradient = float(np.abs(np.diff(grey, axis=0)).mean() + np.abs(np.diff(grey, axis=1)).mean())
    return contrast + gradient


def _iou(first: tuple[int, int], second: tuple[int, int]) -> float:
    ax, ay = first
    bx, by = second
    overlap_w = max(0, CROP_SIZE - abs(ax - bx))
    overlap_h = max(0, CROP_SIZE - abs(ay - by))
    intersection = overlap_w * overlap_h
    return intersection / float(2 * CROP_SIZE * CROP_SIZE - intersection)


def texture_crops(image: Image.Image) -> list[np.ndarray]:
    """Return global input plus two deterministic, preferably non-overlapping texture crops."""
    oriented = ImageOps.exif_transpose(image).convert("RGB")
    global_view = standardized_array(oriented)
    local_image = _cap_and_floor(oriented)
    candidates = []
    for y in _positions(local_image.height):
        for x in _positions(local_image.width):
            array = np.asarray(local_image.crop((x, y, x + CROP_SIZE, y + CROP_SIZE)), dtype=np.uint8)
            candidates.append((_texture_score(array), x, y, array))
    candidates.sort(key=lambda item: (-item[0], item[2], item[1]))
    selected: list[tuple[int, int, np.ndarray]] = []
    for _, x, y, array in candidates:
        if not selected or all(_iou((x, y), (px, py)) < 0.50 for px, py, _ in selected):
            selected.append((x, y, array))
        if len(selected) == 2:
            break
    for _, x, y, array in candidates:
        if len(selected) == 2:
            break
        if not any(x == px and y == py for px, py, _ in selected):
            selected.append((x, y, array))
    while len(selected) < 2:
        selected.append(selected[-1] if selected else (0, 0, global_view))
    return [global_view, selected[0][2], selected[1][2]]


def aggregate_tokens(tokens: np.ndarray, views: int) -> np.ndarray:
    """Aggregate [view*crop, block, dim] into fixed mean+std block features."""
    if tokens.ndim != 3 or tokens.shape[0] != views * CROPS_PER_VIEW or tokens.shape[1] != 4:
        raise ValueError(f"unexpected E42 token shape: {tokens.shape}")
    grouped = tokens.reshape(views, CROPS_PER_VIEW, tokens.shape[1], tokens.shape[2])
    means = grouped.mean(axis=1).reshape(views, -1)
    deviations = grouped.std(axis=1).reshape(views, -1)
    result = np.concatenate([means, deviations], axis=1).astype(np.float32)
    if not np.isfinite(result).all():
        raise ValueError("non-finite E42 aggregate feature")
    return result


def _load_small() -> tuple[Any, tuple[float, ...], tuple[float, ...], str]:
    from huggingface_hub import snapshot_download
    import timm

    snapshot = Path(snapshot_download(DINO_REPO_ID, local_files_only=True))
    if _sha256_file(snapshot / "model.safetensors") != DINO_WEIGHT_SHA256:
        raise ValueError("cached DINOv2-S weights changed")
    model = timm.create_model(DINO_MODEL_ID, pretrained=True, num_classes=0, img_size=CROP_SIZE)
    config = timm.data.resolve_data_config({}, model=model)
    return model, tuple(config["mean"]), tuple(config["std"]), DINO_WEIGHT_SHA256


def _load_large() -> tuple[Any, tuple[float, ...], tuple[float, ...], str]:
    import timm
    import torch
    from pixelproof.dda_candidate import _OfficialDDA, LoRALayer

    if _sha256_file(DDA_CHECKPOINT) != DDA_SHA256:
        raise ValueError("DDA DINOv2-L weight container changed")
    payload = torch.load(DDA_CHECKPOINT, map_location="cpu", weights_only=True)
    wrapper = _OfficialDDA()
    wrapper.load_state_dict(payload["model"], strict=True)
    del payload
    for module in wrapper.modules():
        if isinstance(module, LoRALayer):
            module.alpha = 0.0
    model = wrapper.base_model.model
    config = timm.data.resolve_data_config({}, model=model)
    return model, tuple(config["mean"]), tuple(config["std"]), DDA_SHA256


def _model(backbone: str) -> tuple[Any, tuple[float, ...], tuple[float, ...], str]:
    if backbone == "small":
        return _load_small()
    if backbone == "large":
        return _load_large()
    raise ValueError(f"unknown E42 backbone: {backbone}")


def _save_npz(path: Path, values: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".npz.part")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **values)
    temporary.replace(path)


def extract(backbone: str, batch_views: int) -> dict[str, Any]:
    output = FEATURE_ROOT / f"{backbone}.npz"
    evidence = REPO_ROOT / "evidence" / f"e42_features_{backbone}.json"
    if output.exists() or evidence.exists():
        raise FileExistsError(f"E42 {backbone} feature output already exists")
    if _sha256_file(MANIFEST) != MANIFEST_SHA256 or _sha256_file(CONTRACT) != CONTRACT_SHA256:
        raise ValueError("E42 feature input binding changed")
    manifest = json.loads(MANIFEST.read_text())
    rows = view_plan(manifest["rows"])
    if len(rows) != 20_506:
        raise ValueError(f"E42 view population changed: {len(rows)}")

    import torch

    model, mean_values, std_values, weight_sha256 = _model(backbone)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = model.to(device).eval()
    mean = torch.tensor(mean_values, device=device).view(1, 3, 1, 1)
    std = torch.tensor(std_values, device=device).view(1, 3, 1, 1)
    chunks = []
    verified_paths: dict[str, str] = {}
    with torch.inference_mode():
        for start in range(0, len(rows), batch_views):
            group = rows[start : start + batch_views]
            arrays = []
            for row in group:
                path = Path(str(row["path"]))
                key = str(path)
                if key not in verified_paths:
                    verified_paths[key] = _sha256_file(path)
                if verified_paths[key] != row["sha256"]:
                    raise ValueError(f"E42 feature input changed: {row['parent_id']}")
                with Image.open(path) as opened:
                    transformed = transport_image(opened, str(row["condition"]))
                    arrays.extend(texture_crops(transformed))
            tensor = torch.from_numpy(np.stack(arrays)).to(device)
            tensor = tensor.permute(0, 3, 1, 2).float().div_(255.0)
            intermediates = model.forward_intermediates(
                (tensor - mean) / std,
                indices=list(BLOCKS[backbone]),
                return_prefix_tokens=True,
                norm=True,
                intermediates_only=True,
            )
            tokens = torch.stack([item[1][:, 0, :] for item in intermediates], dim=1)
            chunks.append(aggregate_tokens(tokens.float().cpu().numpy(), len(group)))
            print(f"E42 {backbone} {min(start + batch_views, len(rows))}/{len(rows)}", flush=True)
    features = np.concatenate(chunks)
    expected_width = 2 * len(BLOCKS[backbone]) * DIMENSIONS[backbone]
    if features.shape != (20_506, expected_width):
        raise ValueError(f"E42 {backbone} feature shape changed: {features.shape}")
    values = {
        "features": features,
        "record_ids": np.asarray([str(row["record_id"]) for row in rows]),
        "parent_ids": np.asarray([str(row["parent_id"]) for row in rows]),
        "labels": np.asarray([int(row["label"]) for row in rows], dtype=np.int8),
        "sources": np.asarray([str(row["source"]) for row in rows]),
        "roles": np.asarray([str(row["role"]) for row in rows]),
        "conditions": np.asarray([str(row["condition"]) for row in rows]),
    }
    _save_npz(output, values)
    report = {
        "schema_version": 1,
        "experiment": "E42/texture-intermediate-features",
        "state": f"{backbone}_features_frozen_before_fit",
        "backbone": backbone,
        "model_id": MODEL_IDS[backbone],
        "block_indices": list(BLOCKS[backbone]),
        "weight_binding_sha256": weight_sha256,
        "manifest_sha256": MANIFEST_SHA256,
        "counts": {"views": len(rows), "parents": len(manifest["rows"]), "crops_per_view": 3},
        "feature_shape": list(features.shape),
        "feature_archive_bytes": output.stat().st_size,
        "feature_archive_sha256": _sha256_file(output),
        "boundary": "Consumed TRAIN/DEVELOPMENT features only; no classifier, B-Free or RR-test access.",
    }
    _write_atomic(evidence, report)
    return report


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("backbone", choices=("small", "large"))
    parser.add_argument("--batch-views", type=int)
    args = parser.parse_args(argv)
    batch_views = args.batch_views or (24 if args.backbone == "small" else 4)
    if batch_views <= 0:
        raise ValueError("batch views must be positive")
    print(json.dumps(extract(args.backbone, batch_views), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
