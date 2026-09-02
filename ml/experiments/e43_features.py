"""Extract frozen E42-S features for score-blind E43 RR roles."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from PIL import Image

from experiments.e42_features import BLOCKS, MODEL_IDS, aggregate_tokens, texture_crops, transport_image
from pixelproof.e32_candidate import DINO_REPO_ID, DINO_WEIGHT_SHA256
from pixelproof.project_paths import DATA_ROOT, ML_ROOT


ROOT = DATA_ROOT / "e43"
MANIFEST = ROOT / "rr_roles.json"
MANIFEST_SHA256 = "29dd9b564061098101bcaf178cda0c75cdacc659113ce8b01cc389371bef4b16"
OUTPUT = ROOT / "rr_features_small.npz"
EVIDENCE = ML_ROOT.parent / "evidence" / "e43_rr_features_small.json"
EXPECTED_ROWS = 11_760
EXPECTED_WIDTH = 3_072


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024**2):
            digest.update(chunk)
    return digest.hexdigest()


def feature_rows(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    if manifest.get("state") != "e43_rr_roles_frozen_before_features":
        raise ValueError("E43 RR role manifest state changed")
    rows = [dict(row) for row in manifest.get("rows", [])]
    if len(rows) != EXPECTED_ROWS or len({str(row["record_id"]) for row in rows}) != len(rows):
        raise ValueError("E43 RR feature row population changed")
    if {str(row["e43_role"]) for row in rows} != {"train", "calibration", "development"}:
        raise ValueError("E43 RR feature roles changed")
    return rows


def _prepare(row: Mapping[str, Any]) -> list[np.ndarray]:
    path = Path(str(row["path"]))
    if _digest(path) != str(row["sha256"]):
        raise ValueError(f"E43 RR input changed: {row['record_id']}")
    with Image.open(path) as opened:
        return texture_crops(transport_image(opened, "clean"))


def _save_npz(path: Path, values: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".npz.part")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **values)
    temporary.replace(path)


def extract(batch_views: int = 24) -> dict[str, Any]:
    if OUTPUT.exists() or EVIDENCE.exists():
        raise FileExistsError("E43 RR feature output already exists; no silent replacement")
    if _digest(MANIFEST) != MANIFEST_SHA256:
        raise ValueError("E43 RR role manifest changed")
    rows = feature_rows(json.loads(MANIFEST.read_text()))

    from huggingface_hub import snapshot_download
    import timm
    import torch

    snapshot = Path(snapshot_download(DINO_REPO_ID, local_files_only=True))
    if _digest(snapshot / "model.safetensors") != DINO_WEIGHT_SHA256:
        raise ValueError("cached DINOv2-S weights changed")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = timm.create_model(MODEL_IDS["small"], pretrained=True, num_classes=0, img_size=224).to(device).eval()
    config = timm.data.resolve_data_config({}, model=model)
    mean = torch.tensor(config["mean"], device=device).view(1, 3, 1, 1)
    std = torch.tensor(config["std"], device=device).view(1, 3, 1, 1)
    chunks = []
    with torch.inference_mode(), ThreadPoolExecutor(max_workers=6) as pool:
        for start in range(0, len(rows), batch_views):
            group = rows[start : start + batch_views]
            arrays = [array for pack in pool.map(_prepare, group) for array in pack]
            tensor = torch.from_numpy(np.stack(arrays)).to(device)
            tensor = tensor.permute(0, 3, 1, 2).float().div_(255.0)
            intermediates = model.forward_intermediates(
                (tensor - mean) / std,
                indices=list(BLOCKS["small"]),
                return_prefix_tokens=True,
                norm=True,
                intermediates_only=True,
            )
            tokens = torch.stack([item[1][:, 0, :] for item in intermediates], dim=1)
            chunks.append(aggregate_tokens(tokens.float().cpu().numpy(), len(group)))
            print(f"E43 RR features {min(start + batch_views, len(rows))}/{len(rows)}", flush=True)
    features = np.concatenate(chunks)
    if features.shape != (EXPECTED_ROWS, EXPECTED_WIDTH) or not np.isfinite(features).all():
        raise ValueError(f"E43 RR feature output changed: {features.shape}")
    values = {
        "features": features,
        "record_ids": np.asarray([str(row["record_id"]) for row in rows]),
        "parent_ids": np.asarray([str(row["parent_id"]) for row in rows]),
        "labels": np.asarray([int(row["label"]) for row in rows], dtype=np.int8),
        "sources": np.asarray([str(row["source"]) for row in rows]),
        "conditions": np.asarray([str(row["condition"]) for row in rows]),
        "roles": np.asarray([str(row["e43_role"]) for row in rows]),
    }
    _save_npz(OUTPUT, values)
    report = {
        "schema_version": 1,
        "state": "e43_rr_small_features_complete_before_fit",
        "manifest_sha256": MANIFEST_SHA256,
        "model_id": MODEL_IDS["small"],
        "model_weight_sha256": DINO_WEIGHT_SHA256,
        "block_indices": list(BLOCKS["small"]),
        "feature_contract": "global plus two deterministic texture crops; per-block crop mean+std",
        "shape": list(features.shape),
        "feature_archive_bytes": OUTPUT.stat().st_size,
        "feature_archive_sha256": _digest(OUTPUT),
        "model_scores_created": 0,
    }
    EVIDENCE.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("extract",))
    parser.add_argument("--batch-views", type=int, default=24)
    args = parser.parse_args(argv)
    result = extract(args.batch_views) if args.command == "extract" else None
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
