"""Bind and score E46 SynthWildX without touching the sealed final."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import joblib
import numpy as np
from PIL import Image

from experiments.e42_features import BLOCKS, MODEL_IDS, aggregate_tokens, texture_crops, transport_image
from experiments.e43_train import CANDIDATE as GENERALIST, _digest
from experiments.e44_fusion import CANDIDATE as FUSION
from pixelproof.dda_candidate import CHECKPOINT_SHA256, OfficialDDACandidate
from pixelproof.e32_candidate import DINO_REPO_ID, DINO_WEIGHT_SHA256
from pixelproof.project_paths import DATA_ROOT, ML_ROOT


ROOT = DATA_ROOT / "e46"
MANIFEST = ROOT / "synthwildx" / "audited_manifest_unscored.json"
MANIFEST_SHA256 = "953490a9c63669fac2305e6abcc4259f2f4066e8c58b7851efddedaa7e2da8d4"
EXPECTED_ROWS = 1_708
GENERALIST_SHA256 = "a3aec445926bcc8707b3775f01d2cdd9491ba8495ad8a8ec306840556ca47390"
FUSION_SHA256 = "19fd7bbcfed6ea85b9aa0c620663880f9fed24fbdbb084b09057283ea38bb100"
CONTRACT = ROOT / "development_score_contract.json"
CONTRACT_EVIDENCE = ML_ROOT.parent / "evidence" / "e46_development_score_contract.json"
GENERALIST_SCORES = ROOT / "synthwildx_generalist_scores.jsonl"
GENERALIST_EVIDENCE = ML_ROOT.parent / "evidence" / "e46_synthwildx_generalist_scores.json"
SPECIALIST_SCORES = ROOT / "synthwildx_specialist_scores.jsonl"
SPECIALIST_EVIDENCE = ML_ROOT.parent / "evidence" / "e46_synthwildx_specialist_scores.json"

DEVELOPMENT_GATES = {
    "pooled_real_false_ai_max": 0.20,
    "worst_real_group_false_ai_max": 0.25,
    "pooled_ai_recall_min": 0.80,
    "worst_ai_generator_recall_min": 0.60,
}


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def _write(path: Path, value: Any) -> bytes:
    raw = _json_bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_bytes(raw)
    temporary.replace(path)
    return raw


def _manifest_rows() -> list[dict[str, Any]]:
    raw = MANIFEST.read_bytes()
    if hashlib.sha256(raw).hexdigest() != MANIFEST_SHA256:
        raise ValueError("E46 SynthWildX audited manifest changed")
    payload = json.loads(raw)
    rows = payload.get("rows", [])
    if (
        payload.get("state") != "e46_synthwildx_audited_unscored"
        or payload.get("model_scores_created") != 0
        or len(rows) != EXPECTED_ROWS
        or len({row["record_id"] for row in rows}) != EXPECTED_ROWS
    ):
        raise ValueError("E46 SynthWildX manifest contract changed")
    return rows


def bind() -> dict[str, Any]:
    if CONTRACT.exists() or CONTRACT_EVIDENCE.exists():
        raise FileExistsError("E46 development score contract already exists")
    rows = _manifest_rows()
    checkpoint = DATA_ROOT / "e35_dda_model" / "DDA_ckpt.pth"
    identities = {
        "manifest_sha256": _digest(MANIFEST),
        "generalist_sha256": _digest(GENERALIST),
        "specialist_checkpoint_sha256": _digest(checkpoint),
        "fusion_sha256": _digest(FUSION),
    }
    expected = {
        "manifest_sha256": MANIFEST_SHA256,
        "generalist_sha256": GENERALIST_SHA256,
        "specialist_checkpoint_sha256": CHECKPOINT_SHA256,
        "fusion_sha256": FUSION_SHA256,
    }
    if identities != expected:
        raise ValueError(f"E46 model identity changed: {identities}")
    payload = {
        "schema_version": 1,
        "state": "e46_development_score_contract_frozen_before_model_load",
        "role": "E46_CAL_DEV",
        "identities": identities,
        "counts": {
            "rows": len(rows),
            "CAL": sum(row["role"] == "CAL" for row in rows),
            "DEVELOPMENT": sum(row["role"] == "DEVELOPMENT" for row in rows),
            "real": sum(int(row["label"]) == 0 for row in rows),
            "ai": sum(int(row["label"]) == 1 for row in rows),
        },
        "score_arms": ["E43-S generalist", "official DDA", "frozen E44 fusion"],
        "quality_proxies": ["log_min_dimension", "log_bits_per_pixel", "mean_neighbor_difference"],
        "candidate_methods": ["dda_global", "fusion_global", "fusion_quality_gaussian"],
        "calibration_target": {"real_false_ai_max": 0.10},
        "development_gates": DEVELOPMENT_GATES,
        "forbidden": [
            "E45 use", "TrueFake read or score", "role change", "unregistered candidate arm",
            "backbone retraining", "development-informed threshold refit",
        ],
        "model_scores_created": 0,
    }
    raw = _write(CONTRACT, payload)
    evidence = {**payload, "contract_bytes": len(raw),
                "contract_sha256": hashlib.sha256(raw).hexdigest()}
    _write(CONTRACT_EVIDENCE, evidence)
    return evidence


def _validate_contract() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    raw = CONTRACT.read_bytes()
    contract = json.loads(raw)
    evidence = json.loads(CONTRACT_EVIDENCE.read_text())
    if (
        contract.get("state") != "e46_development_score_contract_frozen_before_model_load"
        or contract.get("model_scores_created") != 0
        or hashlib.sha256(raw).hexdigest() != evidence.get("contract_sha256")
    ):
        raise ValueError("E46 development score contract changed")
    return contract, _manifest_rows()


def quality_proxies(image: Image.Image, payload_bytes: int) -> list[float]:
    rgb = image.convert("RGB")
    width, height = rgb.size
    if width <= 0 or height <= 0 or payload_bytes <= 0:
        raise ValueError("invalid quality-proxy input")
    thumb = rgb.copy()
    thumb.thumbnail((256, 256), Image.Resampling.BILINEAR)
    gray = np.asarray(thumb.convert("L"), dtype=np.float32) / 255.0
    differences = []
    if gray.shape[1] > 1:
        differences.append(np.abs(np.diff(gray, axis=1)).mean())
    if gray.shape[0] > 1:
        differences.append(np.abs(np.diff(gray, axis=0)).mean())
    edge = float(np.mean(differences)) if differences else 0.0
    return [
        math.log1p(min(width, height)),
        math.log1p(8.0 * payload_bytes / (width * height)),
        edge,
    ]


def _prepare(row: Mapping[str, Any]) -> tuple[list[np.ndarray], list[float]]:
    path = Path(str(row["path"]))
    if not path.is_file() or _digest(path) != str(row["sha256"]):
        raise ValueError(f"E46 payload changed: {row['record_id']}")
    with Image.open(path) as opened:
        opened.load()
        quality = quality_proxies(opened, int(row["bytes"]))
        arrays = texture_crops(transport_image(opened, "clean"))
    return arrays, quality


def _resume(path: Path, rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    raw = path.read_bytes()
    if raw and not raw.endswith(b"\n"):
        raise ValueError("incomplete E46 score line")
    scored = [json.loads(line) for line in raw.splitlines() if line]
    if len(scored) > len(rows):
        raise ValueError("E46 score prefix exceeds manifest")
    for index, item in enumerate(scored):
        if (
            item.get("record_id") != rows[index]["record_id"]
            or item.get("role") != rows[index]["role"]
            or int(item.get("label", -1)) != int(rows[index]["label"])
            or not np.isfinite(float(item.get("score", np.nan)))
        ):
            raise ValueError(f"E46 score prefix changed at {index}")
    return scored


def _append(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab") as stream:
        for row in rows:
            stream.write((json.dumps(row, sort_keys=True) + "\n").encode())
        stream.flush()
        os.fsync(stream.fileno())


def _finish(partial: Path, final: Path, evidence_path: Path, rows: Sequence[Mapping[str, Any]], arm: str) -> dict[str, Any]:
    scored = _resume(partial, rows)
    if len(scored) != len(rows):
        raise ValueError(f"E46 {arm} stream incomplete")
    partial.replace(final)
    raw = final.read_bytes()
    result = {
        "schema_version": 1, "state": f"e46_synthwildx_{arm}_scores_complete",
        "rows": len(rows), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest(),
        "coverage": 1.0, "manifest_sha256": MANIFEST_SHA256,
    }
    _write(evidence_path, result)
    return result


def score_generalist(batch_rows: int = 16) -> dict[str, Any]:
    if GENERALIST_SCORES.exists() or GENERALIST_EVIDENCE.exists():
        raise FileExistsError("E46 generalist scores already complete")
    if batch_rows < 1 or batch_rows > 32:
        raise ValueError("generalist batch must be 1..32")
    _, rows = _validate_contract()
    partial = GENERALIST_SCORES.with_suffix(GENERALIST_SCORES.suffix + ".partial")
    completed = _resume(partial, rows)
    artifact = joblib.load(GENERALIST)
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
    with torch.inference_mode(), ThreadPoolExecutor(max_workers=6) as pool:
        for start in range(len(completed), len(rows), batch_rows):
            group = rows[start:start + batch_rows]
            prepared = list(pool.map(_prepare, group))
            arrays = [array for pack, _ in prepared for array in pack]
            tensor = torch.from_numpy(np.stack(arrays)).to(device).permute(0, 3, 1, 2).float().div_(255.0)
            intermediate = model.forward_intermediates(
                (tensor - mean) / std, indices=list(BLOCKS["small"]),
                return_prefix_tokens=True, norm=True, intermediates_only=True,
            )
            tokens = torch.stack([item[1][:, 0, :] for item in intermediate], dim=1)
            features = aggregate_tokens(tokens.float().cpu().numpy(), len(group))
            scores = artifact["head"].predict_proba(features)[:, 1]
            batch = [{
                "record_id": row["record_id"], "role": row["role"], "label": int(row["label"]),
                "source": row["typ"], "score": float(score), "quality": quality, "status": "ok",
            } for row, score, (_, quality) in zip(group, scores, prepared, strict=True)]
            _append(partial, batch)
            done = min(start + len(group), len(rows))
            if done == len(rows) or done // 100 != start // 100:
                print(f"E46 generalist {done}/{len(rows)}", flush=True)
    return _finish(partial, GENERALIST_SCORES, GENERALIST_EVIDENCE, rows, "generalist")


def score_specialist(batch_rows: int = 8) -> dict[str, Any]:
    if SPECIALIST_SCORES.exists() or SPECIALIST_EVIDENCE.exists():
        raise FileExistsError("E46 specialist scores already complete")
    if batch_rows < 1 or batch_rows > 8:
        raise ValueError("specialist batch must be 1..8")
    _, rows = _validate_contract()
    partial = SPECIALIST_SCORES.with_suffix(SPECIALIST_SCORES.suffix + ".partial")
    completed = _resume(partial, rows)
    candidate = OfficialDDACandidate()
    import torch

    with torch.inference_mode():
        for start in range(len(completed), len(rows), batch_rows):
            group = rows[start:start + batch_rows]
            tensors = []
            for row in group:
                path = Path(str(row["path"]))
                if _digest(path) != str(row["sha256"]):
                    raise ValueError(f"E46 payload changed: {row['record_id']}")
                with Image.open(path) as opened:
                    tensors.append(candidate.transform(opened.convert("RGB")))
            scores = candidate.model(torch.stack(tensors).to(candidate.device)).sigmoid().flatten().cpu().numpy()
            batch = [{
                "record_id": row["record_id"], "role": row["role"], "label": int(row["label"]),
                "source": row["typ"], "score": float(score), "status": "ok",
            } for row, score in zip(group, scores, strict=True)]
            _append(partial, batch)
            done = min(start + len(group), len(rows))
            if done == len(rows) or done // 100 != start // 100:
                print(f"E46 specialist {done}/{len(rows)}", flush=True)
    return _finish(partial, SPECIALIST_SCORES, SPECIALIST_EVIDENCE, rows, "specialist")


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("bind", "score-generalist", "score-specialist"))
    parser.add_argument("--batch-rows", type=int, default=16)
    args = parser.parse_args(argv)
    if args.command == "bind":
        result = bind()
    elif args.command == "score-generalist":
        result = score_generalist(args.batch_rows)
    else:
        result = score_specialist(args.batch_rows)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
