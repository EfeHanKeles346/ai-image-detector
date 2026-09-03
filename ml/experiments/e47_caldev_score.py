"""Bind and score the four frozen E47 CAL/DEVELOPMENT arms."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, Sequence

import joblib
import numpy as np
import torch
from PIL import Image

from experiments.e42_features import BLOCKS, MODEL_IDS, aggregate_tokens, texture_crops, transport_image
from experiments.e43_train import CANDIDATE as GENERALIST, _digest
from experiments.e44_fusion import CANDIDATE as FUSION
from experiments.e47_caldev_manifest import MANIFEST
from experiments.e47_unina_diagnostic import CHECKOUT as UNINA_CHECKOUT, LONG_SIDE_CAP, WEIGHTS as UNINA_WEIGHTS
from experiments.e47_univfd_diagnostic import _load_official as load_univfd
from pixelproof.dda_candidate import CHECKPOINT_SHA256, OfficialDDACandidate
from pixelproof.e32_candidate import DINO_REPO_ID, DINO_WEIGHT_SHA256
from pixelproof.project_paths import DATA_ROOT, ML_ROOT


ROOT = DATA_ROOT / "e47"
MANIFEST_SHA256 = "378b83fe56bcf4bbf61d5b626efa71899bea571abeeeca05c74774daa8585739"
GENERALIST_SHA256 = "a3aec445926bcc8707b3775f01d2cdd9491ba8495ad8a8ec306840556ca47390"
FUSION_SHA256 = "19fd7bbcfed6ea85b9aa0c620663880f9fed24fbdbb084b09057283ea38bb100"
UNIVFD_BACKBONE_SHA256 = "b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836"
UNIVFD_HEAD_SHA256 = "477100745713bcc957beb2b40859536859b6483fd6301b3b9293151b194c7847"
UNINA_SHA256 = "65467594eeb53945417c909390a3d872d55b6dbd819aa12cf01e4ced9c4d5a08"
CONTRACT = ROOT / "caldev_score_contract.json"
CONTRACT_EVIDENCE = ML_ROOT.parent / "evidence" / "e47_caldev_score_contract.json"

ARM_PATHS = {
    "generalist": ROOT / "caldev_generalist_scores.jsonl",
    "dda": ROOT / "caldev_dda_scores.jsonl",
    "univfd": ROOT / "caldev_univfd_scores.jsonl",
    "unina": ROOT / "caldev_unina_scores.jsonl",
}


def _write(path: Path, value: Any) -> bytes:
    raw = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_bytes(raw)
    temporary.replace(path)
    return raw


def _rows() -> list[dict[str, Any]]:
    raw = MANIFEST.read_bytes()
    if hashlib.sha256(raw).hexdigest() != MANIFEST_SHA256:
        raise ValueError("E47 CAL/DEVELOPMENT manifest changed")
    payload = json.loads(raw)
    rows = payload.get("rows", [])
    if (payload.get("state") != "e47_caldev_decontaminated_frozen_unscored"
            or payload.get("model_scores_created") != 0 or len(rows) != 2_400):
        raise ValueError("E47 CAL/DEVELOPMENT manifest contract changed")
    return rows


def bind() -> dict[str, Any]:
    if CONTRACT.exists() or CONTRACT_EVIDENCE.exists():
        raise FileExistsError("E47 score contract already exists")
    rows = _rows()
    identities = {
        "manifest_sha256": _digest(MANIFEST),
        "generalist_sha256": _digest(GENERALIST),
        "dda_sha256": _digest(DATA_ROOT / "e35_dda_model" / "DDA_ckpt.pth"),
        "fusion_sha256": _digest(FUSION),
        "univfd_backbone_sha256": _digest(ROOT / "models" / "ViT-L-14.pt"),
        "univfd_head_sha256": _digest(ML_ROOT / "external" / "UniversalFakeDetect" / "pretrained_weights" / "fc_weights.pth"),
        "unina_sha256": _digest(UNINA_WEIGHTS),
    }
    expected = {
        "manifest_sha256": MANIFEST_SHA256,
        "generalist_sha256": GENERALIST_SHA256,
        "dda_sha256": CHECKPOINT_SHA256,
        "fusion_sha256": FUSION_SHA256,
        "univfd_backbone_sha256": UNIVFD_BACKBONE_SHA256,
        "univfd_head_sha256": UNIVFD_HEAD_SHA256,
        "unina_sha256": UNINA_SHA256,
    }
    if identities != expected:
        raise ValueError(f"E47 score identity changed: {identities}")
    payload = {
        "schema_version": 1,
        "state": "e47_caldev_score_contract_frozen_before_model_load",
        "identities": identities,
        "counts": {"rows": len(rows), "CAL": 1_200, "DEVELOPMENT": 1_200,
                   "real": 1_200, "ai": 1_200},
        "arms": {
            "generalist": "unchanged E43-S clean transport",
            "dda": "official DDA preprocessing and probability",
            "univfd": "official CLIP ViT-L/14 center crop and sigmoid linear head",
            "unina": f"official StyleGAN2 ResNet50-NoDown after frozen long-side {LONG_SIDE_CAP}px cap",
        },
        "forbidden": ["role change", "backbone update", "DEVELOPMENT-informed fit or threshold",
                      "E46 score fitting", "failed-row removal after model access"],
        "model_scores_created": 0,
    }
    raw = _write(CONTRACT, payload)
    evidence = {**payload, "contract_bytes": len(raw),
                "contract_sha256": hashlib.sha256(raw).hexdigest()}
    _write(CONTRACT_EVIDENCE, evidence)
    return evidence


def _validate() -> list[dict[str, Any]]:
    raw = CONTRACT.read_bytes()
    receipt = json.loads(CONTRACT_EVIDENCE.read_text())
    if (hashlib.sha256(raw).hexdigest() != receipt.get("contract_sha256")
            or json.loads(raw).get("model_scores_created") != 0):
        raise ValueError("E47 score contract changed")
    return _rows()


def _resume(path: Path, rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    raw = path.read_bytes()
    if raw and not raw.endswith(b"\n"):
        raise ValueError("E47 score partial line truncated")
    scored = [json.loads(line) for line in raw.splitlines() if line]
    if len(scored) > len(rows):
        raise ValueError("E47 score prefix exceeds manifest")
    for index, item in enumerate(scored):
        row = rows[index]
        if (item.get("record_id") != row["record_id"] or item.get("role") != row["role"]
                or item.get("source") != row["source"] or int(item.get("label", -1)) != int(row["label"])
                or not np.isfinite(float(item.get("score", np.nan)))):
            raise ValueError(f"E47 score prefix changed at {index}")
    return scored


def _append(path: Path, output: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab") as stream:
        for row in output:
            stream.write((json.dumps(row, sort_keys=True) + "\n").encode())
        stream.flush()
        os.fsync(stream.fileno())


def _finish(arm: str, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    partial = ARM_PATHS[arm].with_suffix(".jsonl.partial")
    scored = _resume(partial, rows)
    if len(scored) != len(rows):
        raise ValueError(f"E47 {arm} score stream incomplete")
    partial.replace(ARM_PATHS[arm])
    raw = ARM_PATHS[arm].read_bytes()
    evidence = {
        "schema_version": 1, "state": f"e47_caldev_{arm}_scores_complete",
        "rows": len(rows), "coverage": 1.0, "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(), "manifest_sha256": MANIFEST_SHA256,
    }
    _write(ML_ROOT.parent / "evidence" / f"e47_caldev_{arm}_scores.json", evidence)
    return evidence


def _base(row: Mapping[str, Any], score: float) -> dict[str, Any]:
    return {"record_id": row["record_id"], "role": row["role"], "label": int(row["label"]),
            "source": row["source"], "score": float(score), "status": "ok"}


def score_generalist(batch_size: int = 16) -> dict[str, Any]:
    rows = _validate(); partial = ARM_PATHS["generalist"].with_suffix(".jsonl.partial")
    done = _resume(partial, rows)
    artifact = joblib.load(GENERALIST)
    from huggingface_hub import snapshot_download
    import timm
    snapshot = Path(snapshot_download(DINO_REPO_ID, local_files_only=True))
    if _digest(snapshot / "model.safetensors") != DINO_WEIGHT_SHA256:
        raise ValueError("DINOv2-S cache changed")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = timm.create_model(MODEL_IDS["small"], pretrained=True, num_classes=0, img_size=224).to(device).eval()
    config = timm.data.resolve_data_config({}, model=model)
    mean = torch.tensor(config["mean"], device=device).view(1, 3, 1, 1)
    std = torch.tensor(config["std"], device=device).view(1, 3, 1, 1)
    def prepare(row):
        path = Path(row["path"])
        if _digest(path) != row["sha256"]: raise ValueError(f"payload changed: {row['record_id']}")
        with Image.open(path) as opened: return texture_crops(transport_image(opened, "clean"))
    with torch.inference_mode(), ThreadPoolExecutor(max_workers=6) as pool:
        for start in range(len(done), len(rows), batch_size):
            group = rows[start:start + batch_size]
            arrays = [array for pack in pool.map(prepare, group) for array in pack]
            tensor = torch.from_numpy(np.stack(arrays)).to(device).permute(0, 3, 1, 2).float().div_(255)
            inter = model.forward_intermediates((tensor - mean) / std, indices=list(BLOCKS["small"]),
                                                return_prefix_tokens=True, norm=True, intermediates_only=True)
            tokens = torch.stack([item[1][:, 0, :] for item in inter], dim=1)
            values = artifact["head"].predict_proba(aggregate_tokens(tokens.float().cpu().numpy(), len(group)))[:, 1]
            output = [_base(row, value) for row, value in zip(group, values, strict=True)]
            _append(partial, output); done.extend(output)
            if len(done) % 100 == 0 or len(done) == len(rows): print(f"E47 generalist {len(done)}/{len(rows)}", flush=True)
    return _finish("generalist", rows)


def score_dda(batch_size: int = 8) -> dict[str, Any]:
    rows = _validate(); partial = ARM_PATHS["dda"].with_suffix(".jsonl.partial"); done = _resume(partial, rows)
    candidate = OfficialDDACandidate()
    with torch.inference_mode():
        for start in range(len(done), len(rows), batch_size):
            group = rows[start:start + batch_size]; tensors = []
            for row in group:
                path = Path(row["path"])
                if _digest(path) != row["sha256"]: raise ValueError(f"payload changed: {row['record_id']}")
                with Image.open(path) as opened: tensors.append(candidate.transform(opened.convert("RGB")))
            values = candidate.model(torch.stack(tensors).to(candidate.device)).sigmoid().flatten().cpu().numpy()
            output = [_base(row, value) for row, value in zip(group, values, strict=True)]
            _append(partial, output); done.extend(output)
            if len(done) % 100 == 0 or len(done) == len(rows): print(f"E47 DDA {len(done)}/{len(rows)}", flush=True)
    return _finish("dda", rows)


def score_univfd(batch_size: int = 8) -> dict[str, Any]:
    rows = _validate(); partial = ARM_PATHS["univfd"].with_suffix(".jsonl.partial"); done = _resume(partial, rows)
    model, head, preprocess, device = load_univfd()
    with torch.inference_mode():
        for start in range(len(done), len(rows), batch_size):
            group = rows[start:start + batch_size]; tensors = []
            for row in group:
                path = Path(row["path"])
                if _digest(path) != row["sha256"]: raise ValueError(f"payload changed: {row['record_id']}")
                with Image.open(path) as opened: tensors.append(preprocess(opened.convert("RGB")))
            values = torch.sigmoid(head(model.encode_image(torch.stack(tensors).to(device)).float())).flatten().cpu().numpy()
            output = [_base(row, value) for row, value in zip(group, values, strict=True)]
            _append(partial, output); done.extend(output)
            if len(done) % 100 == 0 or len(done) == len(rows): print(f"E47 UnivFD {len(done)}/{len(rows)}", flush=True)
    return _finish("univfd", rows)


def score_unina(batch_size: int = 4) -> dict[str, Any]:
    rows = _validate(); partial = ARM_PATHS["unina"].with_suffix(".jsonl.partial"); done = _resume(partial, rows)
    sys.path.insert(0, str(UNINA_CHECKOUT)); from resnet50nodown import resnet50nodown
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = resnet50nodown(device, str(UNINA_WEIGHTS))
    with torch.inference_mode():
        while len(done) < len(rows):
            start = len(done); tensors = []; group = []
            target_size = None
            while start + len(group) < len(rows) and len(group) < batch_size:
                row = rows[start + len(group)]; path = Path(row["path"])
                if _digest(path) != row["sha256"]: raise ValueError(f"payload changed: {row['record_id']}")
                with Image.open(path) as opened:
                    picture = opened.convert("RGB"); scale = min(1.0, LONG_SIDE_CAP / max(picture.size))
                    if scale < 1: picture = picture.resize((round(picture.width*scale), round(picture.height*scale)), Image.Resampling.LANCZOS)
                    tensor = model.transform(picture)
                if target_size is not None and tuple(tensor.shape) != target_size: break
                target_size = tuple(tensor.shape); tensors.append(tensor); group.append(row)
            values = model(torch.stack(tensors).to(device)).flatten().cpu().numpy()
            output = [_base(row, value) for row, value in zip(group, values, strict=True)]
            _append(partial, output); done.extend(output)
            if len(done) % 100 == 0 or len(done) == len(rows): print(f"E47 UNINA {len(done)}/{len(rows)}", flush=True)
    return _finish("unina", rows)


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("bind", "score-generalist", "score-dda", "score-univfd", "score-unina"))
    parser.add_argument("--batch-size", type=int)
    args = parser.parse_args(argv)
    functions = {"bind": bind, "score-generalist": score_generalist, "score-dda": score_dda,
                 "score-univfd": score_univfd, "score-unina": score_unina}
    result = functions[args.command]() if args.batch_size is None or args.command == "bind" else functions[args.command](args.batch_size)
    print(json.dumps(result, indent=2, sort_keys=True)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
