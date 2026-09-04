"""Bind, score and evaluate the one-shot E50 generalist DEVELOPMENT transfer."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import joblib
import numpy as np
import torch
from PIL import Image

from experiments.e42_features import BLOCKS, MODEL_IDS, aggregate_tokens, texture_crops, transport_image
from experiments.e43_train import CANDIDATE as GENERALIST, _digest
from experiments.e48_decision import GATES, _gate, _rates, select_threshold
from experiments.e48_manifest import MANIFEST
from experiments.e48_score import GENERALIST_SHA256, MANIFEST_SHA256
from pixelproof.benchmark_metrics import evaluate_binary_scores
from pixelproof.e32_candidate import DINO_REPO_ID, DINO_WEIGHT_SHA256
from pixelproof.project_paths import DATA_ROOT, ML_ROOT


ROOT = DATA_ROOT / "e50"
CONTRACT = ROOT / "development_contract.json"
CONTRACT_EVIDENCE = ML_ROOT.parent / "evidence" / "e50_development_contract.json"
FITCAL_SCORES = DATA_ROOT / "e48" / "fitcal_generalist_scores.jsonl"
FITCAL_SCORES_SHA256 = "f2a1be3bf428952a2f8ae166d6bc0f47ba13559607aebb109e4c5adfdd137cf7"
THRESHOLD = 0.07940196245908739
SCORES = ROOT / "development_generalist_scores.jsonl"
SCORE_EVIDENCE = ML_ROOT.parent / "evidence" / "e50_development_scores.json"
REPORT = ROOT / "development_report.json"
REPORT_EVIDENCE = ML_ROOT.parent / "evidence" / "e50_development_result.json"


def _write(path: Path, value: Any) -> bytes:
    raw = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_bytes(raw)
    temporary.replace(path)
    return raw


def _manifest_rows() -> list[dict[str, Any]]:
    raw = MANIFEST.read_bytes()
    if hashlib.sha256(raw).hexdigest() != MANIFEST_SHA256:
        raise ValueError("E50 source manifest changed")
    payload = json.loads(raw)
    rows = payload.get("rows", [])
    if (payload.get("state") != "e48_decontaminated_frozen_unscored"
            or payload.get("model_scores_created") != 0 or len(rows) != 2_400):
        raise ValueError("E50 source manifest boundary changed")
    return rows


def _development_rows() -> list[dict[str, Any]]:
    rows = [row for row in _manifest_rows() if row["role"] == "DEVELOPMENT"]
    if (len(rows) != 1_200 or sum(int(row["label"]) == 0 for row in rows) != 600
            or sum(int(row["label"]) == 1 for row in rows) != 600):
        raise ValueError("E50 DEVELOPMENT boundary changed")
    return rows


def _load_fitcal_scores() -> list[dict[str, Any]]:
    if _digest(FITCAL_SCORES) != FITCAL_SCORES_SHA256:
        raise ValueError("E50 FIT+CAL generalist stream changed")
    rows = [json.loads(line) for line in FITCAL_SCORES.read_text().splitlines() if line]
    if (len(rows) != 1_200 or sum(row["role"] == "FIT" for row in rows) != 600
            or sum(row["role"] == "CAL" for row in rows) != 600
            or any(row["role"] == "DEVELOPMENT" for row in rows)):
        raise ValueError("E50 FIT+CAL role boundary changed")
    return rows


def _calibration_receipt() -> dict[str, Any]:
    rows = [row for row in _load_fitcal_scores() if row["role"] == "CAL"]
    threshold = select_threshold(rows)
    if threshold != THRESHOLD:
        raise ValueError(f"E50 threshold changed: {threshold!r}")
    metrics = evaluate_binary_scores(rows, threshold=THRESHOLD)
    rates = _rates(rows, THRESHOLD)
    gate = _gate(metrics, rates)
    if not gate["passed"]:
        raise ValueError("E50 frozen CAL candidate no longer passes")
    return {"rows": len(rows), "threshold": THRESHOLD, "metrics": metrics,
            "rates": rates, "gate": gate}


def bind() -> dict[str, Any]:
    if CONTRACT.exists() or CONTRACT_EVIDENCE.exists():
        raise FileExistsError("E50 DEVELOPMENT contract already exists")
    rows = _development_rows()
    if _digest(GENERALIST) != GENERALIST_SHA256:
        raise ValueError("E50 frozen generalist changed")
    calibration = _calibration_receipt()
    identities = [{"record_id": row["record_id"], "label": int(row["label"]),
                   "source": row["source"], "camera_pipeline": row.get("camera_pipeline"),
                   "sha256": row["sha256"]} for row in rows]
    identity_raw = json.dumps(identities, sort_keys=True, separators=(",", ":")).encode()
    payload = {
        "schema_version": 1, "state": "e50_candidate_frozen_before_development_model_load",
        "candidate": "exact frozen E43-S generalist only", "threshold": THRESHOLD,
        "gates": GATES, "calibration_receipt": calibration,
        "identities": {
            "source_manifest_sha256": MANIFEST_SHA256,
            "generalist_artifact_sha256": GENERALIST_SHA256,
            "fitcal_generalist_scores_sha256": FITCAL_SCORES_SHA256,
            "development_identity_sha256": hashlib.sha256(identity_raw).hexdigest(),
        },
        "counts": {"DEVELOPMENT": 1_200, "REAL": 600, "AI": 600},
        "sources": {"REAL": {"FODB": 600},
                    "AI": {"FLUX.1": 150, "StableDiffusion3": 150,
                           "StyleGAN": 150, "StyleGAN3": 150}},
        "forbidden": ["FIT or CAL rescore", "second candidate", "specialist or fusion use",
                      "training", "threshold change", "row replacement", "second DEVELOPMENT try"],
        "development_scores_created": 0, "development_metrics_opened": 0,
    }
    raw = _write(CONTRACT, payload)
    evidence = {**payload, "contract_bytes": len(raw),
                "contract_sha256": hashlib.sha256(raw).hexdigest()}
    _write(CONTRACT_EVIDENCE, evidence)
    return evidence


def _validate_contract() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    raw = CONTRACT.read_bytes()
    payload = json.loads(raw)
    evidence = json.loads(CONTRACT_EVIDENCE.read_text())
    if (hashlib.sha256(raw).hexdigest() != evidence.get("contract_sha256")
            or payload.get("development_scores_created") != 0
            or payload.get("development_metrics_opened") != 0
            or float(payload.get("threshold")) != THRESHOLD):
        raise ValueError("E50 DEVELOPMENT contract changed")
    if _digest(GENERALIST) != GENERALIST_SHA256:
        raise ValueError("E50 frozen generalist changed")
    rows = _development_rows()
    identities = [{"record_id": row["record_id"], "label": int(row["label"]),
                   "source": row["source"], "camera_pipeline": row.get("camera_pipeline"),
                   "sha256": row["sha256"]} for row in rows]
    identity_raw = json.dumps(identities, sort_keys=True, separators=(",", ":")).encode()
    if hashlib.sha256(identity_raw).hexdigest() != payload["identities"]["development_identity_sha256"]:
        raise ValueError("E50 DEVELOPMENT identities changed")
    return payload, rows


def _resume(path: Path, rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    raw = path.read_bytes()
    if raw and not raw.endswith(b"\n"):
        raise ValueError("E50 partial score line truncated")
    scored = [json.loads(line) for line in raw.splitlines() if line]
    if len(scored) > len(rows):
        raise ValueError("E50 score prefix exceeds DEVELOPMENT")
    for index, item in enumerate(scored):
        row = rows[index]
        if (item.get("record_id") != row["record_id"] or item.get("role") != "DEVELOPMENT"
                or item.get("source") != row["source"]
                or int(item.get("label", -1)) != int(row["label"])
                or not np.isfinite(float(item.get("score", np.nan)))):
            raise ValueError(f"E50 score prefix changed at {index}")
    return scored


def _append(path: Path, output: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab") as stream:
        for row in output:
            stream.write((json.dumps(row, sort_keys=True) + "\n").encode())
        stream.flush()
        os.fsync(stream.fileno())


def score_development(batch_size: int = 16) -> dict[str, Any]:
    _, rows = _validate_contract()
    if SCORES.exists() or SCORE_EVIDENCE.exists():
        raise FileExistsError("E50 DEVELOPMENT score already complete")
    partial = SCORES.with_suffix(".jsonl.partial")
    done = _resume(partial, rows)
    artifact = joblib.load(GENERALIST)
    from huggingface_hub import snapshot_download
    import timm
    snapshot = Path(snapshot_download(DINO_REPO_ID, local_files_only=True))
    if _digest(snapshot / "model.safetensors") != DINO_WEIGHT_SHA256:
        raise ValueError("DINOv2-S cache changed")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = timm.create_model(MODEL_IDS["small"], pretrained=True, num_classes=0,
                              img_size=224).to(device).eval()
    config = timm.data.resolve_data_config({}, model=model)
    mean = torch.tensor(config["mean"], device=device).view(1, 3, 1, 1)
    std = torch.tensor(config["std"], device=device).view(1, 3, 1, 1)

    def prepare(row: Mapping[str, Any]) -> list[np.ndarray]:
        path = Path(row["path"])
        if _digest(path) != row["sha256"]:
            raise ValueError(f"E50 payload changed: {row['record_id']}")
        with Image.open(path) as opened:
            return texture_crops(transport_image(opened, "clean"))

    with torch.inference_mode(), ThreadPoolExecutor(max_workers=6) as pool:
        for start in range(len(done), len(rows), batch_size):
            group = rows[start:start + batch_size]
            arrays = [array for pack in pool.map(prepare, group) for array in pack]
            tensor = torch.from_numpy(np.stack(arrays)).to(device).permute(0, 3, 1, 2).float().div_(255)
            inter = model.forward_intermediates((tensor - mean) / std, indices=list(BLOCKS["small"]),
                                                return_prefix_tokens=True, norm=True,
                                                intermediates_only=True)
            tokens = torch.stack([item[1][:, 0, :] for item in inter], dim=1)
            values = artifact["head"].predict_proba(
                aggregate_tokens(tokens.float().cpu().numpy(), len(group)))[:, 1]
            output = [{"record_id": row["record_id"], "role": "DEVELOPMENT",
                       "label": int(row["label"]), "source": row["source"],
                       "camera_pipeline": row.get("camera_pipeline"), "status": "ok",
                       "score": float(value)} for row, value in zip(group, values, strict=True)]
            _append(partial, output)
            done.extend(output)
            if len(done) % 100 == 0 or len(done) == len(rows):
                print(f"E50 generalist DEVELOPMENT {len(done)}/{len(rows)}", flush=True)
    if len(done) != len(rows):
        raise ValueError("E50 DEVELOPMENT score stream incomplete")
    partial.replace(SCORES)
    raw = SCORES.read_bytes()
    evidence = {"schema_version": 1, "state": "e50_development_scores_complete_unopened",
                "rows": len(rows), "coverage": 1.0, "bytes": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
                "contract_sha256": _digest(CONTRACT), "metrics_opened": 0}
    _write(SCORE_EVIDENCE, evidence)
    return evidence


def evaluate() -> dict[str, Any]:
    contract, expected = _validate_contract()
    if REPORT.exists() or REPORT_EVIDENCE.exists():
        raise FileExistsError("E50 DEVELOPMENT report already exists")
    score_receipt = json.loads(SCORE_EVIDENCE.read_text())
    if (_digest(SCORES) != score_receipt.get("sha256")
            or score_receipt.get("metrics_opened") != 0):
        raise ValueError("E50 DEVELOPMENT score stream changed")
    rows = [json.loads(line) for line in SCORES.read_text().splitlines() if line]
    _resume(SCORES, expected)
    metrics = evaluate_binary_scores(rows, threshold=THRESHOLD)
    rates = _rates(rows, THRESHOLD)
    gate = _gate(metrics, rates)
    report = {
        "schema_version": 1,
        "state": "e50_development_pass" if gate["passed"] else "e50_development_fail",
        "candidate": contract["candidate"], "threshold": THRESHOLD,
        "rows": len(rows), "metrics": metrics, "rates": rates, "gate": gate,
        "contract_sha256": _digest(CONTRACT), "score_sha256": _digest(SCORES),
        "retry_count": 0, "threshold_changed": False,
        "next": "bind independent E49 final" if gate["passed"] else "archive; do not repair on DEVELOPMENT",
    }
    raw = _write(REPORT, report)
    _write(REPORT_EVIDENCE, {**report, "report_sha256": hashlib.sha256(raw).hexdigest()})
    return report


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("bind", "score-development", "evaluate"))
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args(argv)
    if args.command == "bind":
        result = bind()
    elif args.command == "score-development":
        result = score_development(args.batch_size)
    else:
        result = evaluate()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
