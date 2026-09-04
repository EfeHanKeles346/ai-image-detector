"""Lock, score and open the one-shot E49-C comprehensive final in separate checkpoints."""

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
from PIL import Image
import torch

from experiments.e42_features import BLOCKS, MODEL_IDS, aggregate_tokens, texture_crops, transport_image
from experiments.e43_train import CANDIDATE as GENERALIST, _digest
from experiments.e48_score import GENERALIST_SHA256
from experiments.e49_evaluation import BINARY_THRESHOLD, GATES, REAL_CUT, evaluate_final, validate_paired_final
from experiments.e49_final_manifest import MANIFEST
from pixelproof.e32_candidate import DINO_REPO_ID, DINO_WEIGHT_SHA256
from pixelproof.project_paths import DATA_ROOT, ML_ROOT


ROOT = DATA_ROOT / "e49" / "final"
CONTRACT = ROOT / "score_contract.json"
CONTRACT_EVIDENCE = ML_ROOT.parent / "evidence" / "e49_final_score_contract.json"
SCORES = ROOT / "generalist_scores.jsonl"
SCORE_EVIDENCE = ML_ROOT.parent / "evidence" / "e49_final_scores.json"
REPORT = ROOT / "final_report.json"
REPORT_EVIDENCE = ML_ROOT.parent / "evidence" / "e49_final_result.json"


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def _write_atomic(path: Path, value: Any) -> bytes:
    raw = _json_bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_bytes(raw)
    temporary.replace(path)
    return raw


def _identity_sha(rows: Sequence[Mapping[str, Any]]) -> str:
    identities = [{key: row[key] for key in (
        "record_id", "parent_id", "condition", "label", "source", "sha256",
    )} for row in rows]
    return hashlib.sha256(json.dumps(identities, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _manifest_rows() -> tuple[dict[str, Any], bytes, list[dict[str, Any]]]:
    raw = MANIFEST.read_bytes()
    payload = json.loads(raw)
    rows = payload.get("rows") or []
    if (
        payload.get("state") != "e49_c_comprehensive_final_frozen_unscored"
        or payload.get("model_scores_created") != 0 or payload.get("metrics_opened") != 0
        or len(rows) != 4_000
    ):
        raise ValueError("E49 final manifest boundary changed")
    validate_paired_final(rows)
    for row in rows:
        if not row.get("path") or not row.get("sha256") or row.get("status") != "unscored":
            raise ValueError(f"E49 final payload identity changed: {row.get('record_id')}")
    return payload, raw, rows


def _dino_snapshot() -> Path:
    from huggingface_hub import snapshot_download
    snapshot = Path(snapshot_download(DINO_REPO_ID, local_files_only=True))
    if _digest(snapshot / "model.safetensors") != DINO_WEIGHT_SHA256:
        raise ValueError("E49 final DINOv2-S cache changed")
    return snapshot


def bind_score() -> dict[str, Any]:
    if CONTRACT.exists() or CONTRACT_EVIDENCE.exists():
        raise FileExistsError("E49 final score contract already exists")
    manifest, manifest_raw, rows = _manifest_rows()
    if _digest(GENERALIST) != GENERALIST_SHA256:
        raise ValueError("E49 final E43-S artifact changed")
    _dino_snapshot()
    payload = {
        "schema_version": 1, "state": "e49_c_candidate_locked_before_model_load",
        "candidate": "exact frozen E43-S generalist only",
        "identities": {
            "manifest_sha256": hashlib.sha256(manifest_raw).hexdigest(),
            "observation_identity_sha256": _identity_sha(rows),
            "component_manifest_sha256": manifest["component_manifest_sha256"],
            "generalist_artifact_sha256": GENERALIST_SHA256,
            "dino_weight_sha256": DINO_WEIGHT_SHA256,
        },
        "thresholds": {"binary_ai": BINARY_THRESHOLD, "selective_real": REAL_CUT},
        "gates": GATES,
        "counts": {"parents": 2_000, "observations": 4_000, "conditions": 2, "checks": 20},
        "forbidden": ["training", "threshold change", "row/source removal",
                      "score-based replacement", "second attempt", "metrics before raw-score lock"],
        "scores_created": 0, "metrics_opened": 0,
    }
    raw = _write_atomic(CONTRACT, payload)
    evidence = {**payload, "contract_bytes": len(raw),
                "contract_sha256": hashlib.sha256(raw).hexdigest()}
    _write_atomic(CONTRACT_EVIDENCE, evidence)
    return evidence


def _validate_contract() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    raw = CONTRACT.read_bytes()
    payload = json.loads(raw)
    evidence = json.loads(CONTRACT_EVIDENCE.read_text())
    _, manifest_raw, rows = _manifest_rows()
    if (
        payload.get("state") != "e49_c_candidate_locked_before_model_load"
        or payload.get("scores_created") != 0 or payload.get("metrics_opened") != 0
        or hashlib.sha256(raw).hexdigest() != evidence.get("contract_sha256")
        or payload.get("identities", {}).get("manifest_sha256") != hashlib.sha256(manifest_raw).hexdigest()
        or payload.get("identities", {}).get("observation_identity_sha256") != _identity_sha(rows)
        or payload.get("identities", {}).get("generalist_artifact_sha256") != GENERALIST_SHA256
    ):
        raise ValueError("E49 final score contract changed")
    if _digest(GENERALIST) != GENERALIST_SHA256:
        raise ValueError("E49 final E43-S artifact changed")
    return payload, rows


def _resume(path: Path, expected: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    raw = path.read_bytes()
    if raw and not raw.endswith(b"\n"):
        raise ValueError("E49 final partial score line truncated")
    rows = [json.loads(line) for line in raw.splitlines() if line]
    if len(rows) > len(expected):
        raise ValueError("E49 final score prefix exceeds manifest")
    for index, row in enumerate(rows):
        target = expected[index]
        score = float(row.get("score", np.nan))
        if (
            any(row.get(key) != target[key] for key in ("record_id", "parent_id", "condition", "label", "source"))
            or row.get("status") != "ok" or not np.isfinite(score) or not 0 <= score <= 1
        ):
            raise ValueError(f"E49 final score prefix changed at {index}")
    return rows


def _append(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab") as stream:
        for row in rows:
            stream.write((json.dumps(row, sort_keys=True) + "\n").encode())
        stream.flush()
        os.fsync(stream.fileno())


def validate_scores(rows: Sequence[Mapping[str, Any]]) -> None:
    validate_paired_final(rows)
    for row in rows:
        score = float(row.get("score", np.nan))
        if row.get("status") != "ok" or not np.isfinite(score) or not 0 <= score <= 1:
            raise ValueError(f"E49 final invalid score: {row.get('record_id')}")


def score(batch_size: int = 16) -> dict[str, Any]:
    _, rows = _validate_contract()
    if SCORES.exists() or SCORE_EVIDENCE.exists():
        raise FileExistsError("E49 final scores already complete")
    partial = SCORES.with_suffix(".jsonl.partial")
    done = _resume(partial, rows)
    artifact = joblib.load(GENERALIST)
    _dino_snapshot()
    import timm
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = timm.create_model(MODEL_IDS["small"], pretrained=True, num_classes=0,
                              img_size=224).to(device).eval()
    config = timm.data.resolve_data_config({}, model=model)
    mean = torch.tensor(config["mean"], device=device).view(1, 3, 1, 1)
    std = torch.tensor(config["std"], device=device).view(1, 3, 1, 1)

    def prepare(row: Mapping[str, Any]) -> list[np.ndarray]:
        path = Path(str(row["path"]))
        if _digest(path) != row["sha256"]:
            raise ValueError(f"E49 final payload changed: {row['record_id']}")
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
            output = [{"record_id": row["record_id"], "parent_id": row["parent_id"],
                       "condition": row["condition"], "label": int(row["label"]),
                       "source": row["source"], "status": "ok", "score": float(value)}
                      for row, value in zip(group, values, strict=True)]
            _append(partial, output)
            done.extend(output)
            if len(done) % 100 == 0 or len(done) == len(rows):
                print(f"E49 final E43-S {len(done)}/{len(rows)}", flush=True)
    validate_scores(done)
    partial.replace(SCORES)
    raw = SCORES.read_bytes()
    evidence = {
        "schema_version": 1, "state": "e49_c_scores_complete_unopened",
        "rows": len(done), "coverage": 1.0, "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(), "contract_sha256": _digest(CONTRACT),
        "metrics_opened": 0,
    }
    _write_atomic(SCORE_EVIDENCE, evidence)
    return evidence


def open_metrics() -> dict[str, Any]:
    if REPORT.exists() or REPORT_EVIDENCE.exists():
        raise FileExistsError("E49 final metrics already opened")
    contract, _ = _validate_contract()
    score_evidence = json.loads(SCORE_EVIDENCE.read_text())
    raw = SCORES.read_bytes()
    if (
        score_evidence.get("state") != "e49_c_scores_complete_unopened"
        or score_evidence.get("metrics_opened") != 0
        or score_evidence.get("contract_sha256") != _digest(CONTRACT)
        or score_evidence.get("sha256") != hashlib.sha256(raw).hexdigest()
    ):
        raise ValueError("E49 final raw-score lock changed")
    rows = [json.loads(line) for line in raw.splitlines() if line]
    validate_scores(rows)
    report = evaluate_final(rows)
    report["score_contract_sha256"] = _digest(CONTRACT)
    report["raw_scores_sha256"] = score_evidence["sha256"]
    report["manifest_sha256"] = contract["identities"]["manifest_sha256"]
    report_raw = _write_atomic(REPORT, report)
    conditions = {
        key: {"binary_metrics": value["binary_metrics"], "binary_rates": value["binary_rates"],
              "selective": value["selective"], "gate": value["gate"],
              "bootstrap_95pct": value["bootstrap_95pct"]}
        for key, value in report["conditions"].items()
    }
    evidence = {
        "schema_version": 1, "state": report["state"], "gate": report["gate"],
        "candidate": report["candidate"], "parent_count": 2_000, "observation_count": 4_000,
        "conditions": conditions, "raw_scores_sha256": report["raw_scores_sha256"],
        "manifest_sha256": report["manifest_sha256"], "report_bytes": len(report_raw),
        "report_sha256": hashlib.sha256(report_raw).hexdigest(), "retry_count": 0,
    }
    _write_atomic(REPORT_EVIDENCE, evidence)
    return evidence


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("bind-score", "score", "open-metrics"))
    args = parser.parse_args(argv)
    result = bind_score() if args.command == "bind-score" else score() if args.command == "score" else open_metrics()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
