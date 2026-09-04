"""Lock, score and open the paired E49-D1 AI-only diagnostic exactly once."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
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
from experiments.e48_score import GENERALIST_SHA256
from experiments.e49_dotting import MANIFEST, MODEL_KEYS
from pixelproof.e32_candidate import DINO_REPO_ID, DINO_WEIGHT_SHA256
from pixelproof.project_paths import DATA_ROOT, ML_ROOT


ROOT = DATA_ROOT / "e49_d1_dotting"
MANIFEST_SHA256 = "048572a41b47b65d3d09bd39bee45a40745a40c2e15c50444b41f8384fdccc9d"
BINARY_THRESHOLD = 0.07940196245908739
REAL_CUT = 0.011505939625203613
CONDITIONS = ("publisher_original", "social_q75")
TARGET_PER_MODEL = 160
CONTRACT = ROOT / "score_contract.json"
CONTRACT_EVIDENCE = ML_ROOT.parent / "evidence" / "e49_d1_dotting_score_contract.json"
SCORES = ROOT / "generalist_scores.jsonl"
SCORE_EVIDENCE = ML_ROOT.parent / "evidence" / "e49_d1_dotting_scores.json"
REPORT = ROOT / "diagnostic_report.json"
REPORT_EVIDENCE = ML_ROOT.parent / "evidence" / "e49_d1_dotting_result.json"
GATES = {"score_coverage": 1.0, "pooled_ai_recall_min": 0.80, "worst_model_recall_min": 0.60}


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
        raise ValueError("E49-D1 frozen manifest changed")
    payload = json.loads(raw)
    rows = payload.get("rows", [])
    if (payload.get("state") != "e49_d1_decontaminated_paired_frozen_unscored"
            or payload.get("model_scores_created") != 0 or len(rows) != 1_600):
        raise ValueError("E49-D1 unscored boundary changed")
    validate_rows(rows, require_scores=False)
    return rows


def validate_rows(rows: Sequence[Mapping[str, Any]], *, require_scores: bool) -> None:
    if len(rows) != 1_600:
        raise ValueError("E49-D1 observation count changed")
    parent_maps: dict[str, dict[str, str]] = {}
    expected_sources = set(MODEL_KEYS.values())
    for condition in CONDITIONS:
        selected = [row for row in rows if row.get("condition") == condition]
        if len(selected) != 800:
            raise ValueError(f"E49-D1 {condition} count changed")
        counts = Counter(str(row.get("source")) for row in selected)
        if counts != Counter({source: TARGET_PER_MODEL for source in expected_sources}):
            raise ValueError(f"E49-D1 {condition} source balance changed")
        parents: dict[str, str] = {}
        for row in selected:
            parent_id = str(row.get("parent_id", ""))
            if (not parent_id or parent_id in parents or int(row.get("label", -1)) != 1
                    or not row.get("record_id")):
                raise ValueError(f"E49-D1 {condition} row identity changed")
            parents[parent_id] = str(row["source"])
            if require_scores:
                score = float(row.get("score", np.nan))
                if row.get("status") != "ok" or not np.isfinite(score) or not 0 <= score <= 1:
                    raise ValueError(f"E49-D1 invalid completed score for {row['record_id']}")
            elif not row.get("path") or not row.get("sha256"):
                raise ValueError(f"E49-D1 {condition} payload identity changed")
        parent_maps[condition] = parents
    if parent_maps[CONDITIONS[0]] != parent_maps[CONDITIONS[1]]:
        raise ValueError("E49-D1 transport pairs changed")


def _identity_sha(rows: Sequence[Mapping[str, Any]]) -> str:
    identities = [{key: row[key] for key in ("record_id", "parent_id", "condition", "source", "sha256")}
                  for row in rows]
    return hashlib.sha256(json.dumps(identities, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _dino_snapshot() -> Path:
    from huggingface_hub import snapshot_download
    snapshot = Path(snapshot_download(DINO_REPO_ID, local_files_only=True))
    if _digest(snapshot / "model.safetensors") != DINO_WEIGHT_SHA256:
        raise ValueError("E49-D1 DINOv2-S cache changed")
    return snapshot


def bind_score() -> dict[str, Any]:
    if CONTRACT.exists() or CONTRACT_EVIDENCE.exists():
        raise FileExistsError("E49-D1 score contract already exists")
    rows = _manifest_rows()
    if _digest(GENERALIST) != GENERALIST_SHA256:
        raise ValueError("E49-D1 frozen E43-S artifact changed")
    _dino_snapshot()
    payload = {
        "schema_version": 1,
        "state": "e49_d1_candidate_locked_before_model_load",
        "candidate": "exact frozen E43-S generalist only",
        "identities": {"manifest_sha256": MANIFEST_SHA256,
                       "observation_identity_sha256": _identity_sha(rows),
                       "generalist_artifact_sha256": GENERALIST_SHA256,
                       "dino_weight_sha256": DINO_WEIGHT_SHA256},
        "thresholds": {"binary_ai": BINARY_THRESHOLD, "selective_real": REAL_CUT},
        "gates": GATES,
        "counts": {"parents": 800, "observations": 1_600, "conditions": 2,
                   "per_model_per_condition": TARGET_PER_MODEL},
        "forbidden": ["training", "threshold change", "row/source removal", "score-based replacement",
                      "second attempt", "AUC/BA/REAL-safety claim", "Module-1 promotion"],
        "scores_created": 0,
        "metrics_opened": 0,
    }
    raw = _write(CONTRACT, payload)
    evidence = {**payload, "contract_bytes": len(raw), "contract_sha256": hashlib.sha256(raw).hexdigest()}
    _write(CONTRACT_EVIDENCE, evidence)
    return evidence


def _validate_contract() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    raw = CONTRACT.read_bytes()
    payload = json.loads(raw)
    evidence = json.loads(CONTRACT_EVIDENCE.read_text())
    rows = _manifest_rows()
    if (payload.get("state") != "e49_d1_candidate_locked_before_model_load"
            or payload.get("scores_created") != 0 or payload.get("metrics_opened") != 0
            or hashlib.sha256(raw).hexdigest() != evidence.get("contract_sha256")
            or payload.get("identities", {}).get("observation_identity_sha256") != _identity_sha(rows)
            or payload.get("identities", {}).get("generalist_artifact_sha256") != GENERALIST_SHA256):
        raise ValueError("E49-D1 score contract changed")
    if _digest(GENERALIST) != GENERALIST_SHA256:
        raise ValueError("E49-D1 frozen E43-S artifact changed")
    return payload, rows


def _resume(path: Path, expected: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    raw = path.read_bytes()
    if raw and not raw.endswith(b"\n"):
        raise ValueError("E49-D1 partial score line truncated")
    rows = [json.loads(line) for line in raw.splitlines() if line]
    if len(rows) > len(expected):
        raise ValueError("E49-D1 score prefix exceeds manifest")
    for index, row in enumerate(rows):
        target = expected[index]
        score = float(row.get("score", np.nan))
        if (row.get("record_id") != target["record_id"] or row.get("parent_id") != target["parent_id"]
                or row.get("condition") != target["condition"] or row.get("source") != target["source"]
                or row.get("status") != "ok" or not np.isfinite(score) or not 0 <= score <= 1):
            raise ValueError(f"E49-D1 score prefix changed at {index}")
    return rows


def _append(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab") as stream:
        for row in rows:
            stream.write((json.dumps(row, sort_keys=True) + "\n").encode())
        stream.flush()
        os.fsync(stream.fileno())


def score(batch_size: int = 16) -> dict[str, Any]:
    _, rows = _validate_contract()
    if SCORES.exists() or SCORE_EVIDENCE.exists():
        raise FileExistsError("E49-D1 score already complete")
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
            raise ValueError(f"E49-D1 payload changed: {row['record_id']}")
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
                       "condition": row["condition"], "label": 1, "source": row["source"],
                       "status": "ok", "score": float(value)}
                      for row, value in zip(group, values, strict=True)]
            _append(partial, output)
            done.extend(output)
            if len(done) % 100 == 0 or len(done) == len(rows):
                print(f"E49-D1 E43-S {len(done)}/{len(rows)}", flush=True)
    validate_rows(done, require_scores=True)
    partial.replace(SCORES)
    raw = SCORES.read_bytes()
    evidence = {"schema_version": 1, "state": "e49_d1_scores_complete_unopened",
                "rows": len(done), "coverage": 1.0, "bytes": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(), "contract_sha256": _digest(CONTRACT),
                "metrics_opened": 0}
    _write(SCORE_EVIDENCE, evidence)
    return evidence


def evaluate_diagnostic(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    validate_rows(rows, require_scores=True)
    reports: dict[str, Any] = {}
    score_maps: dict[str, dict[str, float]] = {}
    for condition in CONDITIONS:
        selected = [row for row in rows if row["condition"] == condition]
        per_model: dict[str, float] = {}
        categories: Counter[str] = Counter()
        for row in selected:
            score_value = float(row["score"])
            categories["confident_ai" if score_value >= BINARY_THRESHOLD
                       else "confident_real_miss" if score_value < REAL_CUT else "uncertain"] += 1
        for source in sorted(MODEL_KEYS.values()):
            model_rows = [row for row in selected if row["source"] == source]
            per_model[source] = float(np.mean([float(row["score"]) >= BINARY_THRESHOLD for row in model_rows]))
        pooled = float(np.mean([float(row["score"]) >= BINARY_THRESHOLD for row in selected]))
        checks = {"score_coverage_eq_1": True,
                  "pooled_ai_recall_gte_0_80": pooled >= GATES["pooled_ai_recall_min"],
                  "worst_model_recall_gte_0_60": min(per_model.values()) >= GATES["worst_model_recall_min"]}
        reports[condition] = {
            "rows": len(selected), "score_coverage": 1.0, "pooled_ai_recall": pooled,
            "ai_recall_by_model": per_model, "worst_model_recall": min(per_model.values()),
            "selective_counts": dict(categories),
            "gate": {"passed": all(checks.values()), "checks": checks,
                     "passed_checks": sum(checks.values()), "total_checks": len(checks)},
        }
        score_maps[condition] = {str(row["parent_id"]): float(row["score"]) for row in selected}
    deltas = np.asarray([score_maps["social_q75"][key] - value
                         for key, value in score_maps["publisher_original"].items()])
    passed = all(report["gate"]["passed"] for report in reports.values())
    return {
        "schema_version": 1,
        "state": "e49_d1_ai_diagnostic_pass" if passed else "e49_d1_ai_diagnostic_fail",
        "candidate": {"name": "E43-S", "binary_threshold": BINARY_THRESHOLD,
                      "selective_real_cut": REAL_CUT},
        "parents": 800, "observations": 1_600, "conditions": reports,
        "paired_transport": {"ai_recall_loss_original_to_q75":
                             reports["publisher_original"]["pooled_ai_recall"]
                             - reports["social_q75"]["pooled_ai_recall"],
                             "score_delta_q75_minus_original_mean": float(deltas.mean()),
                             "score_delta_q75_minus_original_median": float(np.median(deltas))},
        "gate": {"passed": passed, "passed_checks": sum(r["gate"]["passed_checks"] for r in reports.values()),
                 "total_checks": 6},
        "boundary": "AI-only current-generator diagnostic; no AUC, BA, REAL-safety or Module-1 promotion claim.",
    }


def evaluate() -> dict[str, Any]:
    _validate_contract()
    if REPORT.exists() or REPORT_EVIDENCE.exists():
        raise FileExistsError("E49-D1 report already exists")
    receipt = json.loads(SCORE_EVIDENCE.read_text())
    if _digest(SCORES) != receipt.get("sha256") or receipt.get("metrics_opened") != 0:
        raise ValueError("E49-D1 score stream changed")
    rows = [json.loads(line) for line in SCORES.read_text().splitlines() if line]
    report = evaluate_diagnostic(rows)
    report.update({"contract_sha256": _digest(CONTRACT), "score_sha256": _digest(SCORES),
                   "retry_count": 0, "threshold_changed": False})
    raw = _write(REPORT, report)
    _write(REPORT_EVIDENCE, {**report, "report_sha256": hashlib.sha256(raw).hexdigest()})
    return report


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("bind-score", "score", "evaluate"))
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args(argv)
    actions = {"bind-score": bind_score, "score": lambda: score(args.batch_size), "evaluate": evaluate}
    print(json.dumps(actions[args.command](), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
