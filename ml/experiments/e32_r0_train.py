"""Extract frozen DINOv2-S features and fit the preregistered E32 R0 head."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from pixelproof.project_paths import DATA_ROOT, ML_ROOT


REPO_ROOT = ML_ROOT.parent
E32_ROOT = DATA_ROOT / "e32"
INPUT_ROOT = E32_ROOT / "model_inputs" / "r0_global_jpeg90"
INPUT_RECEIPT = E32_ROOT / "r0_input_receipt.json"
INPUT_EVIDENCE = REPO_ROOT / "evidence" / "e32_r0_input_receipt.json"
FEATURE_PATH = E32_ROOT / "features" / "r0_dinov2s_features.npz"
ARTIFACT_PATH = E32_ROOT / "models" / "e32_r0_dinov2s.joblib"
COMPACT_EVIDENCE = REPO_ROOT / "evidence" / "e32_r0_dinov2s.json"
MODEL_ID = "vit_small_patch14_dinov2.lvd142m"
INPUT_SIZE = 224
BATCH_SIZE = 48
C_GRID = (0.01, 0.1, 1.0, 10.0)
SEED = 20260826
MACRO_REAL_FP_BUDGET = 0.10
WORST_REAL_FP_BUDGET = 0.20
CURRENT_AI_SOURCES = (
    "qwen-image-2512",
    "flux2-klein-9b",
    "nano-banana-local",
    "gpt-image-1",
    "nano-banana-pro-ash-local",
)


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_atomic(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_bytes(raw)
    temporary.replace(path)


def threshold_at_fp_budget(
    labels: np.ndarray,
    scores: np.ndarray,
    sources: np.ndarray,
    *,
    macro_budget: float = MACRO_REAL_FP_BUDGET,
    worst_budget: float = WORST_REAL_FP_BUDGET,
) -> float:
    real = labels == 0
    names = sorted(set(sources[real].astype(str).tolist()))
    if not names:
        raise ValueError("threshold selection needs authentic rows")
    candidates = np.r_[np.unique(scores), np.nextafter(scores.max(), np.inf)]
    for threshold in candidates:
        rates = [float(np.mean(scores[real & (sources == name)] >= threshold)) for name in names]
        if np.mean(rates) <= macro_budget + 1e-12 and max(rates) <= worst_budget + 1e-12:
            return float(threshold)
    raise RuntimeError("no threshold satisfies authentic FP budgets")


def evaluate(labels: np.ndarray, scores: np.ndarray, sources: np.ndarray, threshold: float) -> dict[str, Any]:
    predictions = scores >= threshold
    real = labels == 0
    ai = labels == 1
    real_fp_by_source = {
        name: float(np.mean(predictions[real & (sources == name)]))
        for name in sorted(set(sources[real].astype(str).tolist()))
    }
    ai_recall_by_source = {
        name: float(np.mean(predictions[ai & (sources == name)]))
        for name in sorted(set(sources[ai].astype(str).tolist()))
    }
    current = [ai_recall_by_source[name] for name in CURRENT_AI_SOURCES]
    return {
        "count": int(len(labels)),
        "auc": float(roc_auc_score(labels, scores)),
        "average_precision": float(average_precision_score(labels, scores)),
        "threshold": float(threshold),
        "ai_recall": float(np.mean(predictions[ai])),
        "real_recall": float(np.mean(~predictions[real])),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "f1": float(f1_score(labels, predictions)),
        "macro_real_fp": float(np.mean(list(real_fp_by_source.values()))),
        "worst_real_fp": float(max(real_fp_by_source.values())),
        "real_fp_by_source": real_fp_by_source,
        "ai_recall_by_source": ai_recall_by_source,
        "current_ai_macro_recall": float(np.mean(current)),
        "current_ai_worst_source_recall": float(min(current)),
    }


def screen_gate(metrics: dict[str, Any]) -> dict[str, Any]:
    checks = {
        "auc_gte_0.85": metrics["auc"] >= 0.85,
        "current_ai_macro_recall_gte_0.60": metrics["current_ai_macro_recall"] >= 0.60,
        "current_ai_worst_source_recall_gte_0.40": metrics["current_ai_worst_source_recall"] >= 0.40,
        "macro_real_fp_lte_0.10": metrics["macro_real_fp"] <= MACRO_REAL_FP_BUDGET + 1e-12,
        "worst_real_fp_lte_0.20": metrics["worst_real_fp"] <= WORST_REAL_FP_BUDGET + 1e-12,
    }
    return {"passed": all(checks.values()), "checks": checks}


def _load_receipt() -> tuple[dict[str, Any], bytes]:
    compact = json.loads(INPUT_EVIDENCE.read_text())
    raw = INPUT_RECEIPT.read_bytes()
    if len(raw) != int(compact["detailed_report_bytes"]) or _sha256(raw) != compact["detailed_report_sha256"]:
        raise ValueError("R0 input receipt binding changed")
    payload = json.loads(raw)
    if payload.get("state") != "r0_input_realization_complete" or int(payload["record_count"]) != 22_688:
        raise ValueError("R0 input receipt is incomplete")
    return payload, raw


def extract_features(receipt: dict[str, Any]) -> dict[str, np.ndarray]:
    import timm
    import torch

    model = timm.create_model(MODEL_ID, pretrained=True, num_classes=0, img_size=INPUT_SIZE)
    config = timm.data.resolve_data_config({}, model=model)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = model.to(device).eval()
    mean = torch.tensor(config["mean"], device=device).view(1, 3, 1, 1)
    std = torch.tensor(config["std"], device=device).view(1, 3, 1, 1)
    rows = receipt["records"]
    chunks: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(rows), BATCH_SIZE):
            batch_rows = rows[start : start + BATCH_SIZE]
            images = []
            for row in batch_rows:
                path = INPUT_ROOT / row["input_path"]
                raw = path.read_bytes()
                if len(raw) != int(row["input_bytes"]) or _sha256(raw) != row["input_sha256"]:
                    raise ValueError(f"standardized input binding changed: {row['record_id']}")
                with Image.open(path) as image:
                    images.append(np.asarray(image.convert("RGB"), dtype=np.uint8).copy())
            tensor = torch.from_numpy(np.stack(images)).to(device)
            tensor = tensor.permute(0, 3, 1, 2).float().div_(255.0)
            chunks.append(model((tensor - mean) / std).float().cpu().numpy())
            print(f"DINOv2-S {min(start + BATCH_SIZE, len(rows))}/{len(rows)}", flush=True)
    return {
        "features": np.concatenate(chunks).astype(np.float32),
        "record_ids": np.asarray([row["record_id"] for row in rows]),
        "labels": np.asarray([1 if row["label"] == "ai" else 0 for row in rows], dtype=np.int8),
        "roles": np.asarray([row["role"] for row in rows]),
        "sources": np.asarray([row["source_id"] for row in rows]),
    }


def _save_features(contract: dict[str, np.ndarray]) -> None:
    FEATURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = FEATURE_PATH.with_suffix(".npz.part")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **contract)
    temporary.replace(FEATURE_PATH)


def _load_or_extract_features(receipt: dict[str, Any]) -> dict[str, np.ndarray]:
    expected_ids = np.asarray([row["record_id"] for row in receipt["records"]])
    if FEATURE_PATH.is_file():
        with np.load(FEATURE_PATH, allow_pickle=False) as stored:
            contract = {name: stored[name] for name in stored.files}
        if np.array_equal(contract.get("record_ids"), expected_ids):
            return contract
        raise ValueError("existing feature archive does not match R0 input receipt")
    contract = extract_features(receipt)
    _save_features(contract)
    return contract


def train_r0() -> dict[str, Any]:
    receipt, receipt_raw = _load_receipt()
    contract = _load_or_extract_features(receipt)
    features = contract["features"]
    labels = contract["labels"].astype(np.int64)
    roles = contract["roles"].astype(str)
    sources = contract["sources"].astype(str)
    train = roles == "TRAIN"
    calibration = roles == "CALIBRATION"
    candidates: list[tuple[float, float, Any, np.ndarray]] = []
    for c_value in C_GRID:
        head = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=c_value,
                class_weight="balanced",
                max_iter=3000,
                random_state=SEED,
                solver="lbfgs",
            ),
        )
        head.fit(features[train], labels[train])
        scores = head.predict_proba(features[calibration])[:, 1]
        auc = float(roc_auc_score(labels[calibration], scores))
        candidates.append((auc, c_value, head, scores))
    best_auc, best_c, head, calibration_scores = max(candidates, key=lambda item: (item[0], -item[1]))
    threshold = threshold_at_fp_budget(labels[calibration], calibration_scores, sources[calibration])
    metrics = evaluate(labels[calibration], calibration_scores, sources[calibration], threshold)
    gate = screen_gate(metrics)
    FEATURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    feature_sha = _sha256_file(FEATURE_PATH)
    artifact = {
        "schema_version": 1,
        "model_name": "E32 R0 DINOv2-S global",
        "model_id": MODEL_ID,
        "input_size": INPUT_SIZE,
        "preprocessing": receipt["preprocessing"],
        "threshold": threshold,
        "head": head,
        "positive_label": "ai",
        "input_receipt_sha256": _sha256(receipt_raw),
        "feature_archive_sha256": feature_sha,
        "selected_c": best_c,
    }
    ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = ARTIFACT_PATH.with_suffix(".joblib.part")
    joblib.dump(artifact, temporary)
    temporary.replace(ARTIFACT_PATH)
    report = {
        "schema_version": 1,
        "experiment": "E32/C4-R0-DINOv2-S",
        "state": "r0_screen_passed_runnable_candidate" if gate["passed"] else "r0_screen_failed_runnable_candidate",
        "model_id": MODEL_ID,
        "input_receipt_sha256": _sha256(receipt_raw),
        "feature_archive_external_path": FEATURE_PATH.relative_to(E32_ROOT).as_posix(),
        "feature_archive_bytes": FEATURE_PATH.stat().st_size,
        "feature_archive_sha256": feature_sha,
        "artifact_external_path": ARTIFACT_PATH.relative_to(E32_ROOT).as_posix(),
        "artifact_bytes": ARTIFACT_PATH.stat().st_size,
        "artifact_sha256": _sha256_file(ARTIFACT_PATH),
        "feature_shape": list(features.shape),
        "train_rows": int(train.sum()),
        "calibration_rows": int(calibration.sum()),
        "c_grid_auc": {str(c): auc for auc, c, _, _ in candidates},
        "selected_c": best_c,
        "metrics": metrics,
        "screen_gate": gate,
        "boundary": "Runnable group-held-out prototype; no DEVELOPMENT or LOCKED FINAL row was accessed.",
    }
    _write_atomic(COMPACT_EVIDENCE, _json_bytes(report))
    return report


def main(argv: Iterable[str] | None = None) -> int:
    del argv
    print(json.dumps(train_r0(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
