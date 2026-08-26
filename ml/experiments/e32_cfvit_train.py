"""E32 R1a: frozen Community-Forensics ViT-S CLS representation screen."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

try:
    import e32_r0_train as r0
except ModuleNotFoundError:
    from experiments import e32_r0_train as r0
from pixelproof.project_paths import DATA_ROOT, ML_ROOT


REPO_ROOT = ML_ROOT.parent
E32_ROOT = DATA_ROOT / "e32"
FEATURE_PATH = E32_ROOT / "features" / "r1a_cfvit_features.npz"
ARTIFACT_PATH = E32_ROOT / "models" / "e32_r1a_cfvit.joblib"
EVIDENCE_PATH = REPO_ROOT / "evidence" / "e32_r1a_cfvit.json"
MODEL_REPO = "buildborderless/CommunityForensics-DeepfakeDet-ViT"
MODEL_REVISION = "ac6ee457bea904a373065754107451793b56db00"
MODEL_WEIGHT_SHA256 = "275ba982236ddd6afddf7131f8133e89f537574b964cf8fa5825b4956d741692"
BATCH_SIZE = 24


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def _write_atomic(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_bytes(raw)
    temporary.replace(path)


def fit_head(
    features: np.ndarray,
    labels: np.ndarray,
    roles: np.ndarray,
    sources: np.ndarray,
) -> tuple[Any, float, float, dict[str, Any], dict[str, float]]:
    train = roles == "TRAIN"
    calibration = roles == "CALIBRATION"
    candidates = []
    for c_value in r0.C_GRID:
        head = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=c_value,
                class_weight="balanced",
                max_iter=3000,
                random_state=r0.SEED,
                solver="lbfgs",
            ),
        )
        head.fit(features[train], labels[train])
        scores = head.predict_proba(features[calibration])[:, 1]
        candidates.append((float(roc_auc_score(labels[calibration], scores)), c_value, head, scores))
    _, best_c, head, scores = max(candidates, key=lambda item: (item[0], -item[1]))
    threshold = r0.threshold_at_fp_budget(labels[calibration], scores, sources[calibration])
    metrics = r0.evaluate(labels[calibration], scores, sources[calibration], threshold)
    return head, best_c, threshold, metrics, {str(c): auc for auc, c, _, _ in candidates}


def extract_features(receipt: dict[str, Any]) -> dict[str, np.ndarray]:
    import torch
    from huggingface_hub import snapshot_download
    from transformers import ViTForImageClassification, ViTImageProcessor

    local = Path(snapshot_download(MODEL_REPO, revision=MODEL_REVISION, local_files_only=True))
    if r0._sha256_file(local / "model.safetensors") != MODEL_WEIGHT_SHA256:
        raise RuntimeError("CF-ViT cached weights changed")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = ViTForImageClassification.from_pretrained(local, local_files_only=True).to(device).eval()
    processor = ViTImageProcessor.from_pretrained(local, local_files_only=True)
    rows = receipt["records"]
    chunks = []
    with torch.inference_mode():
        for start in range(0, len(rows), BATCH_SIZE):
            batch_rows = rows[start : start + BATCH_SIZE]
            images = []
            for row in batch_rows:
                path = r0.INPUT_ROOT / row["input_path"]
                raw = path.read_bytes()
                if len(raw) != int(row["input_bytes"]) or r0._sha256(raw) != row["input_sha256"]:
                    raise ValueError(f"R0 input binding changed: {row['record_id']}")
                with Image.open(path) as image:
                    images.append(image.convert("RGB").copy())
            pixels = processor(images=images, return_tensors="pt")["pixel_values"].to(device)
            hidden = model.vit(pixel_values=pixels).last_hidden_state[:, 0]
            chunks.append(hidden.cpu().numpy())
            print(f"CF-ViT CLS {min(start + BATCH_SIZE, len(rows))}/{len(rows)}", flush=True)
    return {
        "features": np.concatenate(chunks).astype(np.float32),
        "record_ids": np.asarray([row["record_id"] for row in rows]),
        "labels": np.asarray([1 if row["label"] == "ai" else 0 for row in rows], dtype=np.int8),
        "roles": np.asarray([row["role"] for row in rows]),
        "sources": np.asarray([row["source_id"] for row in rows]),
    }


def _load_or_extract(receipt: dict[str, Any]) -> dict[str, np.ndarray]:
    expected = np.asarray([row["record_id"] for row in receipt["records"]])
    if FEATURE_PATH.is_file():
        with np.load(FEATURE_PATH, allow_pickle=False) as stored:
            contract = {name: stored[name] for name in stored.files}
        if not np.array_equal(contract.get("record_ids"), expected):
            raise ValueError("existing CF feature archive is not record-aligned")
        return contract
    contract = extract_features(receipt)
    FEATURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = FEATURE_PATH.with_suffix(".npz.part")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **contract)
    temporary.replace(FEATURE_PATH)
    return contract


def train() -> dict[str, Any]:
    receipt, receipt_raw = r0._load_receipt()
    contract = _load_or_extract(receipt)
    features = contract["features"]
    labels = contract["labels"].astype(np.int64)
    roles = contract["roles"].astype(str)
    sources = contract["sources"].astype(str)
    head, best_c, threshold, metrics, grid = fit_head(features, labels, roles, sources)
    gate = r0.screen_gate(metrics)
    feature_sha = r0._sha256_file(FEATURE_PATH)
    artifact = {
        "schema_version": 1,
        "model_name": "E32 R1a frozen Community-Forensics ViT-S CLS",
        "model_repo": MODEL_REPO,
        "model_revision": MODEL_REVISION,
        "model_weight_sha256": MODEL_WEIGHT_SHA256,
        "input_receipt_sha256": r0._sha256(receipt_raw),
        "feature_archive_sha256": feature_sha,
        "head": head,
        "selected_c": best_c,
        "threshold": threshold,
        "positive_label": "ai",
    }
    ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = ARTIFACT_PATH.with_suffix(".joblib.part")
    joblib.dump(artifact, temporary)
    temporary.replace(ARTIFACT_PATH)
    report = {
        "schema_version": 1,
        "experiment": "E32/C4-R1a-CF-ViT-CLS",
        "state": "r1a_screen_passed_runnable_candidate" if gate["passed"] else "r1a_screen_failed_runnable_candidate",
        "model_repo": MODEL_REPO,
        "model_revision": MODEL_REVISION,
        "model_weight_sha256": MODEL_WEIGHT_SHA256,
        "input_receipt_sha256": r0._sha256(receipt_raw),
        "feature_shape": list(features.shape),
        "feature_archive_bytes": FEATURE_PATH.stat().st_size,
        "feature_archive_sha256": feature_sha,
        "artifact_bytes": ARTIFACT_PATH.stat().st_size,
        "artifact_sha256": r0._sha256_file(ARTIFACT_PATH),
        "c_grid_auc": grid,
        "selected_c": best_c,
        "metrics": metrics,
        "screen_gate": gate,
        "boundary": "Internal group-held-out screen only; owner gallery and locked finals were not accessed.",
    }
    _write_atomic(EVIDENCE_PATH, _json_bytes(report))
    return report


def main(argv: Iterable[str] | None = None) -> int:
    del argv
    print(json.dumps(train(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
