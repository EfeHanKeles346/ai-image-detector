"""Fit one fixed DINOv2-S head using source-held-out E36 adaptation scores."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import joblib
import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from pixelproof.benchmark_metrics import evaluate_binary_scores
from pixelproof.e32_candidate import (
    DINO_MODEL_ID,
    DINO_REPO_ID,
    DINO_WEIGHT_SHA256,
    standardized_array,
)
from pixelproof.project_paths import DATA_ROOT, ML_ROOT


REPO_ROOT = ML_ROOT.parent
E32_ROOT = DATA_ROOT / "e32"
E36_ROOT = DATA_ROOT / "e36"
E37_ROOT = DATA_ROOT / "e37"
E32_FEATURES = E32_ROOT / "features" / "r1b_dino_features.npz"
E32_FEATURES_SHA256 = "ae1737f2114583e6c05c46b265bbda5bce3a2ac36464c84c70625b006a3416de"
E36_MANIFEST = E36_ROOT / "cal_manifest.json"
E36_MANIFEST_SHA256 = "4ed1b7341df39fed1f1fc19a20424f15362c0df224bcc77189045ba4a4af2e03"
E36_FEATURES = E37_ROOT / "e36_dinov2s_features.npz"
OOF_SCORES = E37_ROOT / "oof_scores.jsonl"
REPORT = E37_ROOT / "oof_report.json"
CANDIDATE = E37_ROOT / "e37_dinov2s.joblib"
EVIDENCE = REPO_ROOT / "evidence" / "e37_source_heldout.json"
ROLE_EVIDENCE = REPO_ROOT / "evidence" / "e37_role_amendment.json"
ROLE_EVIDENCE_SHA256 = "2ddafc644f56c49853587e8c9d376bfb170eae027d9d76d57fd079186c1b3014"

INPUT_SIZE = 224
BATCH_SIZE = 48
C_VALUE = 0.1
SEED = 42

FOLDS: tuple[dict[str, frozenset[str]], ...] = (
    {"real": frozenset({"device_001"}), "ai": frozenset({"FLUX.2_max"})},
    {"real": frozenset({"device_002"}), "ai": frozenset({"GLM-Image"})},
    {"real": frozenset({"device_003"}), "ai": frozenset({"Qwen-Image-2.0-pro"})},
    {"real": frozenset({"device_005"}), "ai": frozenset({"Seedream-5.0"})},
    {
        "real": frozenset({"device_009"}),
        "ai": frozenset({"gpt-image-2", "nano-banana-2.0"}),
    },
)


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_bytes(_json_bytes(value))
    temporary.replace(path)


def _save_npz(path: Path, contract: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".npz.part")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **contract)
    temporary.replace(path)


def fold_ids(labels: np.ndarray, sources: np.ndarray) -> np.ndarray:
    """Assign every E36 row to its one preregistered source-held-out fold."""
    assignments = np.full(len(labels), -1, dtype=np.int8)
    for index, (label, source) in enumerate(zip(labels, sources, strict=True)):
        matching = [
            fold_index
            for fold_index, held in enumerate(FOLDS)
            if str(source) in held["ai" if int(label) == 1 else "real"]
        ]
        if len(matching) != 1:
            raise ValueError(f"source {source!r}/label {label} has {len(matching)} fold assignments")
        assignments[index] = matching[0]
    return assignments


def make_head() -> Any:
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=C_VALUE,
            class_weight="balanced",
            max_iter=3_000,
            random_state=SEED,
            solver="lbfgs",
        ),
    )


def source_rates(rows: Sequence[Mapping[str, Any]], threshold: float) -> dict[str, Any]:
    real: dict[str, list[bool]] = {}
    ai: dict[str, list[bool]] = {}
    for row in rows:
        target = ai if int(row["label"]) == 1 else real
        target.setdefault(str(row["source"]), []).append(float(row["score"]) >= threshold)
    real_rates = {source: float(np.mean(values)) for source, values in sorted(real.items())}
    ai_rates = {source: float(np.mean(values)) for source, values in sorted(ai.items())}
    return {
        "threshold": float(threshold),
        "real_fp_by_device": real_rates,
        "real_macro_fp": float(np.mean(list(real_rates.values()))),
        "real_worst_device_fp": max(real_rates.values()),
        "ai_recall_by_family": ai_rates,
        "ai_macro_recall": float(np.mean(list(ai_rates.values()))),
        "ai_worst_family_recall": min(ai_rates.values()),
    }


def select_threshold(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    candidates = sorted(
        {float(row["score"]) for row in rows}
        | {float(np.nextafter(float(row["score"]), np.inf)) for row in rows}
    )
    for threshold in candidates:
        rates = source_rates(rows, threshold)
        if rates["real_macro_fp"] <= 0.10 + 1e-12 and rates["real_worst_device_fp"] <= 0.20 + 1e-12:
            return rates
    raise ValueError("no OOF threshold satisfies authentic source budgets")


def gate(metrics: Mapping[str, Any], rates: Mapping[str, Any]) -> dict[str, Any]:
    checks = {
        "roc_auc_gte_0.90": float(metrics["roc_auc"]) >= 0.90,
        "tpr_at_fpr10_gte_0.80": float(metrics["tpr_at_fpr"]["tpr"]) >= 0.80,
        "eer_lte_0.15": float(metrics["eer"]) <= 0.15,
        "balanced_accuracy_gte_0.85": float(metrics["balanced_accuracy"]) >= 0.85,
        "real_macro_fp_lte_0.10": float(rates["real_macro_fp"]) <= 0.10 + 1e-12,
        "real_worst_device_fp_lte_0.20": float(rates["real_worst_device_fp"]) <= 0.20 + 1e-12,
        "ai_macro_recall_gte_0.80": float(rates["ai_macro_recall"]) >= 0.80,
        "ai_worst_family_recall_gte_0.60": float(rates["ai_worst_family_recall"]) >= 0.60,
        "coverage_eq_1": float(metrics["coverage"]) == 1.0,
    }
    return {"passed": all(checks.values()), "checks": checks}


def _bootstrap_macro(
    rows: Sequence[Mapping[str, Any]], *, threshold: float, label: int, iterations: int = 2_000
) -> dict[str, Any]:
    groups = {
        source: np.asarray(
            [float(row["score"]) >= threshold for row in rows if int(row["label"]) == label and row["source"] == source],
            dtype=np.float64,
        )
        for source in sorted({str(row["source"]) for row in rows if int(row["label"]) == label})
    }
    random = np.random.default_rng(20_260_827 + label)
    samples = [
        float(np.mean([random.choice(values, size=len(values), replace=True).mean() for values in groups.values()]))
        for _ in range(iterations)
    ]
    estimate = float(np.mean([values.mean() for values in groups.values()]))
    return {
        "estimate": estimate,
        "low_95": float(np.quantile(samples, 0.025)),
        "high_95": float(np.quantile(samples, 0.975)),
        "iterations": iterations,
    }


def _load_base() -> dict[str, np.ndarray]:
    if _sha256_file(E32_FEATURES) != E32_FEATURES_SHA256:
        raise ValueError("frozen E32 DINOv2-S feature archive changed")
    with np.load(E32_FEATURES, allow_pickle=False) as stored:
        contract = {name: stored[name] for name in stored.files}
    if contract["features"].shape != (26_682, 384):
        raise ValueError("unexpected E32 feature shape")
    if Counter(contract["roles"].astype(str))["TRAIN"] != 21_349:
        raise ValueError("E32 TRAIN role count changed")
    return contract


def _load_manifest() -> dict[str, Any]:
    if _sha256_file(E36_MANIFEST) != E36_MANIFEST_SHA256:
        raise ValueError("E36 adaptation manifest changed")
    manifest = json.loads(E36_MANIFEST.read_text())
    if manifest.get("state") != "cal_manifest_frozen_unscored" or manifest.get("counts", {}).get("total") != 1_071:
        raise ValueError("E36 adaptation manifest contract changed")
    if _sha256_file(ROLE_EVIDENCE) != ROLE_EVIDENCE_SHA256:
        raise ValueError("E37 role amendment changed")
    return manifest


def _extract_features(rows: Sequence[Mapping[str, Any]], batch_size: int) -> dict[str, np.ndarray]:
    from huggingface_hub import snapshot_download
    import timm
    import torch

    snapshot = Path(snapshot_download(DINO_REPO_ID, local_files_only=True))
    if _sha256_file(snapshot / "model.safetensors") != DINO_WEIGHT_SHA256:
        raise ValueError("cached DINOv2-S weights changed")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = timm.create_model(DINO_MODEL_ID, pretrained=True, num_classes=0, img_size=INPUT_SIZE).to(device).eval()
    config = timm.data.resolve_data_config({}, model=model)
    mean = torch.tensor(config["mean"], device=device).view(1, 3, 1, 1)
    std = torch.tensor(config["std"], device=device).view(1, 3, 1, 1)
    chunks: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            group = rows[start : start + batch_size]
            arrays = []
            for row in group:
                path = E36_ROOT / str(row["path"])
                if _sha256_file(path) != str(row["sha256"]):
                    raise ValueError(f"E36 image binding changed: {row['parent_id']}")
                with Image.open(path) as image:
                    arrays.append(standardized_array(image))
            tensor = torch.from_numpy(np.stack(arrays)).to(device)
            tensor = tensor.permute(0, 3, 1, 2).float().div_(255.0)
            chunks.append(model((tensor - mean) / std).float().cpu().numpy())
            print(f"E37 DINOv2-S {min(start + batch_size, len(rows))}/{len(rows)}", flush=True)
    return {
        "features": np.concatenate(chunks).astype(np.float32),
        "record_ids": np.asarray([str(row["parent_id"]) for row in rows]),
        "labels": np.asarray([int(row["label"]) for row in rows], dtype=np.int8),
        "sources": np.asarray([str(row["source"]) for row in rows]),
    }


def _load_or_extract(rows: Sequence[Mapping[str, Any]], batch_size: int) -> dict[str, np.ndarray]:
    expected = np.asarray([str(row["parent_id"]) for row in rows])
    if E36_FEATURES.is_file():
        with np.load(E36_FEATURES, allow_pickle=False) as stored:
            contract = {name: stored[name] for name in stored.files}
        if not np.array_equal(contract.get("record_ids"), expected):
            raise ValueError("cached E36 features are misaligned")
        return contract
    contract = _extract_features(rows, batch_size)
    _save_npz(E36_FEATURES, contract)
    return contract


def run(batch_size: int = BATCH_SIZE) -> dict[str, Any]:
    if any(path.exists() for path in (OOF_SCORES, REPORT, CANDIDATE, EVIDENCE)):
        raise FileExistsError("E37 result already exists; no silent rerun")
    manifest = _load_manifest()
    rows = manifest["rows"]
    base = _load_base()
    adaptation = _load_or_extract(rows, batch_size)
    base_train = base["roles"].astype(str) == "TRAIN"
    base_x = base["features"][base_train]
    base_y = base["labels"][base_train].astype(np.int64)
    adapt_x = adaptation["features"]
    adapt_y = adaptation["labels"].astype(np.int64)
    adapt_sources = adaptation["sources"].astype(str)
    assignments = fold_ids(adapt_y, adapt_sources)
    oof = np.full(len(rows), np.nan, dtype=np.float64)
    fold_reports = []
    for fold_index in range(len(FOLDS)):
        held = assignments == fold_index
        train_x = np.concatenate([base_x, adapt_x[~held]])
        train_y = np.concatenate([base_y, adapt_y[~held]])
        head = make_head()
        head.fit(train_x, train_y)
        oof[held] = head.predict_proba(adapt_x[held])[:, 1]
        fold_reports.append({
            "fold": fold_index,
            "held_rows": int(held.sum()),
            "held_real_sources": sorted(FOLDS[fold_index]["real"]),
            "held_ai_sources": sorted(FOLDS[fold_index]["ai"]),
            "fit_rows": int(len(train_y)),
        })
        print(f"E37 fold {fold_index + 1}/{len(FOLDS)} scored {int(held.sum())}", flush=True)
    if not np.isfinite(oof).all():
        raise RuntimeError("not every E36 row received exactly one OOF score")
    score_rows = [
        {
            "parent_id": row["parent_id"],
            "path": row["path"],
            "label": int(row["label"]),
            "source": row["source"],
            "condition": row["condition"],
            "fold": int(assignments[index]),
            "status": "ok",
            "score": float(oof[index]),
        }
        for index, row in enumerate(rows)
    ]
    score_raw = b"".join((json.dumps(row, sort_keys=True) + "\n").encode() for row in score_rows)
    score_part = OOF_SCORES.with_suffix(OOF_SCORES.suffix + ".part")
    score_part.parent.mkdir(parents=True, exist_ok=True)
    score_part.write_bytes(score_raw)
    score_part.replace(OOF_SCORES)
    selected = select_threshold(score_rows)
    metrics = evaluate_binary_scores(score_rows, threshold=float(selected["threshold"]))
    result_gate = gate(metrics, selected)
    report = {
        "schema_version": 1,
        "experiment": "E37/DINOv2-S-source-held-out-adaptation",
        "state": "oof_gate_passed_candidate_frozen" if result_gate["passed"] else "oof_gate_failed",
        "model": {
            "backbone": DINO_MODEL_ID,
            "backbone_weight_sha256": DINO_WEIGHT_SHA256,
            "input": "EXIF transpose, RGB, short-side 256, center 224, JPEG q90 4:4:4",
            "head": "StandardScaler + LogisticRegression",
            "c": C_VALUE,
            "class_weight": "balanced",
            "seed": SEED,
        },
        "bindings": {
            "e32_feature_archive_sha256": E32_FEATURES_SHA256,
            "e36_manifest_sha256": E36_MANIFEST_SHA256,
            "e36_feature_archive_sha256": _sha256_file(E36_FEATURES),
            "role_amendment_sha256": ROLE_EVIDENCE_SHA256,
            "oof_scores_sha256": hashlib.sha256(score_raw).hexdigest(),
        },
        "counts": {"base_train": int(len(base_y)), "adaptation": int(len(adapt_y)), "oof": int(len(oof))},
        "folds": fold_reports,
        "selected_frontier": selected,
        "metrics": metrics,
        "bootstrap": {
            "real_macro_fp": _bootstrap_macro(score_rows, threshold=float(selected["threshold"]), label=0),
            "ai_macro_recall": _bootstrap_macro(score_rows, threshold=float(selected["threshold"]), label=1),
        },
        "gate": result_gate,
        "boundary": "E36 source-held-out OOF only; no FINAL byte downloaded or scored.",
    }
    if result_gate["passed"]:
        final_head = make_head()
        final_head.fit(np.concatenate([base_x, adapt_x]), np.concatenate([base_y, adapt_y]))
        artifact = {
            "schema_version": 1,
            "model_name": "E37 DINOv2-S source-held-out adaptation",
            "model_id": DINO_MODEL_ID,
            "model_weight_sha256": DINO_WEIGHT_SHA256,
            "input_size": INPUT_SIZE,
            "positive_label": "ai",
            "threshold": float(selected["threshold"]),
            "head": final_head,
            "preprocessing": {
                "orientation": "PIL ImageOps.exif_transpose",
                "mode": "RGB",
                "resize_short_side": 256,
                "crop": "center-224",
                "encoding": "JPEG quality=90 subsampling=0 optimize=false progressive=false",
            },
            "bindings": report["bindings"],
        }
        temporary = CANDIDATE.with_suffix(".joblib.part")
        joblib.dump(artifact, temporary)
        temporary.replace(CANDIDATE)
        report["candidate"] = {
            "path": CANDIDATE.relative_to(E37_ROOT).as_posix(),
            "bytes": CANDIDATE.stat().st_size,
            "sha256": _sha256_file(CANDIDATE),
            "threshold": float(selected["threshold"]),
        }
    _write_atomic(REPORT, report)
    _write_atomic(EVIDENCE, report)
    return report


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = parser.parse_args(argv)
    print(json.dumps(run(args.batch_size), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
