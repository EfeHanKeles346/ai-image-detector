"""Run the preregistered E40 source/content-balanced DINOv2-S adaptation."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, Sequence

import joblib
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
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


EXPERIMENTS_DIR = Path(__file__).resolve().parent
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))

import e37_source_heldout as e37  # noqa: E402


REPO_ROOT = ML_ROOT.parent
E39_ROOT = DATA_ROOT / "e39"
E40_ROOT = DATA_ROOT / "e40"
E39_MANIFEST = E39_ROOT / "final_manifest.json"
E39_MANIFEST_SHA256 = "1076df20adc700a1893814e477891e9270e4f88502dbad2571e022f3a7127306"
ROLE_EVIDENCE = REPO_ROOT / "evidence" / "e40_role_amendment.json"
ROLE_EVIDENCE_SHA256 = "c475a529e9f9d03761d42bf021dd8a4ba5212b3a44c59d8fce508fc1fff62f0f"
FIXED_CONTRACT = REPO_ROOT / "evidence" / "e40_fixed_contract.json"
FIXED_CONTRACT_SHA256 = "c97361960e38128339655bb5d97a373162338d936f6096aece1502bfec756bf6"
E36_FEATURES_SHA256 = "3a08e0dcacce6441efa0d422531871b964d100b126a6884715f5616c1a26f178"

FEATURES = E40_ROOT / "e39_dinov2s_features.npz"
FEATURE_EVIDENCE = REPO_ROOT / "evidence" / "e40_features.json"
REPORT = E40_ROOT / "development_report.json"
EVIDENCE = REPO_ROOT / "evidence" / "e40_development.json"
DRAFT = E40_ROOT / "e40_head_draft.joblib"

INPUT_SIZE = 224
BATCH_SIZE = 48
C_VALUE = 0.01
CLUSTERS = 16
PRIMARY_SEED = 42
STABILITY_SEEDS = (41, 42, 43)
REPLAY_SALT = "E40_REPLAY_V1"
MODES = ("uniform", "source_balanced", "source_content_balanced")

FOLDS: tuple[dict[str, frozenset[str]], ...] = (
    {"real": frozenset({"D14_Apple_iPhone13mini"}), "ai": frozenset({"Adobe Firefly Image 5"})},
    {"real": frozenset({"D27_DOOGEE_S96Pro"}), "ai": frozenset({"Gemini 3 Pro Image"})},
    {"real": frozenset({"D34_Google_Pixel5"}), "ai": frozenset({"HiDream I1 Dev"})},
    {"real": frozenset({"D43_OnePlus_8T"}), "ai": frozenset({"Ideogram 3"})},
    {"real": frozenset(), "ai": frozenset({"Midjourney v7"})},
    {"real": frozenset(), "ai": frozenset({"Reve Image 1.0"})},
    {"real": frozenset(), "ai": frozenset({"Z Image Turbo"})},
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_bytes((json.dumps(value, indent=2, sort_keys=True) + "\n").encode())
    temporary.replace(path)


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> str:
    raw = b"".join((json.dumps(row, sort_keys=True) + "\n").encode() for row in rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_bytes(raw)
    temporary.replace(path)
    return hashlib.sha256(raw).hexdigest()


def _save_npz(path: Path, values: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".npz.part")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **values)
    temporary.replace(path)


def _load_manifest() -> dict[str, Any]:
    if _sha256_file(E39_MANIFEST) != E39_MANIFEST_SHA256:
        raise ValueError("frozen E39 manifest changed")
    if _sha256_file(ROLE_EVIDENCE) != ROLE_EVIDENCE_SHA256:
        raise ValueError("E40 role amendment changed")
    manifest = json.loads(E39_MANIFEST.read_text())
    if manifest.get("state") != "final_manifest_frozen_unscored" or manifest.get("counts", {}).get("total") != 440:
        raise ValueError("unexpected E39 manifest contract")
    return manifest


def fold_ids(labels: np.ndarray, sources: np.ndarray) -> np.ndarray:
    """Assign every E39 source to exactly one preregistered held-out fold."""
    assignments = np.full(len(labels), -1, dtype=np.int8)
    for index, (label, source) in enumerate(zip(labels, sources, strict=True)):
        side = "ai" if int(label) == 1 else "real"
        matches = [fold for fold, held in enumerate(FOLDS) if str(source) in held[side]]
        if len(matches) != 1:
            raise ValueError(f"source {source!r}/label {label} has {len(matches)} fold assignments")
        assignments[index] = matches[0]
    return assignments


def replay_indices(
    record_ids: np.ndarray,
    labels: np.ndarray,
    roles: np.ndarray,
    sources: np.ndarray,
) -> np.ndarray:
    """Choose the fixed ~5% E32 TRAIN replay, independently inside label/source strata."""
    groups: dict[tuple[int, str], list[int]] = defaultdict(list)
    for index, (label, role, source) in enumerate(zip(labels, roles, sources, strict=True)):
        if str(role) == "TRAIN":
            groups[(int(label), str(source))].append(index)
    selected: list[int] = []
    for key in sorted(groups):
        candidates = groups[key]
        count = round(len(candidates) * 0.05)
        ranked = sorted(
            candidates,
            key=lambda index: hashlib.sha256(
                f"{REPLAY_SALT}|{record_ids[index]}".encode()
            ).digest(),
        )
        selected.extend(ranked[:count])
    return np.asarray(sorted(selected), dtype=np.int64)


def sample_weights(
    labels: np.ndarray,
    sources: np.ndarray,
    mode: str,
    clusters: np.ndarray | None = None,
) -> np.ndarray:
    """Give equal class mass, then equal source and optional content-cell mass."""
    if mode not in MODES:
        raise ValueError(f"unknown weighting mode: {mode}")
    labels = labels.astype(np.int64)
    sources = sources.astype(str)
    if len(labels) == 0 or set(labels.tolist()) != {0, 1}:
        raise ValueError("both classes are required")
    if mode == "source_content_balanced":
        if clusters is None or len(clusters) != len(labels):
            raise ValueError("one cluster is required per training row")
        clusters = np.asarray(clusters, dtype=np.int64)
    weights = np.zeros(len(labels), dtype=np.float64)
    for label in (0, 1):
        label_mask = labels == label
        label_sources = sorted(set(sources[label_mask]))
        for source in label_sources:
            source_mask = label_mask & (sources == source)
            if mode == "uniform":
                weights[source_mask] = 1.0 / int(label_mask.sum())
            elif mode == "source_balanced":
                weights[source_mask] = 1.0 / (len(label_sources) * int(source_mask.sum()))
            else:
                assert clusters is not None
                occupied = sorted(set(clusters[source_mask].tolist()))
                for cluster in occupied:
                    cell = source_mask & (clusters == cluster)
                    weights[cell] = 1.0 / (
                        len(label_sources) * len(occupied) * int(cell.sum())
                    )
    if not np.isfinite(weights).all() or np.any(weights <= 0):
        raise ValueError("invalid sample weights")
    return weights * (len(weights) / weights.sum())


def make_head(seed: int) -> Any:
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=C_VALUE,
            class_weight=None,
            max_iter=3_000,
            random_state=seed,
            solver="lbfgs",
        ),
    )


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
                path = E39_ROOT / str(row["path"])
                if _sha256_file(path) != str(row["sha256"]):
                    raise ValueError(f"E39 image binding changed: {row['parent_id']}")
                with Image.open(path) as image:
                    arrays.append(standardized_array(image))
            tensor = torch.from_numpy(np.stack(arrays)).to(device)
            tensor = tensor.permute(0, 3, 1, 2).float().div_(255.0)
            chunks.append(model((tensor - mean) / std).float().cpu().numpy())
            print(f"E40 DINOv2-S {min(start + batch_size, len(rows))}/{len(rows)}", flush=True)
    return {
        "features": np.concatenate(chunks).astype(np.float32),
        "record_ids": np.asarray([str(row["parent_id"]) for row in rows]),
        "labels": np.asarray([int(row["label"]) for row in rows], dtype=np.int8),
        "sources": np.asarray([str(row["source"]) for row in rows]),
    }


def extract_features(batch_size: int = BATCH_SIZE) -> dict[str, Any]:
    if FEATURES.exists() or FEATURE_EVIDENCE.exists():
        raise FileExistsError("E40 feature output already exists; no silent overwrite")
    rows = _load_manifest()["rows"]
    values = _extract_features(rows, batch_size)
    if values["features"].shape != (440, 384) or not np.isfinite(values["features"]).all():
        raise ValueError("unexpected E40 feature contract")
    _save_npz(FEATURES, values)
    assignments = fold_ids(values["labels"], values["sources"])
    report = {
        "schema_version": 1,
        "experiment": "E40/model-free-feature-cache",
        "state": "features_frozen_before_e40_fit",
        "bindings": {
            "e39_manifest_sha256": E39_MANIFEST_SHA256,
            "e40_role_amendment_sha256": ROLE_EVIDENCE_SHA256,
            "dinov2s_weight_sha256": DINO_WEIGHT_SHA256,
            "feature_archive_sha256": _sha256_file(FEATURES),
        },
        "counts": {
            "total": len(rows),
            "real": int((values["labels"] == 0).sum()),
            "ai": int((values["labels"] == 1).sum()),
            "folds": dict(sorted(Counter(assignments.tolist()).items())),
        },
        "feature_shape": list(values["features"].shape),
        "all_rows_finite": True,
        "boundary": "All 440 consumed E39 parents, unchanged preprocessing/backbone, no filtering, clustering or head fit.",
    }
    _write_atomic(FEATURE_EVIDENCE, report)
    return report


def _load_feature_archive(path: Path, expected_sha256: str | None = None) -> dict[str, np.ndarray]:
    if expected_sha256 is not None and _sha256_file(path) != expected_sha256:
        raise ValueError(f"feature archive changed: {path}")
    with np.load(path, allow_pickle=False) as stored:
        return {name: stored[name] for name in stored.files}


def _training_inputs() -> dict[str, np.ndarray]:
    if not FEATURE_EVIDENCE.exists():
        raise FileNotFoundError("freeze E40 features before training")
    feature_report = json.loads(FEATURE_EVIDENCE.read_text())
    current_feature_sha = _sha256_file(FEATURES)
    if feature_report.get("bindings", {}).get("feature_archive_sha256") != current_feature_sha:
        raise ValueError("E40 feature evidence/archive mismatch")
    base = e37._load_base()
    replay = replay_indices(base["record_ids"], base["labels"], base["roles"], base["sources"])
    if len(replay) != 1_067:
        raise ValueError(f"unexpected E32 replay count: {len(replay)}")
    e36 = _load_feature_archive(e37.E36_FEATURES, E36_FEATURES_SHA256)
    e39 = _load_feature_archive(FEATURES, current_feature_sha)
    return {
        "replay_x": base["features"][replay],
        "replay_y": base["labels"][replay].astype(np.int64),
        "replay_sources": base["sources"][replay].astype(str),
        "replay_ids": base["record_ids"][replay].astype(str),
        "e36_x": e36["features"],
        "e36_y": e36["labels"].astype(np.int64),
        "e36_sources": e36["sources"].astype(str),
        "e39_x": e39["features"],
        "e39_y": e39["labels"].astype(np.int64),
        "e39_sources": e39["sources"].astype(str),
        "e39_ids": e39["record_ids"].astype(str),
    }


def _fit_predict_fold(
    inputs: Mapping[str, np.ndarray],
    held: np.ndarray,
    mode: str,
    seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    included = ~held
    train_x = np.concatenate([inputs["replay_x"], inputs["e36_x"], inputs["e39_x"][included]])
    train_y = np.concatenate([inputs["replay_y"], inputs["e36_y"], inputs["e39_y"][included]])
    train_sources = np.concatenate([
        inputs["replay_sources"], inputs["e36_sources"], inputs["e39_sources"][included]
    ])
    clusters = None
    cluster_report = None
    if mode == "source_content_balanced":
        modern_x = np.concatenate([inputs["e36_x"], inputs["e39_x"][included]])
        clusterer = KMeans(n_clusters=CLUSTERS, init="k-means++", n_init=10, random_state=seed)
        clusterer.fit(modern_x)
        clusters = clusterer.predict(train_x)
        cluster_report = {
            "inertia": float(clusterer.inertia_),
            "occupied": len(set(clusters.tolist())),
        }
    weights = sample_weights(train_y, train_sources, mode, clusters)
    head = make_head(seed)
    head.fit(
        train_x,
        train_y,
        standardscaler__sample_weight=weights,
        logisticregression__sample_weight=weights,
    )
    scores = head.predict_proba(inputs["e39_x"][held])[:, 1]
    return scores, {
        "fit_rows": int(len(train_y)),
        "fit_replay_rows": int(len(inputs["replay_y"])),
        "fit_e36_rows": int(len(inputs["e36_y"])),
        "fit_e39_rows": int(included.sum()),
        "weight_min": float(weights.min()),
        "weight_max": float(weights.max()),
        "cluster": cluster_report,
    }


def _oof(
    inputs: Mapping[str, np.ndarray],
    rows: Sequence[Mapping[str, Any]],
    mode: str,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    assignments = fold_ids(inputs["e39_y"], inputs["e39_sources"])
    scores = np.full(len(rows), np.nan, dtype=np.float64)
    fold_reports = []
    for fold in range(len(FOLDS)):
        held = assignments == fold
        predicted, fit_report = _fit_predict_fold(inputs, held, mode, seed)
        scores[held] = predicted
        fold_reports.append({
            "fold": fold,
            "held_rows": int(held.sum()),
            "held_real_sources": sorted(FOLDS[fold]["real"]),
            "held_ai_sources": sorted(FOLDS[fold]["ai"]),
            **fit_report,
        })
        print(f"E40 {mode} seed={seed} fold {fold + 1}/{len(FOLDS)} scored {int(held.sum())}", flush=True)
    if not np.isfinite(scores).all():
        raise RuntimeError("not every E39 row received one OOF score")
    score_rows = [
        {
            "parent_id": row["parent_id"],
            "path": row["path"],
            "label": int(row["label"]),
            "source": row["source"],
            "condition": row["condition"],
            "fold": int(assignments[index]),
            "status": "ok",
            "score": float(scores[index]),
        }
        for index, row in enumerate(rows)
    ]
    return score_rows, fold_reports


def _evaluate(score_rows: Sequence[Mapping[str, Any]], threshold: float | None = None) -> dict[str, Any]:
    rates = e37.select_threshold(score_rows) if threshold is None else e37.source_rates(score_rows, threshold)
    metrics = evaluate_binary_scores(score_rows, threshold=float(rates["threshold"]))
    return {"selected_frontier": rates, "metrics": metrics, "gate": e37.gate(metrics, rates)}


def _fit_full(inputs: Mapping[str, np.ndarray], mode: str, seed: int, threshold: float) -> dict[str, Any]:
    train_x = np.concatenate([inputs["replay_x"], inputs["e36_x"], inputs["e39_x"]])
    train_y = np.concatenate([inputs["replay_y"], inputs["e36_y"], inputs["e39_y"]])
    sources = np.concatenate([inputs["replay_sources"], inputs["e36_sources"], inputs["e39_sources"]])
    clusters = None
    if mode == "source_content_balanced":
        clusterer = KMeans(n_clusters=CLUSTERS, init="k-means++", n_init=10, random_state=seed)
        clusterer.fit(np.concatenate([inputs["e36_x"], inputs["e39_x"]]))
        clusters = clusterer.predict(train_x)
    weights = sample_weights(train_y, sources, mode, clusters)
    head = make_head(seed)
    head.fit(
        train_x,
        train_y,
        standardscaler__sample_weight=weights,
        logisticregression__sample_weight=weights,
    )
    return {
        "schema_version": 1,
        "model_name": "E40 DINOv2-S adaptation development draft",
        "model_id": DINO_MODEL_ID,
        "model_weight_sha256": DINO_WEIGHT_SHA256,
        "input_size": INPUT_SIZE,
        "positive_label": "ai",
        "threshold": float(threshold),
        "head": head,
        "weighting": mode,
        "seed": seed,
        "status": "development_draft_not_final_validated",
        "preprocessing": {
            "orientation": "PIL ImageOps.exif_transpose",
            "mode": "RGB",
            "resize_short_side": 256,
            "crop": "center-224",
            "encoding": "JPEG quality=90 subsampling=0 optimize=false progressive=false",
        },
    }


def train() -> dict[str, Any]:
    outputs = [REPORT, EVIDENCE, DRAFT]
    outputs.extend(E40_ROOT / f"oof_{mode}_seed{seed}.jsonl" for mode in MODES for seed in STABILITY_SEEDS)
    if any(path.exists() for path in outputs):
        raise FileExistsError("E40 training output already exists; no silent rerun")
    contract_sha = _sha256_file(FIXED_CONTRACT)
    if contract_sha != FIXED_CONTRACT_SHA256:
        raise ValueError("E40 fixed contract changed")
    contract = json.loads(FIXED_CONTRACT.read_text())
    if contract.get("state") != "fixed_before_e40_features_or_fit":
        raise ValueError("E40 fixed contract changed state")
    rows = _load_manifest()["rows"]
    inputs = _training_inputs()
    replay_raw = "\n".join(sorted(inputs["replay_ids"].tolist())).encode()
    primary_results: dict[str, Any] = {}
    primary_scores: dict[str, list[dict[str, Any]]] = {}
    for mode in MODES:
        score_rows, folds = _oof(inputs, rows, mode, PRIMARY_SEED)
        score_path = E40_ROOT / f"oof_{mode}_seed{PRIMARY_SEED}.jsonl"
        score_sha = _write_jsonl(score_path, score_rows)
        primary_scores[mode] = score_rows
        primary_results[mode] = {
            **_evaluate(score_rows),
            "folds": folds,
            "scores_sha256": score_sha,
        }
    selected_mode = next(
        (mode for mode in MODES if primary_results[mode]["gate"]["passed"]),
        None,
    )
    stability: dict[str, Any] = {}
    draft = None
    if selected_mode is not None:
        frozen_threshold = float(primary_results[selected_mode]["selected_frontier"]["threshold"])
        for seed in STABILITY_SEEDS:
            if seed == PRIMARY_SEED:
                score_rows = primary_scores[selected_mode]
                evaluation = _evaluate(score_rows, frozen_threshold)
                folds = primary_results[selected_mode]["folds"]
                score_sha = primary_results[selected_mode]["scores_sha256"]
            else:
                score_rows, folds = _oof(inputs, rows, selected_mode, seed)
                score_path = E40_ROOT / f"oof_{selected_mode}_seed{seed}.jsonl"
                score_sha = _write_jsonl(score_path, score_rows)
                evaluation = _evaluate(score_rows, frozen_threshold)
            stability[str(seed)] = {**evaluation, "folds": folds, "scores_sha256": score_sha}
        if all(result["gate"]["passed"] for result in stability.values()):
            draft = _fit_full(inputs, selected_mode, PRIMARY_SEED, frozen_threshold)
            DRAFT.parent.mkdir(parents=True, exist_ok=True)
            temporary = DRAFT.with_suffix(".joblib.part")
            joblib.dump(draft, temporary)
            temporary.replace(DRAFT)
    passed_stability = bool(stability) and all(result["gate"]["passed"] for result in stability.values())
    report = {
        "schema_version": 1,
        "experiment": "E40/content-balanced-source-held-out-adaptation",
        "state": (
            "development_gate_passed_draft_frozen"
            if selected_mode is not None and passed_stability and draft is not None
            else "development_gate_failed"
        ),
        "bindings": {
            "fixed_contract_sha256": contract_sha,
            "e32_feature_archive_sha256": e37.E32_FEATURES_SHA256,
            "e36_feature_archive_sha256": E36_FEATURES_SHA256,
            "e39_feature_archive_sha256": _sha256_file(FEATURES),
            "e39_manifest_sha256": E39_MANIFEST_SHA256,
            "e40_role_amendment_sha256": ROLE_EVIDENCE_SHA256,
            "replay_record_ids_sha256": hashlib.sha256(replay_raw).hexdigest(),
        },
        "counts": {
            "e32_replay": len(inputs["replay_y"]),
            "e36_adaptation": len(inputs["e36_y"]),
            "e39_adaptation_oof": len(inputs["e39_y"]),
        },
        "model": {
            "backbone": DINO_MODEL_ID,
            "c": C_VALUE,
            "clusters": CLUSTERS,
            "primary_seed": PRIMARY_SEED,
            "stability_seeds": list(STABILITY_SEEDS),
        },
        "primary_ladder": primary_results,
        "selected_mode": selected_mode,
        "stability": stability,
        "boundary": "Consumed development only. E39 is not independent evidence; no E40 FINAL data exists.",
    }
    if draft is not None:
        report["draft"] = {
            "path": DRAFT.relative_to(E40_ROOT).as_posix(),
            "bytes": DRAFT.stat().st_size,
            "sha256": _sha256_file(DRAFT),
            "status": "not packaged; E40-C robustness still required",
        }
    _write_atomic(REPORT, report)
    _write_atomic(EVIDENCE, report)
    return report


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    feature_parser = subparsers.add_parser("features")
    feature_parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    subparsers.add_parser("train")
    args = parser.parse_args(argv)
    result = extract_features(args.batch_size) if args.command == "features" else train()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
