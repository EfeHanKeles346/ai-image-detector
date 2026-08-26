"""E31/B3 source-aware frozen-representation screen.

The script consumes only the accepted E31/B2 tile archive.  Thresholds are
chosen from group-disjoint TRAIN out-of-fold scores; the frozen CALIBRATION
role is read once for arm evaluation.  E30 data is never opened here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

TILE_SHA256 = "508330c2d8318bcd4c8a92c86a86a627ff98ee1bdc97a67772540a68c8569f2b"
SELECTION_SHA256 = "1a3a5c98c4b0614a0af4bd1bc65ca4fbb8ea33404dbb6a2db53b2da17b79df2e"
DINO_MODEL_ID = "vit_small_patch14_dinov2.lvd142m"
DINO_INPUT_PX = 224
SEED = 2026
MACRO_REAL_FP_BUDGET = 0.05
WORST_REAL_FP_BUDGET = 0.10
CURRENT_AI_SOURCES = ("flux-1-dev", "nano-banana", "nano-banana-pro")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def threshold_at_source_fp_budget(
    real_scores: np.ndarray,
    real_sources: np.ndarray,
    *,
    macro_budget: float = MACRO_REAL_FP_BUDGET,
    worst_budget: float = WORST_REAL_FP_BUDGET,
) -> tuple[float, dict[str, Any]]:
    """Lowest threshold satisfying both source-macro and worst-source FP limits."""
    scores = np.asarray(real_scores, dtype=np.float64)
    sources = np.asarray(real_sources).astype(str)
    if not len(scores) or len(scores) != len(sources):
        raise ValueError("real score/source arrays must be non-empty and aligned")
    names = sorted(set(sources.tolist()))
    candidates = np.r_[np.unique(scores), np.nextafter(scores.max(), np.inf)]
    for threshold in candidates:
        rates = {
            name: float(np.mean(scores[sources == name] >= threshold)) for name in names
        }
        macro = float(np.mean(list(rates.values())))
        worst = float(max(rates.values()))
        if macro <= macro_budget + 1e-12 and worst <= worst_budget + 1e-12:
            return float(threshold), {
                "macro_real_fp": macro,
                "worst_real_fp": worst,
                "real_fp_by_source": rates,
                "real_count": int(len(scores)),
            }
    raise RuntimeError("no threshold satisfies the real false-positive budgets")


def _rate_by_group(mask: np.ndarray, predictions: np.ndarray, groups: np.ndarray) -> dict[str, float]:
    return {
        name: float(np.mean(predictions[mask & (groups == name)]))
        for name in sorted(set(groups[mask].astype(str).tolist()))
    }


def evaluate_scores(
    labels: np.ndarray,
    scores: np.ndarray,
    sources: np.ndarray,
    generators: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    sources = np.asarray(sources).astype(str)
    generators = np.asarray(generators).astype(str)
    predictions = scores >= threshold
    real = labels == 0
    ai = labels == 1
    real_fp = _rate_by_group(real, predictions, sources)
    ai_recall = _rate_by_group(ai, predictions, sources)
    current = [ai_recall[name] for name in CURRENT_AI_SOURCES if name in ai_recall]

    generator_recall: dict[str, dict[str, float | int]] = {}
    for name in sorted(set(generators[ai].tolist())):
        if not name:
            continue
        chosen = ai & (generators == name)
        generator_recall[name] = {
            "n": int(chosen.sum()),
            "recall": float(np.mean(predictions[chosen])),
        }

    return {
        "count": int(len(labels)),
        "auc": float(roc_auc_score(labels, scores)),
        "threshold": float(threshold),
        "ai_recall": float(np.mean(predictions[ai])),
        "real_fp": float(np.mean(predictions[real])),
        "macro_real_fp": float(np.mean(list(real_fp.values()))),
        "worst_real_fp": float(max(real_fp.values())),
        "real_fp_by_source": real_fp,
        "ai_recall_by_source": ai_recall,
        "current_ai_macro_recall": float(np.mean(current)),
        "current_ai_worst_source_recall": float(min(current)),
        "generator_recall": generator_recall,
    }


def acceptance(metrics: dict[str, Any]) -> dict[str, Any]:
    checks = {
        "macro_real_fp_lte_0.05": metrics["macro_real_fp"] <= MACRO_REAL_FP_BUDGET + 1e-12,
        "worst_real_fp_lte_0.10": metrics["worst_real_fp"] <= WORST_REAL_FP_BUDGET + 1e-12,
        "current_ai_macro_recall_gte_0.50": metrics["current_ai_macro_recall"] >= 0.50,
        "current_ai_worst_source_recall_gte_0.30": (
            metrics["current_ai_worst_source_recall"] >= 0.30
        ),
    }
    return {"passed": all(checks.values()), "checks": checks}


def load_contract(tiles_path: Path, selection_path: Path) -> dict[str, np.ndarray]:
    if sha256_file(tiles_path) != TILE_SHA256:
        raise ValueError("tile archive SHA-256 does not match accepted E31/B2 evidence")
    selection = json.loads(selection_path.read_text())
    if selection.get("selection_sha256") != SELECTION_SHA256:
        raise ValueError("selection SHA-256 does not match accepted E31/B2 evidence")
    by_record = {row["record_id"]: row for row in selection["records"]}
    with np.load(tiles_path, allow_pickle=False) as stored:
        contract = {name: stored[name] for name in stored.files}
    ids = contract["record_ids"].astype(str)
    if set(ids) != set(by_record):
        raise ValueError("tile archive record ids do not reproduce the frozen selection")
    contract["folds"] = np.array([by_record[value]["fold"] for value in ids], dtype=np.int8)
    contract["groups"] = np.array([by_record[value]["group_id"] for value in ids])
    return contract


def extract_e20_scores(tiles: np.ndarray, checkpoint: Path, batch_size: int) -> np.ndarray:
    import torch
    from pixelproof.models import create_model

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = create_model("resnet18", dropout=0.0, pretrained=False)
    model.load_state_dict(payload["model"])
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = model.to(device).eval()
    mean = torch.tensor((0.485, 0.456, 0.406), device=device).view(1, 3, 1, 1)
    std = torch.tensor((0.229, 0.224, 0.225), device=device).view(1, 3, 1, 1)
    result: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(tiles), batch_size):
            batch = torch.from_numpy(tiles[start : start + batch_size]).to(device)
            batch = batch.permute(0, 3, 1, 2).float().div_(255.0)
            result.append(torch.sigmoid(model((batch - mean) / std)).cpu().numpy())
            print(f"\rR0 E20 {min(start + batch_size, len(tiles)):,}/{len(tiles):,}", end="", flush=True)
    print()
    return np.concatenate(result).astype(np.float32)


def extract_dino_features(tiles: np.ndarray, batch_size: int) -> np.ndarray:
    import timm
    import torch
    import torch.nn.functional as functional

    model = timm.create_model(
        DINO_MODEL_ID,
        pretrained=True,
        num_classes=0,
        img_size=DINO_INPUT_PX,
    )
    config = timm.data.resolve_data_config({}, model=model)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = model.to(device).eval()
    mean = torch.tensor(config["mean"], device=device).view(1, 3, 1, 1)
    std = torch.tensor(config["std"], device=device).view(1, 3, 1, 1)
    result: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(tiles), batch_size):
            batch = torch.from_numpy(tiles[start : start + batch_size]).to(device)
            batch = batch.permute(0, 3, 1, 2).float().div_(255.0)
            batch = functional.interpolate(
                batch,
                size=(DINO_INPUT_PX, DINO_INPUT_PX),
                mode="bicubic",
                align_corners=False,
                antialias=True,
            )
            result.append(model((batch - mean) / std).float().cpu().numpy())
            print(f"\rR1 DINOv2 {min(start + batch_size, len(tiles)):,}/{len(tiles):,}", end="", flush=True)
    print()
    return np.concatenate(result).astype(np.float32)


def extract_forensic_features(tiles: np.ndarray) -> np.ndarray:
    from concurrent.futures import ThreadPoolExecutor
    from pixelproof.features import extract_from_image

    def one(tile: np.ndarray) -> np.ndarray:
        return extract_from_image(Image.fromarray(tile, mode="RGB"))

    chunks: list[np.ndarray] = []
    step = 512
    for start in range(0, len(tiles), step):
        with ThreadPoolExecutor(max_workers=6) as pool:
            chunk = list(pool.map(one, tiles[start : start + step]))
        chunks.append(np.stack(chunk).astype(np.float32))
        print(f"\rR2 forensic {min(start + step, len(tiles)):,}/{len(tiles):,}", end="", flush=True)
    print()
    return np.concatenate(chunks)


def linear_oof(
    features: np.ndarray,
    labels: np.ndarray,
    roles: np.ndarray,
    folds: np.ndarray,
    *,
    seed: int,
) -> tuple[np.ndarray, Any]:
    train = roles == "train"
    oof = np.full(len(labels), np.nan, dtype=np.float64)
    for fold in sorted(set(folds[train].tolist())):
        fit = train & (folds != fold)
        held = train & (folds == fold)
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=1.0,
                class_weight="balanced",
                max_iter=4000,
                random_state=seed,
            ),
        )
        model.fit(features[fit], labels[fit])
        oof[held] = model.predict_proba(features[held])[:, 1]
    if np.isnan(oof[train]).any():
        raise RuntimeError("TRAIN out-of-fold prediction coverage is incomplete")
    final = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=4000,
            random_state=seed,
        ),
    )
    final.fit(features[train], labels[train])
    return oof, final


def run_arm(
    arm: str,
    matrix: np.ndarray,
    contract: dict[str, np.ndarray],
    output_dir: Path,
    seed: int,
) -> dict[str, Any]:
    labels = contract["y"].astype(np.int64)
    roles = contract["roles"].astype(str)
    sources = contract["sources"].astype(str)
    generators = contract["generators"].astype(str)
    train = roles == "train"
    calibration = roles == "calibration"

    artifact: dict[str, Any] = {}
    if arm == "r0_e20":
        train_scores = matrix.astype(np.float64)
        calibration_scores = train_scores[calibration]
        threshold, fit_metrics = threshold_at_source_fp_budget(
            train_scores[train & (labels == 0)], sources[train & (labels == 0)]
        )
    else:
        oof, model = linear_oof(matrix, labels, roles, contract["folds"], seed=seed)
        threshold, fit_metrics = threshold_at_source_fp_budget(
            oof[train & (labels == 0)], sources[train & (labels == 0)]
        )
        calibration_scores = model.predict_proba(matrix[calibration])[:, 1]
        output_dir.mkdir(parents=True, exist_ok=True)
        model_path = output_dir / f"{arm}_seed{seed}.joblib"
        joblib.dump(model, model_path)
        artifact = {
            "path": str(model_path),
            "bytes": model_path.stat().st_size,
            "sha256": sha256_file(model_path),
        }

    metrics = evaluate_scores(
        labels[calibration],
        calibration_scores,
        sources[calibration],
        generators[calibration],
        threshold,
    )
    return {
        "arm": arm,
        "seed": seed,
        "fit_threshold_evidence": fit_metrics,
        "calibration": metrics,
        "acceptance": acceptance(metrics),
        "artifact": artifact,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tiles", type=Path, default=Path("ml/data/e31/train_v2_tiles.npz"))
    parser.add_argument(
        "--selection", type=Path, default=Path("evidence/e31_train_v2_selection_v3.json")
    )
    parser.add_argument("--checkpoint", type=Path, default=Path("ml/artifacts/tile_resnet18_seed2024.pt"))
    parser.add_argument("--cache", type=Path, default=Path("ml/data/e31/b3_features.npz"))
    parser.add_argument("--output", type=Path, default=Path("ml/data/e31/b3_screen_seed2026.json"))
    parser.add_argument("--models", type=Path, default=Path("ml/artifacts/e31"))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    contract = load_contract(args.tiles, args.selection)
    if args.cache.exists():
        with np.load(args.cache, allow_pickle=False) as stored:
            matrices = {name: stored[name] for name in stored.files}
        print(f"loaded feature cache {args.cache}")
    else:
        tiles = contract["x"]
        matrices = {
            "r0_e20": extract_e20_scores(tiles, args.checkpoint, args.batch_size),
            "r1_dinov2": extract_dino_features(tiles, args.batch_size),
            "r2_forensic68": extract_forensic_features(tiles),
        }
        args.cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.cache, **matrices)
        print(f"wrote feature cache {args.cache}")

    results = [
        run_arm(arm, matrices[arm], contract, args.models, args.seed)
        for arm in ("r0_e20", "r1_dinov2", "r2_forensic68")
    ]
    payload = {
        "schema_version": 1,
        "experiment": "E31/B3",
        "state": "single_seed_representation_screen",
        "seed": args.seed,
        "boundaries": {
            "selection_sha256": SELECTION_SHA256,
            "tile_archive_sha256": TILE_SHA256,
            "threshold_source": "group-disjoint TRAIN OOF; R0 uses label-blind TRAIN real scores",
            "evaluation_role": "untouched CALIBRATION",
            "e30_opened": False,
            "dino_model_id": DINO_MODEL_ID,
            "dino_input_px": DINO_INPUT_PX,
        },
        "data_counts": {
            "total": int(len(contract["y"])),
            "labels": dict(Counter(map(int, contract["y"]))),
            "roles": dict(Counter(contract["roles"].astype(str))),
            "sources": dict(Counter(contract["sources"].astype(str))),
        },
        "feature_cache": {
            "path": str(args.cache),
            "bytes": args.cache.stat().st_size,
            "sha256": sha256_file(args.cache),
            "shapes": {name: list(value.shape) for name, value in matrices.items()},
        },
        "arms": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({row["arm"]: row["calibration"] for row in results}, indent=2))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
