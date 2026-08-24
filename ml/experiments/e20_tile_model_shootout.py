# E20 — Phase 2b: three model families on IDENTICAL native tiles.
#
# The question E2 asked with a weak representation and E16 asked on whole images:
# given the same input, does the representation matter? Every arm below trains on
# the same tiles_v1.npz — same crops, same seeds, same evaluation — so the only
# variable is the model.
#
#   A  68 hand-crafted statistics + gradient boosting     the incumbent
#   B  ResNet-18 @128px, ImageNet-pretrained, fine-tuned  §13b's untested half
#   C  SmallCNN @128px, from scratch                      our own, at a new size
#
# WHY THE METRIC IS NOT AUC
# ---------------------------------------------------------------------------
# E11 reported 0.948 and stopped; E13 then showed the same model calls 79% of
# real photographs AI, i.e. it ranks well and cannot be deployed. So the headline
# here is the OPERATING POINT: AI recall at a fixed false-positive budget on real
# photographs from sources the model never trained on. AUC is printed beside it,
# never alone.
#
# EVALUATION IS END-TO-END, NOT PER-TILE
# ---------------------------------------------------------------------------
# A test image is tiled exactly as serve.py tiles it (full coverage, edge
# anchored, texture floor) and every tile is scored ONCE.  The aggregation rule
# is selected on disjoint calibration halves, then measured on untouched images.
# This prevents top-3's variable-tile-count bias from being silently baked into
# every conclusion and makes alternative rules free to compare from raw scores.
# Measuring per-tile accuracy would flatter every arm and answer a question
# nobody asks.
#
# GenImage is reported but discounted: its reals are ImageNet nature photos and
# its fakes are other content, so it is not content-controlled and a model can
# score there by recognising subject matter. Defactify generates its fakes from
# the same MS-COCO captions as its reals, which makes it the fair one.

import argparse
import json
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn

from pixelproof.data import NORMALIZATION
from pixelproof.evaluation_protocol import (
    AGGREGATION_RULES,
    aggregate_tile_scores,
    operating_point,
    records_to_scores,
    stable_calibration_split,
)
from pixelproof.features import THREADS, _vector, select_tiles
from pixelproof.models import create_model
from pixelproof.project_paths import WORK_ROOT

HOME = WORK_ROOT
TILE = 128
TEXTURE_FLOOR = 0.04
SEEDS = [42, 1337, 2024]
FP_BUDGET = 0.10          # false positives allowed on real photographs
LIMIT = 300               # test images per set — E13 used the same

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# archive1 is DELIBERATELY ABSENT, and not only because E10 found metadata alone
# separates it at AUC 1.000. That confound was harmless to the CNNs because
# Resize() destroys dimensions — but this is a TILE experiment, and while a 128px
# tile carries no record of its parent's size, it carries its parent's
# COMPRESSION. archive1's real half sits at 0.190 bytes/pixel and its AI half at
# 1.331, a 7x split that survives tiling intact. A tile model could score well
# there without reading a single generation trace.
#
# (Which also means E13's and E15's archive1 numbers — including v3's
# 0.706 -> 0.902 — should be read with that caveat. Those were tile-model
# measurements on a set with an exploitable compression split.)
TEST_SETS = {
    "genimage (trained-on source)": (HOME / "genimage_split/test/REAL", HOME / "genimage_split/test/FAKE"),
    "defactify (unseen, content-controlled)": (HOME / "defactify_test/real", HOME / "defactify_test/ai"),
}

# Real photographs only, from 10 forensics datasets none of which is in training:
# 2,314 authentic images across CASIA, Columbia, DSO-1, NIST2016,
# RealisticTampering, IMD2020, Coverage, CMFD, VIPP and CocoGlide.
# A real-only set cannot be gamed — there is no second class to shortcut toward —
# so the false-positive rate here is the cleanest version of the number §12b says
# the whole project was getting wrong.
REAL_ONLY = {"forensics auth (10 unseen camera pipelines)": HOME / "manipulation_test"}

GENERATORS = ["dalle3", "midjourney", "sd21", "sd3", "sdxl"]


# --------------------------------------------------------------------------- #
# arms
# --------------------------------------------------------------------------- #
def stratified_holdout(y: np.ndarray, sources: np.ndarray, fraction: float, seed: int):
    """Train/validation indices, stratified by (source, label).

    Stratifying by source as well as label matters here specifically: E14 showed
    a model whose real half comes from one source rejects everything else, so a
    validation slice that happened to under-represent a source would report a
    number about that accident rather than about the epoch.
    """
    rng = np.random.RandomState(seed)
    train, validation = [], []
    for source in sorted(set(sources.tolist())):
        for label in (0, 1):
            group = np.flatnonzero((sources == source) & (y == label))
            rng.shuffle(group)
            cut = max(1, int(len(group) * fraction)) if len(group) > 1 else 0
            validation += group[:cut].tolist()
            train += group[cut:].tolist()
    return np.array(sorted(train)), np.array(sorted(validation))


def tile_features(tiles: np.ndarray) -> np.ndarray:
    """68 statistics per cached uint8 tile, threaded."""
    arrays = [t.astype(np.float32) / 255.0 for t in tiles]
    with ThreadPoolExecutor(max_workers=THREADS) as pool:
        return np.stack(list(pool.map(_vector, arrays)))


def train_statistics(x: np.ndarray, y: np.ndarray, seed: int):
    model = make_pipeline(StandardScaler(),
                          HistGradientBoostingClassifier(random_state=seed)).fit(x, y)
    return lambda features: model.predict_proba(features)[:, 1]


def build_cnn_scorer(name: str, model: nn.Module):
    """Create the exact inference transform for a trained tile CNN."""
    mean, std = NORMALIZATION["imagenet" if name == "resnet18" else "default"]
    mean_t = torch.tensor(mean).view(1, 3, 1, 1).to(DEVICE)
    std_t = torch.tensor(std).view(1, 3, 1, 1).to(DEVICE)
    model = model.to(DEVICE).eval()

    @torch.no_grad()
    def score(batch_tiles: np.ndarray) -> np.ndarray:
        out = []
        for start in range(0, len(batch_tiles), 256):
            chunk = batch_tiles[start:start + 256]
            tensor = (torch.from_numpy(chunk).to(DEVICE)
                      .permute(0, 3, 1, 2).contiguous().float() / 255.0)
            tensor = (tensor - mean_t) / std_t
            out.append(torch.sigmoid(model(tensor)).cpu().numpy())
        return np.concatenate(out)

    return score


def load_cnn_checkpoint(path: Path):
    """Load both legacy E20 and protocol-v2 checkpoints without retraining."""
    checkpoint = torch.load(path, map_location=DEVICE, weights_only=False)
    arm = checkpoint.get("arm")
    if arm not in {"resnet18", "small_cnn"}:
        raise ValueError(f"{path} does not declare a supported CNN arm: {arm!r}")
    kwargs = {"dropout": 0.0, "pretrained": False} if arm == "resnet18" else {"dropout": 0.25}
    model = create_model(arm, **kwargs)
    model.load_state_dict(checkpoint["model"])
    return arm, int(checkpoint.get("seed", 0)), build_cnn_scorer(arm, model), checkpoint


def train_cnn(name: str, tiles: np.ndarray, y: np.ndarray, sources: np.ndarray,
              seed: int, max_epochs: int, batch: int):
    """Fine-tune / train a CNN on native 128px tiles, stopping where validation says.

    ImageNet normalisation for the pretrained backbone, 0.5/0.5 for the
    from-scratch one — each gets the statistics it expects (data.py).
    Augmentation is horizontal flip ONLY: colour or blur would destroy the
    generation traces this whole project is built on reading (HISTORY §3).

    THE EPOCH COUNT IS MEASURED, NOT PICKED. An earlier draft trained for a flat
    3 epochs, inherited from E5's recipe, with nothing behind it — and this
    project has been bitten by exactly that kind of unexamined constant twice
    already (E11's grid sweep stopped at 36 while the score was still climbing;
    serve.py routed on an invented 128px threshold until E11 measured 700). So a
    10% slice stratified by source AND label is held back, validation AUC is
    computed after every epoch, and the weights from the best epoch are the ones
    returned. `--epochs` is now a ceiling rather than a decision.
    """
    torch.manual_seed(seed)
    mean, std = NORMALIZATION["imagenet" if name == "resnet18" else "default"]
    mean_t = torch.tensor(mean).view(1, 3, 1, 1).to(DEVICE)
    std_t = torch.tensor(std).view(1, 3, 1, 1).to(DEVICE)

    model = create_model(name, dropout=0.0 if name == "resnet18" else 0.25).to(DEVICE)
    optimiser = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()
    labels = torch.tensor(y, dtype=torch.float32)

    def to_batch(index: np.ndarray, flip: bool = False) -> torch.Tensor:
        # .contiguous() is required, not cosmetic: permute() returns a view whose
        # strides span two subspaces, and nn.Flatten's backward then fails with
        # "view size is not compatible with input tensor's size and stride".
        batch_t = (torch.from_numpy(tiles[index]).to(DEVICE)
                   .permute(0, 3, 1, 2).contiguous().float() / 255.0)
        if flip:
            mask = torch.rand(len(index), device=DEVICE) < 0.5
            batch_t[mask] = torch.flip(batch_t[mask], dims=[3])
        return (batch_t - mean_t) / std_t

    rng = np.random.RandomState(seed)
    train_index, val_index = stratified_holdout(y, sources, 0.10, seed)

    @torch.no_grad()
    def validation_auc() -> float:
        model.eval()
        scores = []
        for start in range(0, len(val_index), 256):
            chunk = val_index[start:start + 256]
            scores.append(torch.sigmoid(model(to_batch(chunk))).cpu().numpy())
        model.train()
        return float(roc_auc_score(y[val_index], np.concatenate(scores)))

    best = (-1.0, 0, None)
    model.train()
    for epoch in range(max_epochs):
        order = rng.permutation(train_index)
        total, correct, running = 0, 0, 0.0
        for start in range(0, len(order), batch):
            index = order[start:start + batch]
            target = labels[index].to(DEVICE)
            logits = model(to_batch(index, flip=True))
            loss = loss_fn(logits, target)
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            optimiser.step()
            running += loss.item() * len(index)
            correct += ((logits >= 0) == target.bool()).sum().item()
            total += len(index)
        auc = validation_auc()
        mark = ""
        if auc > best[0]:
            best = (auc, epoch + 1, {k: v.detach().clone() for k, v in model.state_dict().items()})
            mark = "  <- best"
        print(f"      epoch {epoch + 1}/{max_epochs}  loss {running / total:.4f}"
              f"  train acc {correct / total:.3f}  val AUC {auc:.4f}{mark}")

    model.load_state_dict(best[2])
    print(f"      keeping epoch {best[1]} (val AUC {best[0]:.4f})")
    model.eval()

    return build_cnn_scorer(name, model), model, {
        "best_epoch": best[1], "validation_auc": best[0]}


# --------------------------------------------------------------------------- #
# evaluation — score tiles ONCE, then compare image-level decision protocols
# --------------------------------------------------------------------------- #
def image_files(folder: Path, limit: int) -> list[Path]:
    files = sorted(p for p in folder.rglob("*")
                   if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
                   and not p.name.startswith("._") and ".mask" not in p.name)
    return files[::max(1, len(files) // limit)][:limit]


def image_tiles(path: Path) -> tuple[np.ndarray, tuple[int, int], list[float]] | None:
    """Uint8 tiles for one test image, same geometry as inference."""
    try:
        with Image.open(path) as image:
            size = image.size
            patches, textures, _ = select_tiles(image, tile=TILE, texture_floor=TEXTURE_FLOOR)
            return np.stack([np.asarray(p, dtype=np.uint8) for p in patches]), size, textures
    except Exception:
        return None


def score_folder_records(folder: Path, arm_kind: str, scorer, dataset: str,
                         source: str, label: int, limit: int = LIMIT) -> list[dict]:
    """Keep every per-tile score so aggregation can change without rescoring."""
    records = []
    for path in image_files(folder, limit):
        loaded = image_tiles(path)
        if loaded is None:
            continue
        tiles, (width, height), textures = loaded
        if len(tiles) == 0:
            continue
        per_tile = scorer(tile_features(tiles) if arm_kind == "stats" else tiles)
        records.append({
            "path": str(path), "dataset": dataset, "source": source, "label": label,
            "width": width, "height": height, "tile_count": len(per_tile),
            "bytes_per_pixel": path.stat().st_size / max(width * height, 1),
            "texture_mean": float(np.mean(textures)),
            "texture_p50": float(np.median(textures)),
            "texture_p90": float(np.percentile(textures, 90)),
            "tile_scores": [float(value) for value in per_tile],
        })
    return records


def safe_auc(real: np.ndarray, ai: np.ndarray) -> float:
    if not len(real) or not len(ai):
        return float("nan")
    truth = np.r_[np.zeros(len(real)), np.ones(len(ai))]
    return float(roc_auc_score(truth, np.r_[real, ai]))


def rate_at(scores: np.ndarray, threshold: float) -> float:
    return float((scores >= threshold).mean()) if len(scores) else float("nan")


def save_raw_records(path: Path, records: list[dict]) -> None:
    """JSONL is intentionally boring: inspectable, streamable and tool-agnostic."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record, separators=(",", ":")) + "\n")


def json_compatible(value):
    """Replace NumPy scalars/non-finite floats so results.json is strict JSON."""
    if isinstance(value, dict):
        return {str(key): json_compatible(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_compatible(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def evaluate_aggregations(real_records: list[dict], generator_records: dict[str, list[dict]],
                          genimage_records: tuple[list[dict], list[dict]],
                          forensic_records: dict[str, list[dict]], rules: list[str],
                          calibration_fraction: float, split_seed: int) -> tuple[str, dict]:
    """Select aggregation on calibration data; report only untouched eval metrics.

    Defactify real and every generator are split separately.  The threshold and
    aggregation choice see only calibration halves.  The evaluation halves and
    all ten forensic real sources remain untouched until the final report.
    """
    real_cal, real_eval = stable_calibration_split(
        real_records, calibration_fraction, split_seed)
    generator_splits = {
        name: stable_calibration_split(records, calibration_fraction, split_seed)
        for name, records in generator_records.items()
    }
    if not real_cal or not real_eval or not generator_splits:
        raise RuntimeError("evaluation needs at least two Defactify real images and one generator")

    evaluated = {}
    for rule in rules:
        cal_real = records_to_scores(real_cal, rule)
        eval_real = records_to_scores(real_eval, rule)
        cal_by_generator = {
            name: records_to_scores(parts[0], rule) for name, parts in generator_splits.items()
        }
        eval_by_generator = {
            name: records_to_scores(parts[1], rule) for name, parts in generator_splits.items()
        }
        cal_ai = np.concatenate([scores for scores in cal_by_generator.values() if len(scores)])
        eval_ai = np.concatenate([scores for scores in eval_by_generator.values() if len(scores)])
        point = operating_point(cal_real, eval_real, eval_ai, FP_BUDGET)
        cut = point["threshold"]

        per_generator = {}
        calibration_recalls = []
        for name in sorted(generator_splits):
            calibration_recall = rate_at(cal_by_generator[name], cut)
            evaluation_recall = rate_at(eval_by_generator[name], cut)
            calibration_recalls.append(calibration_recall)
            per_generator[name] = {
                "calibration_recall": calibration_recall,
                "evaluation_recall": evaluation_recall,
                "evaluation_auc": safe_auc(eval_real, eval_by_generator[name]),
                "n_calibration": len(cal_by_generator[name]),
                "n_evaluation": len(eval_by_generator[name]),
            }

        source_fp = {
            source: rate_at(records_to_scores(records, rule), cut)
            for source, records in forensic_records.items() if records
        }
        gen_real, gen_ai = genimage_records
        gen_real_scores = records_to_scores(gen_real, rule)
        gen_ai_scores = records_to_scores(gen_ai, rule)
        evaluated[rule] = {
            "selection_calibration_macro_recall": float(np.nanmean(calibration_recalls)),
            "threshold": cut,
            "defactify_calibration_fp": point["calibration_fp"],
            "defactify_evaluation_fp": point["evaluation_fp"],
            "defactify_evaluation_recall": point["evaluation_recall"],
            "defactify_evaluation_auc": safe_auc(eval_real, eval_ai),
            "genimage_auc": safe_auc(gen_real_scores, gen_ai_scores),
            "forensics_macro_fp": float(np.mean(list(source_fp.values()))) if source_fp else float("nan"),
            "forensics_worst_fp": float(max(source_fp.values())) if source_fp else float("nan"),
            "forensics_source_fp": source_fp,
            "per_generator": per_generator,
            "counts": {
                "defactify_real_calibration": len(cal_real),
                "defactify_real_evaluation": len(real_eval),
                "defactify_ai_calibration": len(cal_ai),
                "defactify_ai_evaluation": len(eval_ai),
                "forensics_real": sum(len(records) for records in forensic_records.values()),
            },
        }

    # The evaluation half never chooses the winner.  Ties prefer the simpler,
    # historic top-3 rule so a change needs positive calibration evidence.
    preference = {rule: -index for index, rule in enumerate(rules)}
    selected = max(rules, key=lambda rule: (
        evaluated[rule]["selection_calibration_macro_recall"], preference[rule]))
    return selected, evaluated


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tiles", type=Path, default=Path("artifacts/tiles_v1.npz"))
    parser.add_argument("--seeds", type=int, default=3, choices=range(1, len(SEEDS) + 1),
                        help="how many of SEEDS to run; reportable comparisons default to three")
    parser.add_argument("--epochs", type=int, default=8,
                        help="CEILING, not a choice — the best epoch is picked on a "
                             "source-stratified validation slice")
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--limit", type=int, default=LIMIT)
    parser.add_argument("--arms", nargs="+", default=["stats", "resnet18", "small_cnn"],
                        choices=["stats", "resnet18", "small_cnn"])
    parser.add_argument("--aggregations", nargs="+", default=list(AGGREGATION_RULES),
                        choices=AGGREGATION_RULES)
    parser.add_argument("--calibration-fraction", type=float, default=0.5)
    parser.add_argument("--split-seed", type=int, default=2026,
                        help="fixed across model seeds so every arm sees the same evaluation images")
    parser.add_argument("--raw-dir", type=Path, default=Path("artifacts/e20/raw_scores"))
    parser.add_argument("--results", type=Path, default=Path("artifacts/e20/results.json"))
    parser.add_argument("--evaluate-checkpoint", type=Path,
                        help="skip training and evaluate one existing ResNet/SmallCNN checkpoint; "
                             "supports legacy E20 checkpoints")
    parser.add_argument("--train-limit", type=int, default=None,
                        help="subsample the training tiles, stratified by source and label. "
                             "For smoke tests: the point is to prove every arm runs end to end "
                             "before paying for the real thing.")
    args = parser.parse_args()
    if args.limit < 2:
        parser.error("--limit must be at least 2 for disjoint calibration/evaluation")
    if not 0.0 < args.calibration_fraction < 1.0:
        parser.error("--calibration-fraction must be strictly between 0 and 1")

    loaded_run = None
    if args.evaluate_checkpoint:
        arm, seed, scorer, payload = load_cnn_checkpoint(args.evaluate_checkpoint)
        loaded_run = {"arm": arm, "seed": seed, "scorer": scorer, "payload": payload}
        args.arms = [arm]
        tiles = y = sources = None
        print(f"EVALUATION ONLY — {args.evaluate_checkpoint}  arm={arm} seed={seed}")
        print(f"  device: {DEVICE}\n")
    else:
        data = np.load(args.tiles, allow_pickle=True)
        tiles, y, sources = data["x"], data["y"], data["sources"]
        if args.train_limit and args.train_limit < len(y):
            # Stratified, not the first N: the tiles are written in SOURCE ORDER, so
            # tiles[:N] is a subset of one or two sources. That exact slice produced a
            # confidently wrong diagnosis earlier today.
            keep, _ = stratified_holdout(y, sources, args.train_limit / len(y), 0)
            take = np.setdiff1d(np.arange(len(y)), keep)[:args.train_limit]
            tiles, y, sources = tiles[take], y[take], sources[take]
            print(f"SMOKE TEST — training on a stratified {len(y):,} of the full set\n")
        print(f"training tiles: {tiles.shape}  "
              f"{int((y == 0).sum()):,} real / {int((y == 1).sum()):,} ai")
        print(f"  sources: {', '.join(sorted(set(sources.tolist())))}")
        print(f"  device: {DEVICE}\n")

    cached_features = None
    if loaded_run is None and "stats" in args.arms:
        print("extracting 68 statistics for every training tile…")
        start = time.time()
        cached_features = tile_features(tiles)
        print(f"  {cached_features.shape} in {time.time() - start:.0f}s\n")

    results: dict[str, list[dict]] = {}
    for arm in args.arms:
        print(f"=== ARM {arm} ===")
        per_seed = []
        run_seeds = [loaded_run["seed"]] if loaded_run else SEEDS[:args.seeds]
        for seed in run_seeds:
            print(f"  seed {seed}")
            began = time.time()
            training_meta = {"best_epoch": None, "validation_auc": None}
            checkpoint = None
            checkpoint_path = None
            if loaded_run:
                scorer = loaded_run["scorer"]
                training_meta = loaded_run["payload"].get("training", training_meta)
            elif arm == "stats":
                scorer = train_statistics(cached_features, y, seed)
            else:
                scorer, model, training_meta = train_cnn(
                    arm, tiles, y, sources, seed, args.epochs, args.batch)
                checkpoint = {
                    "model": model.state_dict(), "arm": arm, "seed": seed,
                    "training": {
                        **training_meta, "max_epochs": args.epochs, "batch": args.batch,
                        "tile_dataset": str(args.tiles), "tile_dataset_bytes": args.tiles.stat().st_size,
                        "n_tiles": len(y), "labels": dict(Counter(int(v) for v in y)),
                        "sources": dict(Counter(str(v) for v in sources)),
                    },
                    "inference": {
                        "tile_px": TILE, "texture_floor": TEXTURE_FLOOR,
                        "normalization": "imagenet" if arm == "resnet18" else "default",
                        "aggregation_candidates": args.aggregations,
                    },
                }
                checkpoint_path = Path(f"artifacts/tile_{arm}_seed{seed}.pt")
            print(f"    trained in {time.time() - began:.0f}s — evaluating")

            kind = "stats" if arm == "stats" else "cnn"
            all_records = []
            real_def = score_folder_records(
                HOME / "defactify_test/real", kind, scorer,
                "defactify", "real", 0, args.limit)
            all_records += real_def
            generator_records = {}
            for generator in GENERATORS:
                records = score_folder_records(
                    HOME / "defactify_test/ai" / generator, kind, scorer,
                    "defactify", generator, 1, args.limit)
                if records:
                    generator_records[generator] = records
                    all_records += records

            gen_real_dir, gen_ai_dir = TEST_SETS["genimage (trained-on source)"]
            gen_real = score_folder_records(
                gen_real_dir, kind, scorer, "genimage", "real", 0, args.limit)
            gen_ai = score_folder_records(
                gen_ai_dir, kind, scorer, "genimage", "ai", 1, args.limit)
            all_records += gen_real + gen_ai

            forensic_records = {}
            for root in REAL_ONLY.values():
                for folder in sorted(root.glob("*/auth")):
                    source = folder.parent.name
                    records = score_folder_records(
                        folder, kind, scorer, "forensics", source, 0, args.limit)
                    if records:
                        forensic_records[source] = records
                        all_records += records

            raw_path = args.raw_dir / f"{arm}_seed{seed}.jsonl"
            save_raw_records(raw_path, all_records)
            selected, aggregations = evaluate_aggregations(
                real_def, generator_records, (gen_real, gen_ai), forensic_records,
                args.aggregations, args.calibration_fraction, args.split_seed)
            chosen = aggregations[selected]
            measured = {
                "arm": arm, "seed": seed, "training": training_meta,
                "raw_scores": str(raw_path), "selected_aggregation": selected,
                "aggregations": aggregations,
            }
            if checkpoint is not None:
                checkpoint["inference"].update({
                    "selected_aggregation": selected,
                    "threshold": chosen["threshold"],
                    "calibration_fraction": args.calibration_fraction,
                    "split_seed": args.split_seed,
                })
                torch.save(checkpoint, checkpoint_path)
            print("\n      aggregation selected ONLY on calibration halves")
            print(f"      {'rule':<18}{'cal recall':>12}{'eval recall':>13}{'eval FP':>10}"
                  f"{'eval AUC':>11}{'real macro FP':>15}{'worst FP':>11}")
            for rule in args.aggregations:
                row = aggregations[rule]
                mark = "  <- selected" if rule == selected else ""
                print(f"      {rule:<18}{100 * row['selection_calibration_macro_recall']:>11.1f}%"
                      f"{100 * row['defactify_evaluation_recall']:>12.1f}%"
                      f"{100 * row['defactify_evaluation_fp']:>9.1f}%"
                      f"{row['defactify_evaluation_auc']:>11.3f}"
                      f"{100 * row['forensics_macro_fp']:>14.1f}%"
                      f"{100 * row['forensics_worst_fp']:>10.1f}%{mark}")
            print("      selected per-generator evaluation recall: " + ", ".join(
                f"{name} {100 * values['evaluation_recall']:.1f}%"
                for name, values in chosen["per_generator"].items()))
            print(f"      raw per-image/per-tile scores -> {raw_path}")
            if checkpoint_path is not None:
                print(f"      deployable checkpoint contract -> {checkpoint_path}")
            per_seed.append(measured)
        results[arm] = per_seed

    # ---- summary ----------------------------------------------------------
    separator = "=" * 110
    print(f"\n{separator}\nE20 PROTOCOL V2 — aggregation and threshold see calibration only; "
          f"the headline comes from untouched evaluation images.\n{separator}")
    print(f"{'arm':<15}{'seeds':>7}{'eval recall':>16}{'eval AUC':>13}"
          f"{'eval FP':>12}{'real macro FP':>17}{'worst real FP':>16}")
    for arm, runs in results.items():
        selected_rows = [run["aggregations"][run["selected_aggregation"]] for run in runs]
        values = lambda key: np.asarray([row[key] for row in selected_rows], dtype=float)
        cell = lambda key, percent=False: (
            f"{100 * values(key).mean():.1f}%±{100 * values(key).std():.1f}"
            if percent else f"{values(key).mean():.3f}±{values(key).std():.3f}")
        print(f"{arm:<15}{len(runs):>7}{cell('defactify_evaluation_recall', True):>16}"
              f"{cell('defactify_evaluation_auc'):>13}{cell('defactify_evaluation_fp', True):>12}"
              f"{cell('forensics_macro_fp', True):>17}{cell('forensics_worst_fp', True):>16}")

    payload = {
        "protocol": {
            "version": 2, "fp_budget": FP_BUDGET,
            "calibration_fraction": args.calibration_fraction, "split_seed": args.split_seed,
            "aggregations": args.aggregations, "limit_per_set": args.limit,
            "selection": "macro generator recall on calibration halves only",
        },
        "runs": results,
    }
    args.results.parent.mkdir(parents=True, exist_ok=True)
    with args.results.open("w") as handle:
        json.dump(json_compatible(payload), handle, indent=2, allow_nan=False)
    print(f"\nstructured results -> {args.results}")
    print("Reminder: this protocol selects both aggregation and threshold without touching the "
          "evaluation halves. GenImage remains a discounted, non-content-controlled diagnostic.")


if __name__ == "__main__":
    main()
