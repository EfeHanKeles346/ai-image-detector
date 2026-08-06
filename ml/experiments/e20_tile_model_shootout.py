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
# anchored, texture floor), every tile is scored, and the top-3 mean is the
# image's score. Measuring per-tile accuracy would flatter every arm and answer a
# question nobody asks.
#
# GenImage is reported but discounted: its reals are ImageNet nature photos and
# its fakes are other content, so it is not content-controlled and a model can
# score there by recognising subject matter. Defactify generates its fakes from
# the same MS-COCO captions as its reals, which makes it the fair one.

import argparse
import time
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
from pixelproof.features import THREADS, _vector, select_tiles
from pixelproof.models import create_model

HOME = Path.home() / "Desktop"
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


def train_cnn(name: str, tiles: np.ndarray, y: np.ndarray, sources: np.ndarray,
              seed: int, max_epochs: int, batch: int):
    """Fine-tune / train a CNN on native 128px tiles, stopping where validation says.

    ImageNet normalisation for the pretrained backbone, 0.5/0.5 for the
    from-scratch one — each gets the statistics it expects (data.py).
    Augmentation is horizontal flip ONLY: colour or blur would destroy the
    generation traces this whole project is built on reading (ROADMAP §3).

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

    return score, model


# --------------------------------------------------------------------------- #
# evaluation — tile the image the way serve.py does, aggregate top-3
# --------------------------------------------------------------------------- #
def image_files(folder: Path, limit: int) -> list[Path]:
    files = sorted(p for p in folder.rglob("*")
                   if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
                   and not p.name.startswith("._") and ".mask" not in p.name)
    return files[::max(1, len(files) // limit)][:limit]


def image_tiles(path: Path) -> np.ndarray | None:
    """Uint8 tiles for one test image, same geometry as inference."""
    try:
        with Image.open(path) as image:
            patches, _, _ = select_tiles(image, tile=TILE, texture_floor=TEXTURE_FLOOR)
            return np.stack([np.asarray(p, dtype=np.uint8) for p in patches])
    except Exception:
        return None


def score_folder(folder: Path, arm_kind: str, scorer, limit: int = LIMIT) -> np.ndarray:
    """One score per image: top-3 mean over its tiles."""
    scores = []
    for path in image_files(folder, limit):
        tiles = image_tiles(path)
        if tiles is None or len(tiles) == 0:
            continue
        per_tile = scorer(tile_features(tiles) if arm_kind == "stats" else tiles)
        scores.append(float(np.sort(per_tile)[-3:].mean()))
    return np.array(scores)


def operating_point(real: np.ndarray, ai: np.ndarray, budget: float) -> tuple[float, float]:
    """AI recall at the threshold giving `budget` false positives on real photos."""
    if len(real) == 0 or len(ai) == 0:
        return float("nan"), float("nan")
    cut = float(np.percentile(real, 100 * (1 - budget)))
    return cut, float((ai >= cut).mean())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tiles", type=Path, default=Path("artifacts/tiles_v1.npz"))
    parser.add_argument("--seeds", type=int, default=1, help="how many of SEEDS to run")
    parser.add_argument("--epochs", type=int, default=8,
                        help="CEILING, not a choice — the best epoch is picked on a "
                             "source-stratified validation slice")
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--limit", type=int, default=LIMIT)
    parser.add_argument("--arms", nargs="*", default=["stats", "resnet18", "small_cnn"])
    parser.add_argument("--train-limit", type=int, default=None,
                        help="subsample the training tiles, stratified by source and label. "
                             "For smoke tests: the point is to prove every arm runs end to end "
                             "before paying for the real thing.")
    args = parser.parse_args()

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
    print(f"training tiles: {tiles.shape}  {int((y == 0).sum()):,} real / {int((y == 1).sum()):,} ai")
    print(f"  sources: {', '.join(sorted(set(sources.tolist())))}")
    print(f"  device: {DEVICE}\n")

    cached_features = None
    if "stats" in args.arms:
        print("extracting 68 statistics for every training tile…")
        start = time.time()
        cached_features = tile_features(tiles)
        print(f"  {cached_features.shape} in {time.time() - start:.0f}s\n")

    results: dict[str, dict] = {}
    for arm in args.arms:
        print(f"=== ARM {arm} ===")
        per_seed = []
        for seed in SEEDS[:args.seeds]:
            print(f"  seed {seed}")
            began = time.time()
            if arm == "stats":
                scorer = train_statistics(cached_features, y, seed)
            else:
                scorer, model = train_cnn(arm, tiles, y, sources, seed, args.epochs, args.batch)
                torch.save({"model": model.state_dict(), "arm": arm, "seed": seed},
                           f"artifacts/tile_{arm}_seed{seed}.pt")
            print(f"    trained in {time.time() - began:.0f}s — evaluating")

            kind = "stats" if arm == "stats" else "cnn"
            measured = {}

            # Defactify's ai/ folder mixes five generators whose difficulty for a
            # tile method spans almost the whole range — E8 measured crop128 at
            # 0.867 on SDXL and 0.377 on DALL-E 3, i.e. below chance, and E11
            # deliberately optimised the grid on the high-resolution generators
            # only because "low-resolution inputs are the CNNs' domain". Pooling
            # them into one AUC reports their average and hides the split that
            # decides which method should handle which input.
            real_def = score_folder(HOME / "defactify_test/real", kind, scorer, args.limit)
            print(f"      {'per generator (vs defactify real)':<40}")
            for generator in GENERATORS:
                gen = score_folder(HOME / "defactify_test/ai" / generator, kind, scorer, args.limit)
                if not len(gen) or not len(real_def):
                    continue
                truth = np.r_[np.zeros(len(real_def)), np.ones(len(gen))]
                gen_auc = roc_auc_score(truth, np.r_[real_def, gen])
                _, gen_recall = operating_point(real_def, gen, FP_BUDGET)
                measured[f"  defactify/{generator}"] = {
                    "auc": float(gen_auc), "fp_at_half": float((real_def >= 0.5).mean()),
                    "recall_at_budget": gen_recall, "cut": float("nan"),
                    "n_real": len(real_def), "n_ai": len(gen)}
                print(f"        {generator:<20} AUC {gen_auc:.3f}   "
                      f"recall@{int(FP_BUDGET * 100)}%FP {100 * gen_recall:5.1f}%")
            for label, (real_dir, ai_dir) in TEST_SETS.items():
                real = score_folder(real_dir, kind, scorer, args.limit)
                ai = score_folder(ai_dir, kind, scorer, args.limit)
                truth = np.r_[np.zeros(len(real)), np.ones(len(ai))]
                auc = roc_auc_score(truth, np.r_[real, ai]) if len(real) and len(ai) else float("nan")
                cut, recall = operating_point(real, ai, FP_BUDGET)
                measured[label] = {"auc": float(auc), "fp_at_half": float((real >= 0.5).mean()),
                                   "recall_at_budget": recall, "cut": cut,
                                   "n_real": len(real), "n_ai": len(ai)}
                print(f"      {label:<40} AUC {auc:.3f}  FP@0.5 {100 * measured[label]['fp_at_half']:5.1f}%"
                      f"  recall@{int(FP_BUDGET * 100)}%FP {100 * recall:5.1f}%")

            # Real photographs only. No AUC is possible and none is wanted: the
            # question is simply how often a camera photograph from a pipeline the
            # model never saw gets called AI. Reported at 0.5 and at the threshold
            # Defactify's reals calibrated, which is the one a product would ship.
            shipped_cut = measured["defactify (unseen, content-controlled)"]["cut"]
            for label, root in REAL_ONLY.items():
                per_source = {}
                for folder in sorted(root.glob("*/auth")):
                    scores = score_folder(folder, kind, scorer, args.limit)
                    if len(scores):
                        per_source[folder.parent.name] = scores
                if not per_source:
                    continue
                every = np.concatenate(list(per_source.values()))
                measured[label] = {"auc": float("nan"),
                                   "fp_at_half": float((every >= 0.5).mean()),
                                   "recall_at_budget": float("nan"), "cut": float("nan"),
                                   "n_real": len(every), "n_ai": 0,
                                   "fp_at_shipped": float((every >= shipped_cut).mean())}
                print(f"      {label:<40} n={len(every):<5} FP@0.5 "
                      f"{100 * measured[label]['fp_at_half']:5.1f}%   "
                      f"FP@defactify-cut({shipped_cut:.3f}) "
                      f"{100 * measured[label]['fp_at_shipped']:5.1f}%")
                worst = sorted(per_source.items(), key=lambda kv: -(kv[1] >= 0.5).mean())[:3]
                print("        worst sources @0.5: " +
                      ", ".join(f"{k} {100 * (v >= 0.5).mean():.0f}%" for k, v in worst))
            per_seed.append(measured)
        results[arm] = per_seed

    # ---- summary ----------------------------------------------------------
    print(f"\n{'=' * 96}\nE20 — the operating point is the headline. "
          f"AI recall at a {int(FP_BUDGET * 100)}% false-positive budget.\n{'=' * 96}")
    header = f"{'test set':<42}" + "".join(f"{a:>20}" for a in args.arms)
    for metric, title in (("recall_at_budget", f"AI RECALL @ {int(FP_BUDGET * 100)}% FP  ← headline"),
                          ("auc", "AUC (ranking only)"),
                          ("fp_at_half", "FALSE POSITIVES at threshold 0.5")):
        print(f"\n{title}\n{header}")
        rows = list(TEST_SETS) + [f"  defactify/{g}" for g in GENERATORS] + list(REAL_ONLY)
        for label in rows:
            row = f"{label:<42}"
            for arm in args.arms:
                values = [s[label][metric] for s in results[arm] if label in s]
                if not values or np.all(np.isnan(values)):
                    row += f"{'—':>20}"; continue
                mean = float(np.nanmean(values))
                spread = f"±{np.std(values):.3f}" if len(values) > 1 else ""
                row += f"{(f'{mean:.3f}' if metric == 'auc' else f'{100 * mean:.1f}%') + spread:>20}"
            print(row)

    print("\nReminder: GenImage is not content-controlled — its reals are ImageNet nature photos "
          "and its\nfakes are other content, so a high score there can come from recognising "
          "subject matter.\nDefactify generates its fakes from the same captions as its reals. "
          "Weight Defactify.")


if __name__ == "__main__":
    main()
