# =============================================================================
# backbone_features.py — WHAT THIS FILE DOES
# -----------------------------------------------------------------------------
# Runs a FROZEN pretrained backbone over the training pool and the test sets,
# and caches the embeddings. A linear probe on top is then trained in seconds.
#
# WHY A FROZEN BACKBONE RATHER THAN FINE-TUNING
# -----------------------------------------------------------------------------
# E2 established that the representation, not the classifier, is this project's
# bottleneck — but it only ever tested that claim with a weak representation
# (a 300k-parameter CNN's penultimate layer). E15 hit the same ceiling from the
# other side: three statistics models trained on 9.9k, 101k and a balanced
# multi-source pool all land at ~0.69 AUC on Defactify and ~35% AI recall at a
# 10% false-positive budget. The training data is no longer the limit; what the
# features can express is.
#
# IMAGE_FORENSICS_REFERENCE.md 4.4: CLIP-feature detectors are "currently among
# the best out-of-distribution generalizers, even trained on little data".
# Freezing the backbone and fitting a linear probe is that recipe, and it costs
# one forward pass rather than a training run.
#
# WHY DINOv2 AT 518px
# -----------------------------------------------------------------------------
# Every failure this project has measured (E5, E6, E7) came from downscaling
# away the evidence. DINOv2 ViT-S/14 takes 518x518, so a 1024px image is
# downscaled 2.0x instead of the 4.6x a 224px model forces. It is also
# self-supervised, so its features are not tied to ImageNet class semantics —
# which matters here, because E14 showed our models were latching onto exactly
# that kind of source identity.
#
# CODE BLOCKS IN THIS FILE
# -----------------------------------------------------------------------------
# BACKBONES     name -> (timm id, input size). CLIP at 224 is included as the
#               control: if DINOv2 wins, the reader should be able to tell
#               whether that came from the representation or from the extra
#               resolution.
# embed_batch() Frozen forward pass, no gradients, batched on MPS.
# from_folder() Test sets, which live as image folders.
# from_pool()   Training pool. Streams the sources and keeps an image only if
#               its perceptual hash is in the pool index — same matching as
#               pool_features.py, and for the same reason (build_pool records a
#               shard, not a row).
# =============================================================================

import argparse
import csv
from pathlib import Path

import numpy as np
import timm
import torch
from PIL import Image

from pixelproof.build_pool import (SOURCES, _iter_one_folder, _iter_one_parquet,
                                   phash, real_files)

Image.MAX_IMAGE_PIXELS = None

BACKBONES = {
    "dinov2": ("vit_small_patch14_dinov2.lvd142m", 518),
    "clip": ("vit_base_patch16_clip_224.openai", 224),
    "convnext": ("convnext_tiny.fb_in22k", 224),
}
BATCH = 8
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp", ".JPEG"}

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def build(name: str):
    model_id, size = BACKBONES[name]
    model = timm.create_model(model_id, pretrained=True, num_classes=0).eval().to(device)
    config = timm.data.resolve_data_config({}, model=model)
    config["input_size"] = (3, size, size)
    transform = timm.data.create_transform(**config, is_training=False)
    return model, transform, size


@torch.no_grad()
def embed_batch(model, transform, images: list[Image.Image]) -> np.ndarray:
    batch = torch.stack([transform(im.convert("RGB")) for im in images]).to(device)
    return model(batch).float().cpu().numpy()


def from_folder(model, transform, folder: Path, limit: int | None = None) -> np.ndarray:
    files = sorted(p for p in folder.rglob("*")
                   if p.suffix in IMAGE_EXT and not p.name.startswith("._"))
    if limit:
        files = files[:limit]
    out, buffer = [], []
    for i, path in enumerate(files):
        try:
            with Image.open(path) as image:
                image.load()
                buffer.append(image.copy())
        except Exception:
            continue
        if len(buffer) == BATCH:
            out.append(embed_batch(model, transform, buffer))
            buffer = []
            print(f"\r    {i + 1}/{len(files)}", end="", flush=True)
    if buffer:
        out.append(embed_batch(model, transform, buffer))
    print()
    return np.concatenate(out) if out else np.zeros((0, 0), dtype=np.float32)


def from_pool(model, transform, index_csv: Path):
    """Embeddings for the images selected in a pool index, matched by hash."""
    with index_csv.open() as handle:
        keep = {r["phash"]: (int(r["label"]), r["source"]) for r in csv.DictReader(handle)}
    print(f"  pool: {len(keep):,} images")

    vectors, labels, sources, seen = [], [], [], set()
    buffer, pending = [], []
    limit_per_source = 40000

    def flush():
        if not buffer:
            return
        vectors.append(embed_batch(model, transform, buffer))
        for digest in pending:
            labels.append(keep[digest][0])
            sources.append(keep[digest][1])
        buffer.clear()
        pending.clear()

    for name, spec in SOURCES.items():
        units = (sorted(real_files(spec["path"], "*.parquet")) if spec["kind"] == "parquet"
                 else [spec["path"] / folder for folder in spec["folders"]])
        per_unit = max(1, limit_per_source // len(units))
        for unit in units:
            stream = (_iter_one_parquet(spec, unit, per_unit) if spec["kind"] == "parquet"
                      else _iter_one_folder(spec, unit, per_unit))
            for image, _s, _l, _loc, _e in stream:
                digest = phash(image)
                if digest not in keep or digest in seen:
                    continue
                seen.add(digest)
                buffer.append(image.copy())
                pending.append(digest)
                if len(buffer) == BATCH:
                    flush()
                    print(f"\r    {name}: {len(labels):,}", end="", flush=True)
        flush()
        print(f"\r    {name}: {len(labels):,}          ")
    return (np.concatenate(vectors), np.array(labels, dtype=np.int64), np.array(sources))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", default="dinov2", choices=list(BACKBONES))
    parser.add_argument("--index", type=Path, default=Path("artifacts/pool_balanced_v3.csv"))
    parser.add_argument("--limit-test", type=int, default=1200)
    args = parser.parse_args()

    model, transform, size = build(args.backbone)
    dim = model.num_features
    print(f"{args.backbone}: {BACKBONES[args.backbone][0]}  input {size}px  dim {dim}  on {device}")

    home = Path.home() / "Desktop"
    out = Path("artifacts") / f"backbone_{args.backbone}"
    out.mkdir(parents=True, exist_ok=True)

    print("\ntest sets:")
    tests = {
        "genimage_test_real": home / "genimage_split/test/REAL",
        "genimage_test_ai": home / "genimage_split/test/FAKE",
        "archive1_real": home / "archive1/real_dataset",
        "archive1_ai": home / "archive1/Ai_generated_dataset",
        "defactify_real": home / "defactify_test/real",
        **{f"defactify_{g}": home / "defactify_test/ai" / g
           for g in ["dalle3", "midjourney", "sd21", "sd3", "sdxl"]},
    }
    for name, folder in tests.items():
        target = out / f"{name}.npy"
        if target.exists():
            print(f"  {name}: cached")
            continue
        print(f"  {name}")
        np.save(target, from_folder(model, transform, folder, args.limit_test))

    print("\ntraining pool:")
    target = out / "pool.npz"
    if target.exists():
        print("  cached")
    else:
        x, y, s = from_pool(model, transform, args.index)
        np.savez_compressed(target, x=x, y=y, sources=s)
        print(f"  {len(x):,} embeddings -> {target}")


if __name__ == "__main__":
    main()
