# =============================================================================
# build_tile_dataset.py — WHAT THIS FILE DOES
# -----------------------------------------------------------------------------
# Cuts one 128x128 NATIVE tile out of every image in a pool index and caches the
# raw pixels to a single .npz. This is the shared input for Phase 2b: four
# different model families read the SAME tensor, so the comparison between them
# has exactly one variable — the model.
#
# WHY CACHE PIXELS RATHER THAN FEATURES
# -----------------------------------------------------------------------------
# The statistics arm wants 68 numbers per tile; the CNN arms want the pixels.
# Extracting features twice, or re-reading 48k images from parquet on the SSD
# once per epoch, would both be slow and — worse — would let the two arms see
# subtly different crops. One uint8 array, written once, removes that risk.
# 48k tiles at 128x128x3 is 2.4 GB, which is nothing next to correctness here.
#
# TWO DECISIONS THAT ARE NOT OBVIOUS
# -----------------------------------------------------------------------------
# 1. RANDOM tile position, not the centre. `feature_crop128` was fitted on
#    centre crops, and E11 flagged the mismatch as a "mild caveat": at inference
#    a tile can come from anywhere, including the edges. Now that tiles cover the
#    whole image (Phase 2a) the mismatch is no longer mild, and the project's own
#    law says train on what you will be shown. Positions are seeded.
#
# 2. The SAME texture floor as inference. score_tiles() drops tiles below
#    0.04 grey-level std because they cannot enter a top-k aggregate, so a model
#    trained on them would be fitting a population it never sees. An image whose
#    every candidate tile is flat contributes nothing and is skipped — recorded
#    in the summary rather than silently dropped.
#
# CODE BLOCKS IN THIS FILE
# -----------------------------------------------------------------------------
# open_source()   Yields (PIL image, phash) per source, reusing build_pool's
#                 readers so "which image is which row" cannot drift.
# pick_tile()     Chooses a seeded random 128px window that clears the texture
#                 floor, trying a few times before giving up on the image.
# main()          Walks the index, writes tiles.npz (x, y, sources, hashes),
#                 and prints how many images were skipped and why.
# =============================================================================

import argparse
import csv
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

from pixelproof.build_pool import (SOURCES, _iter_one_folder, _iter_one_parquet,
                                   phash, real_files, to_project_label)

Image.MAX_IMAGE_PIXELS = None

TILE = 128
TEXTURE_FLOOR = 0.04      # must match feature_model.TEXTURE_FLOOR
ATTEMPTS = 8              # random windows to try before abandoning an image


def texture_of(patch: Image.Image) -> float:
    return float((np.asarray(patch.convert("L"), dtype=np.float32) / 255.0).std())


def pick_tile(image: Image.Image, rng: np.random.RandomState) -> np.ndarray | None:
    """A seeded random 128px native window that clears the texture floor."""
    image = image.convert("RGB")
    width, height = image.size
    if width < TILE or height < TILE:
        return None                            # excluded by the pool's 128px floor anyway
    best, best_texture = None, -1.0
    for _ in range(ATTEMPTS):
        x = int(rng.randint(0, width - TILE + 1))
        y = int(rng.randint(0, height - TILE + 1))
        patch = image.crop((x, y, x + TILE, y + TILE))
        texture = texture_of(patch)
        if texture >= TEXTURE_FLOOR:
            return np.asarray(patch, dtype=np.uint8)
        if texture > best_texture:
            best, best_texture = patch, texture
    return None if best_texture < TEXTURE_FLOOR / 2 else np.asarray(best, dtype=np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser(description="Cache one native 128px tile per pool image.")
    parser.add_argument("--index", type=Path, default=Path("artifacts/pool_tile_v2.csv"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/tiles_v1.npz"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with args.index.open() as handle:
        wanted = {r["phash"]: (int(r["label"]), r["source"]) for r in csv.DictReader(handle)}
    print(f"index: {len(wanted):,} images from {args.index.name}")

    rng = np.random.RandomState(args.seed)
    tiles, labels, sources, hashes = [], [], [], []
    skipped = Counter()
    seen = set()

    for name, spec in SOURCES.items():
        units = (sorted(real_files(spec["path"], "*.parquet")) if spec["kind"] == "parquet"
                 else [spec["path"] / folder for folder in spec["folders"]])
        if not units:
            print(f"  {name}: NOT FOUND")
            continue
        taken = 0
        for unit in units:
            stream = (_iter_one_parquet(spec, unit, None) if spec["kind"] == "parquet"
                      else _iter_one_folder(spec, unit, None))
            for image, _bytes, raw_label, _loc, _extra in stream:
                digest = phash(image)
                if digest not in wanted or digest in seen:
                    continue
                seen.add(digest)
                patch = pick_tile(image, rng)
                if patch is None:
                    skipped[name] += 1
                    continue
                tiles.append(patch)
                # label from the index, which build_pool already mapped; assert the
                # source's own map agrees, so a stale index cannot slip through
                assert wanted[digest][0] == to_project_label(spec, raw_label), digest
                labels.append(wanted[digest][0])
                sources.append(name)
                hashes.append(digest)
                taken += 1
                if taken % 500 == 0:
                    print(f"\r  {name}: {taken:,}", end="", flush=True)
        print(f"\r  {name}: {taken:,} tiles ({skipped[name]:,} images too flat)      ")

    x = np.stack(tiles)
    y = np.array(labels, dtype=np.int64)
    print(f"\n{len(x):,} tiles  {x.shape}  {x.nbytes / 1e9:.2f} GB in memory")
    print(f"  {int((y == 0).sum()):,} real / {int((y == 1).sum()):,} ai")
    print(f"  by source: " + ", ".join(f"{k}:{v:,}" for k, v in Counter(sources).most_common()))
    if sum(skipped.values()):
        print(f"  skipped as too flat: {sum(skipped.values()):,}")
    missing = len(wanted) - len(x) - sum(skipped.values())
    if missing:
        print(f"  ⚠️ {missing:,} indexed images were never reached — check the readers")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, x=x, y=y, sources=np.array(sources), hashes=np.array(hashes))
    print(f"\nwrote {args.output}  ({args.output.stat().st_size / 1e9:.2f} GB)")


if __name__ == "__main__":
    main()
