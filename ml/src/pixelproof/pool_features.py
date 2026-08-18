# =============================================================================
# pool_features.py — WHAT THIS FILE DOES
# -----------------------------------------------------------------------------
# Extracts the 68 statistics for every image selected in a pool index and caches
# the result, so training and re-training cost seconds instead of an hour.
#
# WHY MATCHING IS DONE BY HASH
# -----------------------------------------------------------------------------
# build_pool.py records a parquet image's location as the shard filename, which
# identifies the file but not the row inside it — so an image cannot be
# re-opened by address. This pass streams the sources again and keeps an image
# only when its perceptual hash appears in the selected pool. One pass does the
# matching and the extraction together, which is what we would pay anyway:
# reading the images is the cost, and hashing on top is free by comparison.
#
# MEMORY DISCIPLINE — the first version of this file crashed an 18 GB machine
# -----------------------------------------------------------------------------
# It asked for ~56 GB and the system died in swap. Four separate mistakes, each
# fixed here, and worth keeping written down:
#
#   1. The 101k-entry hash set was passed inside every one of the 399 job
#      tuples, so ProcessPoolExecutor pickled a full copy per job — gigabytes of
#      queued payload before any work started. Now it is sent ONCE per worker
#      via an initializer.
#   2. Units were read with no row limit, so eleven workers each opened a whole
#      ~3 GB CommunityForensics shard at the same time. Now MAX_ROWS_PER_UNIT
#      bounds what any single worker holds.
#   3. Every feature vector was accumulated in the parent until the end. Now
#      each unit's output is written to disk immediately and concatenated once.
#   4. Deduplication used `digest in list` — O(n^2), which kept workers alive
#      (and their memory resident) far longer than needed. Now a set.
#
# Workers are also capped below the core count: this is memory-bound, not
# CPU-bound, and 11 parallel shard readers is what caused the crash.
#
# CODE BLOCKS IN THIS FILE
# -----------------------------------------------------------------------------
# wanted()       Loads the pool CSV into {phash: label}. Defaults to the
#                BALANCED pool, because the raw index carries a 3.4x resolution
#                gap between classes (HISTORY 1b) that a native-resolution model
#                reads as a shortcut.
# _init_worker() Installs the hash set as a per-process global — sent once.
# _one_unit()    Streams one shard (or class folder), keeps wanted images,
#                writes their vectors to a temp .npz, returns just the path.
# main()         Fans out, concatenates the temp files, writes the final cache.
# =============================================================================

import argparse
import csv
import os
import tempfile
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from pixelproof.build_pool import (SOURCES, _iter_one_folder, _iter_one_parquet,
                                   phash, real_files)
from pixelproof.features import FEATURE_NAMES, extract_from_image

# Memory-bound, not CPU-bound: each worker may hold a decoded image plus a
# parquet read buffer. Eleven of those at once is what killed the first run.
MAX_WORKERS = min(5, (os.cpu_count() or 4))
# build_pool.py sampled each source with per_unit = LIMIT_PER_SOURCE / n_units and
# read only that many rows from the head of every shard. Reading more here is pure
# waste — the extra rows cannot be in the pool. Matching that stride exactly cut
# this pass from ~1.6M reads to ~200k.
LIMIT_PER_SOURCE = 40000

_KEEP: dict[str, int] = {}


def wanted(index_csv: Path) -> dict[str, int]:
    with index_csv.open() as handle:
        return {row["phash"]: int(row["label"]) for row in csv.DictReader(handle)}


def _init_worker(keep: dict[str, int]) -> None:
    global _KEEP
    _KEEP = keep


def _one_unit(job):
    name, spec, unit, scratch, per_unit = job
    stream = (_iter_one_parquet(spec, unit, per_unit) if spec["kind"] == "parquet"
              else _iter_one_folder(spec, unit, per_unit))

    vectors, labels, hashes = [], [], []
    seen: set[str] = set()
    for image, _size, _label, _location, _extra in stream:
        digest = phash(image)
        if digest not in _KEEP or digest in seen:
            continue
        try:
            vectors.append(extract_from_image(image))
        except Exception:
            continue
        seen.add(digest)
        labels.append(_KEEP[digest])
        hashes.append(digest)

    if not vectors:
        return name, None, 0
    target = Path(scratch) / f"{name}_{unit.stem}.npz"
    np.savez(target, x=np.stack(vectors), y=np.array(labels, dtype=np.int64),
             h=np.array(hashes))
    return name, str(target), len(vectors)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, default=Path("artifacts/pool_balanced.csv"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/pool_features.npz"))
    parser.add_argument("--workers", type=int, default=MAX_WORKERS)
    args = parser.parse_args()

    keep = wanted(args.index)
    print(f"pool: {len(keep):,} images  "
          f"({sum(1 for v in keep.values() if v == 0):,} real / {sum(keep.values()):,} ai)")

    jobs_meta = []
    with tempfile.TemporaryDirectory(prefix="poolfeat_") as scratch:
        for name, spec in SOURCES.items():
            units = (sorted(real_files(spec["path"], "*.parquet")) if spec["kind"] == "parquet"
                     else [spec["path"] / folder for folder in spec["folders"]])
            per_unit = max(1, LIMIT_PER_SOURCE // len(units))
            jobs_meta += [(name, spec, unit, scratch, per_unit) for unit in units]
        print(f"streaming {len(jobs_meta)} units, {args.workers} workers "
              f"(per-unit row caps match build_pool's sampling stride)")

        parts, counts, done, total = [], Counter(), 0, 0
        with ProcessPoolExecutor(max_workers=args.workers,
                                 initializer=_init_worker, initargs=(keep,)) as pool:
            for name, path, count in pool.map(_one_unit, jobs_meta, chunksize=1):
                if path:
                    parts.append(path)
                counts[name] += count
                total += count
                done += 1
                print(f"\r  {done}/{len(jobs_meta)} units · {total:,} extracted", end="", flush=True)
        print()

        # Concatenate once, deduplicating across sources (a hash can appear twice).
        xs, ys, hs, seen = [], [], [], set()
        for path in parts:
            with np.load(path) as part:
                mask = np.array([h not in seen for h in part["h"]])
                if mask.any():
                    seen.update(part["h"][mask].tolist())
                    xs.append(part["x"][mask])
                    ys.append(part["y"][mask])
                    hs.append(part["h"][mask])

    x = np.concatenate(xs)
    y = np.concatenate(ys)
    h = np.concatenate(hs)

    for name, count in counts.most_common():
        print(f"  {name:<24}{count:>10,}")
    print(f"\n{len(x):,} unique images  ({int((y == 0).sum()):,} real / {int((y == 1).sum()):,} ai)")
    print(f"not reached: {len(keep) - len(x):,} (outside the sampling stride)")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, x=x, y=y, hashes=h,
                        feature_names=np.array(FEATURE_NAMES))
    print(f"wrote {args.output}  ({args.output.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
