# =============================================================================
# make_balanced_pool.py — WHAT THIS FILE DOES
# -----------------------------------------------------------------------------
# Turns the raw pool index (build_pool.py -> pool_index.csv) into a RESOLUTION-
# BALANCED training index. Nothing is copied and nothing is resized: this only
# decides which rows of the index a training run may use.
#
# WHY THIS EXISTS
# -----------------------------------------------------------------------------
# Five individually-clean sources merged into a pool that was not clean. E12
# measured it: real photographs sat at a median of 1024px and AI images at
# 300px — a 3.4x resolution gap between the classes. A native-resolution model
# reads that as a shortcut (ROADMAP 1b), so it would separate the classes
# without ever looking at generation traces. Band-balancing removes the gap,
# and it costs ~40% of the rows.
#
# THE RULE
# -----------------------------------------------------------------------------
# Split by longest side into bands, and inside each band keep
# min(n_real, n_ai) images from BOTH classes. Consequences:
#   - inside a band the classes are equal, so resolution carries no class
#     information at any scale;
#   - summed over bands the classes are equal, so the pool is balanced;
#   - nothing is invented — we only drop, never upsample or resize. Resolution
#     cannot be manufactured (E5 is what happens when you try).
#
# PROVENANCE — this is a RECONSTRUCTION
# -----------------------------------------------------------------------------
# The script that produced the original artifacts/pool_balanced.csv was never
# committed; five files read that CSV and none wrote it, so E12-E16 were not
# reproducible from the repo. The rule above was recovered by fitting candidate
# band edges against the surviving CSV: BANDS below reproduce it exactly, band
# for band and 51,246 rows per class. Verify with --verify-against.
#
# Note for the log: EXPERIMENTS.md E12 describes "six resolution bands". The
# surviving artifact is consistent with FOUR. The doc is imprecise, not the data.
#
# RESIDUAL GAP — why --bands exists
# -----------------------------------------------------------------------------
# The recovered 4-band structure reproduces the original's counts exactly, but
# not its residual gap: E12 reports 1.08x after balancing and this rule gives
# 1.68x. The real class matches exactly (median 431); the AI class does not
# (256 here against E12's 400). So the original sampled non-uniformly INSIDE a
# band, and that rule cannot be recovered from counts alone.
#
# Rather than imitate a lost rule, the band grid is a parameter. Balancing is
# only ever as tight as the bands are narrow: inside one band the composition is
# free, so a wide band leaves a gap behind. Narrower bands close it and cost
# more rows. Pick the trade-off with --bands and read it off the printed table.
#
# CODE BLOCKS IN THIS FILE
# -----------------------------------------------------------------------------
# BANDS         Default longest-side cut points — the grid recovered from the
#               original artifact, kept as the default so the E12-E16 row counts
#               stay reproducible. Override with --bands for tighter balancing.
#
# longest_side()  One definition of "how big is this image", used everywhere.
#                 Longest side rather than area, because it is what decides how
#                 much a 224px pipeline has to downscale (ROADMAP 4b).
#
# profile()     Per-class median / p10 / p90 and the class gap ratio. Printed
#               before and after, so the effect of balancing is visible rather
#               than asserted.
#
# balance()     The rule itself. Seeded, so two runs give the same pool.
#
# main()        Reads the index, applies optional filters (--min-side,
#               --exclude-source) that Faz 1 needs, balances, prints the
#               before/after audit and writes the CSV.
# =============================================================================

import argparse
import csv
import random
from collections import Counter, defaultdict
from pathlib import Path

# Recovered from artifacts/pool_balanced.csv — see PROVENANCE above.
BANDS = [0, 128, 256, 1024, 10 ** 9]
INFINITY = 10 ** 9

REAL, AI = 0, 1


def longest_side(row: dict) -> int:
    return max(int(row["w"]), int(row["h"]))


def band_of(side: int, bands: list[int]) -> int:
    for index, (low, high) in enumerate(zip(bands[:-1], bands[1:])):
        if low <= side < high:
            return index
    return len(bands) - 2


def _median(values: list[int]) -> float:
    values = sorted(values)
    if not values:
        return 0.0
    middle = len(values) // 2
    return float(values[middle] if len(values) % 2 else (values[middle - 1] + values[middle]) / 2)


def _percentile(values: list[int], q: float) -> float:
    values = sorted(values)
    if not values:
        return 0.0
    return float(values[min(len(values) - 1, int(q / 100 * len(values)))])


def profile(rows: list[dict], title: str) -> float:
    """Per-class resolution profile. Returns the class gap ratio (1.0 = no gap)."""
    sides = {label: [longest_side(r) for r in rows if int(r["label"]) == label]
             for label in (REAL, AI)}
    print(f"\n{title}  —  {len(rows):,} rows")
    print(f"  {'class':<8}{'n':>9}{'p10':>7}{'median':>8}{'p90':>7}")
    for label, name in ((REAL, "real"), (AI, "ai")):
        values = sides[label]
        if not values:
            print(f"  {name:<8}{0:>9}")
            continue
        print(f"  {name:<8}{len(values):>9,}{_percentile(values, 10):>7.0f}"
              f"{_median(values):>8.0f}{_percentile(values, 90):>7.0f}")
    medians = [_median(sides[label]) for label in (REAL, AI)]
    if min(medians) <= 0:
        return float("inf")
    gap = max(medians) / min(medians)
    print(f"  class resolution gap: {gap:.2f}x"
          f"{'   <-- shortcut risk (ROADMAP 1b)' if gap > 2.0 else ''}")
    return gap


def balance(rows: list[dict], seed: int, bands: list[int]) -> tuple[list[dict], list[tuple]]:
    """Keep min(n_real, n_ai) per resolution band, from both classes."""
    buckets: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for row in rows:
        buckets[(band_of(longest_side(row), bands), int(row["label"]))].append(row)

    rng = random.Random(seed)
    kept, report = [], []
    for band in range(len(bands) - 1):
        real, ai = buckets[(band, REAL)], buckets[(band, AI)]
        take = min(len(real), len(ai))
        for group in (real, ai):
            chosen = group[:] if len(group) == take else rng.sample(group, take)
            kept += chosen
        report.append((bands[band], bands[band + 1], len(real), len(ai), take))
    return kept, report


def main() -> None:
    parser = argparse.ArgumentParser(description="Resolution-balance the training pool index.")
    parser.add_argument("--index", type=Path, default=Path("artifacts/pool_index.csv"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/pool_balanced.csv"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-side", type=int, default=0,
                        help="drop images whose longest side is below this. Images under the "
                             "128px tile size are reflection-padded by features.py, i.e. the "
                             "model is shown a synthetic pattern rather than a photograph.")
    parser.add_argument("--exclude-source", action="append", default=[],
                        help="drop a source entirely; repeatable")
    parser.add_argument("--dedupe", action="store_true",
                        help="keep one row per perceptual hash (the original did not)")
    parser.add_argument("--bands", default=",".join(str(b) for b in BANDS[:-1]),
                        help="comma-separated longest-side cut points; 'inf' is appended. "
                             "Narrower bands balance tighter and drop more rows.")
    parser.add_argument("--verify-against", type=Path,
                        help="compare the result with an existing balanced CSV")
    args = parser.parse_args()

    bands = sorted({int(b) for b in args.bands.split(",") if b.strip()} | {0}) + [INFINITY]

    with args.index.open() as handle:
        rows = list(csv.DictReader(handle))
    print(f"read {len(rows):,} rows from {args.index}")
    profile(rows, "BEFORE")

    if args.exclude_source:
        before = len(rows)
        rows = [r for r in rows if r["source"] not in args.exclude_source]
        print(f"\nexcluded {', '.join(args.exclude_source)}: {before - len(rows):,} rows dropped")
    if args.min_side:
        before = len(rows)
        rows = [r for r in rows if longest_side(r) >= args.min_side]
        print(f"min-side {args.min_side}px: {before - len(rows):,} rows dropped")
    if args.dedupe:
        before, seen, unique = len(rows), set(), []
        for row in rows:
            if row["phash"] not in seen:
                seen.add(row["phash"])
                unique.append(row)
        rows = unique
        print(f"dedupe: {before - len(rows):,} duplicate hashes dropped")

    kept, report = balance(rows, args.seed, bands)

    print(f"\n{'band (longest side)':<22}{'real':>10}{'ai':>10}{'kept each':>12}")
    for low, high, n_real, n_ai, take in report:
        label = f"{low}-{high if high < INFINITY else 'inf'}"
        print(f"{label:<22}{n_real:>10,}{n_ai:>10,}{take:>12,}")

    gap = profile(kept, "AFTER")
    cost = 100 * (1 - len(kept) / max(len(rows), 1))
    print(f"  cost: {cost:.1f}% of the filtered rows dropped")
    unique_hashes = len({r['phash'] for r in kept})
    print(f"  unique perceptual hashes: {unique_hashes:,} of {len(kept):,}")

    if args.verify_against and args.verify_against.exists():
        with args.verify_against.open() as handle:
            reference = {r["phash"] for r in csv.DictReader(handle)}
        mine = {r["phash"] for r in kept}
        overlap = len(mine & reference)
        print(f"\nVERIFY against {args.verify_against.name}: "
              f"{len(reference):,} reference hashes, {overlap:,} shared "
              f"({100 * overlap / max(len(reference), 1):.1f}%)")
        print("  (row counts per band are the exact check; hash overlap varies with the seed)")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(kept)
    print(f"\nwrote {len(kept):,} rows -> {args.output}")
    print("  by source: " + ", ".join(f"{k}:{v:,}" for k, v in
                                      Counter(r["source"] for r in kept).most_common()))


if __name__ == "__main__":
    main()
