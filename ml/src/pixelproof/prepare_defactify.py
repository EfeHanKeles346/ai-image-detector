# =============================================================================
# prepare_defactify.py — WHAT THIS FILE DOES
# -----------------------------------------------------------------------------
# Unpacks the Defactify / MS-COCOAI dataset (downloaded from HuggingFace as
# .parquet shards) into plain image folders that evaluate.py can read.
#
# Why this dataset: every generator in it (SD 2.1, SDXL, SD 3, DALL-E 3,
# Midjourney v6) is NEWER than anything our models were trained on
# (SD 1.5, ADM, BigGAN, GLIDE, VQDM, Wukong, old Midjourney). So it answers
# "what happens when a generator our detector has never seen ships?"
#
# CODE BLOCKS IN THIS FILE
# -----------------------------------------------------------------------------
# imports          pyarrow.parquet = reads the .parquet shards in batches
#                  (streaming, so we never hold 16k images in RAM at once).
#
# GENERATORS       Label_B integer -> folder name. Straight from the dataset
#                  card: 0=real, 1=SD2.1, 2=SDXL, 3=SD3, 4=DALL-E 3,
#                  5=Midjourney v6.
#
# extract()        The heart of the file. Walks every shard batch by batch and
#                  writes each image to disk. Crucially it writes the RAW JPEG
#                  BYTES straight from the parquet blob — it does NOT decode
#                  with PIL and re-save. Re-encoding would rewrite the
#                  compression history, which is exactly the forensic signal
#                  we are studying. A per-generator cap (--per-generator) lets
#                  you build a class-balanced subset.
#
# main()           CLI: point --source at the folder holding the .parquet
#                  shards, --output at where the image folders should go.
#                  Prints a per-generator count table at the end.
#
# OUTPUT LAYOUT (matches FolderPairDataset in evaluate.py, which recurses)
# -----------------------------------------------------------------------------
#   <output>/real/                  <- label 0
#   <output>/ai/{sd21,sdxl,sd3,dalle3,midjourney}/   <- label 1
#
#   whole set:      --external-ai <output>/ai        --external-real <output>/real
#   one generator:  --external-ai <output>/ai/dalle3 --external-real <output>/real
# =============================================================================

import argparse
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq

from pixelproof.project_paths import WORK_ROOT

GENERATORS = {0: "real", 1: "sd21", 2: "sdxl", 3: "sd3", 4: "dalle3", 5: "midjourney"}
BATCH_SIZE = 64


def target_dir(output: Path, label_b: int) -> Path:
    name = GENERATORS[label_b]
    return output / "real" if label_b == 0 else output / "ai" / name


def extract(shards: list[Path], output: Path, per_generator: int | None) -> Counter:
    for label_b in GENERATORS:
        target_dir(output, label_b).mkdir(parents=True, exist_ok=True)

    written: Counter = Counter()
    for shard in shards:
        parquet = pq.ParquetFile(shard)
        for batch in parquet.iter_batches(batch_size=BATCH_SIZE, columns=["Image", "Label_A", "Label_B"]):
            for row in batch.to_pylist():
                label_b = row["Label_B"]
                if per_generator is not None and written[label_b] >= per_generator:
                    continue
                blob = row["Image"]
                # Raw bytes only — decoding and re-saving would destroy the
                # original compression history we want to measure.
                raw = blob["bytes"] if isinstance(blob, dict) else blob
                name = GENERATORS[label_b]
                path = target_dir(output, label_b) / f"{name}_{written[label_b]:05d}.jpg"
                path.write_bytes(raw)
                written[label_b] += 1
        print(f"  {shard.name} done — running total {sum(written.values())}")
    return written


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=WORK_ROOT / "defactify/data")
    parser.add_argument("--output", type=Path, default=WORK_ROOT / "defactify_test")
    parser.add_argument("--per-generator", type=int, default=None,
                        help="Cap images per class (use it to build a balanced set)")
    args = parser.parse_args()

    shards = sorted(args.source.glob("test-*.parquet"))
    if not shards:
        raise SystemExit(f"No test-*.parquet shards found in {args.source}")
    print(f"{len(shards)} shard(s) -> {args.output}")

    written = extract(shards, args.output, args.per_generator)
    print("\n=== images written ===")
    for label_b, name in GENERATORS.items():
        print(f"  {name:12s} {written[label_b]:6d}")
    real = written[0]
    ai = sum(count for label_b, count in written.items() if label_b != 0)
    print(f"  {'-' * 20}\n  real {real} / ai {ai}  (total {real + ai})")


if __name__ == "__main__":
    main()
