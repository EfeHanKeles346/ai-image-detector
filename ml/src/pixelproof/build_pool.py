# =============================================================================
# build_pool.py — WHAT THIS FILE DOES
# -----------------------------------------------------------------------------
# Builds ONE index over every training image we hold, so a training run can read
# a single manifest instead of knowing about five dataset layouts. Nothing is
# copied: 100 GB stays where it is and the index records where to find each
# image, what its label is, and which source it came from.
#
# It also does the three checks that must happen BEFORE any training:
#   1. contamination — are any of our test images sitting in the training pool?
#   2. audit         — does the MERGED pool carry a shortcut, even if each
#                      source was clean on its own?
#   3. balance       — how lopsided are the classes?
#
# WHY THIS EXISTS
# -----------------------------------------------------------------------------
# archive1 was used as this project's benchmark through six experiments before
# anyone looked inside it (HISTORY 1b). The lesson generalises: look first.
# Two sources that are individually clean can still combine into a biased pool —
# if one contributes 1024px reals and the other 512px reals, the merged real
# class has a distribution matching neither.
#
# The contamination check matters here specifically: CommunityForensics lists
# `coco` among its real sources, and Defactify's real half IS MS-COCO. Training
# on our own test images would inflate every number we report.
#
# CODE BLOCKS IN THIS FILE
# -----------------------------------------------------------------------------
# SOURCES        One entry per dataset: where it lives, how to read it, and
#                which models may use it. `whole_image_safe=False` marks a set
#                with a shape or resolution shortcut — usable by the CNN (which
#                resizes everything away, measured in E10) and by tile training
#                (fixed tile size), but NOT by the native-resolution statistics
#                model, which can see it.
#
# phash()        64-bit perceptual hash. Survives re-encoding and mild resizing,
#                so it catches a test image that reached the pool through a
#                different pipeline — which an md5 would miss.
#
# iter_parquet() / iter_folder()
#                Streaming readers. Both skip macOS AppleDouble stubs (`._name`),
#                which ExFAT writes beside every real file and which cost us one
#                wrong dataset verdict already (HISTORY 2b.8).
#
# build()        Walks every source, records one row per image, and hashes as it
#                goes so contamination is detected during the pass rather than
#                in a second one.
#
# main()         Writes pool_index.csv, prints the audit, and reports overlap.
# =============================================================================

import argparse
import csv
import io
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from pixelproof.project_paths import DATA_ROOT, WORK_ROOT

Image.MAX_IMAGE_PIXELS = None

HOME = WORK_ROOT
SSD = DATA_ROOT
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".JPEG"}

# LABEL DIRECTION IS NOT A CONVENTION — IT IS A PER-DATASET FACT
# ---------------------------------------------------------------------------
# This project uses 0 = real, 1 = AI everywhere. Datasets do not. Two of the
# five sources below declare the opposite order in their own HuggingFace
# metadata, and until 2026-08-05 this file took int(row[label_col]) raw — so
# 47% of pool_index.csv carried inverted labels, and every experiment trained
# on the pool (E12, E14, E15, E16) learned from a half-poisoned target.
#
#   theminji/AI-vs-Real-balanced   names = ["AiArtData", "RealArt"]  -> 0 is AI
#   theminji/ai-vs-real-200k       names = ["ai", "real"]            -> 0 is AI
#   TheKernel01/AIGC-Detection...  names = ["real", "fake"]          -> 0 is real
#   OwensLab/CommunityForensics    no metadata; resolved from model_name —
#                                  label 0 is 100% "FFHQ" (a real photograph
#                                  set), label 1 carries diffusion model ids
#
# `label_map` translates raw -> project. `label_names` records what we expect
# to find in the file, and verify_labels() fails loudly if a dataset is
# re-exported with a different order: a silent flip is far worse than a crash.
#
# whole_image_safe=False -> has a shape/resolution shortcut a native-resolution
# model could exploit. Still fine for the CNN and for tile training.
SOURCES = {
    "communityforensics": {
        "path": SSD / "OwensLab__CommunityForensics-Small",
        "kind": "parquet", "image_col": "image_data", "label_col": "label",
        # label 0 is 100% model_name="FFHQ" (real photographs); label 1 carries
        # diffusion model ids. Same order as ours.
        "label_map": {0: 0, 1: 1}, "label_names": None,
        # Marked unsafe 2026-08-05. Measured: class 0 is ENTIRELY 1024x1024 and
        # class 1 ENTIRELY 512x512 — p10 = median = p90 in both, so the two size
        # distributions are disjoint constants. A single threshold on image size
        # separates the classes perfectly, which is archive1's AUC 1.000 again.
        # The original audit passed it because the ratio is exactly 2.0 and the
        # rule tested for >2.5; both are fixed in tools/audit_datasets.py.
        "extra_col": "model_name", "whole_image_safe": False,
    },
    "ai_vs_real_balanced": {
        "path": SSD / "theminji__AI-vs-Real-balanced",
        "kind": "parquet", "image_col": "image", "label_col": "label",
        "label_map": {0: 1, 1: 0},                      # INVERTED
        "label_names": ["AiArtData", "RealArt"],
        "extra_col": None, "whole_image_safe": True,
    },
    "genimage": {
        "path": HOME / "genimage_split/train",
        "kind": "folder", "folders": {"REAL": 0, "FAKE": 1},
        "label_map": {0: 0, 1: 1}, "label_names": None,   # folders already carry our order
        "whole_image_safe": True,
    },
    "aigc_benchmark": {
        "path": SSD / "TheKernel01__AIGC-Detection-Benchmark",
        "kind": "parquet", "image_col": "image", "label_col": "label",
        "label_map": {0: 0, 1: 1}, "label_names": ["real", "fake"],
        "extra_col": "generator", "whole_image_safe": False,   # AI 100% square
    },
    "ai_vs_real_200k": {
        "path": SSD / "theminji__ai-vs-real-200k",
        "kind": "parquet", "image_col": "image", "label_col": "label",
        "label_map": {0: 1, 1: 0},                      # INVERTED
        "label_names": ["ai", "real"],
        "extra_col": None, "whole_image_safe": False,          # AI 1024px, real 263px
    },
}

# Never trained on. Any of these appearing in the pool is a contamination bug.
HELD_OUT = {
    "defactify_real": HOME / "defactify_test/real",
    "defactify_ai": HOME / "defactify_test/ai",
}


def real_files(folder: Path, pattern: str = "*"):
    return [p for p in folder.rglob(pattern)
            if p.is_file() and not p.name.startswith("._") and ".cache" not in p.parts]


def phash(image: Image.Image) -> str:
    """64-bit difference hash — stable across re-encoding and mild resizing."""
    grey = np.asarray(image.convert("L").resize((9, 8), Image.LANCZOS), dtype=np.float32)
    bits = (grey[:, 1:] > grey[:, :-1]).ravel()
    return f"{int(''.join('1' if b else '0' for b in bits), 2):016x}"


def to_project_label(spec: dict, raw: int) -> int:
    """Translate a dataset's own label into ours (0 = real, 1 = AI).

    Raises rather than guessing. A source with no declared map is a source
    nobody has checked, and an unchecked map is how 47% of the pool ended up
    inverted for four experiments.
    """
    mapping = spec.get("label_map")
    if not mapping:
        raise ValueError("source has no label_map — declare it in SOURCES before indexing")
    if raw not in mapping:
        raise ValueError(f"unmapped raw label {raw!r}; SOURCES declares {sorted(mapping)}")
    return mapping[raw]


def verify_labels(name: str, spec: dict) -> str:
    """Check the file's own ClassLabel names against what SOURCES expects.

    HuggingFace stores the class order in the parquet schema metadata. If a
    dataset is re-exported with the classes swapped, our map silently inverts
    with it — so compare, and fail loudly on a mismatch. Returns a one-line
    status for the log.
    """
    expected = spec.get("label_names")
    if spec["kind"] != "parquet" or expected is None:
        direction = "inverted" if spec["label_map"][0] == 1 else "same order"
        return f"{name}: no ClassLabel metadata, map asserted by inspection ({direction})"

    import json
    import pyarrow.parquet as pq

    shards = sorted(real_files(spec["path"], "*.parquet"))
    if not shards:
        return f"{name}: NOT FOUND, cannot verify"
    metadata = (pq.ParquetFile(shards[0]).schema_arrow.metadata or {}).get(b"huggingface")
    if not metadata:
        return f"{name}: parquet carries no HuggingFace metadata, map unverified"
    found = json.loads(metadata)["info"]["features"][spec["label_col"]].get("names")
    if found != expected:
        raise ValueError(
            f"{name}: label order changed upstream — SOURCES expects {expected}, "
            f"file declares {found}. Fix label_map before indexing anything.")
    direction = "INVERTED" if spec["label_map"][0] == 1 else "same order"
    return f"{name}: {found} -> {direction}, verified"


def describe(image: Image.Image, byte_size: int) -> dict:
    width, height = image.size
    return {"w": width, "h": height, "fmt": image.format or "?",
            "square": int(width == height), "bpp": round(byte_size / max(width * height, 1), 4)}


def iter_parquet(spec: dict, limit: int | None):
    import pyarrow.parquet as pq

    shards = sorted(real_files(spec["path"], "*.parquet"))
    if not shards:
        return
    per_shard = None if limit is None else max(1, limit // len(shards))
    columns = [spec["image_col"], spec["label_col"]] + ([spec["extra_col"]] if spec["extra_col"] else [])
    for shard in shards:
        taken = 0
        try:
            parquet = pq.ParquetFile(shard)
        except Exception:
            continue
        for batch in parquet.iter_batches(batch_size=64, columns=columns):
            for row in batch.to_pylist():
                blob = row[spec["image_col"]]
                raw = blob.get("bytes") if isinstance(blob, dict) else blob
                if not isinstance(raw, (bytes, bytearray)):
                    continue
                try:
                    image = Image.open(io.BytesIO(raw))
                    image.load()
                except Exception:
                    continue
                extra = str(row.get(spec["extra_col"], "")) if spec["extra_col"] else ""
                yield image, len(raw), int(row[spec["label_col"]]), f"{shard.name}", extra
                taken += 1
                if per_shard and taken >= per_shard:
                    break
            if per_shard and taken >= per_shard:
                break


def iter_folder(spec: dict, limit: int | None):
    for name, label in spec["folders"].items():
        files = [p for p in real_files(spec["path"] / name) if p.suffix in IMAGE_EXT]
        step = 1 if limit is None else max(1, len(files) // max(limit // len(spec["folders"]), 1))
        for path in files[::step]:
            try:
                with Image.open(path) as image:
                    image.load()
                    yield image.copy(), path.stat().st_size, label, str(path), ""
            except Exception:
                continue


def _one_unit(job):
    """Index one shard (or one class folder). Runs in a worker process.

    Parallelised per unit rather than per image: opening and hashing hundreds of
    thousands of images is the whole cost here, and each shard is independent.
    """
    name, spec, unit, per_unit, held_out = job
    local_spec = dict(spec)
    rows, overlaps = [], []
    if spec["kind"] == "parquet":
        local_spec["path"] = unit.parent
        stream = _iter_one_parquet(local_spec, unit, per_unit)
    else:
        stream = _iter_one_folder(local_spec, unit, per_unit)
    for image, byte_size, raw_label, location, extra in stream:
        label = to_project_label(spec, raw_label)   # dataset order -> ours (0=real, 1=AI)
        digest = phash(image)
        if digest in held_out:
            overlaps.append((name, location, held_out[digest]))
            continue                          # never let a test image into the pool
        rows.append({"source": name, "label": label, "location": location,
                     "generator": extra, "phash": digest,
                     "whole_image_safe": int(spec["whole_image_safe"]),
                     **describe(image, byte_size)})
    return rows, overlaps


def _iter_one_parquet(spec, shard: Path, limit: int | None):
    import pyarrow.parquet as pq

    columns = [spec["image_col"], spec["label_col"]] + ([spec["extra_col"]] if spec["extra_col"] else [])
    try:
        parquet = pq.ParquetFile(shard)
    except Exception:
        return
    taken = 0
    for batch in parquet.iter_batches(batch_size=64, columns=columns):
        for row in batch.to_pylist():
            blob = row[spec["image_col"]]
            raw = blob.get("bytes") if isinstance(blob, dict) else blob
            if not isinstance(raw, (bytes, bytearray)):
                continue
            try:
                image = Image.open(io.BytesIO(raw))
                image.load()
            except Exception:
                continue
            extra = str(row.get(spec["extra_col"], "")) if spec["extra_col"] else ""
            yield image, len(raw), int(row[spec["label_col"]]), shard.name, extra
            taken += 1
            if limit and taken >= limit:
                return


def _iter_one_folder(spec, folder: Path, limit: int | None):
    label = spec["folders"][folder.name]
    files = [p for p in real_files(folder) if p.suffix in IMAGE_EXT]
    step = 1 if not limit else max(1, len(files) // limit)
    for path in files[::step][:limit]:
        try:
            with Image.open(path) as image:
                image.load()
                yield image.copy(), path.stat().st_size, label, str(path), ""
        except Exception:
            continue


def build(limit_per_source: int | None, held_out_hashes: dict[str, str]):
    from concurrent.futures import ProcessPoolExecutor

    print("verifying label direction (0 = real, 1 = AI)…")
    for name, spec in SOURCES.items():
        print(f"  {verify_labels(name, spec)}")
    print()

    jobs = []
    for name, spec in SOURCES.items():
        if spec["kind"] == "parquet":
            units = sorted(real_files(spec["path"], "*.parquet"))
        else:
            units = [spec["path"] / folder for folder in spec["folders"]]
        if not units:
            print(f"  {name}: NOT FOUND at {spec['path']}")
            continue
        per_unit = None if limit_per_source is None else max(1, limit_per_source // len(units))
        jobs += [(name, spec, unit, per_unit, held_out_hashes) for unit in units]

    rows, overlaps, counts = [], [], Counter()
    done = 0
    with ProcessPoolExecutor() as pool:
        for unit_rows, unit_overlaps in pool.map(_one_unit, jobs, chunksize=1):
            rows += unit_rows
            overlaps += unit_overlaps
            counts[unit_rows[0]["source"]] += len(unit_rows) if unit_rows else 0
            done += 1
            print(f"\r  {done}/{len(jobs)} units · {len(rows):,} images", end="", flush=True)
    print()
    for name, count in counts.most_common():
        print(f"  {name:<24}{count:>10,}")

    hashes = defaultdict(int)
    for r in rows:
        hashes[r["phash"]] += 1
    return rows, hashes, overlaps


def audit(rows: list[dict]) -> None:
    by_label = defaultdict(list)
    for r in rows:
        by_label[r["label"]].append(r)
    print(f"\n{'class':<8}{'n':>9}{'formats':>28}{'square%':>9}{'median side':>13}{'bpp':>8}")
    print("-" * 76)
    stats = {}
    for label, group in sorted(by_label.items()):
        formats = Counter(r["fmt"] for r in group)
        side = int(np.median([max(r["w"], r["h"]) for r in group]))
        square = round(100 * np.mean([r["square"] for r in group]), 1)
        bpp = round(float(np.mean([r["bpp"] for r in group])), 3)
        stats[label] = {"square": square, "side": side, "bpp": bpp,
                        "n": len(group), "fmt": set(formats)}
        top = ", ".join(f"{k}:{v}" for k, v in formats.most_common(3))
        print(f"{label:<8}{len(group):>9,}{top:>28}{square:>9}{side:>13}{bpp:>8}")

    print("\nMERGED-POOL SHORTCUT CHECK")
    problems = []
    if len(stats) == 2:
        a, b = stats[0], stats[1]
        if all(len(s["fmt"]) == 1 for s in (a, b)) and a["fmt"] != b["fmt"]:
            problems.append(f"FORMAT: {a['fmt']} vs {b['fmt']}")
        if abs(a["square"] - b["square"]) > 60:
            problems.append(f"SHAPE: {a['square']}% vs {b['square']}% square")
        if max(a["side"], b["side"]) / max(min(a["side"], b["side"]), 1) > 2.5:
            problems.append(f"RESOLUTION: median side {a['side']} vs {b['side']}")
        if max(a["bpp"], b["bpp"]) / max(min(a["bpp"], b["bpp"]), 1e-3) > 3:
            problems.append(f"COMPRESSION: {a['bpp']} vs {b['bpp']} bytes/pixel")
        ratio = max(a["n"], b["n"]) / max(min(a["n"], b["n"]), 1)
        if ratio > 1.5:
            problems.append(f"IMBALANCE: {ratio:.1f}:1")
    for p in problems:
        print(f"  !! {p}")
    if not problems:
        print("  no threshold exceeded")
    metadata_probe(rows)


def metadata_probe(rows: list[dict]) -> float:
    """Can a model pick the class from metadata alone, without seeing a pixel?

    The five threshold checks above compare medians and ratios, which is a weak
    instrument: on 2026-08-05 they reported "none detected" for a pool this probe
    separated at AUC 0.956. Medians can coincide while the distributions differ,
    and a boosted tree finds that difference immediately. archive1 scored 1.000
    here, so this is the same test that defines the whole problem (HISTORY 1b).

    Reported, never enforced — a flaw is a usage condition (1b). Size, aspect and
    squareness cannot reach a 128px tile; compression can, so read that column
    separately when building a pool for tile training.
    """
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split

    labels = np.array([int(r["label"]) for r in rows])
    if len(set(labels.tolist())) < 2:
        return float("nan")
    width = np.array([int(r["w"]) for r in rows], dtype=np.float64)
    height = np.array([int(r["h"]) for r in rows], dtype=np.float64)
    longest = np.maximum(width, height)
    aspect = longest / np.maximum(np.minimum(width, height), 1)
    bpp = np.array([float(r["bpp"]) for r in rows])
    square = np.array([int(r["square"]) for r in rows], dtype=np.float64)

    def auc(columns: np.ndarray) -> float:
        x_tr, x_te, y_tr, y_te = train_test_split(
            columns, labels, test_size=0.3, random_state=42, stratify=labels)
        model = HistGradientBoostingClassifier(random_state=42).fit(x_tr, y_tr)
        return float(roc_auc_score(y_te, model.predict_proba(x_te)[:, 1]))

    everything = auc(np.c_[width, height, aspect, bpp, square])
    compression = auc(np.c_[bpp])
    print("\nMETADATA PROBE — class from metadata alone, no pixels "
          "(archive1 scored 1.000 here)")
    print(f"  all metadata      {everything:.3f}"
          f"   {'<-- exploitable by a whole-image model' if everything > 0.75 else ''}")
    print(f"  compression only  {compression:.3f}"
          f"   {'<-- SURVIVES INTO A TILE, fix this one' if compression > 0.6 else ''}")
    return everything


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit-per-source", type=int, default=None,
                        help="cap images per source (omit for everything)")
    parser.add_argument("--output", type=Path, default=Path("artifacts/pool_index.csv"))
    args = parser.parse_args()

    print("hashing held-out test images…")
    held_out: dict[str, str] = {}
    for name, folder in HELD_OUT.items():
        files = [p for p in real_files(folder) if p.suffix.lower() in IMAGE_EXT]
        for path in files:
            try:
                with Image.open(path) as image:
                    held_out[phash(image)] = f"{name}/{path.name}"
            except Exception:
                continue
    print(f"  {len(held_out):,} unique hashes across {len(HELD_OUT)} held-out sets\n")

    print("indexing sources (nothing is copied)…")
    rows, hashes, overlaps = build(args.limit_per_source, held_out)

    print(f"\nCONTAMINATION: {len(overlaps)} pool images matched a held-out test image")
    for source, location, victim in overlaps[:10]:
        print(f"  {source}: {location} == {victim}")
    if overlaps:
        print("  (excluded from the index)")

    duplicates = sum(v - 1 for v in hashes.values() if v > 1)
    print(f"INTERNAL DUPLICATES: {duplicates:,} of {len(rows):,} rows share a hash with another")

    audit(rows)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {len(rows):,} rows -> {args.output}")

    safe = sum(1 for r in rows if r["whole_image_safe"])
    print(f"  usable by the statistics model (native resolution): {safe:,}")
    print(f"  usable by the CNN and by tile training:             {len(rows):,}")


if __name__ == "__main__":
    main()
