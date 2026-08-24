# =============================================================================
# prepare_manipulation.py — WHAT THIS FILE DOES
# -----------------------------------------------------------------------------
# Unpacks the 78 GB manipulation compilation on the SSD into the folder layout
# E17/E18 read. This is Module 2's dataset preparation, the counterpart of
# prepare_defactify.py for Module 1.
#
# WHY THIS EXISTS
# -----------------------------------------------------------------------------
# E17 and E18 were run against /tmp/m2, extracted by hand and never scripted.
# macOS wipes /tmp on reboot, so the only copy of Module 2's data was one
# restart away from gone, and neither experiment could be reproduced from the
# repo. Both problems are the same problem: an unscripted step.
#
# WHAT THE COMPILATION LOOKS LIKE
# -----------------------------------------------------------------------------
# 191 tars named <SubDataset>-<split>-<NNNN>.tar, split being `auth` or `manip`.
# Paths INSIDE a tar already read <SubDataset>/<split>/<file>, so extracting
# preserves the layout and no renaming is needed.
#
#   manip, 4 files per image        auth, 2 files per image
#     <name>.png       the image      <name>.png   the image
#     <name>.mask.png  0/255 mask     <name>.cls   class label
#     <name>.json      {"manip_label": 1, "auth": "CocoGlide/auth/000.png"}
#     <name>.cls       class label
#
# Two details that matter downstream:
#   - <name> keeps the ORIGINAL extension (Tp_D_CND_...tif.png). E18 splits
#     CASIA by it to show that ELA works on .tif originals (0.578) and fails on
#     .jpg ones (0.338), so this must not be normalised away.
#   - the .json `auth` pointer gives before/after pairs of the same scene.
#
# WHICH SUB-DATASETS, AND WHY NOT ALL OF THEM
# -----------------------------------------------------------------------------
# DEFAULT_DATASETS takes every sub-dataset except OpenForensics. OpenForensics
# is 138 of the 191 tars, it is face manipulation rather than splicing or
# inpainting, and nothing in Module 2 currently asks a question about faces.
# Pass --datasets OpenForensics_test to pull it anyway.
#
# One tar per split is the default (--tars-per-split). It is ~500 images, which
# is far more than the 35-39 that survive E17's mask-coverage filter, so more
# tars buy nothing until that filter is loosened.
#
# ⚠️ Report results PER SUB-DATASET, never pooled (DATASETS.md). The tile model
# asks "does this region look generated", which is the right question for a
# diffusion-inpainted region and the wrong one for a Photoshop splice. E17
# measured exactly that split: CocoGlide 0.648, CASIA image-level 0.481. A
# single average would hide it.
#
# CODE BLOCKS IN THIS FILE
# -----------------------------------------------------------------------------
# available()    Reads the tar filenames and returns {dataset: {split: [tars]}}.
#                Derived from what is on disk rather than from summary.json, so
#                it cannot disagree with reality.
# safe_members() Yields tar members that are plain files inside the expected
#                sub-tree. Blocks absolute paths and `..` traversal (a tar can
#                name any path it likes), and skips the macOS AppleDouble stubs
#                ExFAT writes beside every file (HISTORY 2b.8).
# extract()      Unpacks the selected tars, skipping work already done.
# validate()     The point of the whole file: counts images and masks, and
#                reports every manip image whose mask is missing. A missing mask
#                silently drops an image from E17 rather than raising, which is
#                the failure mode this project keeps cataloguing.
# =============================================================================

import argparse
import tarfile
from collections import defaultdict
from pathlib import Path

from pixelproof.project_paths import DATA_ROOT, WORK_ROOT

SSD = DATA_ROOT
SOURCE = SSD / "ductai199x__image-manipulation-dataset-compilation"
OUTPUT = WORK_ROOT / "manipulation_test"

# Everything except OpenForensics — see the header for why.
SKIP_PREFIX = "OpenForensics"
SPLITS = ("manip", "auth")


def available(source: Path) -> dict[str, dict[str, list[Path]]]:
    """{dataset: {split: [tar, ...]}} from the tar filenames on disk."""
    found: dict[str, dict[str, list[Path]]] = defaultdict(lambda: defaultdict(list))
    for path in sorted(source.glob("*.tar")):
        if path.name.startswith("._"):
            continue
        stem = path.stem.rsplit("-", 1)[0]          # drop the NNNN index
        if "-" not in stem:
            continue
        dataset, split = stem.rsplit("-", 1)
        if split in SPLITS:
            found[dataset][split].append(path)
    return {k: dict(v) for k, v in found.items()}


def safe_members(archive: tarfile.TarFile, dataset: str, split: str):
    """Plain files, re-addressed to <dataset>/<split>/<file>.

    The tar's own root folder cannot be trusted to match its filename — three of
    the thirteen sub-datasets disagree, and one of them nests two levels deep:

        NIST2016-*.tar            -> MFCD_NIST2016/manip/...
        RealisticTampering-*.tar  -> Realistic-Tampering/manip/...
        VIPP_Realistic-*.tar      -> VIPP/Realistic/manip/...

    So the split segment is located inside the path and everything after it is
    kept, which normalises all thirteen to one layout. An earlier version keyed
    on the tar filename and silently extracted ZERO files for those three — it
    reported "0 images" rather than failing, which is the quiet-wrong-answer
    failure mode this project keeps cataloguing (HISTORY 2b.8).
    """
    for member in archive:
        if not member.isfile():
            continue
        name = member.name.lstrip("./")
        parts = Path(name).parts
        if not parts or ".." in parts or name.startswith("/"):
            continue                                   # path traversal
        if any(part.startswith("._") for part in parts):
            continue                                   # macOS AppleDouble stub
        if split not in parts:
            continue
        tail = parts[parts.index(split) + 1:]
        if len(tail) != 1:                             # expect exactly one filename
            continue
        member.name = f"{dataset}/{split}/{tail[0]}"
        yield member


def extract(source: Path, output: Path, datasets: list[str], per_split: int, force: bool) -> None:
    catalogue = available(source)
    for dataset in datasets:
        if dataset not in catalogue:
            print(f"  {dataset}: not in the compilation — skipped")
            continue
        for split in SPLITS:
            tars = catalogue[dataset].get(split, [])[:per_split]
            if not tars:
                print(f"  {dataset}/{split}: no tar")
                continue
            target = output / dataset / split
            if target.exists() and any(target.iterdir()) and not force:
                print(f"  {dataset}/{split}: already extracted "
                      f"({sum(1 for _ in target.iterdir()):,} files) — skipped")
                continue
            target.mkdir(parents=True, exist_ok=True)
            written = 0
            for tar_path in tars:
                with tarfile.open(tar_path) as archive:
                    for member in safe_members(archive, dataset, split):
                        # filter="data" is the hardened extractor (Python's default from 3.14):
                        # it strips ownership/permission metadata and refuses anything outside
                        # the destination, on top of the checks in safe_members().
                        archive.extract(member, output, filter="data")
                        written += 1
                        if written % 500 == 0:
                            print(f"\r  {dataset}/{split}: {written:,} files", end="", flush=True)
            note = "  ⚠️ EMPTY — check the tar's internal layout" if not written else ""
            print(f"\r  {dataset}/{split}: {written:,} files from {len(tars)} tar(s){note}      ")


def validate(output: Path) -> None:
    """Count images and masks, and name the manip images missing a mask."""
    print(f"\n{'sub-dataset':<22}{'split':<8}{'images':>9}{'masks':>8}{'no mask':>10}")
    print("-" * 57)
    total_missing = 0
    for dataset in sorted(d for d in output.iterdir() if d.is_dir()):
        for split in SPLITS:
            folder = dataset / split
            if not folder.exists():
                continue
            images = sorted(p for p in folder.glob("*.png") if not p.name.endswith(".mask.png"))
            masks = {p.name for p in folder.glob("*.mask.png")}
            missing = [p for p in images
                       if split == "manip" and p.with_suffix(".mask.png").name not in masks]
            total_missing += len(missing)
            flag = f"{len(missing):>10,}" if missing else f"{'-':>10}"
            print(f"{dataset.name:<22}{split:<8}{len(images):>9,}{len(masks):>8,}{flag}")
            for path in missing[:3]:
                print(f"    missing mask: {path.name}")
    if total_missing:
        print(f"\n⚠️  {total_missing:,} manipulated images have no mask. E17 skips these "
              f"silently, so they would shrink the sample without saying so.")
    else:
        print("\nevery manipulated image has a mask.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Unpack the manipulation compilation for Module 2.")
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--datasets", nargs="*", default=None,
                        help="sub-datasets to extract; default is everything except OpenForensics")
    parser.add_argument("--tars-per-split", type=int, default=1)
    parser.add_argument("--force", action="store_true", help="re-extract even if files exist")
    parser.add_argument("--list", action="store_true", help="show what the compilation holds and exit")
    args = parser.parse_args()

    if not args.source.exists():
        raise SystemExit(f"{args.source} not found — is the SSD mounted?")

    catalogue = available(args.source)
    if args.list:
        print(f"{'sub-dataset':<24}{'manip tars':>12}{'auth tars':>11}")
        for dataset, splits in sorted(catalogue.items()):
            print(f"{dataset:<24}{len(splits.get('manip', [])):>12}{len(splits.get('auth', [])):>11}")
        return

    datasets = args.datasets or [d for d in sorted(catalogue) if not d.startswith(SKIP_PREFIX)]
    print(f"extracting {len(datasets)} sub-datasets, {args.tars_per_split} tar(s) per split "
          f"-> {args.output}")
    extract(args.source, args.output, datasets, args.tars_per_split, args.force)
    validate(args.output)


if __name__ == "__main__":
    main()
