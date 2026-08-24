# =============================================================================
# e24_library_promise.py — WHAT THIS EXPERIMENT DOES
# -----------------------------------------------------------------------------
# E22's product promise: a new real pipeline needs ~100 calibration images and
# a threshold refit — no retraining.  E25 tested that promise on a downloaded
# 2026 set; this experiment tests it on the most deployment-realistic pipeline
# available: 207 camera-original photographs from the project owner's iPhone
# (203 × iPhone 15 Pro, 4 × iPhone 16e, EXIF-verified; median long side
# 4032px — a genuine 12-megapixel pipeline, exactly the class that poisoned
# NIST2016).  The photos never enter the repo; only scores are kept.
#
# PRE-REGISTERED HYPOTHESES
# -----------------------------------------------------------------------------
# H1 (unseen) — under the FROZEN worst-source thresholds the iPhone pipeline
#      stays within the 10% budget for CF; for B-Free the uncapped FP is at
#      risk (12 Mpx, the E23b failure mode) and the 2048 cap contains it.
# H2 (library) — adding the pipeline's calibration half to the 11-source
#      library and refitting keeps its evaluation-half FP within budget at
#      little or no recall cost — the promise, measured end to end.
#
# Device note: runs on CPU by default so the concurrent E20 three-seed
# training keeps the GPU to itself.
#
# Results: printed tables + artifacts/e24/results.json.  Usage:
#   PYTHONPATH=src .venv/bin/python experiments/e24_library_promise.py
# =============================================================================

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from pixelproof.evaluation_protocol import (stable_calibration_split,
                                            threshold_at_fpr)
from pixelproof.project_paths import WORK_ROOT

sys.path.insert(0, str(Path(__file__).parent))
from e21_external_detector_benchmark import (BFreeDetector,  # noqa: E402
                                             CommunityForensicsDetector,
                                             select_device)

PHOTOS = WORK_ROOT / "iphone-gallery"
CAPPED = WORK_ROOT / "e24_iphone_capped"
CAP_PX = 2048
SPLIT_SEED = 2026
CAL_FRACTION = 0.5
FP_BUDGET = 0.10
GENERATORS = ("dalle3", "midjourney", "sd21", "sd3", "sdxl")
ARM_SCORES = {
    "cf_vit": "artifacts/e21/cf_vit_scores.jsonl",
    "bfree": "artifacts/e21/bfree_scores.jsonl",
}


def camera_originals() -> list[Path]:
    """EXIF-bearing JPEGs only — screenshots and downloads are excluded."""
    files = []
    for path in sorted(PHOTOS.iterdir()):
        if path.suffix.lower() not in {".jpg", ".jpeg"}:
            continue
        with Image.open(path) as image:
            if image.getexif().get(272):  # camera model tag
                files.append(path)
    return files


def capped_copy(original: Path) -> Path:
    target = CAPPED / (original.stem + ".png")
    if target.is_file():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(original) as image:
        image = image.convert("RGB")
        width, height = image.size
        long_side = max(width, height)
        if long_side > CAP_PX:
            scale = CAP_PX / long_side
            image = image.resize(
                (round(width * scale), round(height * scale)), Image.LANCZOS
            )
        image.save(target, format="PNG")
    return target


def read_populations(path: str) -> dict[tuple[str, str], list[dict]]:
    populations: dict[tuple[str, str], list[dict]] = defaultdict(list)
    with open(path) as handle:
        for line in handle:
            record = json.loads(line)
            populations[(record["dataset"], record["source"])].append(record)
    return populations


def worst_source_threshold(populations, extra_calibration=None) -> float:
    cuts = []
    for (dataset, source), records in populations.items():
        if not (dataset == "forensics" or (dataset, source) == ("defactify", "real")):
            continue
        calibration, _ = stable_calibration_split(records, CAL_FRACTION, SPLIT_SEED)
        values = np.asarray([r["image_score"] for r in calibration])
        cuts.append(threshold_at_fpr(values, FP_BUDGET))
    if extra_calibration is not None and len(extra_calibration):
        cuts.append(threshold_at_fpr(np.asarray(extra_calibration), FP_BUDGET))
    return float(max(cut for cut in cuts if np.isfinite(cut)))


def macro_recall(populations, t_ai: float) -> float:
    rates = []
    for generator in GENERATORS:
        _, evaluation = stable_calibration_split(
            populations[("defactify", generator)], CAL_FRACTION, SPLIT_SEED
        )
        values = np.asarray([r["image_score"] for r in evaluation])
        rates.append(float((values >= t_ai).mean()))
    return float(np.mean(rates))


def score_files(detector, files: list[Path], raw_path: Path, variant: str) -> list[dict]:
    cache: dict[str, float] = {}
    if raw_path.is_file():
        with raw_path.open() as handle:
            for line in handle:
                record = json.loads(line)
                if (record.get("detector_id") == detector.detector_id
                        and record.get("variant") == variant):
                    cache[record["path"]] = record["image_score"]
    records = []
    with raw_path.open("a") as raw:
        for index, path in enumerate(files, 1):
            key = str(path)
            if key in cache:
                records.append({"path": key, "image_score": cache[key]})
                continue
            try:
                value, width, height = detector.score(path)
            except Exception as error:
                print(f"    ATLANDI {path.name}: {error}")
                continue
            record = {
                "detector_id": detector.detector_id, "variant": variant,
                "path": key, "width": width, "height": height,
                "image_score": value,
            }
            raw.write(json.dumps(record, separators=(",", ":")) + "\n")
            raw.flush()
            records.append(record)
            if index % 25 == 0:
                print(f"    {variant}: {index}/{len(files)}")
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu",
                        help="cpu by default: the GPU belongs to the E20 run tonight")
    args = parser.parse_args()
    device = select_device(args.device)

    files = camera_originals()
    print(f"E24 · {len(files)} kamera-orijinali iPhone fotoğrafı (EXIF doğrulamalı)")
    capped_files = [capped_copy(p) for p in files]
    print(f"     {CAP_PX}px sınırlı kopyalar hazır\n")

    raw_path = Path("artifacts/e24/scores.jsonl")
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, dict] = {}

    for arm, cached in ARM_SCORES.items():
        populations = read_populations(cached)
        frozen_t = worst_source_threshold(populations)
        if arm == "bfree":
            detector = BFreeDetector(
                Path("external/B-Free"), Path("external/B-Free/weights"),
                "BFREE_dino2reg4", device,
            )
            variants = {"uncapped": files, "capped": capped_files}
        else:
            from huggingface_hub import snapshot_download

            local = Path(snapshot_download(
                "buildborderless/CommunityForensics-DeepfakeDet-ViT",
                local_files_only=True,
            ))
            detector = CommunityForensicsDetector(local, device, allow_download=False)
            variants = {"native": files}  # CF preprocessing already shrinks

        arm_out: dict[str, dict] = {"frozen_t_ai": frozen_t}
        for variant, variant_files in variants.items():
            print(f"  {arm} · {variant} skorlanıyor ({device})")
            records = score_files(detector, variant_files, raw_path, f"{arm}:{variant}")
            scores = np.asarray([r["image_score"] for r in records])

            # H1 — unseen pipeline at the frozen threshold
            fp_unseen = float((scores >= frozen_t).mean())

            # H2 — add the calibration half to the library, refit, evaluate
            keyed = [{"path": r["path"], "image_score": r["image_score"]}
                     for r in records]
            calibration, evaluation = stable_calibration_split(
                keyed, CAL_FRACTION, SPLIT_SEED
            )
            refit_t = worst_source_threshold(
                populations,
                extra_calibration=[r["image_score"] for r in calibration],
            )
            ev_scores = np.asarray([r["image_score"] for r in evaluation])
            fp_refit = float((ev_scores >= refit_t).mean())
            arm_out[variant] = {
                "n": len(scores),
                "median_score": float(np.median(scores)),
                "fp_frozen": fp_unseen,
                "refit_t_ai": refit_t,
                "fp_refit_evaluation": fp_refit,
                "recall_frozen": macro_recall(populations, frozen_t),
                "recall_refit": macro_recall(populations, refit_t),
            }
            print(
                f"    H1 dondurulmuş eşik: FP {100 * fp_unseen:.1f}%  ·  "
                f"H2 kütüphane+refit: FP {100 * fp_refit:.1f}%  ·  "
                f"recall {100 * arm_out[variant]['recall_frozen']:.1f}% -> "
                f"{100 * arm_out[variant]['recall_refit']:.1f}%"
            )
        payload[arm] = arm_out

    output = Path("artifacts/e24/results.json")
    output.write_text(json.dumps(payload, indent=2))
    print(f"\nResults -> {output}")


if __name__ == "__main__":
    main()
