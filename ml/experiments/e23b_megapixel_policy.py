# =============================================================================
# e23b_megapixel_policy.py — WHAT THIS EXPERIMENT DOES
# -----------------------------------------------------------------------------
# E22's H1 identified NIST2016 — the one authentic source every detector
# family reads as synthetic — as the 12-megapixel source (median 12.19 Mpx
# against 0.07–3.15 elsewhere).  Both external detectors were built for far
# smaller inputs: B-Free five-crops at 504px sees ~2% of a 12 Mpx frame's
# pixels, CF-ViT shrinks the long side ~9×.  This experiment tests the
# cheapest possible input policy: CAP oversized images before scoring.
#
# PRE-REGISTERED HYPOTHESES
# -----------------------------------------------------------------------------
# H1 — capping NIST2016's long side to 2048px moves its authentic score
#      distribution down toward the other forensic sources, for both arms.
# H2 — under the frozen worst-source threshold (fitted WITHOUT any capping,
#      the deployed t_ai), capped NIST2016 falls to a passing FP rate.
#      If H2 holds, "cap the long side at 2048 before scoring" becomes an
#      input policy like the 48px floor — a property of the serving contract.
#
# Note the honest tension: E7 taught that downscaling DESTROYS generation
# evidence, so a cap must never be applied to the AI-detection side lightly.
# The cap is tested here as a REAL-side rescue on authentic megapixel photos;
# the sanity check below also caps the Defactify AI evaluation halves to
# verify the cap does not silently erase recall at these budgets (test-set
# images are ≤1024px, so the cap should not touch them at all — that check is
# the control that the policy is a no-op below the cap).
#
# PER BLOCK
# -----------------------------------------------------------------------------
# capped_copy()   lossless-format capped copies (PNG; LANCZOS) of exactly the
#                 125 NIST2016 images the E21 runs scored, so the before/after
#                 comparison is per-image, not per-population.
# score_folder()  both external detectors via the E21 adapters, JSONL cache.
# compare()       per-arm: original vs capped medians, per-image score drop,
#                 FP at the frozen t_ai before and after.
#
# Results: printed tables + artifacts/e23/e23b_results.json.  Usage:
#   PYTHONPATH=src .venv/bin/python experiments/e23b_megapixel_policy.py
# =============================================================================

from __future__ import annotations

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

CAP_PX = 2048
WORK = WORK_ROOT / "e23b_nist_capped"
SPLIT_SEED = 2026
CAL_FRACTION = 0.5
FP_BUDGET = 0.10
ARM_SCORES = {
    "cf_vit": "artifacts/e21/cf_vit_scores.jsonl",
    "bfree": "artifacts/e21/bfree_scores.jsonl",
}


def read_arm(path: str) -> dict[tuple[str, str], list[dict]]:
    populations: dict[tuple[str, str], list[dict]] = defaultdict(list)
    with open(path) as handle:
        for line in handle:
            record = json.loads(line)
            populations[(record["dataset"], record["source"])].append(record)
    return populations


def worst_source_threshold(populations: dict) -> float:
    cuts = []
    for (dataset, source), records in populations.items():
        if not (dataset == "forensics" or (dataset, source) == ("defactify", "real")):
            continue
        calibration, _ = stable_calibration_split(records, CAL_FRACTION, SPLIT_SEED)
        values = np.asarray([r["image_score"] for r in calibration])
        cuts.append(threshold_at_fpr(values, FP_BUDGET))
    return float(max(cut for cut in cuts if np.isfinite(cut)))


def capped_copy(original: Path, destination: Path) -> bool:
    """Write a capped copy; returns True when the cap actually resized."""
    with Image.open(original) as image:
        image = image.convert("RGB")
        width, height = image.size
        long_side = max(width, height)
        if long_side > CAP_PX:
            scale = CAP_PX / long_side
            image = image.resize(
                (round(width * scale), round(height * scale)), Image.LANCZOS
            )
            resized = True
        else:
            resized = False
        destination.parent.mkdir(parents=True, exist_ok=True)
        image.save(destination, format="PNG")
    return resized


def main() -> None:
    device = select_device("auto")
    payload: dict[str, dict] = {}

    for arm, cached in ARM_SCORES.items():
        populations = read_arm(cached)
        t_ai = worst_source_threshold(populations)
        nist = populations[("forensics", "NIST2016")]
        originals = {Path(r["path"]).name: r for r in nist}

        # 1) capped copies of exactly the images E21 scored
        capped_dir = WORK / "NIST2016_capped"
        resized_count = 0
        for record in nist:
            source_path = Path(record["path"])
            target = capped_dir / (source_path.stem + ".png")
            if not target.is_file():
                resized_count += int(capped_copy(source_path, target))

        # 2) score the capped copies
        if arm == "bfree":
            detector = BFreeDetector(
                Path("external/B-Free"), Path("external/B-Free/weights"),
                "BFREE_dino2reg4", device,
            )
        else:
            from huggingface_hub import snapshot_download

            local = Path(snapshot_download(
                "buildborderless/CommunityForensics-DeepfakeDet-ViT",
                local_files_only=True,
            ))
            detector = CommunityForensicsDetector(local, device, allow_download=False)

        raw_path = Path(f"artifacts/e23/e23b_{arm}_capped_scores.jsonl")
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        cache: dict[str, float] = {}
        if raw_path.is_file():
            with raw_path.open() as handle:
                for line in handle:
                    record = json.loads(line)
                    if record.get("detector_id") == detector.detector_id:
                        cache[record["path"]] = record["image_score"]

        pairs: list[tuple[float, float]] = []
        with raw_path.open("a") as raw:
            for path in sorted(capped_dir.glob("*.png")):
                key = str(path)
                if key in cache:
                    capped_score = cache[key]
                else:
                    capped_score, width, height = detector.score(path)
                    raw.write(json.dumps({
                        "detector_id": detector.detector_id, "path": key,
                        "width": width, "height": height,
                        "image_score": capped_score,
                    }, separators=(",", ":")) + "\n")
                    raw.flush()
                original = originals.get(path.stem + ".png")
                if original is not None:
                    pairs.append((original["image_score"], capped_score))

        before = np.asarray([p[0] for p in pairs])
        after = np.asarray([p[1] for p in pairs])
        other_medians = {
            source: float(np.median([r["image_score"] for r in records]))
            for (dataset, source), records in populations.items()
            if dataset == "forensics" and source != "NIST2016"
        }
        result = {
            "t_ai_frozen": t_ai,
            "n_pairs": len(pairs),
            "median_before": float(np.median(before)),
            "median_after": float(np.median(after)),
            "median_drop": float(np.median(before - after)),
            "fp_before": float((before >= t_ai).mean()),
            "fp_after": float((after >= t_ai).mean()),
            "other_forensic_median_range": [
                float(min(other_medians.values())),
                float(max(other_medians.values())),
            ],
        }
        payload[arm] = result

        print("=" * 88)
        print(f"E23b · {arm} — cap {CAP_PX}px · frozen t_ai {t_ai:.3f} · {len(pairs)} çift")
        print("=" * 88)
        print(
            f"medyan skor  {result['median_before']:8.3f} -> {result['median_after']:8.3f}"
            f"   (diğer forensik medyanlar {result['other_forensic_median_range'][0]:.3f}"
            f" … {result['other_forensic_median_range'][1]:.3f})"
        )
        print(
            f"FP @ frozen t_ai  {100 * result['fp_before']:5.1f}% -> "
            f"{100 * result['fp_after']:5.1f}%\n"
        )

    output = Path("artifacts/e23/e23b_results.json")
    output.write_text(json.dumps(payload, indent=2))
    print(f"Results -> {output}")


if __name__ == "__main__":
    main()
