# =============================================================================
# e23c_compression_column.py — WHAT THIS EXPERIMENT DOES
# -----------------------------------------------------------------------------
# Every number in this project is measured on images as they sit on disk, and
# E22's H1 showed the calibration domain (Defactify, 0.16 B/px) and the
# transfer domain (forensic sources, 1.1–1.9 B/px) already sit on opposite
# ends of the compression axis.  Meanwhile every image on the internet is
# recompressed in transit.  This experiment — the debt E12 named in 2026-07 —
# measures the whole decision layer under the literature's social-media
# degradation standard: resize to 75%, save as JPEG q50.
#
# PRE-REGISTERED HYPOTHESES
# -----------------------------------------------------------------------------
# H1 — degradation hurts AI recall more than authentic FP: generation traces
#      are high-frequency (E7), authentic "naturalness" is not, so the band
#      should FAIL SAFE (miss more fakes) rather than fail dangerous (accuse
#      more reals).
# H2 — thresholds are compression-domain-specific (E12's gap at the decision
#      layer): the clean-fitted t_ai misbehaves on degraded images, and
#      refitting on degraded calibration halves recovers most of the loss.
#
# PIPELINE NOTE.  The degraded copy is: resize 75% (LANCZOS) → cap the long
# side at 2048px (the E23b input policy, now part of the serving contract) →
# JPEG quality 50.  Halves are inherited from the ORIGINAL paths, so the
# clean and degraded columns compare the same images half by half.
#
# Results: printed tables + artifacts/e23/e23c_results.json.  Usage:
#   PYTHONPATH=src .venv/bin/python experiments/e23c_compression_column.py
# =============================================================================

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from pixelproof.evaluation_protocol import (safe_auc, stable_calibration_split,
                                            threshold_at_fpr)
from pixelproof.project_paths import WORK_ROOT

sys.path.insert(0, str(Path(__file__).parent))
from e21_external_detector_benchmark import (BFreeDetector,  # noqa: E402
                                             CommunityForensicsDetector,
                                             select_device)

WORK = WORK_ROOT / "e23c_degraded"
RESIZE = 0.75
JPEG_QUALITY = 50
CAP_PX = 2048
SPLIT_SEED = 2026
CAL_FRACTION = 0.5
FP_BUDGET = 0.10
GENERATORS = ("dalle3", "midjourney", "sd21", "sd3", "sdxl")
ARM_SCORES = {
    "cf_vit": "artifacts/e21/cf_vit_scores.jsonl",
    "bfree": "artifacts/e21/bfree_scores.jsonl",
}


def read_populations(path: str) -> dict[tuple[str, str], list[dict]]:
    populations: dict[tuple[str, str], list[dict]] = defaultdict(list)
    with open(path) as handle:
        for line in handle:
            record = json.loads(line)
            populations[(record["dataset"], record["source"])].append(record)
    return populations


def degraded_path(record: dict) -> Path:
    stem = Path(record["path"]).stem
    return WORK / f"{record['dataset']}__{record['source']}" / f"{stem}.jpg"


def build_degraded(populations: dict) -> None:
    total, built = 0, 0
    for records in populations.values():
        for record in records:
            total += 1
            target = degraded_path(record)
            if target.is_file():
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with Image.open(record["path"]) as image:
                image = image.convert("RGB")
                width, height = image.size
                width, height = round(width * RESIZE), round(height * RESIZE)
                long_side = max(width, height)
                if long_side > CAP_PX:  # E23b input policy
                    scale = CAP_PX / long_side
                    width, height = round(width * scale), round(height * scale)
                image = image.resize((width, height), Image.LANCZOS)
                image.save(target, format="JPEG", quality=JPEG_QUALITY)
            built += 1
    print(f"degraded kopyalar: {built} yeni / {total} toplam")


def worst_source_threshold(
    populations: dict, score_of, exclude: str | None = None
) -> float:
    cuts = []
    for (dataset, source), records in populations.items():
        if not (dataset == "forensics" or (dataset, source) == ("defactify", "real")):
            continue
        if source == exclude:
            continue
        calibration, _ = stable_calibration_split(records, CAL_FRACTION, SPLIT_SEED)
        values = np.asarray(
            [score_of(r) for r in calibration if score_of(r) is not None],
            dtype=np.float64,
        )
        if len(values):
            cuts.append(threshold_at_fpr(values, FP_BUDGET))
    return float(max(cut for cut in cuts if np.isfinite(cut)))


def main() -> None:
    device = select_device("auto")
    payload: dict[str, dict] = {}
    reference = read_populations(ARM_SCORES["bfree"])
    build_degraded(reference)

    for arm, cached in ARM_SCORES.items():
        populations = read_populations(cached)
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

        # score all degraded copies (JSONL-cached, resumable)
        raw_path = Path(f"artifacts/e23/e23c_{arm}_scores.jsonl")
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        cache: dict[str, float] = {}
        if raw_path.is_file():
            with raw_path.open() as handle:
                for line in handle:
                    record = json.loads(line)
                    if record.get("detector_id") == detector.detector_id:
                        cache[record["path"]] = record["image_score"]
        scored = 0
        with raw_path.open("a") as raw:
            for (dataset, source), records in sorted(populations.items()):
                for record in records:
                    target = degraded_path(record)
                    key = str(target)
                    if key in cache or not target.is_file():
                        continue
                    try:
                        value, width, height = detector.score(target)
                    except Exception as error:
                        print(f"    ATLANDI {target.name}: {error}")
                        continue
                    raw.write(json.dumps({
                        "detector_id": detector.detector_id, "path": key,
                        "orig_path": record["path"], "dataset": dataset,
                        "source": source, "label": record["label"],
                        "width": width, "height": height,
                        "image_score": value,
                    }, separators=(",", ":")) + "\n")
                    scored += 1
                    if scored % 200 == 0:
                        raw.flush()
                        print(f"  {arm}: {scored} yeni skor")
        print(f"  {arm}: skorlama tamam ({scored} yeni)")

        degraded_scores = {}
        with raw_path.open() as handle:
            for line in handle:
                record = json.loads(line)
                if record.get("detector_id") == detector.detector_id:
                    degraded_scores[record.get("orig_path", record["path"])] = record[
                        "image_score"
                    ]

        def score_clean(record: dict) -> float:
            return record["image_score"]

        def score_degraded(record: dict) -> float | None:
            return degraded_scores.get(record["path"])

        def band_metrics(t_ai: float, score_of) -> dict:
            real_fp = {}
            for (dataset, source), records in populations.items():
                if not (dataset == "forensics"
                        or (dataset, source) == ("defactify", "real")):
                    continue
                _, evaluation = stable_calibration_split(
                    records, CAL_FRACTION, SPLIT_SEED
                )
                values = np.asarray(
                    [score_of(r) for r in evaluation if score_of(r) is not None]
                )
                real_fp[source] = float((values >= t_ai).mean()) if len(values) else None
            recalls = {}
            for generator in GENERATORS:
                _, evaluation = stable_calibration_split(
                    populations[("defactify", generator)], CAL_FRACTION, SPLIT_SEED
                )
                values = np.asarray(
                    [score_of(r) for r in evaluation if score_of(r) is not None]
                )
                recalls[generator] = (
                    float((values >= t_ai).mean()) if len(values) else None
                )
            fps = [v for v in real_fp.values() if v is not None]
            return {
                "t_ai": t_ai,
                "macro_fp": float(np.mean(fps)),
                "worst_fp": float(np.max(fps)),
                "worst_source": max(real_fp, key=lambda s: real_fp[s] or -1),
                "macro_recall": float(
                    np.mean([v for v in recalls.values() if v is not None])
                ),
                "per_generator_recall": recalls,
                "per_source_fp": real_fp,
            }

        clean_t = worst_source_threshold(populations, score_clean)
        frozen = band_metrics(clean_t, score_degraded)
        refit_t = worst_source_threshold(populations, score_degraded)
        refit = band_metrics(refit_t, score_degraded)
        clean_reference = band_metrics(clean_t, score_clean)

        # ranking under degradation: pooled generators vs defactify real
        def pooled_auc(score_of) -> float:
            real = [score_of(r) for r in populations[("defactify", "real")]]
            ai = [
                score_of(r)
                for g in GENERATORS
                for r in populations[("defactify", g)]
            ]
            real = [v for v in real if v is not None]
            ai = [v for v in ai if v is not None]
            return safe_auc(np.asarray(real), np.asarray(ai))

        payload[arm] = {
            "clean": clean_reference,
            "degraded_frozen_threshold": frozen,
            "degraded_refit_threshold": refit,
            "defactify_auc_clean": pooled_auc(score_clean),
            "defactify_auc_degraded": pooled_auc(score_degraded),
        }

        print("=" * 96)
        print(f"E23c · {arm} — resize 75% + cap {CAP_PX} + JPEG q{JPEG_QUALITY}")
        print("=" * 96)
        rows = [
            ("temiz (referans)", clean_reference),
            ("bozulmuş · donmuş eşik", frozen),
            ("bozulmuş · yeniden fit", refit),
        ]
        print(f"{'senaryo':26s} {'worst FP':>9s} {'macro FP':>9s} {'recall':>8s}  worst kaynak")
        for label, values in rows:
            print(
                f"{label:26s} {100 * values['worst_fp']:8.1f}% "
                f"{100 * values['macro_fp']:8.1f}% {100 * values['macro_recall']:7.1f}%"
                f"  {values['worst_source']}"
            )
        print(
            f"Defactify AUC: temiz {payload[arm]['defactify_auc_clean']:.3f} -> "
            f"bozulmuş {payload[arm]['defactify_auc_degraded']:.3f}\n"
        )

    output = Path("artifacts/e23/e23c_results.json")
    output.write_text(json.dumps(payload, indent=2))
    print(f"Results -> {output}")


if __name__ == "__main__":
    main()
