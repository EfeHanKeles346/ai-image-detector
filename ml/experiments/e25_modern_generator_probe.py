# =============================================================================
# e25_modern_generator_probe.py — WHAT THIS EXPERIMENT DOES
# -----------------------------------------------------------------------------
# The deployable band (E22 + E23a) has never met a generator newer than
# Midjourney v6 (2024).  The SSD holds five never-used modern sources — Nano
# Banana (Gemini 2.5 Flash Image), Nano Banana Pro, FLUX.1-dev, GPT Image 4K,
# and the julienlucas mixed set with a real half — all 2025–26 era.  This
# probe asks the question the literature answers pessimistically (frozen
# detectors collapse to 18–30% on 2026 generators): does OUR band see them?
#
# PRE-REGISTERED HYPOTHESES
# -----------------------------------------------------------------------------
# H1 — recall varies strongly by generator family (B-Free trains on SD-family
#      reconstructions, so diffusion-family generators should be caught and
#      native multimodal ones (GPT Image, Nano Banana) are the risk).
# H2 — julienlucas' real half behaves as a fresh unseen pipeline: its FP rate
#      under the frozen worst-source threshold is the honest 12th-pipeline
#      test, before E24's personal photos.
#
# PER BLOCK
# -----------------------------------------------------------------------------
# SOURCES         where each sample comes from on the SSD, and its label
#                 contract.  julienlucas declares label_names and the script
#                 RAISES if the parquet metadata disagrees (the E19b guard).
# extract()       deterministic stride-sample of up to 200 images per source,
#                 written as RAW BYTES (no re-encode — compression history is
#                 part of what the detectors read) to PIXELPROOF_WORK_ROOT/e25_modern_probe.
# score()         both external detectors via the untouched E21 adapters,
#                 JSONL-cached per arm under artifacts/e25/.
# band()          the asymmetric band of E23a: t_ai = worst-source threshold
#                 over the eleven known pipelines' calibration halves; verdicts
#                 are "AI" or "insufficient evidence", never "real".
#
# Results: printed tables + artifacts/e25/results.json.  Usage (SSD mounted):
#   PYTHONPATH=src .venv/bin/python experiments/e25_modern_generator_probe.py
# =============================================================================

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from pixelproof.evaluation_protocol import (safe_auc, stable_calibration_split,
                                            threshold_at_fpr)
from pixelproof.project_paths import DATA_ROOT, WORK_ROOT

sys.path.insert(0, str(Path(__file__).parent))
from e21_external_detector_benchmark import (BFreeDetector,  # noqa: E402
                                             CommunityForensicsDetector,
                                             select_device)

SSD = DATA_ROOT
PROBE = WORK_ROOT / "e25_modern_probe"
PER_SOURCE = 200
SPLIT_SEED = 2026
CAL_FRACTION = 0.5
FP_BUDGET = 0.10

# label contract: parquet sources with a label column must declare the class
# names we expect at index 0/1; extraction RAISES on a mismatch (E19b).
SOURCES = {
    "julienlucas": {
        "dir": "julienlucas__midjourney-dalle-sd-nanobananapro-dataset/data",
        "label_names": ["fake", "real"],  # 0=fake — INVERTED vs project labels
    },
    "flux1_dev": {"dir": "ash12321__flux-1-dev-generated-10k/data"},
    "nano_banana_pro": {"dir": "kaupane__nano-banana-pro-gen/data"},
    "nano_banana": {"dir": "bitmind__nano-banana/data"},
    "gpt_image_4k": {"glob": "a3xrfgb__gpt-image-mega-4k/*.png"},
}
ARM_SCORES = {
    "cf_vit": "artifacts/e21/cf_vit_scores.jsonl",
    "bfree": "artifacts/e21/bfree_scores.jsonl",
}


def verify_label_names(parquet: pq.ParquetFile, expected: list[str], src: str) -> None:
    meta = (parquet.schema_arrow.metadata or {}).get(b"huggingface", b"{}")
    features = json.loads(meta.decode("utf8", "ignore") or "{}")
    names = (
        features.get("info", {}).get("features", {}).get("label", {}).get("names")
    )
    if names != expected:
        raise RuntimeError(
            f"{src}: parquet declares label names {names!r}, expected {expected!r} "
            "— the dataset may have been re-exported; refusing to guess (E19b)"
        )


def stride(items: list, count: int) -> list:
    if len(items) <= count:
        return items
    index = np.linspace(0, len(items) - 1, count).round().astype(int)
    return [items[i] for i in np.unique(index)]


def source_done(name: str, spec: dict) -> bool:
    """Per-source resume check — a killed extraction must not skip the rest."""
    folders = (
        [PROBE / f"{name}_ai", PROBE / f"{name}_real"]
        if "label_names" in spec
        else [PROBE / name]
    )
    return all(f.is_dir() and any(f.iterdir()) for f in folders)


def extract() -> None:
    if not SSD.is_dir():
        raise RuntimeError(f"SSD bağlı değil: {SSD}")
    for name, spec in SOURCES.items():
        if source_done(name, spec):
            print(f"{name}: mevcut, atlanıyor")
            continue
        if "glob" in spec:  # loose image files
            files = sorted(
                p for p in SSD.glob(spec["glob"]) if not p.name.startswith("._")
            )
            out = PROBE / name
            out.mkdir(parents=True, exist_ok=True)
            for path in stride(files, PER_SOURCE):
                (out / path.name).write_bytes(path.read_bytes())
            print(f"{name}: {min(len(files), PER_SOURCE)} görsel (dosyadan)")
            continue

        shards = sorted(
            p
            for p in (SSD / spec["dir"]).glob("*.parquet")
            if not p.name.startswith("._")
        )
        # Two passes so a 14 GB source never sits in memory: count rows per
        # shard from metadata, choose global stride indices, then read only
        # the needed batches shard by shard.
        counts = []
        for shard in shards:
            parquet = pq.ParquetFile(shard)
            if "label_names" in spec:
                verify_label_names(parquet, spec["label_names"], name)
            counts.append(parquet.metadata.num_rows)
        total = sum(counts)
        budget = PER_SOURCE * (2 if "label_names" in spec else 1)
        # oversample: labelled sets are filtered per class after reading
        wanted_global = set(
            np.linspace(0, total - 1, min(total, budget * 3)).round().astype(int)
        )
        written: dict[str, int] = defaultdict(int)
        offset = 0
        for shard, count in zip(shards, counts):
            wanted_local = [i - offset for i in wanted_global
                            if offset <= i < offset + count]
            offset += count
            if not wanted_local:
                continue
            parquet = pq.ParquetFile(shard)
            columns = [c for c in ("image", "label")
                       if c in parquet.schema_arrow.names]
            position = 0
            wanted_iter = sorted(wanted_local)
            for batch in parquet.iter_batches(batch_size=64, columns=columns):
                rows = batch.to_pylist()
                for local, row in enumerate(rows):
                    index = position + local
                    if wanted_iter and index == wanted_iter[0]:
                        wanted_iter.pop(0)
                        if "label_names" in spec:
                            class_name = spec["label_names"][row["label"]]
                            folder = f"{name}_{'real' if class_name == 'real' else 'ai'}"
                        else:
                            folder = name
                        if written[folder] >= PER_SOURCE:
                            continue
                        out = PROBE / folder
                        out.mkdir(parents=True, exist_ok=True)
                        image = row["image"]
                        suffix = Path(image.get("path") or "x.png").suffix or ".png"
                        (out / f"{written[folder]:04d}{suffix}").write_bytes(
                            image["bytes"]
                        )
                        written[folder] += 1
                position += len(rows)
                if not wanted_iter:
                    break
        for folder, count in sorted(written.items()):
            print(f"{folder}: {count} görsel (parquetten)")


def worst_source_threshold(scores_path: str) -> float:
    populations: dict[tuple[str, str], list[dict]] = defaultdict(list)
    with open(scores_path) as handle:
        for line in handle:
            record = json.loads(line)
            populations[(record["dataset"], record["source"])].append(record)
    cuts = []
    for (dataset, source), records in populations.items():
        if not (dataset == "forensics" or (dataset, source) == ("defactify", "real")):
            continue
        calibration, _ = stable_calibration_split(records, CAL_FRACTION, SPLIT_SEED)
        values = np.asarray([r["image_score"] for r in calibration])
        cuts.append(threshold_at_fpr(values, FP_BUDGET))
    return float(max(cut for cut in cuts if np.isfinite(cut)))


def score_folders(detector, raw_path: Path) -> dict[str, list[float]]:
    cache: dict[str, float] = {}
    if raw_path.is_file():
        with raw_path.open() as handle:
            for line in handle:
                record = json.loads(line)
                if record.get("detector_id") == detector.detector_id:
                    cache[record["path"]] = record["image_score"]
    results: dict[str, list[float]] = {}
    with raw_path.open("a") as raw:
        for folder in sorted(p for p in PROBE.iterdir() if p.is_dir()):
            scores: list[float] = []
            files = sorted(
                p for p in folder.iterdir()
                if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
            )
            for path in files:
                key = str(path)
                if key in cache:
                    scores.append(cache[key])
                    continue
                try:
                    value, width, height = detector.score(path)
                except Exception as error:
                    print(f"    ATLANDI {path.name}: {error}")
                    continue
                raw.write(json.dumps({
                    "detector_id": detector.detector_id, "path": key,
                    "source": folder.name, "width": width, "height": height,
                    "bytes_per_pixel": path.stat().st_size / max(width * height, 1),
                    "image_score": value,
                }, separators=(",", ":")) + "\n")
                raw.flush()
                scores.append(value)
            results[folder.name] = scores
            print(f"  {folder.name}: {len(scores)} skor")
    return results


def main() -> None:
    extract()
    device = select_device("auto")
    output: dict[str, dict] = {}
    for arm, cached in ARM_SCORES.items():
        threshold = worst_source_threshold(cached)
        print(f"\n===== {arm} · t_ai (worst-source) = {threshold:.3f} =====")
        if arm == "bfree":
            detector = BFreeDetector(
                Path("external/B-Free"), Path("external/B-Free/weights"),
                "BFREE_dino2reg4", device,
            )
        else:
            from huggingface_hub import snapshot_download

            # cache-only resolution: works offline, never downloads
            local = Path(snapshot_download(
                "buildborderless/CommunityForensics-DeepfakeDet-ViT",
                local_files_only=True,
            ))
            detector = CommunityForensicsDetector(local, device, allow_download=False)
        raw_path = Path(f"artifacts/e25/{arm}_scores.jsonl")
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        folder_scores = score_folders(detector, raw_path)

        real = np.asarray(folder_scores.get("julienlucas_real", []))
        table: dict[str, dict] = {}
        for source, values in folder_scores.items():
            scores = np.asarray(values)
            row = {
                "n": int(len(scores)),
                "median_score": float(np.median(scores)) if len(scores) else None,
                "ai_verdict": float((scores >= threshold).mean()) if len(scores) else None,
            }
            if source != "julienlucas_real" and len(real) and len(scores):
                row["auc_vs_julienlucas_real"] = safe_auc(real, scores)
            table[source] = row
        output[arm] = {"t_ai": threshold, "sources": table}

        print(f"{'kaynak':22s} {'n':>4s} {'medyan':>8s} {'AI kararı':>10s} {'AUC':>6s}")
        for source, row in sorted(table.items()):
            auc = row.get("auc_vs_julienlucas_real")
            print(
                f"{source:22s} {row['n']:4d} {row['median_score']:8.3f} "
                f"{100 * row['ai_verdict']:9.1f}% {auc:6.3f}" if auc is not None else
                f"{source:22s} {row['n']:4d} {row['median_score']:8.3f} "
                f"{100 * row['ai_verdict']:9.1f}%      —"
            )

    results = Path("artifacts/e25/results.json")
    results.write_text(json.dumps(output, indent=2))
    print(f"\nResults -> {results}")


if __name__ == "__main__":
    main()
