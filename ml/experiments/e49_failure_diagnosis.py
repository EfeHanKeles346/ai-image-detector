"""Reproduce model-blind E49 failure diagnostics without selecting a successor."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from experiments.e49_evaluation import BINARY_THRESHOLD
from pixelproof.project_paths import DATA_ROOT, ML_ROOT


ROOT = DATA_ROOT / "e49" / "final"
MANIFEST = ROOT / "manifest_unscored.json"
SCORES = ROOT / "generalist_scores.jsonl"
FINAL_REPORT = ROOT / "final_report.json"
REPORT = ROOT / "failure_diagnosis.json"
EVIDENCE = ML_ROOT.parent / "evidence" / "e49_failure_diagnosis.json"

MANIFEST_SHA256 = "9744a9d2385ef2f105b7a132bfee76a7099d280ac9d075d81220f572429c5909"
SCORES_SHA256 = "249f005c83acbc65ac987c9fdeea91a9a3c7dfc4a388fdd4e20a436bc53610a8"
FINAL_REPORT_SHA256 = "10fc0649a68d31c57f815949e3c8d52f2d2bdfaeaadc9422c0ae83c32d525573"
RESOLUTION_BINS = ((0, 4), (4, 8), (8, 12), (12, 20), (20, 100))


def _write(path: Path, value: Any) -> bytes:
    raw = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_bytes(raw)
    temporary.replace(path)
    return raw


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def diagnose_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    originals = [row for row in rows if int(row["label"]) == 0
                 and row["condition"] == "publisher_original"]
    formats = {}
    for image_format in sorted({str(row["format"]) for row in originals}):
        selected = [row for row in originals if row["format"] == image_format]
        formats[image_format] = {
            "rows": len(selected),
            "false_ai_rate": float(np.mean([float(row["score"]) >= BINARY_THRESHOLD for row in selected])),
            "median_score": float(np.median([float(row["score"]) for row in selected])),
        }
    resolution = []
    megapixels = np.asarray([int(row["width"]) * int(row["height"]) / 1e6 for row in originals])
    real_scores = np.asarray([float(row["score"]) for row in originals])
    for low, high in RESOLUTION_BINS:
        selected = (megapixels >= low) & (megapixels < high)
        selected_scores = real_scores[selected]
        resolution.append({
            "megapixels_gte": low, "megapixels_lt": high, "rows": int(selected.sum()),
            "false_ai_rate": (
                float(np.mean(selected_scores >= BINARY_THRESHOLD))
                if len(selected_scores) else None
            ),
            "median_score": float(np.median(selected_scores)) if len(selected_scores) else None,
        })
    by_parent: dict[str, dict[str, Mapping[str, Any]]] = {}
    for row in rows:
        by_parent.setdefault(str(row["parent_id"]), {})[str(row["condition"])] = row
    paired = {}
    for label in (0, 1):
        pairs = [pair for pair in by_parent.values()
                 if int(pair["publisher_original"]["label"]) == label]
        original = np.asarray([float(pair["publisher_original"]["score"]) for pair in pairs])
        q75 = np.asarray([float(pair["social_q75"]["score"]) for pair in pairs])
        paired[str(label)] = {
            "parents": len(pairs),
            "both_at_or_above_binary_cut": float(np.mean(
                (original >= BINARY_THRESHOLD) & (q75 >= BINARY_THRESHOLD)
            )),
            "either_at_or_above_binary_cut": float(np.mean(
                (original >= BINARY_THRESHOLD) | (q75 >= BINARY_THRESHOLD)
            )),
            "q75_minus_original_score_mean": float(np.mean(q75 - original)),
            "q75_minus_original_score_median": float(np.median(q75 - original)),
        }
    log_megapixels = np.log1p(megapixels)
    correlation = None
    if (len(originals) >= 2 and np.std(log_megapixels) > 0
            and np.std(real_scores) > 0):
        correlation = float(np.corrcoef(log_megapixels, real_scores)[0, 1])
    return {
        "real_original_format": formats,
        "real_original_resolution_bins": resolution,
        "real_original_log_megapixel_score_correlation": correlation,
        "paired_existing_cut": paired,
        "counts": {"rows": len(rows), "parents": len(by_parent),
                   "real_originals": len(originals),
                   "format_rows": dict(Counter(str(row["format"]) for row in originals))},
    }


def diagnose() -> dict[str, Any]:
    if REPORT.exists() or EVIDENCE.exists():
        raise FileExistsError("E49 failure diagnosis already exists")
    for path, expected in ((MANIFEST, MANIFEST_SHA256), (SCORES, SCORES_SHA256),
                           (FINAL_REPORT, FINAL_REPORT_SHA256)):
        if _sha(path) != expected:
            raise ValueError(f"E49 consumed-final evidence changed: {path}")
    manifest = json.loads(MANIFEST.read_text())
    scores = [json.loads(line) for line in SCORES.read_text().splitlines() if line]
    metadata = {str(row["record_id"]): row for row in manifest["rows"]}
    if len(scores) != 4_000 or set(metadata) != {str(row["record_id"]) for row in scores}:
        raise ValueError("E49 diagnosis identity coverage changed")
    rows = [{**metadata[str(score["record_id"])], **score} for score in scores]
    final = json.loads(FINAL_REPORT.read_text())
    result = diagnose_rows(rows)
    result.update({
        "schema_version": 1, "state": "e49_consumed_failure_diagnosis_only",
        "source_hashes": {"manifest": MANIFEST_SHA256, "scores": SCORES_SHA256,
                          "final_report": FINAL_REPORT_SHA256},
        "tpr_at_fpr_0_10": {
            condition: float(final["conditions"][condition]["binary_metrics"]["tpr_at_fpr"]["tpr"])
            for condition in ("publisher_original", "social_q75")
        },
        "candidate_thresholds_created": 0, "new_model_scores_created": 0,
        "boundary": "Post-final diagnosis only; cannot select E51 data, threshold, model or feature.",
    })
    raw = _write(REPORT, result)
    evidence = {**result, "report_bytes": len(raw), "report_sha256": hashlib.sha256(raw).hexdigest()}
    _write(EVIDENCE, evidence)
    return evidence


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("diagnose",))
    parser.parse_args(argv)
    print(json.dumps(diagnose(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
