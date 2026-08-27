"""Post-hoc DDA threshold feasibility diagnostic; never selects a deployable cut."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from pixelproof.project_paths import DATA_ROOT, ML_ROOT


E35_ROOT = DATA_ROOT / "e35_dda_model"
SCORES = E35_ROOT / "development_scores.jsonl"
REPORT = E35_ROOT / "development_report.json"
EVIDENCE = ML_ROOT.parent / "evidence" / "e35_dda_threshold_diagnostic.json"
SNAPSHOTS = (0.5, 0.75, 0.8, 0.85, 0.9, 0.95)


def _rate(rows: Sequence[dict[str, Any]], threshold: float) -> float:
    return float(np.mean([float(row["score"]) >= threshold for row in rows]))


def threshold_summary(rows: Sequence[dict[str, Any]], threshold: float) -> dict[str, Any]:
    rr = [row for row in rows if row["population"] == "rr"]
    rr_real = [row for row in rr if row["label"] == 0]
    rr_ai = [row for row in rr if row["label"] == 1]
    owner = [row for row in rows if row["population"] == "owner"]
    ipn = [row for row in rows if row["population"] == "ipn"]
    devices = sorted({str(row["source"]) for row in ipn})
    device_fp = {
        device: _rate([row for row in ipn if row["source"] == device], threshold)
        for device in devices
    }
    return {
        "threshold": threshold,
        "rr_real_fp": _rate(rr_real, threshold),
        "rr_ai_recall": _rate(rr_ai, threshold),
        "owner_fp": _rate(owner, threshold),
        "ipn_macro_device_fp": float(np.mean(list(device_fp.values()))),
        "ipn_worst_device_fp": max(device_fp.values()),
    }


def run() -> dict[str, Any]:
    if EVIDENCE.exists():
        raise FileExistsError("DDA threshold diagnostic already exists; no silent rerun")
    source_bytes = SCORES.read_bytes()
    report = json.loads(REPORT.read_text())
    if hashlib.sha256(source_bytes).hexdigest() != report["scores_sha256"]:
        raise ValueError("DDA score stream no longer matches the frozen DEVELOPMENT report")
    rows = [json.loads(line) for line in source_bytes.splitlines()]
    frontiers = [threshold_summary(rows, threshold) for threshold in SNAPSHOTS]
    candidates = []
    for threshold in sorted({float(row["score"]) for row in rows}):
        summary = threshold_summary(rows, threshold)
        if (
            summary["rr_real_fp"] <= 0.20
            and summary["owner_fp"] <= 0.20
            and summary["ipn_worst_device_fp"] <= 0.20
        ):
            candidates.append(summary)
    result = {
        "schema_version": 1,
        "experiment": "E35/official-DDA-posthoc-threshold-diagnostic",
        "source_scores_sha256": report["scores_sha256"],
        "source_state": report["state"],
        "fixed_frontiers": frontiers,
        "first_all_real_safe_score_boundary": candidates[0] if candidates else None,
        "interpretation": (
            "A conservative region exists on consumed DEVELOPMENT, but every displayed cut is "
            "ineligible for deployment because RR/IPN/owner outcomes were inspected. E36 must "
            "re-estimate a threshold on new CAL and validate it once on a new LOCKED FINAL set."
        ),
    }
    EVIDENCE.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main(argv: Iterable[str] | None = None) -> int:
    del argv
    print(json.dumps(run(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
