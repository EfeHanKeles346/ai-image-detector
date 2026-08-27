"""One-shot DEVELOPMENT gate for the frozen official DDA candidate."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from pixelproof.benchmark_metrics import evaluate_binary_scores
from pixelproof.dda_candidate import CHECKPOINT_SHA256, OfficialDDACandidate, THRESHOLD
from pixelproof.e32_candidate import image_paths, sha256_file
from pixelproof.project_paths import DATA_ROOT, ML_ROOT


E35_ROOT = DATA_ROOT / "e35_dda_model"
IPN_AUDIT = DATA_ROOT / "e32" / "r1b_ipn_realization.json"
RR_MANIFEST = DATA_ROOT / "e33_rrdataset" / "r1c_cal_manifest.json"
EXPECTED_IPN_SHA = "f5827dcea69484421b15a6608555a7c64556d206c991d02606222b1892bc243b"
EXPECTED_RR_MANIFEST_SHA = "5d575a08a214002b962a128f8684de9ef161f85ecaf06a53b8dbfb7fede8b521"
EXPECTED_OWNER_IDENTITY = "390e3c210ee61d70252d7e4714b8640463f44d57760942d25a1bdf7eab5aac09"
SCORES = E35_ROOT / "development_scores.jsonl"
REPORT = E35_ROOT / "development_report.json"
EVIDENCE = ML_ROOT.parent / "evidence" / "e35_dda_development.json"


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def _write_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_bytes(_json_bytes(value))
    temporary.replace(path)


def _summary(values: Sequence[float]) -> dict[str, float]:
    scores = np.asarray(values, dtype=np.float64)
    return {
        "min": float(scores.min()),
        "p25": float(np.quantile(scores, 0.25)),
        "median": float(np.median(scores)),
        "p75": float(np.quantile(scores, 0.75)),
        "p95": float(np.quantile(scores, 0.95)),
        "max": float(scores.max()),
    }


def development_gate(
    *,
    rr_auc: float,
    rr_real_fp: float,
    rr_ai_macro: float,
    rr_ai_worst: float,
    ipn_worst_fp: float,
    owner_fp: float,
) -> dict[str, Any]:
    checks = {
        "rr_auc_gte_0.85": rr_auc >= 0.85,
        "rr_real_fp_lte_0.20": rr_real_fp <= 0.20,
        "rr_ai_macro_gte_0.80": rr_ai_macro >= 0.80,
        "rr_ai_worst_gte_0.60": rr_ai_worst >= 0.60,
        "ipn_worst_device_fp_lte_0.20": ipn_worst_fp <= 0.20,
        "owner_fp_lte_0.20": owner_fp <= 0.20,
    }
    return {"passed": all(checks.values()), "checks": checks}


def run(owner_gallery: Path, batch_size: int = 2) -> dict[str, Any]:
    if SCORES.exists() or REPORT.exists() or EVIDENCE.exists():
        raise FileExistsError("official DDA DEVELOPMENT output already exists; no silent rerun")
    if sha256_file(IPN_AUDIT) != EXPECTED_IPN_SHA:
        raise ValueError("IPN realization binding changed")
    if sha256_file(RR_MANIFEST) != EXPECTED_RR_MANIFEST_SHA:
        raise ValueError("RR validation manifest binding changed")
    ipn = json.loads(IPN_AUDIT.read_text())
    rr = json.loads(RR_MANIFEST.read_text())
    owner_paths = image_paths([str(owner_gallery)])
    owner_identity = [
        {"name": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in owner_paths
    ]
    owner_sha = hashlib.sha256(_json_bytes(owner_identity)).hexdigest()
    if owner_sha != EXPECTED_OWNER_IDENTITY or len(owner_paths) != 210:
        raise ValueError("owner gallery identity changed")

    candidate = OfficialDDACandidate()
    rows: list[dict[str, Any]] = []
    rr_paths = [DATA_ROOT / "e33_rrdataset" / row["path"] for row in rr["rows"]]
    for source_row, score in zip(rr["rows"], candidate.score_paths(rr_paths, batch_size), strict=True):
        rows.append({**source_row, "population": "rr", "status": "ok", "score": score.score})
    print(f"DDA RR {len(rr_paths)}/{len(rr_paths)}", flush=True)

    ipn_by_device: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in ipn["records"]:
        ipn_by_device[str(row["device"])].append(row)
    for device, source_rows in sorted(ipn_by_device.items()):
        paths = [DATA_ROOT / "e32" / row["source_key"] for row in source_rows]
        for source_row, score in zip(source_rows, candidate.score_paths(paths, batch_size), strict=True):
            rows.append({
                "path": source_row["source_key"],
                "population": "ipn",
                "label": 0,
                "source": device,
                "condition": "native",
                "status": "ok",
                "score": score.score,
            })
        print(f"DDA IPN {device} {len(paths)}/{len(paths)}", flush=True)

    owner_scores = candidate.score_paths(owner_paths, batch_size)
    for path, score in zip(owner_paths, owner_scores, strict=True):
        rows.append({
            "path": path.name,
            "population": "owner",
            "label": 0,
            "source": "owner_gallery",
            "condition": "native",
            "status": "ok",
            "score": score.score,
        })
    print(f"DDA owner {len(owner_paths)}/{len(owner_paths)}", flush=True)

    score_raw = b"".join((json.dumps(row, sort_keys=True) + "\n").encode() for row in rows)
    score_part = SCORES.with_suffix(SCORES.suffix + ".part")
    score_part.write_bytes(score_raw)
    score_part.replace(SCORES)

    rr_rows = [row for row in rows if row["population"] == "rr"]
    ipn_rows = [row for row in rows if row["population"] == "ipn"]
    owner_rows = [row for row in rows if row["population"] == "owner"]
    rr_metrics = evaluate_binary_scores(rr_rows, threshold=THRESHOLD)
    ai_groups = [
        group for group in rr_metrics["per_group"]
        if group["label"] == 1 and group["count"] >= 20
    ]
    rr_ai_rates = {group["source"]: group["ai_recall"] for group in ai_groups}
    ipn_fp = {
        source: float(np.mean([row["score"] >= THRESHOLD for row in ipn_rows if row["source"] == source]))
        for source in sorted({row["source"] for row in ipn_rows})
    }
    owner_fp = float(np.mean([row["score"] >= THRESHOLD for row in owner_rows]))
    gate = development_gate(
        rr_auc=float(rr_metrics["roc_auc"]),
        rr_real_fp=float(rr_metrics["real_false_positive_rate"]),
        rr_ai_macro=float(np.mean(list(rr_ai_rates.values()))),
        rr_ai_worst=min(rr_ai_rates.values()),
        ipn_worst_fp=max(ipn_fp.values()),
        owner_fp=owner_fp,
    )
    report = {
        "schema_version": 1,
        "experiment": "E35/official-DDA-development",
        "state": "dda_development_passed" if gate["passed"] else "dda_development_failed",
        "contract": candidate.contract(),
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "scores_sha256": hashlib.sha256(score_raw).hexdigest(),
        "scores_bytes": len(score_raw),
        "counts": {"rr": len(rr_rows), "ipn": len(ipn_rows), "owner": len(owner_rows)},
        "rr": {
            "metrics": rr_metrics,
            "ai_recall_by_scenario": rr_ai_rates,
            "ai_macro_recall": float(np.mean(list(rr_ai_rates.values()))),
            "ai_worst_recall": min(rr_ai_rates.values()),
        },
        "ipn": {
            "device_fp": ipn_fp,
            "macro_device_fp": float(np.mean(list(ipn_fp.values()))),
            "worst_device_fp": max(ipn_fp.values()),
            "score_summary": _summary([row["score"] for row in ipn_rows]),
        },
        "owner": {
            "false_positive_rate": owner_fp,
            "real_recall": 1.0 - owner_fp,
            "score_summary": _summary([row["score"] for row in owner_rows]),
            "gallery_identity_sha256": owner_sha,
        },
        "development_gate": gate,
        "boundary": "Published threshold DEVELOPMENT only; no threshold fit, retry arm or locked DDA-COCO access.",
    }
    _write_atomic(REPORT, report)
    _write_atomic(EVIDENCE, report)
    return report


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("owner_gallery", type=Path)
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args(argv)
    result = run(args.owner_gallery.resolve(), args.batch_size)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
