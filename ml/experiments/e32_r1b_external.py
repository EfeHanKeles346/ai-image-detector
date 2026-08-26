"""Score the frozen R1b candidate once on IPN and owner DEVELOPMENT reals."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from pixelproof.e32_candidate import image_paths
from pixelproof.e32_r1b_candidate import ARTIFACT_SHA256, E32R1bCandidate
from pixelproof.project_paths import DATA_ROOT, ML_ROOT


IPN_AUDIT = DATA_ROOT / "e32" / "r1b_ipn_realization.json"
EXPECTED_IPN_SHA = "f5827dcea69484421b15a6608555a7c64556d206c991d02606222b1892bc243b"
EXPECTED_OWNER_IDENTITY = "390e3c210ee61d70252d7e4714b8640463f44d57760942d25a1bdf7eab5aac09"
OUTPUT = ML_ROOT.parent / "evidence" / "e32_r1b_external_development.json"


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _summary(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "min": float(array.min()),
        "p25": float(np.quantile(array, 0.25)),
        "median": float(np.median(array)),
        "p75": float(np.quantile(array, 0.75)),
        "p90": float(np.quantile(array, 0.90)),
        "p95": float(np.quantile(array, 0.95)),
        "max": float(array.max()),
    }


def external_gate(ipn_device_fp: dict[str, float], owner_fp: float, current_ai_macro: float) -> dict[str, Any]:
    checks = {
        "ipn_worst_device_fp_lte_0.20": max(ipn_device_fp.values()) <= 0.20 + 1e-12,
        "owner_fp_lte_0.20": owner_fp <= 0.20 + 1e-12,
        "internal_current_ai_macro_gte_0.90": current_ai_macro >= 0.90,
    }
    return {"passed": all(checks.values()), "checks": checks}


def run(owner_gallery: Path) -> dict[str, Any]:
    if _sha256_file(IPN_AUDIT) != EXPECTED_IPN_SHA:
        raise ValueError("IPN realization binding changed")
    ipn = json.loads(IPN_AUDIT.read_text())
    if ipn.get("state") != "development_realization_passed_unscored":
        raise ValueError("IPN is not the frozen unscored DEVELOPMENT set")
    ipn_paths = [DATA_ROOT / "e32" / row["source_key"] for row in ipn["records"]]
    device_by_path = {str(DATA_ROOT / "e32" / row["source_key"]): row["device"] for row in ipn["records"]}
    owner_paths = image_paths([str(owner_gallery)])
    owner_identity = [
        {"name": path.name, "bytes": path.stat().st_size, "sha256": _sha256_file(path)}
        for path in owner_paths
    ]
    owner_identity_sha = hashlib.sha256(_json_bytes(owner_identity)).hexdigest()
    if owner_identity_sha != EXPECTED_OWNER_IDENTITY or len(owner_paths) != 210:
        raise ValueError("owner gallery identity changed")

    candidate = E32R1bCandidate()
    ipn_scores = candidate.score_paths(ipn_paths)
    owner_scores = candidate.score_paths(owner_paths)
    by_device: dict[str, list[bool]] = defaultdict(list)
    for row in ipn_scores:
        by_device[str(device_by_path[row.path])].append(row.predicted_ai)
    ipn_device_fp = {device: float(np.mean(values)) for device, values in sorted(by_device.items())}
    owner_fp = float(np.mean([row.predicted_ai for row in owner_scores]))
    cf_internal_path = ML_ROOT.parent / "evidence" / "e32_r1b_cf.json"
    selection = json.loads((ML_ROOT.parent / "evidence" / "e32_r1b_selection.json").read_text())
    if selection["arm_receipts"]["cf"]["evidence_sha256"] != _sha256_file(cf_internal_path):
        raise ValueError("selected CF internal evidence changed")
    current_ai_macro = float(json.loads(cf_internal_path.read_text())["metrics"]["current_ai_macro_recall"])
    gate = external_gate(ipn_device_fp, owner_fp, current_ai_macro)
    report = {
        "schema_version": 1,
        "experiment": "E32/C4-R1b-external-development",
        "state": "r1b_external_passed" if gate["passed"] else "r1b_external_failed",
        "artifact_sha256": ARTIFACT_SHA256,
        "threshold": candidate.threshold,
        "ipn": {
            "real_images": len(ipn_scores),
            "false_positives": sum(row.predicted_ai for row in ipn_scores),
            "real_recall": 1.0 - float(np.mean([row.predicted_ai for row in ipn_scores])),
            "device_fp": ipn_device_fp,
            "macro_device_fp": float(np.mean(list(ipn_device_fp.values()))),
            "worst_device_fp": max(ipn_device_fp.values()),
            "score_summary": _summary([row.score for row in ipn_scores]),
            "highest_scores": [
                {"name": Path(row.path).name, "device": device_by_path[row.path], "score": row.score}
                for row in sorted(ipn_scores, key=lambda item: item.score, reverse=True)[:10]
            ],
            "realization_sha256": EXPECTED_IPN_SHA,
        },
        "owner": {
            "real_images": len(owner_scores),
            "false_positives": sum(row.predicted_ai for row in owner_scores),
            "real_recall": 1.0 - owner_fp,
            "score_summary": _summary([row.score for row in owner_scores]),
            "highest_scores": [
                {"name": Path(row.path).name, "score": row.score}
                for row in sorted(owner_scores, key=lambda item: item.score, reverse=True)[:10]
            ],
            "gallery_identity_sha256": owner_identity_sha,
            "comparison": {"r0_real_recall": 0.24285714285714288, "r1a_real_recall": 0.2666666666666667},
        },
        "internal_current_ai_macro_recall": current_ai_macro,
        "external_gate": gate,
        "boundary": "Frozen DEVELOPMENT scoring only; no refit, threshold change, DINO fallback or test-derived policy.",
    }
    temporary = OUTPUT.with_suffix(OUTPUT.suffix + ".part")
    temporary.write_bytes(_json_bytes(report))
    temporary.replace(OUTPUT)
    return report


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("owner_gallery", type=Path)
    args = parser.parse_args(argv)
    print(json.dumps(run(args.owner_gallery.resolve()), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
