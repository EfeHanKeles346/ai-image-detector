"""Score owner-gallery DEVELOPMENT once with the frozen E32 R1a artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from pixelproof.e32_candidate import image_paths
from pixelproof.e32_cfvit_candidate import ARTIFACT_SHA256, E32CFViTCandidate
from pixelproof.project_paths import ML_ROOT


OUTPUT = ML_ROOT.parent / "evidence" / "e32_r1a_owner_gallery_smoke.json"


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def run(folder: Path) -> dict[str, Any]:
    paths = image_paths([str(folder)])
    candidate = E32CFViTCandidate()
    scores = candidate.score_paths(paths)
    values = np.asarray([row.score for row in scores])
    positives = [row for row in scores if row.predicted_ai]
    identity = [
        {"name": path.name, "bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        for path in paths
    ]
    report = {
        "schema_version": 1,
        "experiment": "E32/C4-R1a-owner-gallery-development-smoke",
        "state": "r1a_development_owner_gallery_scored_no_refit",
        "artifact_sha256": ARTIFACT_SHA256,
        "supported_still_count": len(paths),
        "excluded_nonimage_count": sum(1 for path in folder.rglob("*") if path.is_file()) - len(paths),
        "gallery_identity_sha256": hashlib.sha256(_json_bytes(identity)).hexdigest(),
        "threshold": candidate.threshold,
        "false_positive_count": len(positives),
        "real_recall": 1.0 - len(positives) / len(paths),
        "score_summary": {
            "min": float(values.min()),
            "p25": float(np.quantile(values, 0.25)),
            "median": float(np.median(values)),
            "p75": float(np.quantile(values, 0.75)),
            "p90": float(np.quantile(values, 0.90)),
            "p95": float(np.quantile(values, 0.95)),
            "max": float(values.max()),
        },
        "highest_scores": [
            {"name": Path(row.path).name, "score": row.score, "predicted_ai": row.predicted_ai}
            for row in sorted(scores, key=lambda row: row.score, reverse=True)[:10]
        ],
        "comparison": {
            "r0_real_recall": 0.24285714285714288,
            "historical_frozen_cf_unique_gallery_real_recall": 205 / 206,
        },
        "boundary": "Previously consumed owner-real DEVELOPMENT; no refit or threshold change.",
    }
    temporary = OUTPUT.with_suffix(OUTPUT.suffix + ".part")
    temporary.write_bytes(_json_bytes(report))
    temporary.replace(OUTPUT)
    return report


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    args = parser.parse_args(argv)
    print(json.dumps(run(args.folder.resolve()), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
