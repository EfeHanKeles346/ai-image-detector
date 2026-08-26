"""Score the already-consumed owner gallery as DEVELOPMENT without refitting E32."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from pixelproof.e32_candidate import ARTIFACT_SHA256, E32Candidate, image_paths
from pixelproof.project_paths import ML_ROOT


OUTPUT = ML_ROOT.parent / "evidence" / "e32_owner_gallery_smoke.json"


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def _write_atomic(path: Path, raw: bytes) -> None:
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_bytes(raw)
    temporary.replace(path)


def run(folder: Path) -> dict[str, Any]:
    paths = image_paths([str(folder)])
    candidate = E32Candidate()
    scores = candidate.score_paths(paths)
    values = np.asarray([row.score for row in scores])
    false_positives = [row for row in scores if row.predicted_ai]
    identity = [
        {"name": path.name, "bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        for path in paths
    ]
    report = {
        "schema_version": 1,
        "experiment": "E32/C4-owner-gallery-development-smoke",
        "state": "development_owner_gallery_scored_no_refit",
        "artifact_sha256": ARTIFACT_SHA256,
        "supported_still_count": len(paths),
        "excluded_nonimage_count": sum(1 for path in folder.rglob("*") if path.is_file()) - len(paths),
        "gallery_identity_sha256": hashlib.sha256(_json_bytes(identity)).hexdigest(),
        "threshold": candidate.threshold,
        "false_positive_count": len(false_positives),
        "real_recall": 1.0 - len(false_positives) / len(paths),
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
        "boundary": "Previously consumed owner-real DEVELOPMENT only; no refit or threshold change is allowed.",
    }
    _write_atomic(OUTPUT, _json_bytes(report))
    return report


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    args = parser.parse_args(argv)
    print(json.dumps(run(args.folder.resolve()), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
