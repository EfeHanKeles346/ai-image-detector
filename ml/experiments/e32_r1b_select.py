"""Apply the preregistered internal-only R1b arm selection rule."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from pixelproof.project_paths import ML_ROOT


EVIDENCE = ML_ROOT.parent / "evidence"
OUTPUT = EVIDENCE / "e32_r1b_selection.json"


def choose(reports: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    passing = [row for row in reports if row.get("screen_gate", {}).get("passed") is True]
    if not passing:
        raise ValueError("no R1b arm passed the internal gate")
    return max(
        passing,
        key=lambda row: (
            float(row["metrics"]["auc"]),
            -float(row["selected_c"]),
            1 if row["arm"] == "dino" else 0,
        ),
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze() -> dict[str, Any]:
    paths = [EVIDENCE / "e32_r1b_dino.json", EVIDENCE / "e32_r1b_cf.json"]
    reports = [json.loads(path.read_text()) for path in paths]
    selected = choose(reports)
    report = {
        "schema_version": 1,
        "experiment": "E32/C4-R1b-internal-arm-selection",
        "state": "r1b_internal_arm_frozen_external_unopened",
        "rule": "passing arm with higher CALIBRATION AUC; exact tie smaller selected C; exact tie DINO",
        "arm_receipts": {
            row["arm"]: {
                "evidence_sha256": _sha256(path),
                "auc": row["metrics"]["auc"],
                "selected_c": row["selected_c"],
                "artifact_sha256": row["artifact_sha256"],
                "passed": row["screen_gate"]["passed"],
            }
            for row, path in zip(reports, paths, strict=True)
        },
        "selected_arm": selected["arm"],
        "selected_artifact_sha256": selected["artifact_sha256"],
        "selected_threshold": selected["metrics"]["threshold"],
        "boundary": "Selection uses internal TRAIN/CALIBRATION evidence only; IPN and owner model scores remain unopened.",
    }
    temporary = OUTPUT.with_suffix(OUTPUT.suffix + ".part")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    temporary.replace(OUTPUT)
    return report


def main(argv: Iterable[str] | None = None) -> int:
    argparse.ArgumentParser(description=__doc__).parse_args(argv)
    print(json.dumps(freeze(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
