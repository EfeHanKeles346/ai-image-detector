"""Build, score and freeze the threshold-only E33/R1c calibration candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable, Mapping

from pixelproof.benchmark_metrics import (
    BenchmarkContractError,
    evaluate_binary_scores,
    select_source_robust_threshold,
)
from pixelproof.e32_candidate import sha256_file
from pixelproof.e32_r1b_candidate import ARTIFACT_SHA256, E32R1bCandidate
from pixelproof.project_paths import DATA_ROOT, ML_ROOT


OUTPUT_ROOT = DATA_ROOT / "e33_rrdataset"
CAL_ROOT = OUTPUT_ROOT / "calibration"
MANIFEST = OUTPUT_ROOT / "r1c_cal_manifest.json"
SCORES = OUTPUT_ROOT / "r1c_cal_scores.jsonl"
CANDIDATE = OUTPUT_ROOT / "r1c_threshold_candidate.json"
MANIFEST_EVIDENCE = ML_ROOT.parent / "evidence" / "e33_r1c_cal_manifest.json"
CANDIDATE_EVIDENCE = ML_ROOT.parent / "evidence" / "e33_r1c_threshold.json"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
SCENARIOS = {
    "real": "rrdataset_real_pool",
    "normal": "everyday_life",
    "production": "labor_and_production",
    "Culture_&_Religion": "culture_and_religion",
    "Medical_&_Public_Health": "medical_and_public_health",
    "Political_&_Social_Events": "political_and_social_events",
    "War_&_Conflict_Scenes": "war_and_conflict",
    "Natural_Disasters_&_Accidents": "natural_disasters_and_accidents",
}
NUMBERED_STEM = re.compile(r"^(?P<prefix>.+)_(?P<index>\d{6})$")


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def _write_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_bytes(_json_bytes(value))
    temporary.replace(path)


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def scenario_from_filename(name: str) -> str:
    match = NUMBERED_STEM.fullmatch(Path(name).stem)
    if match is None or match.group("prefix") not in SCENARIOS:
        raise ValueError(f"undeclared RRDataset scenario filename: {name}")
    return SCENARIOS[match.group("prefix")]


def build_manifest(root: Path = CAL_ROOT) -> dict[str, Any]:
    extraction = json.loads((OUTPUT_ROOT / "calibration_extraction_receipt.json").read_text())
    if extraction.get("state") != "calibration_extraction_complete":
        raise ValueError("RRDataset calibration extraction is not complete")
    rows = []
    for class_name, label in (("real", 0), ("ai", 1)):
        class_root = root / class_name
        if not class_root.is_dir():
            raise FileNotFoundError(f"missing calibration class directory: {class_root}")
        for path in sorted(class_root.iterdir()):
            if (
                not path.is_file()
                or path.name.startswith("._")
                or path.suffix.lower() not in IMAGE_SUFFIXES
            ):
                continue
            scenario = scenario_from_filename(path.name)
            rows.append({
                "path": str(path.relative_to(OUTPUT_ROOT)),
                "label": label,
                "class_name": class_name,
                "source": scenario,
                "condition": "original",
                "parent_id": f"{class_name}:{path.stem}",
                "bytes": path.stat().st_size,
            })
    if not rows or {row["label"] for row in rows} != {0, 1}:
        raise ValueError("calibration manifest needs explicit REAL and AI rows")
    if len(rows) != int(extraction["image_count"]):
        raise ValueError(
            f"calibration manifest count mismatch: {len(rows)} != {extraction['image_count']}"
        )
    if len({row["path"] for row in rows}) != len(rows):
        raise ValueError("calibration manifest repeats a path")
    if len({row["parent_id"] for row in rows}) != len(rows):
        raise ValueError("calibration manifest repeats a parent")
    by_class_source: dict[str, int] = {}
    for row in rows:
        key = f"{row['class_name']}/{row['source']}"
        by_class_source[key] = by_class_source.get(key, 0) + 1
    detailed = {
        "schema_version": 1,
        "state": "r1c_cal_manifest_frozen_unscored",
        "label_invariant": {"real": 0, "ai": 1},
        "role": "calibration_only",
        "source_semantics": "REAL is one undisclosed upstream pool; AI exposes seven scenario groups",
        "limitations": [
            "RRDataset validation filenames expose AI scenario but not REAL scenario or camera pipeline.",
            "The REAL false-positive budget is aggregate and cannot claim cross-camera calibration.",
            "Owner-gallery and IPN scores are absent and cannot select this threshold.",
            "The RRDataset test archive is absent and locked at this stage.",
        ],
        "rows": rows,
    }
    raw = _json_bytes(detailed)
    _write_atomic(MANIFEST, detailed)
    compact = {
        "schema_version": 1,
        "state": detailed["state"],
        "role": detailed["role"],
        "row_count": len(rows),
        "real_count": sum(row["label"] == 0 for row in rows),
        "ai_count": sum(row["label"] == 1 for row in rows),
        "by_class_source": dict(sorted(by_class_source.items())),
        "detailed_manifest_bytes": len(raw),
        "detailed_manifest_sha256": _sha256(raw),
        "source_semantics": detailed["source_semantics"],
        "production_scores_created": 0,
    }
    _write_atomic(MANIFEST_EVIDENCE, compact)
    return compact


def _manifest() -> dict[str, Any]:
    payload = json.loads(MANIFEST.read_text())
    if payload.get("state") != "r1c_cal_manifest_frozen_unscored":
        raise ValueError("R1c CAL manifest is not frozen")
    return payload


def _score_chunk(candidate: E32R1bCandidate, rows: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    paths = [OUTPUT_ROOT / str(row["path"]) for row in rows]
    try:
        scores = candidate.score_paths(paths, batch_size=len(paths))
        return [
            {**dict(row), "status": "ok", "score": score.score, "error": None}
            for row, score in zip(rows, scores, strict=True)
        ]
    except Exception as error:
        if len(rows) == 1:
            return [{**dict(rows[0]), "status": "error", "score": None, "error": f"{type(error).__name__}: {error}"}]
        midpoint = len(rows) // 2
        return _score_chunk(candidate, rows[:midpoint]) + _score_chunk(candidate, rows[midpoint:])


def score_calibration(batch_size: int = 24) -> dict[str, Any]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    manifest = _manifest()
    if SCORES.exists() or CANDIDATE.exists():
        raise FileExistsError("R1c calibration score/candidate already exists; no silent rerun")
    candidate = E32R1bCandidate()
    rows = list(manifest["rows"])
    output: list[dict[str, Any]] = []
    for start in range(0, len(rows), batch_size):
        output.extend(_score_chunk(candidate, rows[start : start + batch_size]))
        if len(output) % 500 < batch_size:
            print(f"R1c CAL {len(output)}/{len(rows)}", flush=True)
    raw = b"".join((json.dumps(row, sort_keys=True) + "\n").encode() for row in output)
    temporary = SCORES.with_suffix(SCORES.suffix + ".part")
    temporary.write_bytes(raw)
    temporary.replace(SCORES)
    receipt = {
        "schema_version": 1,
        "state": "r1c_cal_scores_complete",
        "manifest_sha256": sha256_file(MANIFEST),
        "r1b_contract": candidate.contract(),
        "score_rows": len(output),
        "successful": sum(row["status"] == "ok" for row in output),
        "failed": sum(row["status"] == "error" for row in output),
        "scores_bytes": len(raw),
        "scores_sha256": _sha256(raw),
    }
    _write_atomic(OUTPUT_ROOT / "r1c_cal_score_receipt.json", receipt)
    return receipt


def _score_rows() -> list[dict[str, Any]]:
    receipt = json.loads((OUTPUT_ROOT / "r1c_cal_score_receipt.json").read_text())
    if receipt.get("state") != "r1c_cal_scores_complete" or receipt.get("manifest_sha256") != sha256_file(MANIFEST):
        raise ValueError("R1c CAL score receipt is not bound to the manifest")
    if receipt.get("scores_sha256") != sha256_file(SCORES):
        raise ValueError("R1c CAL scores changed")
    return [json.loads(line) for line in SCORES.read_text().splitlines() if line]


def freeze_threshold() -> dict[str, Any]:
    rows = _score_rows()
    # The official 500-row validation split exposes 250 pooled REAL rows and roughly 22-93
    # AI rows per scenario. Twenty is frozen from filenames, before any model score is opened.
    try:
        result = select_source_robust_threshold(rows, min_group_size=20)
        passed = True
        failure = None
    except BenchmarkContractError as error:
        # Preserve the first REAL-safe frontier point as diagnostic evidence without promoting it.
        result = select_source_robust_threshold(
            rows,
            min_group_size=20,
            macro_ai_recall_floor=0.0,
            worst_ai_recall_floor=0.0,
        )
        passed = False
        failure = str(error)
    receipt = json.loads((OUTPUT_ROOT / "r1c_cal_score_receipt.json").read_text())
    old_metrics = evaluate_binary_scores(rows, threshold=0.1259349137544632)
    selected_metrics = evaluate_binary_scores(rows, threshold=float(result["threshold"]))
    detailed = {
        "schema_version": 1,
        "state": "r1c_threshold_candidate_frozen" if passed else "r1c_threshold_rejected_on_calibration",
        "candidate": "E32/R1c-T",
        "passed": passed,
        "failure": failure,
        "base_artifact_sha256": ARTIFACT_SHA256,
        "score_receipt_sha256": sha256_file(OUTPUT_ROOT / "r1c_cal_score_receipt.json"),
        "scores_sha256": receipt["scores_sha256"],
        "selection": result,
        "metrics": {
            "frozen_r1b_threshold": old_metrics,
            "first_real_safe_threshold": selected_metrics,
        },
        "boundary": "threshold selected only on RRDataset validation scenario groups; DEVELOPMENT remains unopened",
    }
    _write_atomic(CANDIDATE, detailed)
    compact = {
        **detailed,
        "detailed_candidate_sha256": sha256_file(CANDIDATE),
        "detailed_candidate_bytes": CANDIDATE.stat().st_size,
    }
    _write_atomic(CANDIDATE_EVIDENCE, compact)
    return compact


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("build-manifest", "score-cal", "freeze-threshold"))
    parser.add_argument("--batch-size", type=int, default=24)
    args = parser.parse_args(argv)
    if args.command == "build-manifest":
        result = build_manifest()
    elif args.command == "score-cal":
        result = score_calibration(args.batch_size)
    else:
        result = freeze_threshold()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
