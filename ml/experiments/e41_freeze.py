"""Package the byte-identical E40 head with E41's consumed-calibration threshold."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np

from pixelproof.e32_candidate import DINO_MODEL_ID, DINO_WEIGHT_SHA256
from pixelproof.project_paths import DATA_ROOT, ML_ROOT


REPO_ROOT = ML_ROOT.parent
E40_ROOT = DATA_ROOT / "e40"
E41_ROOT = DATA_ROOT / "e41"
DRAFT = E40_ROOT / "e40_head_draft.joblib"
DRAFT_SHA256 = "72b8d8cd05f0f4d51ab5008e19bf4ab3418988b3444b2e640a25fdd0e54819c0"
DIAGNOSTIC = REPO_ROOT / "evidence" / "e40_threshold_diagnostic.json"
DIAGNOSTIC_SHA256 = "53829c3bbadaec13f3562030b15891aaa215aa91e15d08d69eeeaf23f69baeba"
ROLE = REPO_ROOT / "evidence" / "e41_role_amendment.json"
CONTRACT = REPO_ROOT / "evidence" / "e41_fixed_contract.json"
CONTRACT_SHA256 = "482cb58d48766c9697de817fa85668be90c4e7dbb09aa04a9213b199cb665c39"
CANDIDATE = E41_ROOT / "e41_dinov2s.joblib"
REPORT = E41_ROOT / "candidate_report.json"
EVIDENCE = REPO_ROOT / "evidence" / "e41_candidate.json"
OLD_THRESHOLD = 0.17080099880695346
NEW_THRESHOLD = 0.6195540428161622


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_bytes((json.dumps(value, indent=2, sort_keys=True) + "\n").encode())
    temporary.replace(path)


def head_numeric_sha256(head: Any) -> str:
    """Bind every learned scaler/logistic numeric field, independently of artifact metadata."""
    scaler = head.named_steps["standardscaler"]
    logistic = head.named_steps["logisticregression"]
    digest = hashlib.sha256()
    for name, value in (
        ("scaler.mean_", scaler.mean_),
        ("scaler.var_", scaler.var_),
        ("scaler.scale_", scaler.scale_),
        ("logistic.classes_", logistic.classes_),
        ("logistic.coef_", logistic.coef_),
        ("logistic.intercept_", logistic.intercept_),
    ):
        array = np.ascontiguousarray(value)
        digest.update(name.encode())
        digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def run() -> dict[str, Any]:
    if any(path.exists() for path in (CANDIDATE, REPORT, EVIDENCE)):
        raise FileExistsError("E41 candidate output already exists; no silent rerun")
    if _sha256_file(DRAFT) != DRAFT_SHA256:
        raise ValueError("E40 draft changed")
    if _sha256_file(DIAGNOSTIC) != DIAGNOSTIC_SHA256:
        raise ValueError("E40 threshold diagnostic changed")
    if _sha256_file(CONTRACT) != CONTRACT_SHA256:
        raise ValueError("E41 fixed contract changed")
    role = json.loads(ROLE.read_text())
    contract = json.loads(CONTRACT.read_text())
    if role.get("state") != "role_amended_before_e41_candidate":
        raise ValueError("E41 role amendment changed state")
    if contract.get("state") != "fixed_before_e41_candidate":
        raise ValueError("E41 fixed contract changed state")
    role_sha = _sha256_file(ROLE)
    if contract.get("inputs", {}).get("e41_role_amendment_sha256") != role_sha:
        raise ValueError("E41 contract does not bind the current role amendment")
    diagnostic = json.loads(DIAGNOSTIC.read_text())
    if diagnostic.get("state") != "posthoc_diagnostic_not_e40_evidence":
        raise ValueError("unexpected threshold diagnostic state")
    if float(diagnostic["selected_frontier"]["threshold"]) != NEW_THRESHOLD:
        raise ValueError("E41 threshold changed")

    artifact = joblib.load(DRAFT)
    if artifact.get("model_id") != DINO_MODEL_ID or artifact.get("model_weight_sha256") != DINO_WEIGHT_SHA256:
        raise ValueError("unexpected E40 model identity")
    if float(artifact.get("threshold")) != OLD_THRESHOLD:
        raise ValueError("unexpected E40 draft threshold")
    before_head_sha = head_numeric_sha256(artifact["head"])
    artifact.update({
        "schema_version": 1,
        "model_name": "E41 DINOv2-S broad-real threshold-transfer candidate",
        "threshold": NEW_THRESHOLD,
        "status": "research_candidate_awaiting_independent_final",
        "calibration": {
            "role": "E41_BROAD_REAL_CALIBRATION",
            "count": 650,
            "diagnostic_sha256": DIAGNOSTIC_SHA256,
            "role_amendment_sha256": role_sha,
            "fixed_contract_sha256": _sha256_file(CONTRACT),
        },
    })
    after_head_sha = head_numeric_sha256(artifact["head"])
    if before_head_sha != after_head_sha:
        raise RuntimeError("E41 packaging changed learned head values")
    CANDIDATE.parent.mkdir(parents=True, exist_ok=True)
    temporary = CANDIDATE.with_suffix(".joblib.part")
    joblib.dump(artifact, temporary)
    temporary.replace(CANDIDATE)
    reloaded = joblib.load(CANDIDATE)
    if head_numeric_sha256(reloaded["head"]) != before_head_sha:
        raise RuntimeError("serialized E41 head differs from E40 draft")

    report = {
        "schema_version": 1,
        "experiment": "E41/broad-real-threshold-transfer",
        "state": "candidate_frozen_awaiting_independent_final",
        "candidate": {
            "path": CANDIDATE.relative_to(E41_ROOT).as_posix(),
            "bytes": CANDIDATE.stat().st_size,
            "sha256": _sha256_file(CANDIDATE),
            "threshold": NEW_THRESHOLD,
            "head_numeric_sha256": before_head_sha,
            "head_numeric_identical_to_e40": True,
        },
        "bindings": {
            "e40_draft_sha256": DRAFT_SHA256,
            "e40_threshold_diagnostic_sha256": DIAGNOSTIC_SHA256,
            "e41_role_amendment_sha256": role_sha,
            "e41_fixed_contract_sha256": _sha256_file(CONTRACT),
        },
        "calibration_measurements": {
            "counts": diagnostic["counts"],
            "metrics": diagnostic["metrics"],
            "selected_frontier": diagnostic["selected_frontier"],
            "claim": "consumed calibration only; not independent evidence",
        },
        "boundary": "No retraining or preprocessing change. No E41 FINAL byte exists; candidate is not served or validated.",
    }
    _write_atomic(REPORT, report)
    _write_atomic(EVIDENCE, report)
    return report


def main(argv: Iterable[str] | None = None) -> int:
    del argv
    print(json.dumps(run(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
