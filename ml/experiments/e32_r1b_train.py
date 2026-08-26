"""Append iPhone features to frozen R0/R1a caches and refit controlled R1b heads."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import joblib
import numpy as np
from huggingface_hub import snapshot_download

from pixelproof.project_paths import DATA_ROOT, ML_ROOT


EXPERIMENTS_DIR = Path(__file__).resolve().parent
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))

import e32_cfvit_train as cf  # noqa: E402
import e32_r0_train as r0  # noqa: E402
from pixelproof.e32_candidate import DINO_REPO_ID, DINO_WEIGHT_SHA256  # noqa: E402


E32_ROOT = DATA_ROOT / "e32"
INPUT_RECEIPT = E32_ROOT / "r1b_input_receipt.json"
INPUT_EVIDENCE = ML_ROOT.parent / "evidence" / "e32_r1b_input_receipt.json"
OLD_RECEIPT = E32_ROOT / "r0_input_receipt.json"
OLD_DINO = E32_ROOT / "features" / "r0_dinov2s_features.npz"
OLD_CF = E32_ROOT / "features" / "r1a_cfvit_features.npz"
OLD_FEATURE_SHA = {
    "dino": "716df956fc40b1cf557b30c34e1adb216d2abeb2314c20b2edf10671a387be3b",
    "cf": "c170a1f6688421f73c72c3b9ed6f1de10a57bf9850a535246e64a15bc71bbc6b",
}


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _write_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_bytes(_json_bytes(value))
    temporary.replace(path)


def _save_npz(path: Path, contract: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".npz.part")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **contract)
    temporary.replace(path)


def merge_features(
    old: Mapping[str, np.ndarray], new: Mapping[str, np.ndarray], rows: Sequence[Mapping[str, Any]]
) -> dict[str, np.ndarray]:
    old_ids = old["record_ids"].astype(str)
    new_ids = new["record_ids"].astype(str)
    if len(set(old_ids) & set(new_ids)):
        raise ValueError("old/new feature ids overlap")
    vectors = {
        record_id: vector for record_id, vector in zip(old_ids, old["features"], strict=True)
    }
    vectors.update(
        {record_id: vector for record_id, vector in zip(new_ids, new["features"], strict=True)}
    )
    expected = [str(row["record_id"]) for row in rows]
    if set(vectors) != set(expected):
        raise ValueError("merged features do not match R1b receipt")
    return {
        "features": np.stack([vectors[record_id] for record_id in expected]).astype(np.float32),
        "record_ids": np.asarray(expected),
        "labels": np.asarray([1 if row["label"] == "ai" else 0 for row in rows], dtype=np.int8),
        "roles": np.asarray([row["role"] for row in rows]),
        "sources": np.asarray([row["source_id"] for row in rows]),
    }


def _load_receipts() -> tuple[dict[str, Any], bytes, dict[str, Any]]:
    raw = INPUT_RECEIPT.read_bytes()
    compact = json.loads(INPUT_EVIDENCE.read_text())
    if compact.get("detailed_report_sha256") != _sha256(raw):
        raise ValueError("R1b input receipt binding changed")
    receipt = json.loads(raw)
    if receipt.get("state") != "r1b_input_realization_complete" or receipt.get("record_count") != 26_682:
        raise ValueError("R1b input receipt incomplete")
    old = json.loads(OLD_RECEIPT.read_text())
    return receipt, raw, old


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as stored:
        return {name: stored[name] for name in stored.files}


def _old_contract(arm: str, old_receipt: Mapping[str, Any]) -> dict[str, np.ndarray]:
    path = OLD_DINO if arm == "dino" else OLD_CF
    if r0._sha256_file(path) != OLD_FEATURE_SHA[arm]:
        raise ValueError(f"frozen {arm} feature archive changed")
    contract = _load_npz(path)
    expected = np.asarray([row["record_id"] for row in old_receipt["records"]])
    if not np.array_equal(contract["record_ids"], expected):
        raise ValueError(f"frozen {arm} features lost old receipt alignment")
    return contract


def _new_features(arm: str, rows: Sequence[Mapping[str, Any]]) -> tuple[dict[str, np.ndarray], Path]:
    path = E32_ROOT / "features" / f"r1b_{arm}_iphone14_features.npz"
    expected = np.asarray([row["record_id"] for row in rows])
    if path.is_file():
        contract = _load_npz(path)
        if not np.array_equal(contract["record_ids"], expected):
            raise ValueError(f"cached R1b {arm} iPhone features are misaligned")
        return contract, path
    if arm == "dino":
        local = Path(snapshot_download(DINO_REPO_ID, local_files_only=True))
        if r0._sha256_file(local / "model.safetensors") != DINO_WEIGHT_SHA256:
            raise ValueError("cached DINO weights changed")
        contract = r0.extract_features({"records": list(rows)})
    else:
        contract = cf.extract_features({"records": list(rows)})
    _save_npz(path, contract)
    return contract, path


def train(arm: str) -> dict[str, Any]:
    receipt, receipt_raw, old_receipt = _load_receipts()
    new_rows = [row for row in receipt["records"] if row["source_id"] == "csafe-mcsidb-iphone14"]
    if len(new_rows) != 3_994:
        raise ValueError("R1b iPhone input count changed")
    old = _old_contract(arm, old_receipt)
    new, new_path = _new_features(arm, new_rows)
    full_path = E32_ROOT / "features" / f"r1b_{arm}_features.npz"
    expected = np.asarray([row["record_id"] for row in receipt["records"]])
    if full_path.is_file():
        contract = _load_npz(full_path)
        if not np.array_equal(contract["record_ids"], expected):
            raise ValueError(f"cached R1b {arm} full features are misaligned")
    else:
        contract = merge_features(old, new, receipt["records"])
        _save_npz(full_path, contract)

    labels = contract["labels"].astype(np.int64)
    roles = contract["roles"].astype(str)
    sources = contract["sources"].astype(str)
    head, best_c, threshold, metrics, grid = cf.fit_head(
        contract["features"], labels, roles, sources
    )
    gate = r0.screen_gate(metrics)
    model_contract = (
        {
            "model_id": r0.MODEL_ID,
            "input_size": r0.INPUT_SIZE,
            "model_weight_sha256": DINO_WEIGHT_SHA256,
        }
        if arm == "dino"
        else {
            "model_repo": cf.MODEL_REPO,
            "model_revision": cf.MODEL_REVISION,
            "model_weight_sha256": cf.MODEL_WEIGHT_SHA256,
        }
    )
    artifact_path = E32_ROOT / "models" / f"e32_r1b_{arm}.joblib"
    artifact = {
        "schema_version": 1,
        "model_name": f"E32 R1b controlled iPhone correction ({arm})",
        **model_contract,
        "input_receipt_sha256": _sha256(receipt_raw),
        "feature_archive_sha256": r0._sha256_file(full_path),
        "new_feature_archive_sha256": r0._sha256_file(new_path),
        "head": head,
        "selected_c": best_c,
        "threshold": threshold,
        "positive_label": "ai",
        "preprocessing": receipt["preprocessing"],
    }
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = artifact_path.with_suffix(".joblib.part")
    joblib.dump(artifact, temporary)
    temporary.replace(artifact_path)
    report = {
        "schema_version": 1,
        "experiment": f"E32/C4-R1b-{arm}-controlled-head",
        "state": "r1b_internal_screen_passed" if gate["passed"] else "r1b_internal_screen_failed",
        "arm": arm,
        **model_contract,
        "input_receipt_sha256": _sha256(receipt_raw),
        "old_feature_archive_sha256": OLD_FEATURE_SHA[arm],
        "new_feature_shape": list(new["features"].shape),
        "new_feature_archive_bytes": new_path.stat().st_size,
        "new_feature_archive_sha256": r0._sha256_file(new_path),
        "feature_shape": list(contract["features"].shape),
        "feature_archive_bytes": full_path.stat().st_size,
        "feature_archive_sha256": r0._sha256_file(full_path),
        "artifact_bytes": artifact_path.stat().st_size,
        "artifact_sha256": r0._sha256_file(artifact_path),
        "train_rows": int(np.sum(roles == "TRAIN")),
        "calibration_rows": int(np.sum(roles == "CALIBRATION")),
        "c_grid_auc": grid,
        "selected_c": best_c,
        "metrics": metrics,
        "screen_gate": gate,
        "boundary": "Internal device-held-out/source-stratified screen only; IPN and owner model scores remain unopened.",
    }
    _write_atomic(ML_ROOT.parent / "evidence" / f"e32_r1b_{arm}.json", report)
    return report


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True, choices=("dino", "cf"))
    args = parser.parse_args(argv)
    print(json.dumps(train(args.arm), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
