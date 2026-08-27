"""Score the frozen E39 candidate once on the independent 440-parent FINAL."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, Sequence

import joblib
import numpy as np
from PIL import Image

from pixelproof.benchmark_metrics import evaluate_binary_scores
from pixelproof.e32_candidate import DINO_MODEL_ID, DINO_REPO_ID, DINO_WEIGHT_SHA256, standardized_array
from pixelproof.project_paths import DATA_ROOT, ML_ROOT


EXPERIMENTS_DIR = Path(__file__).resolve().parent
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))

import e37_source_heldout as e37  # noqa: E402


E38_ROOT = DATA_ROOT / "e38"
E39_ROOT = DATA_ROOT / "e39"
MANIFEST = E39_ROOT / "final_manifest.json"
MANIFEST_SHA256 = "1076df20adc700a1893814e477891e9270e4f88502dbad2571e022f3a7127306"
DECISION_CONTRACT = E39_ROOT / "e39_threshold_candidate.json"
DECISION_CONTRACT_SHA256 = "7d49792911b9b24acca2ad58d08d5ecc14bded5ee85174218d08d1a4d3712cef"
MODEL_ARTIFACT = E38_ROOT / "e38_dinov2s.joblib"
MODEL_ARTIFACT_SHA256 = "fddbe475adb807bcf523127095b2a8221443761ad4afa71d8c163829afc44067"
THRESHOLD = 0.27006945013999945
SCORES = E39_ROOT / "final_scores.jsonl"
REPORT = E39_ROOT / "final_report.json"
EVIDENCE = ML_ROOT.parent / "evidence" / "e39_final_result.json"
BATCH_SIZE = 48


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def final_gate(metrics: Mapping[str, Any], rates: Mapping[str, Any]) -> dict[str, Any]:
    return e37.gate(metrics, rates)


def _load_contract() -> tuple[list[dict[str, Any]], Any]:
    bindings = (
        (MANIFEST, MANIFEST_SHA256, "E39 FINAL manifest"),
        (DECISION_CONTRACT, DECISION_CONTRACT_SHA256, "E39 decision contract"),
        (MODEL_ARTIFACT, MODEL_ARTIFACT_SHA256, "E39 underlying model"),
    )
    for path, expected, label in bindings:
        if not path.is_file() or _sha256_file(path) != expected:
            raise ValueError(f"{label} changed")
    manifest = json.loads(MANIFEST.read_text())
    if manifest.get("state") != "final_manifest_frozen_unscored" or manifest.get("counts") != {"real": 160, "ai": 280, "total": 440}:
        raise ValueError("E39 FINAL role/count contract changed")
    decision = json.loads(DECISION_CONTRACT.read_text())
    if (
        decision.get("state") != "research_candidate_frozen_final_eligible"
        or decision.get("underlying_artifact_sha256") != MODEL_ARTIFACT_SHA256
        or float(decision.get("threshold", -1)) != THRESHOLD
        or decision.get("positive_label") != "ai"
    ):
        raise ValueError("E39 decision contract changed")
    artifact = joblib.load(MODEL_ARTIFACT)
    if artifact.get("positive_label") != "ai":
        raise ValueError("E39 model score direction changed")
    return manifest["rows"], artifact["head"]


def _score(rows: Sequence[Mapping[str, Any]], head: Any, batch_size: int) -> list[dict[str, Any]]:
    from huggingface_hub import snapshot_download
    import timm
    import torch

    snapshot = Path(snapshot_download(DINO_REPO_ID, local_files_only=True))
    if _sha256_file(snapshot / "model.safetensors") != DINO_WEIGHT_SHA256:
        raise ValueError("cached DINOv2-S weights changed")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = timm.create_model(DINO_MODEL_ID, pretrained=True, num_classes=0, img_size=224).to(device).eval()
    config = timm.data.resolve_data_config({}, model=model)
    mean = torch.tensor(config["mean"], device=device).view(1, 3, 1, 1)
    std = torch.tensor(config["std"], device=device).view(1, 3, 1, 1)
    results = []
    with torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            group = rows[start : start + batch_size]
            arrays = []
            for row in group:
                path = E39_ROOT / str(row["path"])
                if _sha256_file(path) != str(row["sha256"]):
                    raise ValueError(f"E39 FINAL image binding changed: {row['path']}")
                with Image.open(path) as image:
                    arrays.append(standardized_array(image))
            tensor = torch.from_numpy(np.stack(arrays)).to(device)
            tensor = tensor.permute(0, 3, 1, 2).float().div_(255.0)
            embeddings = model((tensor - mean) / std).float().cpu().numpy()
            scores = head.predict_proba(embeddings)[:, 1]
            for row, score in zip(group, scores, strict=True):
                value = float(score)
                if not np.isfinite(value):
                    raise ValueError(f"non-finite E39 FINAL score: {row['path']}")
                results.append({
                    "parent_id": row["parent_id"],
                    "path": row["path"],
                    "label": int(row["label"]),
                    "source": row["source"],
                    "condition": row["condition"],
                    "status": "ok",
                    "score": value,
                })
            print(f"E39 FINAL {min(start + batch_size, len(rows))}/{len(rows)}", flush=True)
    return results


def run(batch_size: int = BATCH_SIZE) -> dict[str, Any]:
    if any(path.exists() for path in (SCORES, REPORT, EVIDENCE)):
        raise FileExistsError("E39 FINAL output already exists; rerun/retry is forbidden")
    rows, head = _load_contract()
    scored = _score(rows, head, batch_size)
    score_raw = b"".join((json.dumps(row, sort_keys=True) + "\n").encode() for row in scored)
    score_part = SCORES.with_suffix(SCORES.suffix + ".part")
    score_part.write_bytes(score_raw)
    score_part.replace(SCORES)
    rates = e37.source_rates(scored, THRESHOLD)
    metrics = evaluate_binary_scores(scored, threshold=THRESHOLD)
    gate = final_gate(metrics, rates)
    report = {
        "schema_version": 1,
        "experiment": "E39/independent-native-modern-final",
        "state": "final_passed" if gate["passed"] else "final_failed",
        "decision_contract_sha256": DECISION_CONTRACT_SHA256,
        "underlying_model_sha256": MODEL_ARTIFACT_SHA256,
        "manifest_sha256": MANIFEST_SHA256,
        "scores_sha256": hashlib.sha256(score_raw).hexdigest(),
        "scores_bytes": len(score_raw),
        "threshold": THRESHOLD,
        "counts": {"real": 160, "ai": 280, "total": 440},
        "source_rates": rates,
        "metrics": metrics,
        "bootstrap": {
            "real_macro_fp": e37._bootstrap_macro(scored, threshold=THRESHOLD, label=0),
            "ai_macro_recall": e37._bootstrap_macro(scored, threshold=THRESHOLD, label=1),
        },
        "gate": gate,
        "boundary": "First and only score of the frozen E39 threshold candidate on independent FINAL; no retry or retuning permitted.",
    }
    _write_atomic(REPORT, report)
    _write_atomic(EVIDENCE, report)
    return report


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = parser.parse_args(argv)
    print(json.dumps(run(args.batch_size), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
