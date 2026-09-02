"""Bind and execute the one-shot E43-S score on frozen DDA-COCO pairs."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
from io import BytesIO
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
import zipfile

import joblib
import numpy as np
from PIL import Image

from experiments.e42_features import BLOCKS, MODEL_IDS, aggregate_tokens, texture_crops, transport_image
from experiments.e43_dda_coco import (
    CANDIDATE,
    CANDIDATE_SHA256,
    COCO_ARCHIVE,
    DDA_ARCHIVE,
    MANIFEST,
    THRESHOLD,
    VARIANTS,
)
from pixelproof.benchmark_metrics import evaluate_binary_scores
from pixelproof.e32_candidate import DINO_REPO_ID, DINO_WEIGHT_SHA256
from pixelproof.project_paths import DATA_ROOT, ML_ROOT


ROOT = DATA_ROOT / "e43_dda_coco"
MANIFEST_SHA256 = "e663d679f86ba69a545659203e11528d8998c9a362198a19f5f269a1ef97a3db"
DECLARED_ROWS = 34_755
DECLARED_PARENTS = 4_965
CONTRACT = ROOT / "score_contract.json"
CONTRACT_EVIDENCE = ML_ROOT.parent / "evidence" / "e43_dda_coco_score_contract.json"
SCORES = ROOT / "scores.jsonl"
REPORT = ROOT / "report.json"
RESULT_EVIDENCE = ML_ROOT.parent / "evidence" / "e43_dda_coco_result.json"
GATES = {
    "roc_auc_min": 0.90,
    "tpr_at_fpr10_min": 0.80,
    "eer_max": 0.15,
    "balanced_accuracy_min": 0.85,
    "real_false_positive_rate_max": 0.10,
    "ai_macro_recall_min": 0.80,
    "ai_worst_variant_recall_min": 0.60,
    "coverage": 1.0,
}


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def _write(path: Path, value: Any) -> bytes:
    raw = _json_bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_bytes(raw)
    temporary.replace(path)
    return raw


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024**2):
            digest.update(chunk)
    return digest.hexdigest()


def bind_score_contract() -> dict[str, Any]:
    if CONTRACT.exists() or CONTRACT_EVIDENCE.exists():
        raise FileExistsError("DDA-COCO score contract already exists")
    manifest = json.loads(MANIFEST.read_text())
    if (
        _digest(MANIFEST) != MANIFEST_SHA256
        or manifest.get("state") != "e43_dda_coco_manifest_frozen_unscored"
        or int(manifest.get("counts", {}).get("rows", -1)) != DECLARED_ROWS
        or int(manifest.get("counts", {}).get("parents", -1)) != DECLARED_PARENTS
    ):
        raise ValueError("DDA-COCO frozen manifest changed")
    if _digest(CANDIDATE) != CANDIDATE_SHA256:
        raise ValueError("E43-S candidate changed before score binding")
    payload = {
        "schema_version": 1,
        "experiment": "E43/DDA-COCO-open-independent-test",
        "state": "e43_dda_coco_score_contract_frozen",
        "manifest_sha256": MANIFEST_SHA256,
        "candidate_sha256": CANDIDATE_SHA256,
        "threshold": THRESHOLD,
        "declared_rows": DECLARED_ROWS,
        "declared_parents": DECLARED_PARENTS,
        "conditions": ["REAL", *VARIANTS],
        "gates": GATES,
        "inference": "unchanged E43-S clean global+two-texture-crop DINOv2-S feature path",
        "forbidden": [
            "threshold change",
            "row or parent removal after score",
            "test-informed fitting",
            "retry after a complete score stream",
        ],
        "model_scores_created": 0,
    }
    raw = _write(CONTRACT, payload)
    evidence = {
        **payload,
        "detailed_contract_bytes": len(raw),
        "detailed_contract_sha256": hashlib.sha256(raw).hexdigest(),
    }
    _write(CONTRACT_EVIDENCE, evidence)
    return evidence


def _prepare_payload(payload: bytes, expected_sha256: str) -> list[np.ndarray]:
    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise ValueError("DDA-COCO member changed after manifest freeze")
    with Image.open(BytesIO(payload)) as opened:
        opened.load()
        clean = transport_image(opened, "clean")
    return texture_crops(clean)


def _score_rows(
    rows: Sequence[Mapping[str, Any]], head: Any, batch_rows: int
) -> list[dict[str, Any]]:
    from huggingface_hub import snapshot_download
    import timm
    import torch

    snapshot = Path(snapshot_download(DINO_REPO_ID, local_files_only=True))
    if _digest(snapshot / "model.safetensors") != DINO_WEIGHT_SHA256:
        raise ValueError("cached DINOv2-S weights changed")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = timm.create_model(MODEL_IDS["small"], pretrained=True, num_classes=0, img_size=224)
    model = model.to(device).eval()
    config = timm.data.resolve_data_config({}, model=model)
    mean = torch.tensor(config["mean"], device=device).view(1, 3, 1, 1)
    std = torch.tensor(config["std"], device=device).view(1, 3, 1, 1)
    output: list[dict[str, Any]] = []
    with (
        zipfile.ZipFile(COCO_ARCHIVE) as real_bundle,
        zipfile.ZipFile(DDA_ARCHIVE) as synthetic_bundle,
        ThreadPoolExecutor(max_workers=6) as pool,
        torch.inference_mode(),
    ):
        for start in range(0, len(rows), batch_rows):
            group = rows[start : start + batch_rows]
            payloads = [
                (real_bundle if str(row["archive"]) == "COCO-val2017" else synthetic_bundle).read(
                    str(row["member"])
                )
                for row in group
            ]
            pairs = [
                (payload, str(row["sha256"]))
                for payload, row in zip(payloads, group, strict=True)
            ]
            packs = pool.map(lambda pair: _prepare_payload(pair[0], pair[1]), pairs)
            arrays = [array for pack in packs for array in pack]
            tensor = torch.from_numpy(np.stack(arrays)).to(device)
            tensor = tensor.permute(0, 3, 1, 2).float().div_(255.0)
            intermediates = model.forward_intermediates(
                (tensor - mean) / std,
                indices=list(BLOCKS["small"]),
                return_prefix_tokens=True,
                norm=True,
                intermediates_only=True,
            )
            tokens = torch.stack([item[1][:, 0, :] for item in intermediates], dim=1)
            features = aggregate_tokens(tokens.float().cpu().numpy(), len(group))
            scores = head.predict_proba(features)[:, 1]
            for row, value in zip(group, scores, strict=True):
                score = float(value)
                if not np.isfinite(score):
                    raise ValueError("non-finite DDA-COCO score")
                output.append({
                    "record_id": row["record_id"],
                    "parent_id": row["parent_id"],
                    "label": int(row["label"]),
                    "source": "COCO-val2017" if int(row["label"]) == 0 else str(row["condition"]),
                    "condition": str(row["condition"]),
                    "score": score,
                    "status": "ok",
                })
            completed = min(start + batch_rows, len(rows))
            if completed == len(rows) or completed // 250 != start // 250:
                print(f"E43 DDA score {completed}/{len(rows)}", flush=True)
    return output


def variant_rates(rows: Sequence[Mapping[str, Any]], threshold: float = THRESHOLD) -> dict[str, Any]:
    real = [float(row["score"]) >= threshold for row in rows if int(row["label"]) == 0]
    ai = {
        variant: [
            float(row["score"]) >= threshold
            for row in rows
            if int(row["label"]) == 1 and str(row["condition"]) == variant
        ]
        for variant in VARIANTS
    }
    if not real or any(not values for values in ai.values()):
        raise ValueError("DDA-COCO rates require REAL and every variant")
    recalls = {variant: float(np.mean(values)) for variant, values in ai.items()}
    return {
        "real_false_positive_rate": float(np.mean(real)),
        "ai_recall_by_variant": recalls,
        "ai_macro_recall": float(np.mean(list(recalls.values()))),
        "ai_worst_variant_recall": min(recalls.values()),
    }


def final_gate(metrics: Mapping[str, Any], rates: Mapping[str, Any]) -> dict[str, Any]:
    checks = {
        "roc_auc_gte_0_90": float(metrics["roc_auc"]) >= GATES["roc_auc_min"],
        "tpr_at_fpr10_gte_0_80": float(metrics["tpr_at_fpr"]["tpr"]) >= GATES["tpr_at_fpr10_min"],
        "eer_lte_0_15": float(metrics["eer"]) <= GATES["eer_max"],
        "balanced_accuracy_gte_0_85": float(metrics["balanced_accuracy"]) >= GATES["balanced_accuracy_min"],
        "real_fp_lte_0_10": float(rates["real_false_positive_rate"]) <= GATES["real_false_positive_rate_max"] + 1e-12,
        "ai_macro_recall_gte_0_80": float(rates["ai_macro_recall"]) >= GATES["ai_macro_recall_min"],
        "ai_worst_variant_recall_gte_0_60": float(rates["ai_worst_variant_recall"]) >= GATES["ai_worst_variant_recall_min"],
        "coverage_eq_1": float(metrics["coverage"]) == GATES["coverage"],
    }
    return {"passed": all(checks.values()), "checks": checks}


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> tuple[int, str]:
    raw = b"".join((json.dumps(row, sort_keys=True) + "\n").encode() for row in rows)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_bytes(raw)
    temporary.replace(path)
    return len(raw), hashlib.sha256(raw).hexdigest()


def evaluate(batch_rows: int = 16) -> dict[str, Any]:
    if batch_rows < 1 or batch_rows > 64:
        raise ValueError("batch_rows must be between 1 and 64")
    if any(path.exists() for path in (SCORES, REPORT, RESULT_EVIDENCE)):
        raise FileExistsError("DDA-COCO result already exists; no retry")
    contract = json.loads(CONTRACT.read_text())
    contract_evidence = json.loads(CONTRACT_EVIDENCE.read_text())
    if (
        contract.get("state") != "e43_dda_coco_score_contract_frozen"
        or _digest(CONTRACT) != contract_evidence.get("detailed_contract_sha256")
        or contract.get("manifest_sha256") != MANIFEST_SHA256
    ):
        raise ValueError("DDA-COCO score contract changed")
    if _digest(MANIFEST) != MANIFEST_SHA256 or _digest(CANDIDATE) != CANDIDATE_SHA256:
        raise ValueError("DDA-COCO manifest or candidate changed")
    manifest = json.loads(MANIFEST.read_text())
    rows = manifest["rows"]
    if len(rows) != DECLARED_ROWS:
        raise ValueError("DDA-COCO declared row count changed")
    artifact = joblib.load(CANDIDATE)
    if float(artifact["threshold"]) != THRESHOLD:
        raise ValueError("E43-S artifact threshold changed")
    scored = _score_rows(rows, artifact["head"], batch_rows)
    metrics = evaluate_binary_scores(scored, threshold=THRESHOLD)
    rates = variant_rates(scored)
    by_variant = {
        variant: evaluate_binary_scores(
            [row for row in scored if int(row["label"]) == 0 or row["condition"] == variant],
            threshold=THRESHOLD,
        )
        for variant in VARIANTS
    }
    gate = final_gate(metrics, rates)
    score_bytes, score_sha256 = _write_jsonl(SCORES, scored)
    report = {
        "schema_version": 1,
        "state": "e43_dda_coco_passed_once" if gate["passed"] else "e43_dda_coco_failed_once",
        "contract_sha256": _digest(CONTRACT),
        "manifest_sha256": MANIFEST_SHA256,
        "candidate_sha256": CANDIDATE_SHA256,
        "threshold": THRESHOLD,
        "pooled_metrics": metrics,
        "rates": rates,
        "by_variant": by_variant,
        "gate": gate,
        "score_stream": {"rows": len(scored), "bytes": score_bytes, "sha256": score_sha256},
        "scope": "Independent aligned content/frequency benchmark; does not replace ITW-SM/NIST real-world evidence.",
        "decision": "Preserve this one-shot result without threshold tuning, row removal or retry.",
    }
    raw = _write(REPORT, report)
    evidence = {
        **report,
        "detailed_report_bytes": len(raw),
        "detailed_report_sha256": hashlib.sha256(raw).hexdigest(),
    }
    _write(RESULT_EVIDENCE, evidence)
    return evidence


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("bind-score", "evaluate"))
    parser.add_argument("--batch-rows", type=int, default=16)
    args = parser.parse_args(argv)
    result = bind_score_contract() if args.command == "bind-score" else evaluate(args.batch_rows)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
