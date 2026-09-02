"""Screen the pinned official DDA checkpoint on a score-blind consumed DDA sample."""

from __future__ import annotations

import argparse
from io import BytesIO
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
import zipfile

import numpy as np
from PIL import Image

from experiments.e43_dda_coco import COCO_ARCHIVE, DDA_ARCHIVE, MANIFEST, VARIANTS
from pixelproof.benchmark_metrics import evaluate_binary_scores
from pixelproof.dda_candidate import CHECKPOINT_SHA256, OfficialDDACandidate, THRESHOLD
from pixelproof.project_paths import DATA_ROOT, ML_ROOT


ROOT = DATA_ROOT / "e44"
MANIFEST_SHA256 = "e663d679f86ba69a545659203e11528d8998c9a362198a19f5f269a1ef97a3db"
SCREEN_PARENTS = 700
SCREEN_ROWS = SCREEN_PARENTS * 7
CONTRACT = ROOT / "dda_specialist_contract.json"
CONTRACT_EVIDENCE = ML_ROOT.parent / "evidence" / "e44_dda_specialist_contract.json"
SCORES = ROOT / "dda_specialist_scores.jsonl"
REPORT = ROOT / "dda_specialist_report.json"
RESULT_EVIDENCE = ML_ROOT.parent / "evidence" / "e44_dda_specialist_result.json"
CORE_VARIANTS = (
    "sd-vae-ft-ema",
    "sd-vae-ft-mse",
    "sdxl-vae",
    "stable-diffusion-2-1",
)
GATES = {
    "coverage": 1.0,
    "roc_auc_min": 0.85,
    "balanced_accuracy_min": 0.80,
    "real_false_positive_rate_max": 0.20,
    "core_variant_macro_recall_min": 0.80,
    "all_variant_macro_recall_min": 0.70,
    "worst_variant_recall_min": 0.40,
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


def select_parent_ids(rows: Sequence[Mapping[str, Any]], count: int = SCREEN_PARENTS) -> list[str]:
    parents = sorted({str(row["parent_id"]) for row in rows})
    if len(parents) < count:
        raise ValueError("insufficient complete DDA parents for E44 screen")
    ranked = sorted(
        parents,
        key=lambda parent: (hashlib.sha256(f"E44_DDA_SCREEN|{parent}".encode()).digest(), parent),
    )
    return sorted(ranked[:count])


def selected_rows(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    if manifest.get("state") != "e43_dda_coco_manifest_frozen_unscored":
        raise ValueError("E44 requires the exact frozen DDA manifest")
    rows = manifest.get("rows", [])
    selected = set(select_parent_ids(rows))
    output = [dict(row) for row in rows if str(row["parent_id"]) in selected]
    if len(output) != SCREEN_ROWS or len({str(row["parent_id"]) for row in output}) != SCREEN_PARENTS:
        raise ValueError("E44 DDA screen is not complete by parent")
    expected = {"REAL", *VARIANTS}
    for parent in selected:
        conditions = {str(row["condition"]) for row in output if row["parent_id"] == parent}
        if conditions != expected:
            raise ValueError(f"E44 DDA screen parent is incomplete: {parent}")
    return sorted(output, key=lambda row: str(row["record_id"]))


def bind_contract() -> dict[str, Any]:
    if CONTRACT.exists() or CONTRACT_EVIDENCE.exists():
        raise FileExistsError("E44 DDA specialist contract already exists")
    if _digest(MANIFEST) != MANIFEST_SHA256:
        raise ValueError("consumed DDA manifest changed")
    checkpoint = DATA_ROOT / "e35_dda_model" / "DDA_ckpt.pth"
    if _digest(checkpoint) != CHECKPOINT_SHA256:
        raise ValueError("official DDA checkpoint changed")
    rows = selected_rows(json.loads(MANIFEST.read_text()))
    parent_ids = sorted({str(row["parent_id"]) for row in rows})
    payload = {
        "schema_version": 1,
        "experiment": "E44-A/official-DDA-consumed-specialist-screen",
        "state": "e44_dda_specialist_contract_frozen",
        "role": "consumed_development_comparison_not_final",
        "manifest_sha256": MANIFEST_SHA256,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "threshold": THRESHOLD,
        "declared_parents": SCREEN_PARENTS,
        "declared_rows": SCREEN_ROWS,
        "parent_ids_sha256": hashlib.sha256(("\n".join(parent_ids) + "\n").encode()).hexdigest(),
        "conditions": ["REAL", *VARIANTS],
        "gates": GATES,
        "forbidden": ["parent reselection", "threshold change", "test-informed fitting", "final claim"],
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


def rates(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    real = [float(row["score"]) >= THRESHOLD for row in rows if int(row["label"]) == 0]
    by_variant = {
        variant: [
            float(row["score"]) >= THRESHOLD
            for row in rows
            if int(row["label"]) == 1 and str(row["condition"]) == variant
        ]
        for variant in VARIANTS
    }
    if not real or any(not values for values in by_variant.values()):
        raise ValueError("E44 DDA rates require REAL and all variants")
    recall = {variant: float(np.mean(values)) for variant, values in by_variant.items()}
    return {
        "real_false_positive_rate": float(np.mean(real)),
        "ai_recall_by_variant": recall,
        "core_variant_macro_recall": float(np.mean([recall[name] for name in CORE_VARIANTS])),
        "all_variant_macro_recall": float(np.mean(list(recall.values()))),
        "worst_variant_recall": min(recall.values()),
    }


def screen_gate(metrics: Mapping[str, Any], found: Mapping[str, Any]) -> dict[str, Any]:
    checks = {
        "coverage_eq_1": float(metrics["coverage"]) == GATES["coverage"],
        "roc_auc_gte_0_85": float(metrics["roc_auc"]) >= GATES["roc_auc_min"],
        "balanced_accuracy_gte_0_80": (
            float(metrics["balanced_accuracy"]) >= GATES["balanced_accuracy_min"]
        ),
        "real_fp_lte_0_20": (
            float(found["real_false_positive_rate"])
            <= GATES["real_false_positive_rate_max"] + 1e-12
        ),
        "core_macro_recall_gte_0_80": (
            float(found["core_variant_macro_recall"])
            >= GATES["core_variant_macro_recall_min"]
        ),
        "all_macro_recall_gte_0_70": (
            float(found["all_variant_macro_recall"])
            >= GATES["all_variant_macro_recall_min"]
        ),
        "worst_recall_gte_0_40": (
            float(found["worst_variant_recall"]) >= GATES["worst_variant_recall_min"]
        ),
    }
    return {"passed": all(checks.values()), "checks": checks}


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> tuple[int, str]:
    raw = b"".join((json.dumps(row, sort_keys=True) + "\n").encode() for row in rows)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_bytes(raw)
    temporary.replace(path)
    return len(raw), hashlib.sha256(raw).hexdigest()


def evaluate(batch_size: int = 2) -> dict[str, Any]:
    if batch_size < 1 or batch_size > 8:
        raise ValueError("batch_size must be between 1 and 8")
    if any(path.exists() for path in (SCORES, REPORT, RESULT_EVIDENCE)):
        raise FileExistsError("E44 DDA specialist result already exists")
    contract = json.loads(CONTRACT.read_text())
    evidence = json.loads(CONTRACT_EVIDENCE.read_text())
    if (
        contract.get("state") != "e44_dda_specialist_contract_frozen"
        or _digest(CONTRACT) != evidence.get("detailed_contract_sha256")
        or contract.get("threshold") != THRESHOLD
    ):
        raise ValueError("E44 DDA specialist contract changed")
    rows = selected_rows(json.loads(MANIFEST.read_text()))
    candidate = OfficialDDACandidate()
    import torch

    output: list[dict[str, Any]] = []
    with zipfile.ZipFile(COCO_ARCHIVE) as real_bundle, zipfile.ZipFile(DDA_ARCHIVE) as synthetic_bundle:
        with torch.inference_mode():
            for start in range(0, len(rows), batch_size):
                group = rows[start : start + batch_size]
                tensors = []
                for row in group:
                    bundle = real_bundle if str(row["archive"]) == "COCO-val2017" else synthetic_bundle
                    payload = bundle.read(str(row["member"]))
                    if hashlib.sha256(payload).hexdigest() != str(row["sha256"]):
                        raise ValueError("E44 DDA member changed after manifest")
                    with Image.open(BytesIO(payload)) as image:
                        tensors.append(candidate.transform(image.convert("RGB")))
                values = candidate.model(torch.stack(tensors).to(candidate.device)).sigmoid().flatten()
                for row, value in zip(group, values.cpu().numpy(), strict=True):
                    score = float(value)
                    if not np.isfinite(score):
                        raise ValueError("non-finite official DDA specialist score")
                    output.append({
                        "record_id": row["record_id"],
                        "parent_id": row["parent_id"],
                        "label": int(row["label"]),
                        "source": "COCO-val2017" if int(row["label"]) == 0 else str(row["condition"]),
                        "condition": str(row["condition"]),
                        "score": score,
                        "status": "ok",
                    })
                completed = min(start + batch_size, len(rows))
                if completed == len(rows) or completed // 100 != start // 100:
                    print(f"E44 DDA specialist {completed}/{len(rows)}", flush=True)

    metrics = evaluate_binary_scores(output, threshold=THRESHOLD)
    found_rates = rates(output)
    by_variant = {
        variant: evaluate_binary_scores(
            [row for row in output if int(row["label"]) == 0 or row["condition"] == variant],
            threshold=THRESHOLD,
        )
        for variant in VARIANTS
    }
    gate = screen_gate(metrics, found_rates)
    score_bytes, score_sha256 = _write_jsonl(SCORES, output)
    report = {
        "schema_version": 1,
        "state": "e44_dda_specialist_screen_passed" if gate["passed"] else "e44_dda_specialist_screen_failed",
        "role": contract["role"],
        "contract_sha256": _digest(CONTRACT),
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "threshold": THRESHOLD,
        "metrics": metrics,
        "rates": found_rates,
        "by_variant": by_variant,
        "gate": gate,
        "score_stream": {"rows": len(output), "bytes": score_bytes, "sha256": score_sha256},
        "next": (
            "adapt official DDA representation with real-camera safeguards"
            if gate["passed"]
            else "construct separate full DDA-aligned TRAIN pairs"
        ),
        "boundary": "Consumed comparative DEVELOPMENT only; cannot validate or promote a detector.",
    }
    raw = _write(REPORT, report)
    tracked = {
        **report,
        "detailed_report_bytes": len(raw),
        "detailed_report_sha256": hashlib.sha256(raw).hexdigest(),
    }
    _write(RESULT_EVIDENCE, tracked)
    return tracked


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("bind", "evaluate"))
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args(argv)
    result = bind_contract() if args.command == "bind" else evaluate(args.batch_size)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
