"""Run E40's preregistered grouped transport and owner-gallery DEVELOPMENT gate."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import io
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, Sequence

import joblib
import numpy as np
from PIL import Image, ImageOps

from pixelproof.benchmark_metrics import evaluate_binary_scores
from pixelproof.e32_candidate import (
    DINO_MODEL_ID,
    DINO_REPO_ID,
    DINO_WEIGHT_SHA256,
    image_paths,
    standardized_array,
)
from pixelproof.project_paths import DATA_ROOT, ML_ROOT


EXPERIMENTS_DIR = Path(__file__).resolve().parent
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))

import e37_source_heldout as e37  # noqa: E402


REPO_ROOT = ML_ROOT.parent
E39_ROOT = DATA_ROOT / "e39"
E40_ROOT = DATA_ROOT / "e40"
E39_MANIFEST = E39_ROOT / "final_manifest.json"
E39_MANIFEST_SHA256 = "1076df20adc700a1893814e477891e9270e4f88502dbad2571e022f3a7127306"
DEVELOPMENT_REPORT = E40_ROOT / "development_report.json"
DEVELOPMENT_REPORT_SHA256 = "93f98f07663855c9fc918406c93fad8ca27b7434e29019ec88afd49279c2ebcc"
DRAFT = E40_ROOT / "e40_head_draft.joblib"
DRAFT_SHA256 = "72b8d8cd05f0f4d51ab5008e19bf4ab3418988b3444b2e640a25fdd0e54819c0"
CONTRACT = REPO_ROOT / "evidence" / "e40_robustness_contract.json"
CONTRACT_SHA256 = "acc8408c5a24d1899437d37c3c8e4f4769ab06455483acbbe891b4103ba6a4de"

SCORES = E40_ROOT / "robustness_scores.jsonl"
REPORT = E40_ROOT / "robustness_report.json"
EVIDENCE = REPO_ROOT / "evidence" / "e40_robustness.json"
CANDIDATE = E40_ROOT / "e40_candidate.joblib"

EXPECTED_OWNER_IDENTITY = "390e3c210ee61d70252d7e4714b8640463f44d57760942d25a1bdf7eab5aac09"
OWNER_COUNT = 210
RESERVE = {
    "name": "WhatsApp Image 2026-08-25 at 17.14.51.jpeg",
    "bytes": 206_418,
    "sha256": "e04755bfa5ef63da5536dc10395bd3f1faf2f79a6e304bc4eeba87a5e4ec57e3",
}
THRESHOLD = 0.17080099880695346
TRANSPORTS = ("native", "jpeg_q50", "resize75_q50")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def _write_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_bytes(_json_bytes(value))
    temporary.replace(path)


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> tuple[int, str]:
    raw = b"".join((json.dumps(row, sort_keys=True) + "\n").encode() for row in rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_bytes(raw)
    temporary.replace(path)
    return len(raw), hashlib.sha256(raw).hexdigest()


def _summary(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "min": float(array.min()),
        "p25": float(np.quantile(array, 0.25)),
        "median": float(np.median(array)),
        "p75": float(np.quantile(array, 0.75)),
        "p95": float(np.quantile(array, 0.95)),
        "max": float(array.max()),
        "mean": float(array.mean()),
    }


def transport_image(image: Image.Image, transport: str) -> Image.Image:
    """Create the fixed robustness view; the candidate preprocessing runs afterward."""
    oriented = ImageOps.exif_transpose(image).convert("RGB")
    if transport == "native":
        return oriented.copy()
    if transport == "resize75_q50":
        width = max(1, round(oriented.width * 0.75))
        height = max(1, round(oriented.height * 0.75))
        long_side = max(width, height)
        if long_side > 2_048:
            scale = 2_048 / long_side
            width, height = max(1, round(width * scale)), max(1, round(height * scale))
        oriented = oriented.resize((width, height), Image.Resampling.LANCZOS)
    elif transport != "jpeg_q50":
        raise ValueError(f"unknown transport: {transport}")
    encoded = io.BytesIO()
    oriented.save(
        encoded,
        format="JPEG",
        quality=50,
        subsampling=2,
        optimize=False,
        progressive=False,
    )
    with Image.open(io.BytesIO(encoded.getvalue())) as decoded:
        return decoded.convert("RGB").copy()


def robustness_gate(
    transports: Mapping[str, Mapping[str, Any]],
    agreements: Mapping[str, Mapping[str, float]],
    owner_fp: float,
) -> dict[str, Any]:
    checks: dict[str, bool] = {"owner_gallery_fp_lte_0.20": owner_fp <= 0.20 + 1e-12}
    for name in ("jpeg_q50", "resize75_q50"):
        result = transports[name]
        metrics = result["metrics"]
        rates = result["source_rates"]
        checks.update({
            f"{name}_coverage_eq_1": float(metrics["coverage"]) == 1.0,
            f"{name}_auc_gte_0.85": float(metrics["roc_auc"]) >= 0.85,
            f"{name}_tpr_at_fpr10_gte_0.80": float(metrics["tpr_at_fpr"]["tpr"]) >= 0.80,
            f"{name}_balanced_accuracy_gte_0.80": float(metrics["balanced_accuracy"]) >= 0.80,
            f"{name}_real_macro_fp_lte_0.20": float(rates["real_macro_fp"]) <= 0.20 + 1e-12,
            f"{name}_real_worst_fp_lte_0.30": float(rates["real_worst_device_fp"]) <= 0.30 + 1e-12,
            f"{name}_ai_macro_recall_gte_0.80": float(rates["ai_macro_recall"]) >= 0.80,
            f"{name}_ai_worst_recall_gte_0.60": float(rates["ai_worst_family_recall"]) >= 0.60,
            f"{name}_real_agreement_gte_0.80": float(agreements[name]["real"]) >= 0.80,
            f"{name}_ai_agreement_gte_0.80": float(agreements[name]["ai"]) >= 0.80,
        })
    for name in TRANSPORTS:
        summary = transports[name]["score_summary"]
        checks[f"{name}_mean_ai_gt_real"] = float(summary["ai"]["mean"]) > float(summary["real"]["mean"])
    return {"passed": all(checks.values()), "checks": checks}


def _owner_paths(folder: Path) -> tuple[list[Path], str, dict[str, Any] | None]:
    discovered = image_paths([str(folder)])
    reserve = [
        path for path in discovered
        if path.name == RESERVE["name"]
        and path.stat().st_size == RESERVE["bytes"]
        and _sha256_file(path) == RESERVE["sha256"]
    ]
    if len(discovered) == 211 and len(reserve) == 1:
        selected = [path for path in discovered if path != reserve[0]]
    elif len(discovered) == 210 and not reserve:
        selected = discovered
    else:
        raise ValueError("owner gallery has an undeclared membership change")
    identity = [
        {"name": path.name, "bytes": path.stat().st_size, "sha256": _sha256_file(path)}
        for path in selected
    ]
    identity_sha = hashlib.sha256(_json_bytes(identity)).hexdigest()
    if len(selected) != OWNER_COUNT or identity_sha != EXPECTED_OWNER_IDENTITY:
        raise ValueError("owner gallery identity changed")
    return selected, identity_sha, RESERVE if reserve else None


class DraftScorer:
    def __init__(self) -> None:
        from huggingface_hub import snapshot_download
        import timm
        import torch

        if _sha256_file(DRAFT) != DRAFT_SHA256:
            raise ValueError("E40 draft changed")
        artifact = joblib.load(DRAFT)
        if artifact.get("model_id") != DINO_MODEL_ID or float(artifact.get("threshold")) != THRESHOLD:
            raise ValueError("unexpected E40 draft contract")
        snapshot = Path(snapshot_download(DINO_REPO_ID, local_files_only=True))
        if _sha256_file(snapshot / "model.safetensors") != DINO_WEIGHT_SHA256:
            raise ValueError("cached DINOv2-S weights changed")
        self.torch = torch
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.model = timm.create_model(
            DINO_MODEL_ID, pretrained=True, num_classes=0, img_size=224
        ).to(self.device).eval()
        config = timm.data.resolve_data_config({}, model=self.model)
        self.mean = torch.tensor(config["mean"], device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor(config["std"], device=self.device).view(1, 3, 1, 1)
        self.head = artifact["head"]

    def score_arrays(self, arrays: Sequence[np.ndarray], batch_size: int) -> list[float]:
        scores: list[float] = []
        with self.torch.inference_mode():
            for start in range(0, len(arrays), batch_size):
                batch = np.stack(arrays[start : start + batch_size])
                tensor = self.torch.from_numpy(batch).to(self.device)
                tensor = tensor.permute(0, 3, 1, 2).float().div_(255.0)
                features = self.model((tensor - self.mean) / self.std).float().cpu().numpy()
                scores.extend(float(value) for value in self.head.predict_proba(features)[:, 1])
        return scores


def run(owner_gallery: Path, batch_size: int = 48) -> dict[str, Any]:
    if any(path.exists() for path in (SCORES, REPORT, EVIDENCE, CANDIDATE)):
        raise FileExistsError("E40 robustness output already exists; no silent rerun")
    contract_sha = _sha256_file(CONTRACT)
    if contract_sha != CONTRACT_SHA256:
        raise ValueError("E40 robustness contract changed")
    contract = json.loads(CONTRACT.read_text())
    if contract.get("state") != "fixed_before_e40_robustness_scores":
        raise ValueError("E40 robustness contract changed state")
    if _sha256_file(E39_MANIFEST) != E39_MANIFEST_SHA256:
        raise ValueError("E39 manifest changed")
    if _sha256_file(DEVELOPMENT_REPORT) != DEVELOPMENT_REPORT_SHA256:
        raise ValueError("E40 development report changed")
    manifest = json.loads(E39_MANIFEST.read_text())
    rows = manifest["rows"]
    owner_paths, owner_identity, excluded_reserve = _owner_paths(owner_gallery)
    scorer = DraftScorer()

    score_rows: list[dict[str, Any]] = []
    by_transport: dict[str, list[dict[str, Any]]] = {}
    for transport in TRANSPORTS:
        arrays = []
        for row in rows:
            path = E39_ROOT / str(row["path"])
            if _sha256_file(path) != str(row["sha256"]):
                raise ValueError(f"E39 image binding changed: {row['parent_id']}")
            with Image.open(path) as image:
                arrays.append(standardized_array(transport_image(image, transport)))
        scores = scorer.score_arrays(arrays, batch_size)
        transport_rows = [
            {
                "population": "e39_parent_linked",
                "parent_id": row["parent_id"],
                "label": int(row["label"]),
                "source": row["source"],
                "condition": row["condition"],
                "transport": transport,
                "status": "ok",
                "score": score,
            }
            for row, score in zip(rows, scores, strict=True)
        ]
        by_transport[transport] = transport_rows
        score_rows.extend(transport_rows)
        print(f"E40 robustness {transport} {len(transport_rows)}/{len(rows)}", flush=True)

    owner_arrays = []
    for path in owner_paths:
        with Image.open(path) as image:
            owner_arrays.append(standardized_array(transport_image(image, "native")))
    owner_scores = scorer.score_arrays(owner_arrays, batch_size)
    owner_rows = [
        {
            "population": "owner_gallery",
            "parent_id": path.name,
            "label": 0,
            "source": "owner_gallery",
            "condition": "native",
            "transport": "native",
            "status": "ok",
            "score": score,
        }
        for path, score in zip(owner_paths, owner_scores, strict=True)
    ]
    score_rows.extend(owner_rows)
    print(f"E40 robustness owner {len(owner_rows)}/{len(owner_paths)}", flush=True)
    scores_bytes, scores_sha = _write_jsonl(SCORES, score_rows)

    native_decisions = {
        str(row["parent_id"]): float(row["score"]) >= THRESHOLD
        for row in by_transport["native"]
    }
    agreements: dict[str, dict[str, float]] = {}
    transport_results: dict[str, Any] = {}
    for transport, transport_rows in by_transport.items():
        rates = e37.source_rates(transport_rows, THRESHOLD)
        metrics = evaluate_binary_scores(transport_rows, threshold=THRESHOLD)
        agreements[transport] = {
            name: float(np.mean([
                (float(row["score"]) >= THRESHOLD) == native_decisions[str(row["parent_id"])]
                for row in transport_rows if int(row["label"]) == label
            ]))
            for name, label in (("real", 0), ("ai", 1))
        }
        transport_results[transport] = {
            "parent_count": len(transport_rows),
            "metrics": metrics,
            "source_rates": rates,
            "score_summary": {
                name: _summary([float(row["score"]) for row in transport_rows if int(row["label"]) == label])
                for name, label in (("real", 0), ("ai", 1))
            },
        }
    owner_fp = float(np.mean([score >= THRESHOLD for score in owner_scores]))
    result_gate = robustness_gate(transport_results, agreements, owner_fp)
    report = {
        "schema_version": 1,
        "experiment": "E40/grouped-transport-and-owner-robustness",
        "state": "robustness_passed_candidate_packaged" if result_gate["passed"] else "robustness_failed",
        "bindings": {
            "contract_sha256": contract_sha,
            "development_report_sha256": DEVELOPMENT_REPORT_SHA256,
            "draft_sha256": DRAFT_SHA256,
            "e39_manifest_sha256": E39_MANIFEST_SHA256,
            "scores_sha256": scores_sha,
        },
        "counts": {
            "unique_e39_parents": len(rows),
            "parent_linked_views": sum(len(group) for group in by_transport.values()),
            "owner_gallery": len(owner_rows),
        },
        "threshold": THRESHOLD,
        "transports": transport_results,
        "decision_agreement_with_native": agreements,
        "owner_gallery": {
            "false_positive_rate": owner_fp,
            "real_recall": 1.0 - owner_fp,
            "score_summary": _summary(owner_scores),
            "identity_sha256": owner_identity,
            "excluded_unscored_reserve": excluded_reserve,
        },
        "gate": result_gate,
        "scores": {"bytes": scores_bytes, "sha256": scores_sha},
        "boundary": "Consumed DEVELOPMENT robustness only; derivatives share 440 parents and owner gallery cannot validate or tune E40.",
    }
    if result_gate["passed"]:
        artifact = joblib.load(DRAFT)
        artifact.update({
            "model_name": "E40 DINOv2-S source-held-out adaptation research candidate",
            "status": "research_candidate_awaiting_independent_final",
            "bindings": {
                "e40_development_report_sha256": DEVELOPMENT_REPORT_SHA256,
                "e40_robustness_contract_sha256": contract_sha,
                "e40_robustness_scores_sha256": scores_sha,
            },
        })
        temporary = CANDIDATE.with_suffix(".joblib.part")
        joblib.dump(artifact, temporary)
        temporary.replace(CANDIDATE)
        report["candidate"] = {
            "path": CANDIDATE.relative_to(E40_ROOT).as_posix(),
            "bytes": CANDIDATE.stat().st_size,
            "sha256": _sha256_file(CANDIDATE),
            "status": artifact["status"],
        }
    _write_atomic(REPORT, report)
    _write_atomic(EVIDENCE, report)
    return report


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("owner_gallery", type=Path)
    parser.add_argument("--batch-size", type=int, default=48)
    args = parser.parse_args(argv)
    result = run(args.owner_gallery.resolve(), args.batch_size)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
