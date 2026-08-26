"""E30/A4 — resumable, contract-frozen scoring of DEVELOPMENT rows only."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import numpy as np
import torch
from huggingface_hub import snapshot_download
from PIL import Image
from scipy.stats import beta
from sklearn.metrics import roc_auc_score

from pixelproof.data_contract import DataRecord, DataRole, load_manifest, sha256_bytes
from pixelproof.e31_candidate import E31Candidate
from pixelproof.project_model import load_project_model
from pixelproof.project_paths import DATA_ROOT, ML_ROOT
from pixelproof.verdict import CF_REVISION, CF_T_AI, CommunityForensicsArm


DATASET_DIR = DATA_ROOT / "e30/mllm_development"
MANIFEST_PATH = DATASET_DIR / "derived_manifest.json"
OUTPUT_DIR = DATA_ROOT / "e30/development_scores"
EXPECTED_CONTENT_SET = "7634755c75f855064e0d6c3c2731c6fe63d6582f2b6065654698adbf32e924b8"
EXPECTED_CF_WEIGHTS = "275ba982236ddd6afddf7131f8133e89f537574b964cf8fa5825b4956d741692"
CF_HUB_ID = "buildborderless/CommunityForensics-DeepfakeDet-ViT"
TRANSPORTS = (
    "standardized_jpeg",
    "jpeg_q90",
    "jpeg_q75",
    "jpeg_q50",
    "resize256_q90",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def exact_interval(successes: int, total: int, alpha: float = 0.05) -> list[float] | None:
    """Two-sided Clopper-Pearson interval."""
    if total <= 0 or not 0 <= successes <= total:
        return None
    lower = 0.0 if successes == 0 else float(beta.ppf(alpha / 2, successes, total - successes + 1))
    upper = 1.0 if successes == total else float(beta.ppf(1 - alpha / 2, successes + 1, total - successes))
    return [lower, upper]


def _rate(
    rows: Iterable[Mapping[str, Any]], *, independent: bool = True
) -> dict[str, Any]:
    values = list(rows)
    successes = sum(bool(row["predicted_ai"]) for row in values)
    total = len(values)
    result = {
        "n": total,
        "ai_triggers": successes,
        "rate": successes / total if total else None,
        "exact_95_ci": exact_interval(successes, total) if independent else None,
    }
    if not independent:
        result["interval_reason"] = "transport views repeat underlying parent content"
    return result


def summarize(
    rows: Iterable[Mapping[str, Any]],
    threshold: float,
    *,
    abstains_below_threshold: bool = True,
) -> dict[str, Any]:
    values = list(rows)
    failures = [row for row in values if row["status"] != "ok"]
    ok = [row for row in values if row["status"] == "ok"]
    labels = np.asarray([row["label"] == "ai" for row in ok], dtype=np.int64)
    scores = np.asarray([row["score"] for row in ok], dtype=np.float64)
    real = [row for row in ok if row["label"] == "real"]
    ai = [row for row in ok if row["label"] == "ai"]

    per_transport = {}
    for transport in TRANSPORTS:
        transport_rows = [row for row in ok if row["transport"] == transport]
        transport_real = [row for row in transport_rows if row["label"] == "real"]
        transport_ai = [row for row in transport_rows if row["label"] == "ai"]
        per_transport[transport] = {
            "real_false_positive": _rate(transport_real),
            "ai_recall": _rate(transport_ai),
            "roc_auc": (
                float(
                    roc_auc_score(
                        [row["label"] == "ai" for row in transport_rows],
                        [row["score"] for row in transport_rows],
                    )
                )
                if transport_real and transport_ai
                else None
            ),
        }

    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in ok:
        grouped[f"{row['transport']}::{row['group']}"].append(row)
    per_group = {name: _rate(group) for name, group in sorted(grouped.items())}

    by_generator: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    by_regime: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in ok:
        if row["label"] == "ai":
            by_generator[str(row["generator"])].append(row)
        regime = str(row["group"]).split(":", 1)[0]
        by_regime[f"{row['label']}::{regime}"].append(row)

    base_recall = per_transport["standardized_jpeg"]["ai_recall"]["rate"]
    deltas = {
        transport: (
            None
            if base_recall is None or metrics["ai_recall"]["rate"] is None
            else metrics["ai_recall"]["rate"] - base_recall
        )
        for transport, metrics in per_transport.items()
        if transport != "standardized_jpeg"
    }
    real_group_rates = [
        metrics["rate"]
        for name, metrics in per_group.items()
        if ":MLLMGenSet matched real" in name and metrics["rate"] is not None
    ]
    ai_group_rates = [
        metrics["rate"]
        for name, metrics in per_group.items()
        if ":MLLMGenSet matched real" not in name and metrics["rate"] is not None
    ]
    return {
        "threshold": threshold,
        "accounting": {"expected": len(values), "succeeded": len(ok), "failed": len(failures)},
        "overall": {
            "roc_auc": float(roc_auc_score(labels, scores)),
            "real_false_positive": _rate(real, independent=False),
            "ai_recall": _rate(ai, independent=False),
            "abstention_rate": 1.0 - _rate(ok, independent=False)["rate"] if abstains_below_threshold else 0.0,
        },
        "macro": {
            "real_false_positive": float(np.mean(real_group_rates)),
            "worst_real_group_false_positive": max(real_group_rates),
            "ai_recall": float(np.mean(ai_group_rates)),
            "worst_ai_group_recall": min(ai_group_rates),
        },
        "per_transport": per_transport,
        "per_generator": {
            name: _rate(group, independent=False)
            for name, group in sorted(by_generator.items())
        },
        "per_artifact_regime": {
            name: _rate(group, independent=False)
            for name, group in sorted(by_regime.items())
        },
        "recall_delta_vs_standardized": deltas,
        "per_transport_group": per_group,
        "failures": failures,
    }


def development_gate(summary: Mapping[str, Any]) -> dict[str, Any]:
    deltas = summary["recall_delta_vs_standardized"]
    checks = {
        "all_900_rows_scored": summary["accounting"]["failed"] == 0,
        "macro_real_fp_lte_0.05": summary["macro"]["real_false_positive"] <= 0.05,
        "worst_real_group_fp_lte_0.10": (
            summary["macro"]["worst_real_group_false_positive"] <= 0.10
        ),
        "macro_ai_recall_gte_0.50": summary["macro"]["ai_recall"] >= 0.50,
        "worst_ai_group_recall_gte_0.30": summary["macro"]["worst_ai_group_recall"] >= 0.30,
        "jpeg_q75_recall_loss_lte_0.15": deltas["jpeg_q75"] >= -0.15,
        "resize256_q90_recall_loss_lte_0.15": deltas["resize256_q90"] >= -0.15,
    }
    return {"passed": all(checks.values()), "checks": checks}


def _contract_sha(payload: Mapping[str, Any]) -> str:
    return sha256_bytes(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode())


def _device(name: str) -> torch.device:
    if name == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    selected = torch.device(name)
    if selected.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS requested but unavailable")
    if selected.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return selected


def _e20(device: torch.device) -> tuple[dict[str, Any], Callable[[Image.Image], float]]:
    model = load_project_model(ML_ROOT, device)
    contract = {
        "detector_id": "e20_project",
        **model.metadata.to_dict(),
        "max_tiles": 256,
        "decision": "binary AI/real at checkpoint threshold; no abstention",
    }
    return contract, lambda image: float(model.score_image(image, max_tiles=256).score)


def _cf(device: torch.device) -> tuple[dict[str, Any], Callable[[Image.Image], float]]:
    snapshot = Path(
        snapshot_download(CF_HUB_ID, revision=CF_REVISION, local_files_only=True)
    )
    weights_sha = sha256_file(snapshot / "model.safetensors")
    if weights_sha != EXPECTED_CF_WEIGHTS:
        raise RuntimeError("CF-ViT cached weights changed")
    arm = CommunityForensicsArm(device)
    contract = {
        "arm": "cf_vit",
        "source_id": CF_HUB_ID,
        "source_revision": CF_REVISION,
        "weights_sha256": weights_sha,
        "threshold": CF_T_AI,
        "preprocessing": "shortest edge 440, center crop 384, CLIP normalization",
        "decision": "ai at/above threshold; otherwise insufficient",
    }
    return contract, lambda image: float(arm.score(image))


def _e31(device: torch.device) -> tuple[dict[str, Any], Callable[[Image.Image, str], float]]:
    model = E31Candidate(device=device)
    return model.contract(), lambda image, content_key: float(
        model.score_image(image, content_key).score
    )


def _load_rows() -> tuple[dict[str, Any], list[DataRecord]]:
    manifest, rows = load_manifest(MANIFEST_PATH, require_hashes=True)
    if manifest.get("content_set_sha256") != EXPECTED_CONTENT_SET:
        raise RuntimeError("E30 development content set changed")
    if len(rows) != 900 or {row.role for row in rows} != {DataRole.DEVELOPMENT_TEST}:
        raise RuntimeError("E30 development manifest must contain 900 DEVELOPMENT rows")
    return manifest, rows


def _read_cache(path: Path, contract_sha: str) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    output = {}
    with path.open() as stream:
        for line in stream:
            row = json.loads(line)
            if row["contract_sha256"] != contract_sha:
                raise RuntimeError("score cache belongs to another model contract")
            if row["record_id"] in output:
                raise RuntimeError("duplicate score-cache record")
            output[row["record_id"]] = row
    return output


def run_arm(name: str, device: torch.device, rows: list[DataRecord]) -> dict[str, Any]:
    if name == "e20":
        contract, scorer = _e20(device)
    elif name == "cf_vit":
        contract, scorer = _cf(device)
    else:
        contract, scorer = _e31(device)
    contract_sha = _contract_sha(contract)
    cache_path = OUTPUT_DIR / f"{name}.jsonl"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cached = _read_cache(cache_path, contract_sha)
    print(f"{name}: {len(cached)}/900 cached", flush=True)
    with cache_path.open("a") as stream:
        for index, record in enumerate(rows, 1):
            if record.record_id in cached:
                continue
            path = DATASET_DIR / record.path
            raw = path.read_bytes()
            base = {
                "record_id": record.record_id,
                "parent_id": record.parent_id,
                "label": record.label,
                "group": record.group,
                "generator": record.generator,
                "transport": record.transport,
                "contract_sha256": contract_sha,
            }
            try:
                if sha256_bytes(raw) != record.sha256:
                    raise RuntimeError("image hash changed")
                with Image.open(path) as opened:
                    image = opened.convert("RGB")
                score = (
                    scorer(image, record.content_id or record.record_id)
                    if name == "dinov2_e31"
                    else scorer(image)
                )
                if not np.isfinite(score):
                    raise RuntimeError("non-finite score")
                result = {
                    **base,
                    "status": "ok",
                    "score": score,
                    "predicted_ai": bool(score >= float(contract["threshold"])),
                    "error": None,
                }
            except Exception as error:
                result = {
                    **base,
                    "status": "error",
                    "score": None,
                    "predicted_ai": False,
                    "error": f"{type(error).__name__}: {error}"[:300],
                }
            stream.write(json.dumps(result, separators=(",", ":")) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
            cached[record.record_id] = result
            if index % 20 == 0 or index == len(rows):
                print(f"{name}: {index}/900 accounted", flush=True)
    ordered = [cached[row.record_id] for row in rows]
    summary = summarize(
        ordered,
        float(contract["threshold"]),
        abstains_below_threshold=name in {"cf_vit", "dinov2_e31"},
    )
    return {
        "contract": contract,
        "contract_sha256": contract_sha,
        "summary": summary,
        "development_gate": development_gate(summary),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--arm", choices=("e20", "cf_vit", "dinov2_e31", "all"), default="all"
    )
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    args = parser.parse_args()
    manifest, rows = _load_rows()
    device = _device(args.device)
    arms = ("e20", "cf_vit", "dinov2_e31") if args.arm == "all" else (args.arm,)
    results = {
        "experiment": "E30-A4-development",
        "dataset_content_set_sha256": manifest["content_set_sha256"],
        "role": "development_test",
        "device": str(device),
        "shortcut_audits_by_transport": manifest["shortcut_audits_by_transport"],
        "arms": {name: run_arm(name, device, rows) for name in arms},
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    destination = OUTPUT_DIR / "results.json"
    temporary = destination.with_suffix(".json.part")
    temporary.write_text(json.dumps(results, indent=2) + "\n")
    temporary.replace(destination)
    compact = {
        name: {
            "contract_sha256": value["contract_sha256"],
            "overall": value["summary"]["overall"],
            "macro": value["summary"]["macro"],
            "recall_delta_vs_standardized": value["summary"]["recall_delta_vs_standardized"],
            "development_gate": value["development_gate"],
        }
        for name, value in results["arms"].items()
    }
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()
