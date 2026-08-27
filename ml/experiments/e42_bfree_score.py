"""Score frozen E41 once on the parent-grouped B-Free viral stress set."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
from PIL import Image

from pixelproof.benchmark_metrics import evaluate_binary_scores
from pixelproof.e32_candidate import DINO_MODEL_ID, DINO_REPO_ID, DINO_WEIGHT_SHA256, standardized_array
from pixelproof.project_paths import DATA_ROOT, ML_ROOT


ROOT = DATA_ROOT / "e42_external" / "bfree_viral"
MANIFEST = ROOT / "unscored_manifest.json"
MANIFEST_SHA256 = "338a2f2b2135a4bbfcb8ce0ceef7da5d8cbe2a5b1ffbe745c0e05a1248f37ca2"
CANDIDATE = DATA_ROOT / "e41" / "e41_dinov2s.joblib"
CANDIDATE_SHA256 = "9bcc021e74b617ee48cf297bd384a8dbe946240ec04822323af1e7c3fe63ab65"
THRESHOLD = 0.6195540428161622
SCORES = ROOT / "e41_scores.jsonl"
REPORT = ROOT / "e41_report.json"
EVIDENCE = ML_ROOT.parent / "evidence" / "e42_bfree_result.json"


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024**2):
            digest.update(chunk)
    return digest.hexdigest()


def _write_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def parent_weighted_summary(rows: Sequence[Mapping[str, Any]], threshold: float) -> dict[str, Any]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["parent_id"]), []).append(row)
    event_rows = []
    event_decision_rates: dict[str, float] = {}
    for parent_id, versions in sorted(grouped.items()):
        labels = {int(row["label"]) for row in versions}
        if len(labels) != 1:
            raise ValueError(f"parent event crosses labels: {parent_id}")
        label = labels.pop()
        scores = np.asarray([float(row["score"]) for row in versions], dtype=np.float64)
        if not np.isfinite(scores).all():
            raise ValueError(f"parent event contains non-finite scores: {parent_id}")
        decision_rate = float(np.mean(scores >= threshold))
        event_decision_rates[parent_id] = decision_rate
        event_rows.append({
            "parent_id": parent_id,
            "source": parent_id,
            "condition": "mean_across_web_versions",
            "label": label,
            "score": float(scores.mean()),
            "status": "ok",
            "version_count": len(versions),
            "ai_decision_rate": decision_rate,
        })
    real_rates = [row["ai_decision_rate"] for row in event_rows if row["label"] == 0]
    ai_rates = [row["ai_decision_rate"] for row in event_rows if row["label"] == 1]
    if not real_rates or not ai_rates:
        raise ValueError("parent-weighted stress metrics need both labels")
    real_fp = float(np.mean(real_rates))
    ai_recall = float(np.mean(ai_rates))
    return {
        "event_rows": event_rows,
        "event_metrics": evaluate_binary_scores(event_rows, threshold=threshold),
        "real_parent_weighted_fp": real_fp,
        "real_parent_weighted_recall": 1.0 - real_fp,
        "ai_parent_weighted_recall": ai_recall,
        "parent_weighted_balanced_accuracy": ((1.0 - real_fp) + ai_recall) / 2.0,
        "event_decision_rates": event_decision_rates,
    }


def stress_gate(summary: Mapping[str, Any]) -> dict[str, Any]:
    checks = {
        "parent_weighted_balanced_accuracy_gte_0_80": summary["parent_weighted_balanced_accuracy"] >= 0.80,
        "real_parent_weighted_recall_gte_0_75": summary["real_parent_weighted_recall"] >= 0.75,
        "ai_parent_weighted_recall_gte_0_75": summary["ai_parent_weighted_recall"] >= 0.75,
    }
    return {"passed": all(checks.values()), "checks": checks}


def _bootstrap(rows: Sequence[Mapping[str, Any]], *, iterations: int = 10_000, seed: int = 42) -> dict[str, list[float]]:
    grouped: dict[int, list[float]] = {0: [], 1: []}
    for row in rows:
        grouped[int(row["label"])].append(float(row["ai_decision_rate"]))
    rng = np.random.default_rng(seed)
    balanced, real_recall, ai_recall = [], [], []
    for _ in range(iterations):
        real = np.asarray(grouped[0])
        ai = np.asarray(grouped[1])
        sampled_real = real[rng.integers(0, len(real), len(real))]
        sampled_ai = ai[rng.integers(0, len(ai), len(ai))]
        rr = 1.0 - float(sampled_real.mean())
        ar = float(sampled_ai.mean())
        real_recall.append(rr)
        ai_recall.append(ar)
        balanced.append((rr + ar) / 2.0)
    percentile = lambda values: [float(value) for value in np.percentile(values, [2.5, 97.5])]  # noqa: E731
    return {
        "parent_weighted_balanced_accuracy_95ci": percentile(balanced),
        "real_parent_weighted_recall_95ci": percentile(real_recall),
        "ai_parent_weighted_recall_95ci": percentile(ai_recall),
    }


def _load() -> tuple[list[dict[str, Any]], Any]:
    if _digest(MANIFEST) != MANIFEST_SHA256:
        raise ValueError("B-Free unscored manifest changed")
    manifest = json.loads(MANIFEST.read_text())
    if (
        manifest.get("state") != "wild_stress_manifest_frozen_unscored"
        or manifest.get("candidate_sha256") != CANDIDATE_SHA256
        or manifest.get("counts", {}).get("source_events") != 34
    ):
        raise ValueError("B-Free manifest role/candidate contract changed")
    if _digest(CANDIDATE) != CANDIDATE_SHA256:
        raise ValueError("E41 candidate changed")
    artifact = joblib.load(CANDIDATE)
    if (
        artifact.get("model_id") != DINO_MODEL_ID
        or artifact.get("model_weight_sha256") != DINO_WEIGHT_SHA256
        or artifact.get("positive_label") != "ai"
        or float(artifact.get("threshold", -1)) != THRESHOLD
    ):
        raise ValueError("E41 inference contract changed")
    return manifest["rows"], artifact["head"]


def _score(rows: Sequence[Mapping[str, Any]], head: Any, batch_size: int) -> list[dict[str, Any]]:
    from huggingface_hub import snapshot_download
    import timm
    import torch

    snapshot = Path(snapshot_download(DINO_REPO_ID, local_files_only=True))
    if _digest(snapshot / "model.safetensors") != DINO_WEIGHT_SHA256:
        raise ValueError("cached DINOv2-S weights changed")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = timm.create_model(DINO_MODEL_ID, pretrained=True, num_classes=0, img_size=224).to(device).eval()
    config = timm.data.resolve_data_config({}, model=model)
    mean = torch.tensor(config["mean"], device=device).view(1, 3, 1, 1)
    std = torch.tensor(config["std"], device=device).view(1, 3, 1, 1)
    output = []
    with torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            group = rows[start : start + batch_size]
            arrays = []
            for row in group:
                path = Path(str(row["path"]))
                if _digest(path) != row["sha256"]:
                    raise ValueError(f"B-Free input changed: {row['relative_path']}")
                with Image.open(path) as image:
                    arrays.append(standardized_array(image))
            tensor = torch.from_numpy(np.stack(arrays)).to(device)
            tensor = tensor.permute(0, 3, 1, 2).float().div_(255.0)
            embeddings = model((tensor - mean) / std).float().cpu().numpy()
            scores = head.predict_proba(embeddings)[:, 1]
            for row, score in zip(group, scores, strict=True):
                output.append({
                    "record_id": row["record_id"],
                    "parent_id": row["parent_id"],
                    "relative_path": row["relative_path"],
                    "label": int(row["label"]),
                    "source": row["source_id"],
                    "condition": row["condition"],
                    "score": float(score),
                    "status": "ok",
                })
            print(f"B-Free E41 {min(start + batch_size, len(rows))}/{len(rows)}", flush=True)
    return output


def run(batch_size: int = 48) -> dict[str, Any]:
    if any(path.exists() for path in (SCORES, REPORT, EVIDENCE)):
        raise FileExistsError("B-Free E41 result already exists; retry is forbidden")
    rows, head = _load()
    scored = _score(rows, head, batch_size)
    raw_scores = b"".join((json.dumps(row, sort_keys=True) + "\n").encode() for row in scored)
    temporary = SCORES.with_suffix(SCORES.suffix + ".part")
    temporary.write_bytes(raw_scores)
    temporary.replace(SCORES)
    version_metrics = evaluate_binary_scores(scored, threshold=THRESHOLD)
    parent = parent_weighted_summary(scored, THRESHOLD)
    gate = stress_gate(parent)
    report = {
        "schema_version": 1,
        "experiment": "E41/B-Free-viral-external-stress",
        "state": "wild_stress_passed" if gate["passed"] else "wild_stress_failed",
        "candidate_sha256": CANDIDATE_SHA256,
        "manifest_sha256": MANIFEST_SHA256,
        "scores_sha256": hashlib.sha256(raw_scores).hexdigest(),
        "scores_bytes": len(raw_scores),
        "threshold": THRESHOLD,
        "counts": {"versions": len(scored), "source_events": len(parent["event_rows"])},
        "version_weighted_metrics_diagnostic": version_metrics,
        "parent_event_metrics": parent["event_metrics"],
        "parent_weighted_decisions": {
            key: parent[key]
            for key in (
                "real_parent_weighted_fp", "real_parent_weighted_recall",
                "ai_parent_weighted_recall", "parent_weighted_balanced_accuracy",
            )
        },
        "bootstrap": _bootstrap(parent["event_rows"]),
        "gate": gate,
        "limitations": [
            "811/1111 URL versions were recoverable; all 34 source events are represented.",
            "Images date through 2024, so this is web-propagation stress rather than a current-generator final.",
            "Confidence intervals are wide because effective independent N is 17 events per class.",
        ],
        "boundary": "First and only E41 score on the frozen B-Free manifest; no retry, threshold change or tuning.",
    }
    _write_atomic(REPORT, report)
    _write_atomic(EVIDENCE, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=48)
    args = parser.parse_args()
    print(json.dumps(run(args.batch_size), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
