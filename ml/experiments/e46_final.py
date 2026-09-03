"""Bind and execute the one-shot E46 TrueFake Facebook final."""

from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import joblib
import numpy as np
from PIL import Image

from experiments.e42_features import BLOCKS, MODEL_IDS, aggregate_tokens, texture_crops, transport_image
from experiments.e43_train import CANDIDATE as GENERALIST, _digest
from experiments.e44_fusion import CANDIDATE as FUSION, _feature
from experiments.e46_calibration import CANDIDATE
from pixelproof.benchmark_metrics import evaluate_binary_scores
from pixelproof.dda_candidate import CHECKPOINT_SHA256, OfficialDDACandidate
from pixelproof.e32_candidate import DINO_REPO_ID, DINO_WEIGHT_SHA256
from pixelproof.project_paths import DATA_ROOT, ML_ROOT


ROOT = DATA_ROOT / "e46"
MANIFEST = ROOT / "truefake_facebook" / "final_manifest_unscored.json"
MANIFEST_SHA256 = "4572339ebe15821c6c86d50178ed31aa80f60cf98f1bd710d73e7265c15b225b"
CANDIDATE_SHA256 = "9fec91b8a28a842784562654ec49f74add63ce4c5d588d5170c75a4fd2bb84a1"
GENERALIST_SHA256 = "a3aec445926bcc8707b3775f01d2cdd9491ba8495ad8a8ec306840556ca47390"
FUSION_SHA256 = "19fd7bbcfed6ea85b9aa0c620663880f9fed24fbdbb084b09057283ea38bb100"
EXPECTED_ROWS = 2_000
BINARY_THRESHOLD = 0.6688565012954346
REAL_CUT = 0.5185430496088231
AI_CUT = BINARY_THRESHOLD
BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_SEED = 46

CONTRACT = ROOT / "truefake_final_score_contract.json"
CONTRACT_EVIDENCE = ML_ROOT.parent / "evidence" / "e46_final_score_contract.json"
GENERALIST_SCORES = ROOT / "truefake_generalist_scores.jsonl"
GENERALIST_EVIDENCE = ML_ROOT.parent / "evidence" / "e46_truefake_generalist_scores.json"
SPECIALIST_SCORES = ROOT / "truefake_specialist_scores.jsonl"
SPECIALIST_EVIDENCE = ML_ROOT.parent / "evidence" / "e46_truefake_specialist_scores.json"
FUSED_SCORES = ROOT / "truefake_fused_scores.jsonl"
REPORT = ROOT / "truefake_final_report.json"
RESULT_EVIDENCE = ML_ROOT.parent / "evidence" / "e46_final_result.json"

GATES = {
    "score_coverage": 1.0,
    "binary_auc_min": 0.90,
    "binary_balanced_accuracy_min": 0.85,
    "pooled_real_false_ai_max": 0.10,
    "pooled_ai_recall_min": 0.80,
    "source_real_false_ai_worst_max": 0.20,
    "source_ai_recall_worst_min": 0.60,
    "automatic_coverage_min": 0.80,
    "covered_accuracy_min": 0.95,
    "uncertain_rate_max": 0.20,
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


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _manifest_rows() -> list[dict[str, Any]]:
    raw = MANIFEST.read_bytes()
    if hashlib.sha256(raw).hexdigest() != MANIFEST_SHA256:
        raise ValueError("E46 TrueFake final manifest changed")
    payload = json.loads(raw)
    rows = payload.get("rows", [])
    if (
        payload.get("state") != "e46_truefake_decontaminated_final_frozen_unscored"
        or payload.get("model_scores_created") != 0
        or len(rows) != EXPECTED_ROWS
        or len({row["record_id"] for row in rows}) != EXPECTED_ROWS
    ):
        raise ValueError("E46 TrueFake final manifest contract changed")
    return rows


def bind() -> dict[str, Any]:
    if CONTRACT.exists() or CONTRACT_EVIDENCE.exists():
        raise FileExistsError("E46 final score contract already exists")
    rows = _manifest_rows()
    artifact = joblib.load(CANDIDATE)
    identities = {
        "manifest_sha256": _digest(MANIFEST),
        "candidate_sha256": _digest(CANDIDATE),
        "generalist_sha256": _digest(GENERALIST),
        "specialist_checkpoint_sha256": _digest(DATA_ROOT / "e35_dda_model" / "DDA_ckpt.pth"),
        "fusion_sha256": _digest(FUSION),
    }
    expected = {
        "manifest_sha256": MANIFEST_SHA256,
        "candidate_sha256": CANDIDATE_SHA256,
        "generalist_sha256": GENERALIST_SHA256,
        "specialist_checkpoint_sha256": CHECKPOINT_SHA256,
        "fusion_sha256": FUSION_SHA256,
    }
    if identities != expected:
        raise ValueError(f"E46 final identities changed: {identities}")
    if (
        artifact.get("selected_method") != "fusion_global"
        or float(artifact.get("threshold")) != BINARY_THRESHOLD
        or float(artifact["selective_policy"]["real_if_score_lt"]) != REAL_CUT
        or float(artifact["selective_policy"]["ai_if_score_gte"]) != AI_CUT
    ):
        raise ValueError("E46 candidate decision policy changed")
    payload = {
        "schema_version": 1,
        "state": "e46_truefake_final_score_contract_frozen_before_model_load",
        "role": "E46_UNTOUCHED_FINAL",
        "identities": identities,
        "counts": {
            "rows": len(rows), "real": sum(int(row["label"]) == 0 for row in rows),
            "ai": sum(int(row["label"]) == 1 for row in rows),
            "by_source": dict(sorted({source: sum(row["source"] == source for row in rows)
                                      for source in {row["source"] for row in rows}}.items())),
        },
        "selected_method": "fusion_global",
        "binary_threshold": BINARY_THRESHOLD,
        "selective_policy": {"real_if_score_lt": REAL_CUT,
                             "ai_if_score_gte": AI_CUT, "otherwise": "uncertain"},
        "gates": GATES,
        "confidence_intervals": {
            "method": "publisher-source-and-label-stratified row bootstrap",
            "samples": BOOTSTRAP_SAMPLES,
            "seed": BOOTSTRAP_SEED,
            "percentiles": [0.025, 0.975],
        },
        "forbidden": ["threshold/policy change", "row removal after score", "refit",
                      "completed-run retry", "training on TrueFake", "E45 repair"],
        "model_scores_created": 0,
    }
    raw = _write(CONTRACT, payload)
    evidence = {**payload, "contract_bytes": len(raw),
                "contract_sha256": hashlib.sha256(raw).hexdigest()}
    _write(CONTRACT_EVIDENCE, evidence)
    return evidence


def _validate_contract() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    raw = CONTRACT.read_bytes()
    contract = json.loads(raw)
    evidence = json.loads(CONTRACT_EVIDENCE.read_text())
    if (
        contract.get("state") != "e46_truefake_final_score_contract_frozen_before_model_load"
        or contract.get("model_scores_created") != 0
        or hashlib.sha256(raw).hexdigest() != evidence.get("contract_sha256")
    ):
        raise ValueError("E46 final score contract changed")
    return contract, _manifest_rows()


def _resume(path: Path, rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    raw = path.read_bytes()
    if raw and not raw.endswith(b"\n"):
        raise ValueError("incomplete E46 final score line")
    scored = [json.loads(line) for line in raw.splitlines() if line]
    if len(scored) > len(rows):
        raise ValueError("E46 final score prefix exceeds manifest")
    for index, item in enumerate(scored):
        expected = rows[index]
        if (
            item.get("record_id") != expected["record_id"]
            or item.get("source") != expected["source"]
            or int(item.get("label", -1)) != int(expected["label"])
            or not np.isfinite(float(item.get("score", np.nan)))
        ):
            raise ValueError(f"E46 final score prefix changed at {index}")
    return scored


def _append(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab") as stream:
        for row in rows:
            stream.write((json.dumps(row, sort_keys=True) + "\n").encode())
        stream.flush()
        os.fsync(stream.fileno())


def _finish(partial: Path, final: Path, evidence_path: Path, rows: Sequence[Mapping[str, Any]], arm: str) -> dict[str, Any]:
    scored = _resume(partial, rows)
    if len(scored) != len(rows):
        raise ValueError(f"E46 final {arm} score stream incomplete")
    partial.replace(final)
    raw = final.read_bytes()
    result = {
        "schema_version": 1, "state": f"e46_truefake_{arm}_scores_complete",
        "rows": len(rows), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest(),
        "coverage": 1.0, "manifest_sha256": MANIFEST_SHA256,
    }
    _write(evidence_path, result)
    return result


def _prepare(row: Mapping[str, Any]) -> list[np.ndarray]:
    path = Path(str(row["path"]))
    if not path.is_file() or _digest(path) != row["sha256"]:
        raise ValueError(f"E46 TrueFake payload changed: {row['record_id']}")
    with Image.open(path) as opened:
        return texture_crops(transport_image(opened, "clean"))


def score_generalist(batch_rows: int = 16) -> dict[str, Any]:
    if GENERALIST_SCORES.exists() or GENERALIST_EVIDENCE.exists():
        raise FileExistsError("E46 final generalist scores already complete")
    _, rows = _validate_contract()
    partial = GENERALIST_SCORES.with_suffix(GENERALIST_SCORES.suffix + ".partial")
    completed = _resume(partial, rows)
    artifact = joblib.load(GENERALIST)
    from huggingface_hub import snapshot_download
    import timm
    import torch

    snapshot = Path(snapshot_download(DINO_REPO_ID, local_files_only=True))
    if _digest(snapshot / "model.safetensors") != DINO_WEIGHT_SHA256:
        raise ValueError("cached DINOv2-S weights changed")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = timm.create_model(MODEL_IDS["small"], pretrained=True, num_classes=0, img_size=224).to(device).eval()
    config = timm.data.resolve_data_config({}, model=model)
    mean = torch.tensor(config["mean"], device=device).view(1, 3, 1, 1)
    std = torch.tensor(config["std"], device=device).view(1, 3, 1, 1)
    with torch.inference_mode(), ThreadPoolExecutor(max_workers=6) as pool:
        for start in range(len(completed), len(rows), batch_rows):
            group = rows[start:start + batch_rows]
            arrays = [array for pack in pool.map(_prepare, group) for array in pack]
            tensor = torch.from_numpy(np.stack(arrays)).to(device).permute(0, 3, 1, 2).float().div_(255.0)
            intermediate = model.forward_intermediates(
                (tensor - mean) / std, indices=list(BLOCKS["small"]),
                return_prefix_tokens=True, norm=True, intermediates_only=True,
            )
            tokens = torch.stack([item[1][:, 0, :] for item in intermediate], dim=1)
            features = aggregate_tokens(tokens.float().cpu().numpy(), len(group))
            scores = artifact["head"].predict_proba(features)[:, 1]
            batch = [{"record_id": row["record_id"], "label": int(row["label"]),
                      "source": row["source"], "score": float(score), "status": "ok"}
                     for row, score in zip(group, scores, strict=True)]
            _append(partial, batch)
            done = min(start + len(group), len(rows))
            if done == len(rows) or done // 100 != start // 100:
                print(f"E46 final generalist {done}/{len(rows)}", flush=True)
    return _finish(partial, GENERALIST_SCORES, GENERALIST_EVIDENCE, rows, "generalist")


def score_specialist(batch_rows: int = 8) -> dict[str, Any]:
    if SPECIALIST_SCORES.exists() or SPECIALIST_EVIDENCE.exists():
        raise FileExistsError("E46 final specialist scores already complete")
    _, rows = _validate_contract()
    partial = SPECIALIST_SCORES.with_suffix(SPECIALIST_SCORES.suffix + ".partial")
    completed = _resume(partial, rows)
    candidate = OfficialDDACandidate()
    import torch

    with torch.inference_mode():
        for start in range(len(completed), len(rows), batch_rows):
            group = rows[start:start + batch_rows]
            tensors = []
            for row in group:
                path = Path(row["path"])
                if _digest(path) != row["sha256"]:
                    raise ValueError(f"E46 TrueFake payload changed: {row['record_id']}")
                with Image.open(path) as opened:
                    tensors.append(candidate.transform(opened.convert("RGB")))
            scores = candidate.model(torch.stack(tensors).to(candidate.device)).sigmoid().flatten().cpu().numpy()
            batch = [{"record_id": row["record_id"], "label": int(row["label"]),
                      "source": row["source"], "score": float(score), "status": "ok"}
                     for row, score in zip(group, scores, strict=True)]
            _append(partial, batch)
            done = min(start + len(group), len(rows))
            if done == len(rows) or done // 100 != start // 100:
                print(f"E46 final specialist {done}/{len(rows)}", flush=True)
    return _finish(partial, SPECIALIST_SCORES, SPECIALIST_EVIDENCE, rows, "specialist")


def source_rates(rows: Sequence[Mapping[str, Any]], threshold: float) -> dict[str, Any]:
    real: dict[str, list[bool]] = defaultdict(list)
    ai: dict[str, list[bool]] = defaultdict(list)
    for row in rows:
        predicted = float(row["score"]) >= threshold
        (ai if int(row["label"]) == 1 else real)[str(row["source"])].append(predicted)
    real_rates = {source: float(np.mean(values)) for source, values in sorted(real.items())}
    ai_rates = {source: float(np.mean(values)) for source, values in sorted(ai.items())}
    return {
        "pooled_real_false_ai": float(np.mean([value for values in real.values() for value in values])),
        "pooled_ai_recall": float(np.mean([value for values in ai.values() for value in values])),
        "real_false_ai_by_source": real_rates,
        "ai_recall_by_source": ai_rates,
        "worst_real_false_ai": max(real_rates.values()),
        "worst_ai_recall": min(ai_rates.values()),
    }


def selective_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    labels = np.asarray([int(row["label"]) for row in rows])
    scores = np.asarray([float(row["score"]) for row in rows])
    decisions = np.where(scores < REAL_CUT, 0, np.where(scores >= AI_CUT, 1, -1))
    automatic = decisions >= 0
    return {
        "automatic_coverage": float(automatic.mean()),
        "covered_accuracy": float((decisions[automatic] == labels[automatic]).mean()),
        "uncertain_rate": float((~automatic).mean()),
        "automatic_rows": int(automatic.sum()),
        "uncertain_rows": int((~automatic).sum()),
    }


def bootstrap_primary(rows: Sequence[Mapping[str, Any]], samples: int = BOOTSTRAP_SAMPLES,
                      seed: int = BOOTSTRAP_SEED) -> dict[str, list[float]]:
    labels = np.asarray([int(row["label"]) for row in rows], dtype=np.int8)
    scores = np.asarray([float(row["score"]) for row in rows], dtype=np.float64)
    sources = np.asarray([str(row["source"]) for row in rows])
    binary = scores >= BINARY_THRESHOLD
    decisions = np.where(scores < REAL_CUT, 0, np.where(scores >= AI_CUT, 1, -1))
    order = np.argsort(scores, kind="stable")
    ordered_scores = scores[order]
    starts = np.r_[0, np.flatnonzero(np.diff(ordered_scores)) + 1]
    strata = [np.flatnonzero(sources == source) for source in sorted(set(sources))]
    if any(len(indices) == 0 for indices in strata):
        raise ValueError("E46 bootstrap stratum missing")
    rng = np.random.default_rng(seed)
    output = {key: [] for key in ("roc_auc", "balanced_accuracy", "real_false_ai", "ai_recall",
                                   "automatic_coverage", "covered_accuracy", "uncertain_rate")}
    for offset in range(0, samples, 100):
        chunk = min(100, samples - offset)
        counts = np.zeros((chunk, len(rows)), dtype=np.int16)
        for indices in strata:
            counts[:, indices] = rng.multinomial(len(indices), np.full(len(indices), 1 / len(indices)), size=chunk)
        real_n = counts[:, labels == 0].sum(axis=1)
        ai_n = counts[:, labels == 1].sum(axis=1)
        real_fp = (counts[:, labels == 0] * binary[labels == 0]).sum(axis=1) / real_n
        ai_recall = (counts[:, labels == 1] * binary[labels == 1]).sum(axis=1) / ai_n
        automatic = decisions != -1
        automatic_n = (counts * automatic).sum(axis=1)
        correct_n = (counts * automatic * (decisions == labels)).sum(axis=1)
        ordered = counts[:, order]
        pos = np.add.reduceat(ordered * (labels[order] == 1), starts, axis=1)
        neg = np.add.reduceat(ordered * (labels[order] == 0), starts, axis=1)
        cumulative_neg = np.cumsum(neg, axis=1)
        auc = (pos * (cumulative_neg - neg + 0.5 * neg)).sum(axis=1) / (real_n * ai_n)
        output["roc_auc"].extend(auc.tolist())
        output["balanced_accuracy"].extend(((1 - real_fp + ai_recall) / 2).tolist())
        output["real_false_ai"].extend(real_fp.tolist())
        output["ai_recall"].extend(ai_recall.tolist())
        output["automatic_coverage"].extend((automatic_n / len(rows)).tolist())
        output["covered_accuracy"].extend((correct_n / automatic_n).tolist())
        output["uncertain_rate"].extend((1 - automatic_n / len(rows)).tolist())
    return {key: [float(value) for value in np.quantile(values, [0.025, 0.975])]
            for key, values in output.items()}


def evaluate() -> dict[str, Any]:
    if REPORT.exists() or RESULT_EVIDENCE.exists():
        raise FileExistsError("E46 final result already exists; no retry")
    contract, manifest = _validate_contract()
    generalist = {row["record_id"]: row for row in _jsonl(GENERALIST_SCORES)}
    specialist = {row["record_id"]: row for row in _jsonl(SPECIALIST_SCORES)}
    if len(generalist) != EXPECTED_ROWS or set(generalist) != set(specialist):
        raise ValueError("E46 final score arms do not join")
    for path, evidence_path in ((GENERALIST_SCORES, GENERALIST_EVIDENCE),
                                (SPECIALIST_SCORES, SPECIALIST_EVIDENCE)):
        if _digest(path) != json.loads(evidence_path.read_text())["sha256"]:
            raise ValueError(f"E46 final arm changed: {path}")
    fusion = joblib.load(FUSION)
    features = np.asarray([_feature(generalist[row["record_id"]]["score"],
                                    specialist[row["record_id"]]["score"]) for row in manifest])
    values = fusion["head"].predict_proba(features)[:, 1]
    rows = [{
        "record_id": row["record_id"], "label": int(row["label"]), "source": row["source"],
        "condition": "Facebook", "generalist_score": float(generalist[row["record_id"]]["score"]),
        "specialist_score": float(specialist[row["record_id"]]["score"]),
        "score": float(score), "status": "ok",
    } for row, score in zip(manifest, values, strict=True)]
    score_raw = b"".join((json.dumps(row, sort_keys=True) + "\n").encode() for row in rows)
    temporary = FUSED_SCORES.with_suffix(FUSED_SCORES.suffix + ".part")
    temporary.write_bytes(score_raw)
    temporary.replace(FUSED_SCORES)
    metrics = evaluate_binary_scores(rows, threshold=BINARY_THRESHOLD)
    rates = source_rates(rows, BINARY_THRESHOLD)
    selective = selective_metrics(rows)
    intervals = bootstrap_primary(rows)
    checks = {
        "score_coverage_eq_1": float(metrics["coverage"]) == GATES["score_coverage"],
        "binary_auc_gte_0_90": metrics["roc_auc"] >= GATES["binary_auc_min"],
        "binary_balanced_accuracy_gte_0_85": metrics["balanced_accuracy"] >= GATES["binary_balanced_accuracy_min"],
        "pooled_real_false_ai_lte_0_10": rates["pooled_real_false_ai"] <= GATES["pooled_real_false_ai_max"],
        "pooled_ai_recall_gte_0_80": rates["pooled_ai_recall"] >= GATES["pooled_ai_recall_min"],
        "worst_source_real_false_ai_lte_0_20": rates["worst_real_false_ai"] <= GATES["source_real_false_ai_worst_max"],
        "worst_source_ai_recall_gte_0_60": rates["worst_ai_recall"] >= GATES["source_ai_recall_worst_min"],
        "automatic_coverage_gte_0_80": selective["automatic_coverage"] >= GATES["automatic_coverage_min"],
        "covered_accuracy_gte_0_95": selective["covered_accuracy"] >= GATES["covered_accuracy_min"],
        "uncertain_rate_lte_0_20": selective["uncertain_rate"] <= GATES["uncertain_rate_max"],
    }
    gate = {"passed": all(checks.values()), "passed_checks": sum(checks.values()),
            "total_checks": len(checks), "checks": checks}
    report = {
        "schema_version": 1,
        "state": "e46_independent_final_passed" if gate["passed"] else "e46_independent_final_failed",
        "role": contract["role"],
        "contract_sha256": _digest(CONTRACT),
        "manifest_sha256": MANIFEST_SHA256,
        "candidate": {"sha256": CANDIDATE_SHA256, "selected_method": "fusion_global",
                      "threshold": BINARY_THRESHOLD, "selective_policy": contract["selective_policy"]},
        "counts": {"rows": len(rows), "real": 1000, "ai": 1000},
        "binary_metrics": metrics,
        "binary_rates": rates,
        "selective": selective,
        "bootstrap_95pct": intervals,
        "gate": gate,
        "score_stream": {"rows": len(rows), "bytes": len(score_raw),
                         "sha256": hashlib.sha256(score_raw).hexdigest()},
        "boundary": "First and only E46 TrueFake result; no repair, row removal, refit or retry is permitted.",
    }
    raw = _write(REPORT, report)
    _write(RESULT_EVIDENCE, {**report, "report_bytes": len(raw),
                             "report_sha256": hashlib.sha256(raw).hexdigest()})
    return report


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("bind", "score-generalist", "score-specialist", "evaluate"))
    parser.add_argument("--batch-rows", type=int)
    args = parser.parse_args(argv)
    actions: dict[str, Callable[[], dict[str, Any]]] = {
        "bind": bind,
        "score-generalist": lambda: score_generalist(args.batch_rows or 16),
        "score-specialist": lambda: score_specialist(args.batch_rows or 8),
        "evaluate": evaluate,
    }
    print(json.dumps(actions[args.command](), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
