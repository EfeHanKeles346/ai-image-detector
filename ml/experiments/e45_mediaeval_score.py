"""Bind, score and evaluate the untouched E45 MediaEval final exactly once."""

from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence
import zipfile

import joblib
import numpy as np
from PIL import Image

from experiments.e42_features import BLOCKS, MODEL_IDS, aggregate_tokens, texture_crops, transport_image
from experiments.e43_train import CANDIDATE as GENERALIST, _digest
from experiments.e44_fusion import CANDIDATE as FUSION, _feature
from pixelproof.benchmark_metrics import evaluate_binary_scores
from pixelproof.dda_candidate import CHECKPOINT_SHA256, OfficialDDACandidate
from pixelproof.e32_candidate import DINO_REPO_ID, DINO_WEIGHT_SHA256
from pixelproof.project_paths import DATA_ROOT, ML_ROOT


ROOT = DATA_ROOT / "e45_mediaeval_itwsm"
ARCHIVE = ROOT / "archives" / "itw-sm-sid-val.zip"
MANIFEST = ROOT / "unscored_manifest.json"
MANIFEST_SHA256 = "3e7c1d7e815a252d454d36c78f2a6ad6381983edb9494c31951bdb683c6d7e03"
CONTRACT = ROOT / "score_contract.json"
CONTRACT_EVIDENCE = ML_ROOT.parent / "evidence" / "e45_score_contract.json"
GENERALIST_SCORES = ROOT / "generalist_scores.jsonl"
GENERALIST_EVIDENCE = ML_ROOT.parent / "evidence" / "e45_generalist_scores.json"
SPECIALIST_SCORES = ROOT / "specialist_scores.jsonl"
SPECIALIST_EVIDENCE = ML_ROOT.parent / "evidence" / "e45_specialist_scores.json"
FUSED_SCORES = ROOT / "fused_scores.jsonl"
REPORT = ROOT / "final_report.json"
RESULT_EVIDENCE = ML_ROOT.parent / "evidence" / "e45_final_result.json"

GENERALIST_SHA256 = "a3aec445926bcc8707b3775f01d2cdd9491ba8495ad8a8ec306840556ca47390"
FUSION_SHA256 = "19fd7bbcfed6ea85b9aa0c620663880f9fed24fbdbb084b09057283ea38bb100"
BINARY_THRESHOLD = 0.34779336534869326
REAL_CUT = 0.254571216982131
AI_CUT = 0.6938513176357805
EXPECTED_ROWS = 9_978
PLATFORMS = ("Facebook", "Instagram", "LinkedIn", "X")
BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_SEED = 45
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


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> tuple[int, str]:
    raw = b"".join((json.dumps(row, sort_keys=True) + "\n").encode() for row in rows)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_bytes(raw)
    temporary.replace(path)
    return len(raw), hashlib.sha256(raw).hexdigest()


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _manifest_rows() -> list[dict[str, Any]]:
    if _digest(MANIFEST) != MANIFEST_SHA256:
        raise ValueError("E45 final manifest changed")
    payload = json.loads(MANIFEST.read_text())
    if payload.get("state") != "e45_mediaeval_decontaminated_manifest_frozen_unscored":
        raise ValueError("E45 final manifest state changed")
    rows = payload.get("rows", [])
    if len(rows) != EXPECTED_ROWS or len({row["record_id"] for row in rows}) != EXPECTED_ROWS:
        raise ValueError("E45 final manifest row count or uniqueness changed")
    return rows


def bind() -> dict[str, Any]:
    if CONTRACT.exists() or CONTRACT_EVIDENCE.exists():
        raise FileExistsError("E45 score contract already exists; no silent replacement")
    rows = _manifest_rows()
    checkpoint = DATA_ROOT / "e35_dda_model" / "DDA_ckpt.pth"
    identities = {
        "manifest_sha256": _digest(MANIFEST),
        "generalist_sha256": _digest(GENERALIST),
        "specialist_checkpoint_sha256": _digest(checkpoint),
        "fusion_sha256": _digest(FUSION),
    }
    expected = {
        "manifest_sha256": MANIFEST_SHA256,
        "generalist_sha256": GENERALIST_SHA256,
        "specialist_checkpoint_sha256": CHECKPOINT_SHA256,
        "fusion_sha256": FUSION_SHA256,
    }
    if identities != expected:
        raise ValueError(f"E45 candidate identity changed: {identities}")
    counts = {
        "rows": len(rows),
        "real": sum(int(row["label"]) == 0 for row in rows),
        "ai": sum(int(row["label"]) == 1 for row in rows),
        "by_platform": {
            platform: sum(str(row["platform"]) == platform for row in rows)
            for platform in PLATFORMS
        },
    }
    payload = {
        "schema_version": 1,
        "state": "e45_final_score_contract_frozen_before_model_load",
        "role": "UNTOUCHED_FINAL",
        "identities": identities,
        "counts": counts,
        "binary_threshold": BINARY_THRESHOLD,
        "selective_policy": {
            "real_if_score_lt": REAL_CUT,
            "ai_if_score_gte": AI_CUT,
            "otherwise": "uncertain",
        },
        "gates": GATES,
        "confidence_intervals": {
            "method": "platform-and-label-stratified row bootstrap",
            "samples": BOOTSTRAP_SAMPLES,
            "seed": BOOTSTRAP_SEED,
            "percentiles": [0.025, 0.975],
        },
        "score_arms": ["E43-S generalist", "official DDA specialist", "frozen E44 fusion"],
        "forbidden": [
            "threshold or policy change",
            "row removal after score",
            "test-informed refit",
            "completed-run retry",
            "training on E45",
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


def _validate_contract() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    contract = json.loads(CONTRACT.read_text())
    evidence = json.loads(CONTRACT_EVIDENCE.read_text())
    if (
        contract.get("state") != "e45_final_score_contract_frozen_before_model_load"
        or contract.get("model_scores_created") != 0
        or _digest(CONTRACT) != evidence.get("detailed_contract_sha256")
        or contract.get("identities", {}).get("manifest_sha256") != MANIFEST_SHA256
    ):
        raise ValueError("E45 final score contract changed")
    return contract, _manifest_rows()


def _resume_prefix(path: Path, rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    raw = path.read_bytes()
    if raw and not raw.endswith(b"\n"):
        raise ValueError(f"E45 partial score has an incomplete line: {path}")
    completed = [json.loads(line) for line in raw.splitlines() if line]
    if len(completed) > len(rows):
        raise ValueError("E45 partial score exceeds manifest")
    for index, scored in enumerate(completed):
        expected = rows[index]
        if (
            scored.get("record_id") != expected["record_id"]
            or int(scored.get("label", -1)) != int(expected["label"])
            or scored.get("platform") != expected["platform"]
            or not np.isfinite(float(scored.get("score", np.nan)))
        ):
            raise ValueError(f"E45 partial score prefix changed at row {index}")
    return completed


def _append_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab") as stream:
        for row in rows:
            stream.write((json.dumps(row, sort_keys=True) + "\n").encode())
        stream.flush()
        os.fsync(stream.fileno())


def _complete_arm(
    partial: Path, final: Path, evidence_path: Path, rows: Sequence[Mapping[str, Any]], arm: str
) -> dict[str, Any]:
    completed = _resume_prefix(partial, rows)
    if len(completed) != len(rows):
        raise ValueError(f"E45 {arm} score stream is incomplete")
    partial.replace(final)
    raw = final.read_bytes()
    result = {
        "schema_version": 1,
        "state": f"e45_{arm}_scores_complete",
        "rows": len(completed),
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "coverage": 1.0,
        "manifest_sha256": MANIFEST_SHA256,
    }
    _write(evidence_path, result)
    return result


def _payload(bundle: zipfile.ZipFile, row: Mapping[str, Any]) -> bytes:
    payload = bundle.read(str(row["member"]))
    if hashlib.sha256(payload).hexdigest() != str(row["sha256"]):
        raise ValueError(f"E45 ZIP member changed: {row['record_id']}")
    return payload


def _generalist_arrays(item: tuple[bytes, Mapping[str, Any]]) -> list[np.ndarray]:
    payload, row = item
    with Image.open(BytesIO(payload)) as opened:
        return texture_crops(transport_image(opened, "clean"))


def score_generalist(batch_rows: int = 16) -> dict[str, Any]:
    if GENERALIST_SCORES.exists() or GENERALIST_EVIDENCE.exists():
        raise FileExistsError("E45 generalist scores already completed")
    if batch_rows < 1 or batch_rows > 32:
        raise ValueError("E45 generalist batch must be 1..32")
    _, rows = _validate_contract()
    partial = GENERALIST_SCORES.with_suffix(GENERALIST_SCORES.suffix + ".partial")
    completed = _resume_prefix(partial, rows)
    artifact = joblib.load(GENERALIST)
    from huggingface_hub import snapshot_download
    import timm
    import torch

    snapshot = Path(snapshot_download(DINO_REPO_ID, local_files_only=True))
    if _digest(snapshot / "model.safetensors") != DINO_WEIGHT_SHA256:
        raise ValueError("cached DINOv2-S weights changed")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = timm.create_model(
        MODEL_IDS["small"], pretrained=True, num_classes=0, img_size=224
    ).to(device).eval()
    config = timm.data.resolve_data_config({}, model=model)
    mean = torch.tensor(config["mean"], device=device).view(1, 3, 1, 1)
    std = torch.tensor(config["std"], device=device).view(1, 3, 1, 1)
    with zipfile.ZipFile(ARCHIVE) as bundle, torch.inference_mode(), ThreadPoolExecutor(max_workers=6) as pool:
        for start in range(len(completed), len(rows), batch_rows):
            group = rows[start : start + batch_rows]
            items = [(_payload(bundle, row), row) for row in group]
            arrays = [array for pack in pool.map(_generalist_arrays, items) for array in pack]
            tensor = torch.from_numpy(np.stack(arrays)).to(device).permute(0, 3, 1, 2).float().div_(255.0)
            intermediates = model.forward_intermediates(
                (tensor - mean) / std,
                indices=list(BLOCKS["small"]),
                return_prefix_tokens=True,
                norm=True,
                intermediates_only=True,
            )
            tokens = torch.stack([item[1][:, 0, :] for item in intermediates], dim=1)
            features = aggregate_tokens(tokens.float().cpu().numpy(), len(group))
            values = artifact["head"].predict_proba(features)[:, 1]
            batch = [
                {
                    "record_id": row["record_id"],
                    "label": int(row["label"]),
                    "platform": row["platform"],
                    "score": float(value),
                    "status": "ok",
                }
                for row, value in zip(group, values, strict=True)
            ]
            _append_rows(partial, batch)
            done = min(start + len(group), len(rows))
            if done == len(rows) or done // 100 != start // 100:
                print(f"E45 generalist {done}/{len(rows)}", flush=True)
    return _complete_arm(partial, GENERALIST_SCORES, GENERALIST_EVIDENCE, rows, "generalist")


def score_specialist(batch_rows: int = 2) -> dict[str, Any]:
    if SPECIALIST_SCORES.exists() or SPECIALIST_EVIDENCE.exists():
        raise FileExistsError("E45 specialist scores already completed")
    if batch_rows < 1 or batch_rows > 8:
        raise ValueError("E45 specialist batch must be 1..8")
    _, rows = _validate_contract()
    partial = SPECIALIST_SCORES.with_suffix(SPECIALIST_SCORES.suffix + ".partial")
    completed = _resume_prefix(partial, rows)
    candidate = OfficialDDACandidate()
    import torch

    with zipfile.ZipFile(ARCHIVE) as bundle, torch.inference_mode():
        for start in range(len(completed), len(rows), batch_rows):
            group = rows[start : start + batch_rows]
            tensors = []
            for row in group:
                with Image.open(BytesIO(_payload(bundle, row))) as opened:
                    tensors.append(candidate.transform(opened.convert("RGB")))
            values = candidate.model(torch.stack(tensors).to(candidate.device)).sigmoid().flatten().cpu().numpy()
            batch = [
                {
                    "record_id": row["record_id"],
                    "label": int(row["label"]),
                    "platform": row["platform"],
                    "score": float(value),
                    "status": "ok",
                }
                for row, value in zip(group, values, strict=True)
            ]
            _append_rows(partial, batch)
            done = min(start + len(group), len(rows))
            if done == len(rows) or done // 100 != start // 100:
                print(f"E45 specialist {done}/{len(rows)}", flush=True)
    return _complete_arm(partial, SPECIALIST_SCORES, SPECIALIST_EVIDENCE, rows, "specialist")


def binary_platform_rates(rows: Sequence[Mapping[str, Any]], threshold: float) -> dict[str, Any]:
    real_by_platform: dict[str, list[bool]] = defaultdict(list)
    ai_by_platform: dict[str, list[bool]] = defaultdict(list)
    for row in rows:
        predicted_ai = float(row["score"]) >= threshold
        target = ai_by_platform if int(row["label"]) == 1 else real_by_platform
        target[str(row["platform"])].append(predicted_ai)
    if set(real_by_platform) != set(PLATFORMS) or set(ai_by_platform) != set(PLATFORMS):
        raise ValueError("E45 binary rates require both labels on every platform")
    real = {platform: float(np.mean(real_by_platform[platform])) for platform in PLATFORMS}
    ai = {platform: float(np.mean(ai_by_platform[platform])) for platform in PLATFORMS}
    pooled_real = [value for values in real_by_platform.values() for value in values]
    pooled_ai = [value for values in ai_by_platform.values() for value in values]
    return {
        "pooled_real_false_ai": float(np.mean(pooled_real)),
        "pooled_ai_recall": float(np.mean(pooled_ai)),
        "real_false_ai_by_platform": real,
        "ai_recall_by_platform": ai,
        "worst_real_false_ai": max(real.values()),
        "worst_ai_recall": min(ai.values()),
    }


def selective_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    decisions = []
    by_platform: dict[str, list[tuple[int, str]]] = defaultdict(list)
    for row in rows:
        score = float(row["score"])
        decision = "real" if score < REAL_CUT else "ai" if score >= AI_CUT else "uncertain"
        label = int(row["label"])
        decisions.append((label, decision))
        by_platform[str(row["platform"])].append((label, decision))
    automatic = [(label, decision) for label, decision in decisions if decision != "uncertain"]
    correct = sum((decision == "ai") == (label == 1) for label, decision in automatic)
    platform = {}
    for name in PLATFORMS:
        values = by_platform[name]
        auto = [(label, decision) for label, decision in values if decision != "uncertain"]
        platform[name] = {
            "automatic_coverage": len(auto) / len(values),
            "covered_accuracy": (
                sum((decision == "ai") == (label == 1) for label, decision in auto) / len(auto)
                if auto else None
            ),
            "uncertain_rate": sum(decision == "uncertain" for _, decision in values) / len(values),
            "real_false_ai": (
                sum(label == 0 and decision == "ai" for label, decision in values)
                / sum(label == 0 for label, _ in values)
            ),
            "ai_false_real": (
                sum(label == 1 and decision == "real" for label, decision in values)
                / sum(label == 1 for label, _ in values)
            ),
        }
    return {
        "automatic_coverage": len(automatic) / len(decisions),
        "covered_accuracy": correct / len(automatic) if automatic else None,
        "uncertain_rate": 1 - len(automatic) / len(decisions),
        "automatic_rows": len(automatic),
        "uncertain_rows": len(decisions) - len(automatic),
        "by_platform": platform,
    }


def bootstrap_primary(
    rows: Sequence[Mapping[str, Any]], samples: int = BOOTSTRAP_SAMPLES, seed: int = BOOTSTRAP_SEED
) -> dict[str, list[float]]:
    labels = np.asarray([int(row["label"]) for row in rows], dtype=np.int8)
    scores = np.asarray([float(row["score"]) for row in rows], dtype=np.float64)
    platforms = np.asarray([str(row["platform"]) for row in rows])
    binary = scores >= BINARY_THRESHOLD
    decisions = np.where(scores < REAL_CUT, 0, np.where(scores >= AI_CUT, 1, -1))
    order = np.argsort(scores, kind="stable")
    ordered_scores = scores[order]
    starts = np.r_[0, np.flatnonzero(np.diff(ordered_scores)) + 1]
    rng = np.random.default_rng(seed)
    output: dict[str, list[float]] = {
        key: [] for key in (
            "roc_auc", "balanced_accuracy", "real_false_ai", "ai_recall",
            "automatic_coverage", "covered_accuracy", "uncertain_rate",
        )
    }
    strata = [np.flatnonzero((platforms == platform) & (labels == label))
              for platform in PLATFORMS for label in (0, 1)]
    if any(len(indices) == 0 for indices in strata):
        raise ValueError("E45 bootstrap requires all platform/label strata")
    chunk_size = 100
    for offset in range(0, samples, chunk_size):
        chunk = min(chunk_size, samples - offset)
        counts = np.zeros((chunk, len(rows)), dtype=np.int16)
        for indices in strata:
            counts[:, indices] = rng.multinomial(
                len(indices), np.full(len(indices), 1 / len(indices)), size=chunk
            )
        real_n = counts[:, labels == 0].sum(axis=1)
        ai_n = counts[:, labels == 1].sum(axis=1)
        real_fp = (counts[:, labels == 0] * binary[labels == 0]).sum(axis=1) / real_n
        ai_recall = (counts[:, labels == 1] * binary[labels == 1]).sum(axis=1) / ai_n
        automatic_mask = decisions != -1
        correct_mask = decisions == labels
        automatic_n = (counts * automatic_mask).sum(axis=1)
        covered_correct = (counts * automatic_mask * correct_mask).sum(axis=1)

        ordered = counts[:, order]
        positive_groups = np.add.reduceat(ordered * (labels[order] == 1), starts, axis=1)
        negative_groups = np.add.reduceat(ordered * (labels[order] == 0), starts, axis=1)
        cumulative_negative = np.cumsum(negative_groups, axis=1)
        auc = (
            positive_groups * (cumulative_negative - negative_groups + 0.5 * negative_groups)
        ).sum(axis=1) / (real_n * ai_n)

        output["roc_auc"].extend(auc.tolist())
        output["balanced_accuracy"].extend(((1 - real_fp + ai_recall) / 2).tolist())
        output["real_false_ai"].extend(real_fp.tolist())
        output["ai_recall"].extend(ai_recall.tolist())
        output["automatic_coverage"].extend((automatic_n / len(rows)).tolist())
        output["covered_accuracy"].extend((covered_correct / automatic_n).tolist())
        output["uncertain_rate"].extend((1 - automatic_n / len(rows)).tolist())
    return {
        key: [float(value) for value in np.quantile(values, [0.025, 0.975])]
        for key, values in output.items()
    }


def evaluate() -> dict[str, Any]:
    if REPORT.exists() or RESULT_EVIDENCE.exists():
        raise FileExistsError("E45 final result already exists; no completed-run retry")
    contract, manifest_rows = _validate_contract()
    generalist = _load_jsonl(GENERALIST_SCORES)
    specialist = _load_jsonl(SPECIALIST_SCORES)
    for path, evidence_path, values in (
        (GENERALIST_SCORES, GENERALIST_EVIDENCE, generalist),
        (SPECIALIST_SCORES, SPECIALIST_EVIDENCE, specialist),
    ):
        evidence = json.loads(evidence_path.read_text())
        if _digest(path) != evidence.get("sha256") or len(values) != EXPECTED_ROWS:
            raise ValueError(f"E45 arm identity changed: {path}")
    generalist_by_id = {row["record_id"]: row for row in generalist}
    specialist_by_id = {row["record_id"]: row for row in specialist}
    if set(generalist_by_id) != set(specialist_by_id) or len(generalist_by_id) != EXPECTED_ROWS:
        raise ValueError("E45 score arms do not join one-to-one")
    artifact = joblib.load(FUSION)
    features = np.asarray([
        _feature(generalist_by_id[row["record_id"]]["score"], specialist_by_id[row["record_id"]]["score"])
        for row in manifest_rows
    ])
    values = artifact["head"].predict_proba(features)[:, 1]
    expected_fused = [
        {
            "record_id": row["record_id"],
            "label": int(row["label"]),
            "platform": row["platform"],
            "generalist_score": float(generalist_by_id[row["record_id"]]["score"]),
            "specialist_score": float(specialist_by_id[row["record_id"]]["score"]),
            "score": float(score),
            "status": "ok",
        }
        for row, score in zip(manifest_rows, values, strict=True)
    ]
    if FUSED_SCORES.exists():
        fused = _load_jsonl(FUSED_SCORES)
        if len(fused) != len(expected_fused):
            raise ValueError("E45 preserved fused stream count changed")
        for index, (found, expected) in enumerate(zip(fused, expected_fused, strict=True)):
            if found != expected:
                raise ValueError(f"E45 preserved fused stream changed at row {index}")
        raw_scores = FUSED_SCORES.read_bytes()
        score_bytes = len(raw_scores)
        score_sha256 = hashlib.sha256(raw_scores).hexdigest()
    else:
        fused = expected_fused
        score_bytes, score_sha256 = _write_jsonl(FUSED_SCORES, fused)
    metric_rows = [
        {**row, "source": row["platform"], "condition": "original"} for row in fused
    ]
    metrics = evaluate_binary_scores(metric_rows, threshold=BINARY_THRESHOLD)
    rates = binary_platform_rates(fused, BINARY_THRESHOLD)
    selective = selective_metrics(fused)
    intervals = bootstrap_primary(fused)
    checks = {
        "score_coverage_eq_1": float(metrics["coverage"]) == GATES["score_coverage"],
        "binary_auc_gte_0_90": float(metrics["roc_auc"]) >= GATES["binary_auc_min"],
        "binary_balanced_accuracy_gte_0_85": float(metrics["balanced_accuracy"]) >= GATES["binary_balanced_accuracy_min"],
        "pooled_real_false_ai_lte_0_10": rates["pooled_real_false_ai"] <= GATES["pooled_real_false_ai_max"],
        "pooled_ai_recall_gte_0_80": rates["pooled_ai_recall"] >= GATES["pooled_ai_recall_min"],
        "worst_platform_real_false_ai_lte_0_20": rates["worst_real_false_ai"] <= GATES["source_real_false_ai_worst_max"],
        "worst_platform_ai_recall_gte_0_60": rates["worst_ai_recall"] >= GATES["source_ai_recall_worst_min"],
        "automatic_coverage_gte_0_80": selective["automatic_coverage"] >= GATES["automatic_coverage_min"],
        "covered_accuracy_gte_0_95": selective["covered_accuracy"] >= GATES["covered_accuracy_min"],
        "uncertain_rate_lte_0_20": selective["uncertain_rate"] <= GATES["uncertain_rate_max"],
    }
    gate = {"passed": all(checks.values()), "checks": checks, "passed_checks": sum(checks.values()), "total_checks": len(checks)}
    report = {
        "schema_version": 1,
        "state": "e45_independent_final_passed" if gate["passed"] else "e45_independent_final_failed",
        "role": contract["role"],
        "contract_sha256": _digest(CONTRACT),
        "manifest_sha256": MANIFEST_SHA256,
        "candidate": {
            "generalist_sha256": GENERALIST_SHA256,
            "specialist_checkpoint_sha256": CHECKPOINT_SHA256,
            "fusion_sha256": FUSION_SHA256,
            "binary_threshold": BINARY_THRESHOLD,
            "selective_policy": contract["selective_policy"],
        },
        "counts": {"rows": len(fused), "official_rows": 10_000, "official_coverage": len(fused) / 10_000},
        "binary_metrics": metrics,
        "binary_rates": rates,
        "selective": selective,
        "bootstrap_95pct": intervals,
        "gate": gate,
        "score_stream": {"rows": len(fused), "bytes": score_bytes, "sha256": score_sha256},
        "boundary": "First and only E45 result; no threshold repair, row removal, refit or retry is permitted.",
    }
    raw = _write(REPORT, report)
    _write(RESULT_EVIDENCE, {**report, "detailed_report_bytes": len(raw), "detailed_report_sha256": hashlib.sha256(raw).hexdigest()})
    return report


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("bind", "score-generalist", "score-specialist", "evaluate"))
    parser.add_argument("--batch-size", type=int)
    args = parser.parse_args(argv)
    actions: dict[str, Callable[[], dict[str, Any]]] = {
        "bind": bind,
        "score-generalist": lambda: score_generalist(args.batch_size or 16),
        "score-specialist": lambda: score_specialist(args.batch_size or 2),
        "evaluate": evaluate,
    }
    print(json.dumps(actions[args.command](), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
