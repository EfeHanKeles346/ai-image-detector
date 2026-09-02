"""Build a conservative E43-S + official-DDA two-score fusion on consumed data."""

from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import joblib
import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from experiments.e42_features import BLOCKS, MODEL_IDS, aggregate_tokens, texture_crops, transport_image
from experiments.e43_dda_score import SCORES as E43_DDA_SCORES
from experiments.e43_train import CANDIDATE as GENERALIST, _digest
from experiments.e44_dda_specialist import SCORES as E44_DDA_SCORES, VARIANTS
from pixelproof.benchmark_metrics import evaluate_binary_scores
from pixelproof.e32_candidate import DINO_REPO_ID, DINO_WEIGHT_SHA256
from pixelproof.project_paths import DATA_ROOT, ML_ROOT


ROOT = DATA_ROOT / "e44"
E35_ROOT = DATA_ROOT / "e35_dda_model"
E35_SCORES = E35_ROOT / "development_scores.jsonl"
E35_REPORT = E35_ROOT / "development_report.json"
E43_DDA_REPORT = DATA_ROOT / "e43_dda_coco" / "report.json"
E44_DDA_REPORT = ROOT / "dda_specialist_report.json"
GENERALIST_SHA256 = "a3aec445926bcc8707b3775f01d2cdd9491ba8495ad8a8ec306840556ca47390"
CONTRACT = ROOT / "fusion_contract.json"
CONTRACT_EVIDENCE = ML_ROOT.parent / "evidence" / "e44_fusion_contract.json"
GENERALIST_SCORES = ROOT / "e35_generalist_scores.jsonl"
GENERALIST_EVIDENCE = ML_ROOT.parent / "evidence" / "e44_fusion_joint_scores.json"
CANDIDATE = ROOT / "e44_fusion.joblib"
CANDIDATE_REPORT = ROOT / "fusion_candidate.json"
CANDIDATE_EVIDENCE = ML_ROOT.parent / "evidence" / "e44_fusion_candidate.json"
DEVELOPMENT_REPORT = ROOT / "fusion_development.json"
DEVELOPMENT_SCORES = ROOT / "fusion_development_scores.jsonl"
DEVELOPMENT_EVIDENCE = ML_ROOT.parent / "evidence" / "e44_fusion_development.json"
OWNER_GALLERY_DEFAULT = Path("/Users/efehankeles/Desktop/PixelProof Workspace/Samples/fotoğraf galeri")
EPSILON = 1e-5
SEED = 42
C_VALUE = 0.1


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def _write(path: Path, value: Any) -> bytes:
    raw = _json_bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_bytes(raw)
    temporary.replace(path)
    return raw


def _jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> tuple[int, str]:
    raw = b"".join((json.dumps(row, sort_keys=True) + "\n").encode() for row in rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_bytes(raw)
    temporary.replace(path)
    return len(raw), hashlib.sha256(raw).hexdigest()


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _rank(values: Sequence[str], namespace: str) -> list[str]:
    return sorted(
        set(values),
        key=lambda value: (hashlib.sha256(f"{namespace}|{value}".encode()).digest(), value),
    )


def split_exact(values: Sequence[str], namespace: str, fit: int, calibration: int) -> dict[str, str]:
    ranked = _rank(values, namespace)
    if fit < 1 or calibration < 1 or fit + calibration >= len(ranked):
        raise ValueError("invalid exact role allocation")
    return {
        value: "fit" if index < fit else "calibration" if index < fit + calibration else "development"
        for index, value in enumerate(ranked)
    }


def split_fractional(values: Sequence[str], namespace: str) -> dict[str, str]:
    ranked = _rank(values, namespace)
    if len(ranked) < 5:
        raise ValueError("fractional split requires at least five identities")
    fit = max(1, int(np.floor(0.60 * len(ranked))))
    calibration = max(1, int(np.floor(0.20 * len(ranked))))
    if fit + calibration >= len(ranked):
        calibration = 1
        fit = len(ranked) - 2
    return {
        value: "fit" if index < fit else "calibration" if index < fit + calibration else "development"
        for index, value in enumerate(ranked)
    }


def role_maps(e44_rows: Sequence[Mapping[str, Any]], e35_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    dda_parents = sorted({str(row["parent_id"]) for row in e44_rows})
    dda_roles = split_exact(dda_parents, "E44_FUSION_DDA", 400, 150)
    rr_roles: dict[str, str] = {}
    rr = [row for row in e35_rows if row["population"] == "rr"]
    groups: dict[tuple[int, str], list[str]] = defaultdict(list)
    for row in rr:
        groups[(int(row["label"]), str(row["source"]))].append(str(row["parent_id"]))
    for (label, source), parents in sorted(groups.items()):
        rr_roles.update(split_fractional(parents, f"E44_FUSION_RR|{label}|{source}"))
    devices = sorted({str(row["source"]) for row in e35_rows if row["population"] == "ipn"})
    ipn_roles = split_exact(devices, "E44_FUSION_IPN_DEVICE", 7, 2)
    return {"dda_parent": dda_roles, "rr_parent": rr_roles, "ipn_device": ipn_roles}


def _resolve_e35(row: Mapping[str, Any], owner_gallery: Path) -> Path:
    population = str(row["population"])
    if population == "rr":
        return DATA_ROOT / "e33_rrdataset" / str(row["path"])
    if population == "ipn":
        return DATA_ROOT / "e32" / str(row["path"])
    if population == "owner":
        return owner_gallery / str(row["path"])
    raise ValueError(f"unsupported E35 population: {population}")


def _validate_stream(path: Path, report: Path, key: str = "scores_sha256") -> list[dict[str, Any]]:
    payload = json.loads(report.read_text())
    expected = payload["score_stream"]["sha256"] if key == "score_stream" else payload[key]
    if _digest(path) != expected:
        raise ValueError(f"score stream changed: {path}")
    return _load_jsonl(path)


def bind(owner_gallery: Path = OWNER_GALLERY_DEFAULT) -> dict[str, Any]:
    if CONTRACT.exists() or CONTRACT_EVIDENCE.exists():
        raise FileExistsError("E44 fusion contract already exists")
    if _digest(GENERALIST) != GENERALIST_SHA256:
        raise ValueError("E43-S candidate changed")
    e35 = _validate_stream(E35_SCORES, E35_REPORT)
    e43_dda = _validate_stream(E43_DDA_SCORES, E43_DDA_REPORT, "score_stream")
    e44_dda = _validate_stream(E44_DDA_SCORES, E44_DDA_REPORT, "score_stream")
    e43_by_id = {str(row["record_id"]): row for row in e43_dda}
    if len(e35) != 1_670 or len(e44_dda) != 4_900:
        raise ValueError("E44 fusion source count changed")
    if any(str(row["record_id"]) not in e43_by_id for row in e44_dda):
        raise ValueError("E44 specialist rows do not join the E43 DDA stream")
    roles = role_maps(e44_dda, e35)
    identity_rows = []
    for row in e35:
        path = _resolve_e35(row, owner_gallery)
        if not path.is_file():
            raise FileNotFoundError(path)
        identity_rows.append({
            "record_id": f"e35:{row['population']}:{row['path']}",
            "population": row["population"],
            "path": str(path),
            "sha256": _digest(path),
            "label": int(row["label"]),
            "source": str(row["source"]),
            "condition": str(row["condition"]),
            "parent_id": str(row.get("parent_id", row["path"])),
            "role": (
                roles["rr_parent"][str(row["parent_id"])] if row["population"] == "rr"
                else roles["ipn_device"][str(row["source"])] if row["population"] == "ipn"
                else "development"
            ),
        })
    dda_role_rows = [
        {"record_id": str(row["record_id"]), "parent_id": str(row["parent_id"]),
         "role": roles["dda_parent"][str(row["parent_id"])]}
        for row in e44_dda
    ]
    payload = {
        "schema_version": 1,
        "state": "e44_fusion_contract_frozen_before_missing_joint_scores",
        "role": "consumed_development_only_not_final",
        "generalist_sha256": GENERALIST_SHA256,
        "inputs": {
            "e35_dda_scores_sha256": _digest(E35_SCORES),
            "e43_dda_scores_sha256": _digest(E43_DDA_SCORES),
            "e44_dda_scores_sha256": _digest(E44_DDA_SCORES),
        },
        "method": {"features": "clamped logits of E43-S and official-DDA probabilities",
                   "head": "StandardScaler + LogisticRegression", "C": C_VALUE, "seed": SEED},
        "e35_rows": identity_rows,
        "dda_roles": dda_role_rows,
        "forbidden": ["source/device inference feature", "owner fit", "role change after score",
                      "post-development threshold repair", "final claim"],
        "model_scores_created": 0,
    }
    raw = _write(CONTRACT, payload)
    summary = {
        "schema_version": 1, "state": payload["state"], "role": payload["role"],
        "generalist_sha256": GENERALIST_SHA256, "inputs": payload["inputs"],
        "counts": {"e35": len(identity_rows), "dda": len(dda_role_rows)},
        "e35_identity_sha256": hashlib.sha256(_json_bytes(identity_rows)).hexdigest(),
        "dda_roles_sha256": hashlib.sha256(_json_bytes(dda_role_rows)).hexdigest(),
        "detailed_contract_bytes": len(raw), "detailed_contract_sha256": hashlib.sha256(raw).hexdigest(),
        "model_scores_created": 0, "forbidden": payload["forbidden"],
    }
    _write(CONTRACT_EVIDENCE, summary)
    return summary


def _prepare(row: Mapping[str, Any]) -> list[np.ndarray]:
    path = Path(str(row["path"]))
    if _digest(path) != str(row["sha256"]):
        raise ValueError(f"E44 fusion input changed: {row['record_id']}")
    with Image.open(path) as opened:
        return texture_crops(transport_image(opened, "clean"))


def score_generalist(batch_rows: int = 16) -> dict[str, Any]:
    if GENERALIST_SCORES.exists() or GENERALIST_EVIDENCE.exists():
        raise FileExistsError("E44 fusion generalist scores already exist")
    contract = json.loads(CONTRACT.read_text())
    evidence = json.loads(CONTRACT_EVIDENCE.read_text())
    if _digest(CONTRACT) != evidence["detailed_contract_sha256"] or contract["model_scores_created"] != 0:
        raise ValueError("E44 fusion contract changed")
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
    output = []
    rows = contract["e35_rows"]
    with torch.inference_mode(), ThreadPoolExecutor(max_workers=6) as pool:
        for start in range(0, len(rows), batch_rows):
            group = rows[start:start + batch_rows]
            arrays = [array for pack in pool.map(_prepare, group) for array in pack]
            tensor = torch.from_numpy(np.stack(arrays)).to(device).permute(0, 3, 1, 2).float().div_(255.0)
            intermediates = model.forward_intermediates(
                (tensor - mean) / std, indices=list(BLOCKS["small"]),
                return_prefix_tokens=True, norm=True, intermediates_only=True,
            )
            tokens = torch.stack([item[1][:, 0, :] for item in intermediates], dim=1)
            features = aggregate_tokens(tokens.float().cpu().numpy(), len(group))
            scores = artifact["head"].predict_proba(features)[:, 1]
            for row, score in zip(group, scores, strict=True):
                output.append({**{key: row[key] for key in (
                    "record_id", "population", "parent_id", "label", "source", "condition", "role")},
                    "score": float(score), "status": "ok"})
            print(f"E44 fusion generalist {min(start + batch_rows, len(rows))}/{len(rows)}", flush=True)
    size, digest = _jsonl(GENERALIST_SCORES, output)
    result = {"schema_version": 1, "state": "e44_fusion_joint_scores_complete",
              "rows": len(output), "bytes": size, "sha256": digest, "coverage": 1.0}
    _write(GENERALIST_EVIDENCE, result)
    return result


def _feature(generalist: float, specialist: float) -> list[float]:
    values = np.clip(np.asarray([generalist, specialist], dtype=np.float64), EPSILON, 1 - EPSILON)
    return np.log(values / (1 - values)).tolist()


def source_label_weights(rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
    groups: dict[tuple[int, str, str], list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        groups[(int(row["label"]), str(row["population"]), str(row["source"]))].append(index)
    weights = np.zeros(len(rows), dtype=np.float64)
    for label in (0, 1):
        label_groups = [indices for key, indices in groups.items() if key[0] == label]
        for indices in label_groups:
            weights[indices] = 1.0 / (2 * len(label_groups) * len(indices))
    if not np.all(weights > 0):
        raise ValueError("fusion fit requires both labels and valid source groups")
    return weights * (len(weights) / weights.sum())


def _joined_rows() -> list[dict[str, Any]]:
    contract = json.loads(CONTRACT.read_text())
    e35_dda = _load_jsonl(E35_SCORES)
    e35_gen = {row["record_id"]: row for row in _load_jsonl(GENERALIST_SCORES)}
    e35_contract = {row["record_id"]: row for row in contract["e35_rows"]}
    output = []
    for row in e35_dda:
        record_id = f"e35:{row['population']}:{row['path']}"
        bound = e35_contract[record_id]
        output.append({**{key: bound[key] for key in (
            "record_id", "population", "parent_id", "label", "source", "condition", "role")},
            "generalist_score": float(e35_gen[record_id]["score"]), "specialist_score": float(row["score"])})
    e43 = {row["record_id"]: row for row in _load_jsonl(E43_DDA_SCORES)}
    dda_roles = {row["record_id"]: row["role"] for row in contract["dda_roles"]}
    for row in _load_jsonl(E44_DDA_SCORES):
        peer = e43[row["record_id"]]
        output.append({"record_id": row["record_id"], "population": "dda", "parent_id": row["parent_id"],
                       "label": int(row["label"]), "source": row["source"], "condition": row["condition"],
                       "role": dda_roles[row["record_id"]], "generalist_score": float(peer["score"]),
                       "specialist_score": float(row["score"])})
    if len(output) != 6_570 or len({row["record_id"] for row in output}) != len(output):
        raise ValueError("E44 fusion join changed")
    return output


def _rates(rows: Sequence[Mapping[str, Any]], threshold: float) -> dict[str, Any]:
    real_groups: dict[str, list[bool]] = defaultdict(list)
    ai_groups: dict[str, list[bool]] = defaultdict(list)
    for row in rows:
        predicted = float(row["score"]) >= threshold
        key = f"{row['population']}:{row['source']}"
        (ai_groups if int(row["label"]) == 1 else real_groups)[key].append(predicted)
    real_fp = {key: float(np.mean(values)) for key, values in real_groups.items()}
    ai_recall = {key: float(np.mean(values)) for key, values in ai_groups.items()}
    return {
        "real_fp_by_group": real_fp, "ai_recall_by_group": ai_recall,
        "real_macro_fp": float(np.mean(list(real_fp.values()))), "real_worst_fp": max(real_fp.values()),
        "ai_macro_recall": float(np.mean(list(ai_recall.values()))), "ai_worst_recall": min(ai_recall.values()),
    }


def select_threshold(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    candidates = sorted({float(row["score"]) for row in rows}, reverse=True)
    feasible = []
    for threshold in candidates:
        rates = _rates(rows, threshold)
        metrics = evaluate_binary_scores(rows, threshold=threshold)
        if (rates["real_macro_fp"] <= 0.10 and rates["real_worst_fp"] <= 0.25
                and rates["ai_macro_recall"] >= 0.75 and rates["ai_worst_recall"] >= 0.40):
            feasible.append((float(metrics["balanced_accuracy"]), threshold, rates, metrics))
    if not feasible:
        raise RuntimeError("no E44 fusion CAL threshold satisfies frozen constraints")
    _, threshold, rates, metrics = max(feasible, key=lambda item: (item[0], item[1]))
    return {"threshold": threshold, "rates": rates, "metrics": metrics}


def fit() -> dict[str, Any]:
    if any(path.exists() for path in (CANDIDATE, CANDIDATE_REPORT, CANDIDATE_EVIDENCE)):
        raise FileExistsError("E44 fusion candidate already exists")
    rows = _joined_rows()
    fit_rows = [row for row in rows if row["role"] == "fit"]
    cal_rows = [row for row in rows if row["role"] == "calibration"]
    x_fit = np.asarray([_feature(row["generalist_score"], row["specialist_score"]) for row in fit_rows])
    y_fit = np.asarray([row["label"] for row in fit_rows], dtype=np.int64)
    head = make_pipeline(StandardScaler(), LogisticRegression(C=C_VALUE, max_iter=2000,
                                                               random_state=SEED, solver="lbfgs"))
    weights = source_label_weights(fit_rows)
    head.fit(x_fit, y_fit, standardscaler__sample_weight=weights,
             logisticregression__sample_weight=weights)
    cal_scores = head.predict_proba(np.asarray([
        _feature(row["generalist_score"], row["specialist_score"]) for row in cal_rows]))[:, 1]
    scored_cal = [{**row, "score": float(score), "status": "ok"}
                  for row, score in zip(cal_rows, cal_scores, strict=True)]
    selected = select_threshold(scored_cal)
    artifact = {"schema_version": 1, "model_name": "E44 two-specialist logistic fusion",
                "positive_label": "ai", "inputs": ["e43s_probability", "official_dda_probability"],
                "transform": {"kind": "clamped_logit", "epsilon": EPSILON}, "head": head,
                "threshold": float(selected["threshold"]), "C": C_VALUE, "seed": SEED,
                "generalist_sha256": GENERALIST_SHA256,
                "specialist_checkpoint_sha256": "b27a31d39374803ddeff02bfabb2be76e190b04300490cddfafb24f683f37e3e"}
    temporary = CANDIDATE.with_suffix(".joblib.part")
    joblib.dump(artifact, temporary)
    temporary.replace(CANDIDATE)
    report = {"schema_version": 1, "state": "e44_fusion_candidate_frozen_before_development",
              "fit_rows": len(fit_rows), "calibration_rows": len(cal_rows), "C": C_VALUE,
              "selected": selected, "candidate_bytes": CANDIDATE.stat().st_size,
              "candidate_sha256": _digest(CANDIDATE), "development_scores_created": 0,
              "boundary": "Consumed FIT/CAL only; DEVELOPMENT aggregation remains unread."}
    raw = _write(CANDIDATE_REPORT, report)
    _write(CANDIDATE_EVIDENCE, {**report, "detailed_report_sha256": hashlib.sha256(raw).hexdigest()})
    return report


def development_gate(rows: Sequence[Mapping[str, Any]], threshold: float) -> dict[str, Any]:
    metrics = evaluate_binary_scores(rows, threshold=threshold)
    dda = [row for row in rows if row["population"] == "dda"]
    rr = [row for row in rows if row["population"] == "rr"]
    ipn = [row for row in rows if row["population"] == "ipn"]
    owner = [row for row in rows if row["population"] == "owner"]
    dda_recall = {variant: float(np.mean([float(row["score"]) >= threshold for row in dda
                                         if int(row["label"]) == 1 and row["condition"] == variant]))
                  for variant in VARIANTS}
    rr_rates = _rates(rr, threshold)
    ipn_fp = {source: float(np.mean([float(row["score"]) >= threshold for row in ipn if row["source"] == source]))
              for source in sorted({row["source"] for row in ipn})}
    owner_fp = float(np.mean([float(row["score"]) >= threshold for row in owner]))
    found = {"metrics": metrics, "dda_recall": dda_recall,
             "dda_macro_recall": float(np.mean(list(dda_recall.values()))),
             "dda_worst_recall": min(dda_recall.values()), "rr": rr_rates,
             "ipn_fp_by_device": ipn_fp, "ipn_worst_device_fp": max(ipn_fp.values()),
             "owner_false_positive_rate": owner_fp}
    checks = {
        "coverage_eq_1": float(metrics["coverage"]) == 1.0,
        "auc_gte_0_90": float(metrics["roc_auc"]) >= 0.90,
        "balanced_accuracy_gte_0_85": float(metrics["balanced_accuracy"]) >= 0.85,
        "dda_macro_recall_gte_0_75": found["dda_macro_recall"] >= 0.75,
        "dda_worst_recall_gte_0_50": found["dda_worst_recall"] >= 0.50,
        "rr_ai_macro_gte_0_80": rr_rates["ai_macro_recall"] >= 0.80,
        "rr_ai_worst_gte_0_60": rr_rates["ai_worst_recall"] >= 0.60,
        "rr_real_fp_lte_0_10": rr_rates["real_macro_fp"] <= 0.10,
        "ipn_worst_fp_lte_0_20": found["ipn_worst_device_fp"] <= 0.20,
        "owner_fp_lte_0_20": owner_fp <= 0.20,
    }
    return {"passed": all(checks.values()), "checks": checks, "found": found}


def evaluate() -> dict[str, Any]:
    if any(path.exists() for path in (DEVELOPMENT_REPORT, DEVELOPMENT_SCORES, DEVELOPMENT_EVIDENCE)):
        raise FileExistsError("E44 fusion DEVELOPMENT result already exists")
    candidate_report = json.loads(CANDIDATE_REPORT.read_text())
    if _digest(CANDIDATE) != candidate_report["candidate_sha256"]:
        raise ValueError("E44 fusion candidate changed")
    artifact = joblib.load(CANDIDATE)
    rows = [row for row in _joined_rows() if row["role"] == "development"]
    x = np.asarray([_feature(row["generalist_score"], row["specialist_score"]) for row in rows])
    scores = artifact["head"].predict_proba(x)[:, 1]
    scored = [{**row, "score": float(score), "status": "ok"}
              for row, score in zip(rows, scores, strict=True)]
    size, digest = _jsonl(DEVELOPMENT_SCORES, scored)
    gate = development_gate(scored, float(artifact["threshold"]))
    report = {"schema_version": 1,
              "state": "e44_fusion_development_passed" if gate["passed"] else "e44_fusion_development_failed",
              "candidate_sha256": candidate_report["candidate_sha256"],
              "threshold": float(artifact["threshold"]), "counts": {"rows": len(scored)},
              "score_stream": {"bytes": size, "sha256": digest}, "gate": gate,
              "boundary": "Consumed DEVELOPMENT only; not independent final evidence."}
    raw = _write(DEVELOPMENT_REPORT, report)
    _write(DEVELOPMENT_EVIDENCE, {**report, "detailed_report_sha256": hashlib.sha256(raw).hexdigest()})
    return report


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("bind", "score-generalist", "fit", "evaluate"))
    parser.add_argument("--owner-gallery", type=Path, default=OWNER_GALLERY_DEFAULT)
    parser.add_argument("--batch-rows", type=int, default=16)
    args = parser.parse_args(argv)
    if args.command == "bind":
        result = bind(args.owner_gallery.resolve())
    elif args.command == "score-generalist":
        result = score_generalist(args.batch_rows)
    elif args.command == "fit":
        result = fit()
    else:
        result = evaluate()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
