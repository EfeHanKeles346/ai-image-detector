"""Evaluate the verified project model on user-supplied real/ and ai/ folders."""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import subprocess
import sys
from collections import defaultdict
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import PIL
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score

from pixelproof.artifact_registry import DEFAULT_REPO_ROOT
from pixelproof.image_input import DEFAULT_LIMITS, ImageLimits, ImagePolicyError, decode_image
from pixelproof.project_model import (
    ProjectImageScore,
    ProjectModelContractError,
    ProjectModelMetadata,
    load_project_model,
)


SCHEMA_VERSION = 1
IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".webp"})
CSV_FIELDS = (
    "path",
    "class_name",
    "label",
    "source",
    "status",
    "score",
    "threshold",
    "predicted_ai",
    "tile_count",
    "width",
    "height",
    "bytes",
    "error_stage",
    "error",
)


class EvaluationConfigurationError(ValueError):
    """The requested dataset or output location cannot produce an honest evaluation."""


class ProjectScorer(Protocol):
    metadata: ProjectModelMetadata

    def score_image(self, picture: Image.Image, max_tiles: int = 256) -> ProjectImageScore: ...


def _resolved_directory(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_dir():
        raise EvaluationConfigurationError(f"{label} folder does not exist: {resolved}")
    return resolved


def _validate_roots(real_root: Path, ai_root: Path) -> tuple[Path, Path]:
    real = _resolved_directory(real_root, "real")
    ai = _resolved_directory(ai_root, "ai")
    if real == ai or real in ai.parents or ai in real.parents:
        raise EvaluationConfigurationError(
            "real and ai folders must be separate and neither may contain the other"
        )
    return real, ai


def _discover(root: Path, class_name: str, label: int) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        relative = path.relative_to(root)
        source = relative.parent.as_posix()
        rows.append({
            "absolute_path": path,
            "path": f"{class_name}/{relative.as_posix()}",
            "class_name": class_name,
            "label": label,
            "source": source,
        })
    return rows


def _read_bounded(path: Path, limits: ImageLimits) -> bytes:
    with path.open("rb") as stream:
        raw = stream.read(limits.max_upload_bytes + 1)
    if len(raw) > limits.max_upload_bytes:
        raise ImagePolicyError(
            413,
            f"file exceeds the {limits.max_upload_bytes // (1024 * 1024)} MB limit",
        )
    return raw


def _failure(candidate: dict[str, Any], stage: str, error: Exception) -> dict[str, Any]:
    return {
        **{key: value for key, value in candidate.items() if key != "absolute_path"},
        "status": "error",
        "score": None,
        "threshold": None,
        "predicted_ai": None,
        "tile_count": None,
        "width": None,
        "height": None,
        "bytes": None,
        "error_stage": stage,
        "error": f"{type(error).__name__}: {error}",
    }


def _score_candidate(
    model: ProjectScorer,
    candidate: dict[str, Any],
    max_tiles: int,
    limits: ImageLimits,
) -> dict[str, Any]:
    try:
        raw = _read_bounded(candidate["absolute_path"], limits)
    except (OSError, ImagePolicyError) as error:
        return _failure(candidate, "read", error)
    try:
        picture = decode_image(raw, limits)
    except ImagePolicyError as error:
        failure = _failure(candidate, "decode", error)
        failure["bytes"] = len(raw)
        return failure
    try:
        measured = model.score_image(picture, max_tiles=max_tiles)
        score = float(measured.score)
        if not np.isfinite(score) or not 0.0 <= score <= 1.0:
            raise ValueError(f"model returned invalid score {score!r}")
        if measured.tile_count <= 0:
            raise ValueError("model returned no scored tiles")
    except Exception as error:
        failure = _failure(candidate, "inference", error)
        failure.update({"bytes": len(raw), "width": picture.width, "height": picture.height})
        return failure

    threshold = float(model.metadata.threshold)
    return {
        **{key: value for key, value in candidate.items() if key != "absolute_path"},
        "status": "ok",
        "score": score,
        "threshold": threshold,
        "predicted_ai": bool(score >= threshold),
        "tile_count": measured.tile_count,
        "width": picture.width,
        "height": picture.height,
        "bytes": len(raw),
        "error_stage": None,
        "error": None,
    }


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _metrics(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    successful = [row for row in rows if row["status"] == "ok"]
    labels = np.asarray([row["label"] for row in successful], dtype=np.int64)
    scores = np.asarray([row["score"] for row in successful], dtype=np.float64)
    predictions = np.asarray([row["predicted_ai"] for row in successful], dtype=np.bool_)

    tp = int(np.sum((labels == 1) & predictions))
    fn = int(np.sum((labels == 1) & ~predictions))
    fp = int(np.sum((labels == 0) & predictions))
    tn = int(np.sum((labels == 0) & ~predictions))
    classes = set(labels.tolist())
    auc = float(roc_auc_score(labels, scores)) if classes == {0, 1} else None
    return {
        "roc_auc": auc,
        "recall_at_stored_threshold": _rate(tp, tp + fn),
        "false_positive_rate_at_stored_threshold": _rate(fp, fp + tn),
        "accuracy_at_stored_threshold": _rate(tp + tn, len(successful)),
        "confusion": {"tp": tp, "fn": fn, "fp": fp, "tn": tn},
    }


def _counts(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    def one(group: Sequence[dict[str, Any]]) -> dict[str, int]:
        return {
            "discovered": len(group),
            "succeeded": sum(row["status"] == "ok" for row in group),
            "failed": sum(row["status"] == "error" for row in group),
            "read_failures": sum(row["error_stage"] == "read" for row in group),
            "decode_failures": sum(row["error_stage"] == "decode" for row in group),
            "inference_failures": sum(row["error_stage"] == "inference" for row in group),
        }

    return {
        "total": one(rows),
        "real": one([row for row in rows if row["class_name"] == "real"]),
        "ai": one([row for row in rows if row["class_name"] == "ai"]),
    }


def _per_folder(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["class_name"], row["source"])].append(row)

    result = []
    for (class_name, source), group in sorted(grouped.items()):
        successful = [row for row in group if row["status"] == "ok"]
        hits = sum(bool(row["predicted_ai"]) for row in successful)
        result.append({
            "class_name": class_name,
            "source": source,
            **_counts(group)[class_name],
            "ai_signal_rate": _rate(hits, len(successful)),
        })
    return result


def _git_provenance(repo_root: Path) -> dict[str, Any]:
    def git(*args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", "-C", str(repo_root), *args],
            check=False,
            capture_output=True,
            text=True,
        )

    revision = git("rev-parse", "HEAD")
    status = git("status", "--porcelain")
    return {
        "commit": revision.stdout.strip() if revision.returncode == 0 else None,
        "dirty": bool(status.stdout.strip()) if status.returncode == 0 else None,
    }


def evaluate_folders(
    model: ProjectScorer,
    real_root: Path,
    ai_root: Path,
    *,
    repo_root: Path = DEFAULT_REPO_ROOT,
    max_tiles: int = 256,
    limits: ImageLimits = DEFAULT_LIMITS,
    command: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Score every supported image once and return a JSON-serializable report."""
    if max_tiles <= 0:
        raise EvaluationConfigurationError("max_tiles must be positive")
    real, ai = _validate_roots(real_root, ai_root)
    candidates = _discover(real, "real", 0) + _discover(ai, "ai", 1)
    if not any(candidate["class_name"] == "real" for candidate in candidates):
        raise EvaluationConfigurationError(f"real folder has no supported images: {real}")
    if not any(candidate["class_name"] == "ai" for candidate in candidates):
        raise EvaluationConfigurationError(f"ai folder has no supported images: {ai}")

    rows = [
        _score_candidate(model, candidate, max_tiles, limits)
        for candidate in candidates
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": model.metadata.to_dict(),
        "configuration": {
            "real_root": str(real),
            "ai_root": str(ai),
            "max_tiles": max_tiles,
            "image_limits": {
                "max_upload_bytes": limits.max_upload_bytes,
                "max_pixels": limits.max_pixels,
                "max_dimension": limits.max_dimension,
                "max_aspect_ratio": limits.max_aspect_ratio,
            },
        },
        "provenance": {
            "command": list(command if command is not None else sys.argv),
            "cwd": os.getcwd(),
            "git": _git_provenance(repo_root.resolve()),
            "environment": {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "torch": torch.__version__,
                "pillow": PIL.__version__,
                "numpy": np.__version__,
                "device": str(getattr(model, "device", "unknown")),
            },
        },
        "counts": _counts(rows),
        "metrics": _metrics(rows),
        "per_folder": _per_folder(rows),
        "predictions": rows,
    }


def _report_destination(output: Path) -> Path:
    destination = output.expanduser().resolve()
    if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
        raise EvaluationConfigurationError(f"output must be absent or empty: {destination}")
    return destination


def write_report(report: dict[str, Any], output: Path) -> tuple[Path, Path]:
    """Write results.json and predictions.csv without replacing a prior run."""
    destination = _report_destination(output)
    destination.mkdir(parents=True, exist_ok=True)

    json_path = destination / "results.json"
    json_temporary = destination / ".results.json.tmp"
    json_temporary.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    json_temporary.replace(json_path)

    csv_path = destination / "predictions.csv"
    csv_temporary = destination / ".predictions.csv.tmp"
    with csv_temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(report["predictions"])
    csv_temporary.replace(csv_path)
    return json_path, csv_path


def _select_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(name)
    if name == "cuda" and not torch.cuda.is_available():
        raise EvaluationConfigurationError("CUDA was requested but is unavailable")
    if name == "mps" and not torch.backends.mps.is_available():
        raise EvaluationConfigurationError("MPS was requested but is unavailable")
    return device


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real", type=Path, required=True, help="folder of authentic images")
    parser.add_argument("--ai", type=Path, required=True, help="folder of AI-generated images")
    parser.add_argument("--output", type=Path, required=True, help="new or empty report directory")
    parser.add_argument("--repo-root", type=Path, default=DEFAULT_REPO_ROOT)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--max-tiles", type=int, default=256)
    args = parser.parse_args()

    try:
        _report_destination(args.output)
        device = _select_device(args.device)
        model = load_project_model(args.repo_root, device)
        report = evaluate_folders(
            model,
            args.real,
            args.ai,
            repo_root=args.repo_root,
            max_tiles=args.max_tiles,
            command=sys.argv,
        )
        json_path, csv_path = write_report(report, args.output)
    except (EvaluationConfigurationError, ProjectModelContractError, OSError) as error:
        parser.error(str(error))

    counts = report["counts"]["total"]
    metrics = report["metrics"]
    print(f"results: {json_path}")
    print(f"predictions: {csv_path}")
    print(
        f"images: {counts['succeeded']}/{counts['discovered']} succeeded; "
        f"roc_auc={metrics['roc_auc']}; "
        f"recall={metrics['recall_at_stored_threshold']}; "
        f"false_positive_rate={metrics['false_positive_rate_at_stored_threshold']}"
    )
    if counts["failed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
