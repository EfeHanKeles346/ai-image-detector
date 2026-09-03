"""Post-final GAN-signal triage for E47; never a replacement E46 result."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from PIL import Image, ImageOps
from sklearn.metrics import roc_auc_score

from pixelproof.data import NORMALIZATION
from pixelproof.models import create_model
from pixelproof.project_paths import DATA_ROOT, ML_ROOT


ROOT = DATA_ROOT / "e47"
MANIFEST = DATA_ROOT / "e46" / "truefake_facebook" / "final_manifest_unscored.json"
MANIFEST_SHA256 = "4572339ebe15821c6c86d50178ed31aa80f60cf98f1bd710d73e7265c15b225b"
E46_RESULT = ML_ROOT.parent / "evidence" / "e46_final_result.json"
E46_RESULT_SHA256 = "e7e14fdfae15f20aba0b668acbc70ac34db68689adc905dcaa4594fa85d7a7ed"
E46_FUSED = DATA_ROOT / "e46" / "truefake_fused_scores.jsonl"
E46_FUSED_SHA256 = "6a51a9b11163fc8bb45889e38cc400a1b210cff7bcd6f36d6a760edc1fa68c97"
CHECKPOINT = ML_ROOT / "artifacts" / "best_genimage.pt"
CHECKPOINT_SHA256 = "997e5727eca25b96c2007645eb28667a4bf8d3a46e55c58917e4424aa6144315"
E46_THRESHOLD = 0.6688565012954346
SCORES = ROOT / "legacy_genimage_scores.jsonl"
REPORT = ROOT / "legacy_genimage_diagnostic.json"
EVIDENCE = ML_ROOT.parent / "evidence" / "e47_legacy_gan_diagnostic.json"
STYLEGAN_SOURCES = ("StyleGAN", "StyleGAN2", "StyleGAN3")


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write(path: Path, value: Any) -> bytes:
    raw = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_bytes(raw)
    temporary.replace(path)
    return raw


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _real_safe_threshold(real_scores: np.ndarray, budget: float = 0.10) -> float:
    candidates = sorted(
        {float(value) for value in real_scores}
        | {float(np.nextafter(value, np.inf)) for value in real_scores}
    )
    for threshold in candidates:
        if float(np.mean(real_scores >= threshold)) <= budget + 1e-12:
            return threshold
    return float(np.nextafter(real_scores.max(), np.inf))


def analyze(
    rows: Sequence[Mapping[str, Any]],
    fused_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if len(rows) != len(fused_rows) or not rows:
        raise ValueError("E47 diagnostic streams must be non-empty and aligned")
    fused_by_id = {str(row["record_id"]): row for row in fused_rows}
    if len(fused_by_id) != len(fused_rows):
        raise ValueError("duplicate E46 fused identity")
    if any(str(row["record_id"]) not in fused_by_id for row in rows):
        raise ValueError("E47 diagnostic identity does not join E46")

    labels = np.asarray([int(row["label"]) for row in rows], dtype=np.int8)
    scores = np.asarray([float(row["score"]) for row in rows], dtype=np.float64)
    sources = np.asarray([str(row["source"]) for row in rows])
    if set(labels.tolist()) != {0, 1} or not np.isfinite(scores).all():
        raise ValueError("E47 diagnostic needs finite REAL and AI scores")
    threshold = _real_safe_threshold(scores[labels == 0])
    legacy = scores >= threshold
    fused = np.asarray(
        [float(fused_by_id[str(row["record_id"])]["score"]) >= E46_THRESHOLD for row in rows]
    )
    combined = fused | legacy

    recalls = {
        source: float(np.mean(legacy[(labels == 1) & (sources == source)]))
        for source in sorted(set(sources[labels == 1].tolist()))
    }
    real_mask = labels == 0
    per_source_auc = {}
    for source in sorted(set(sources[labels == 1].tolist())):
        ai_mask = (labels == 1) & (sources == source)
        pair_mask = real_mask | ai_mask
        per_source_auc[source] = float(roc_auc_score(labels[pair_mask], scores[pair_mask]))

    stylegan_rates = [recalls[source] for source in STYLEGAN_SOURCES]
    missed = (labels == 1) & ~fused
    recovered = missed & legacy
    extra_real = (labels == 0) & ~fused & legacy
    combined_real_fp = float(np.mean(combined[labels == 0]))
    unlocked = (
        float(np.mean(stylegan_rates)) >= 0.50
        and min(stylegan_rates) >= 0.30
        and combined_real_fp <= 0.15
    )
    return {
        "diagnostic_threshold": threshold,
        "legacy": {
            "roc_auc": float(roc_auc_score(labels, scores)),
            "real_false_ai": float(np.mean(legacy[labels == 0])),
            "ai_recall": float(np.mean(legacy[labels == 1])),
            "ai_recall_by_source": recalls,
            "auc_by_ai_source_vs_all_real": per_source_auc,
        },
        "complementarity": {
            "e46_ai_misses": int(missed.sum()),
            "misses_recovered": int(recovered.sum()),
            "miss_recovery_rate": float(recovered.sum() / missed.sum()) if missed.any() else None,
            "extra_real_false_ai": int(extra_real.sum()),
            "or_fusion_real_false_ai": combined_real_fp,
            "or_fusion_ai_recall": float(np.mean(combined[labels == 1])),
        },
        "unlock": {
            "mean_stylegan_recall_gte_0_50": float(np.mean(stylegan_rates)) >= 0.50,
            "each_stylegan_recall_gte_0_30": min(stylegan_rates) >= 0.30,
            "or_fusion_real_false_ai_lte_0_15": combined_real_fp <= 0.15,
            "passed": unlocked,
        },
    }


def run(batch_size: int = 64) -> dict[str, Any]:
    if SCORES.exists() or REPORT.exists() or EVIDENCE.exists():
        raise FileExistsError("E47 legacy diagnostic already exists; do not overwrite")
    identities = {
        "manifest_sha256": _digest(MANIFEST),
        "e46_result_sha256": _digest(E46_RESULT),
        "e46_fused_scores_sha256": _digest(E46_FUSED),
        "checkpoint_sha256": _digest(CHECKPOINT),
    }
    expected = {
        "manifest_sha256": MANIFEST_SHA256,
        "e46_result_sha256": E46_RESULT_SHA256,
        "e46_fused_scores_sha256": E46_FUSED_SHA256,
        "checkpoint_sha256": CHECKPOINT_SHA256,
    }
    if identities != expected:
        raise ValueError(f"E47 input identity changed: {identities}")
    if json.loads(E46_RESULT.read_text()).get("state") != "e46_independent_final_failed":
        raise ValueError("E47 requires the archived consumed E46 failure")
    manifest = json.loads(MANIFEST.read_text())
    rows = manifest.get("rows", [])
    if len(rows) != 2_000:
        raise ValueError("E47 diagnostic expects the exact 2,000 consumed E46 rows")

    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    config = checkpoint.get("config", {})
    if config.get("model", {}).get("name") != "resnet18" or config.get("data", {}).get("image_size") != 224:
        raise ValueError("legacy GenImage checkpoint contract changed")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = create_model("resnet18", dropout=0.0, pretrained=False).to(device).eval()
    model.load_state_dict(checkpoint["model"], strict=True)
    mean, std = NORMALIZATION["imagenet"]
    mean_t = torch.tensor(mean, device=device).view(1, 3, 1, 1)
    std_t = torch.tensor(std, device=device).view(1, 3, 1, 1)

    scored: list[dict[str, Any]] = []
    with torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            group = rows[start : start + batch_size]
            arrays = []
            for row in group:
                path = Path(row["path"])
                if _digest(path) != row["sha256"]:
                    raise ValueError(f"E47 payload changed: {row['record_id']}")
                with Image.open(path) as opened:
                    rgb = ImageOps.exif_transpose(opened).convert("RGB")
                    arrays.append(np.asarray(rgb.resize((224, 224), Image.Resampling.BILINEAR), dtype=np.uint8))
            tensor = torch.from_numpy(np.stack(arrays)).to(device).permute(0, 3, 1, 2).float().div_(255.0)
            values = torch.sigmoid(model((tensor - mean_t) / std_t)).cpu().numpy()
            scored.extend(
                {"record_id": row["record_id"], "label": int(row["label"]),
                 "source": row["source"], "score": float(value), "status": "ok"}
                for row, value in zip(group, values, strict=True)
            )
            print(f"E47 legacy GAN diagnostic {min(start + len(group), len(rows))}/{len(rows)}", flush=True)

    score_raw = b"".join((json.dumps(row, sort_keys=True) + "\n").encode() for row in scored)
    SCORES.parent.mkdir(parents=True, exist_ok=True)
    temporary = SCORES.with_suffix(".jsonl.part")
    temporary.write_bytes(score_raw)
    temporary.replace(SCORES)
    diagnostic = analyze(scored, _read_jsonl(E46_FUSED))
    report = {
        "schema_version": 1,
        "state": "e47_legacy_gan_diagnostic_complete",
        "boundary": "Post-final architecture triage only; cannot repair E46 or fit E47.",
        "identities": identities,
        "counts": {"rows": len(scored), "real": 1_000, "ai": 1_000},
        "score_stream": {"bytes": len(score_raw), "sha256": hashlib.sha256(score_raw).hexdigest()},
        **diagnostic,
        "next_action": "build new CAL/DEVELOPMENT around legacy GAN arm" if diagnostic["unlock"]["passed"]
        else "reject legacy arm and acquire an official frozen GAN specialist",
    }
    report_raw = _write(REPORT, report)
    evidence = {**report, "report_bytes": len(report_raw),
                "report_sha256": hashlib.sha256(report_raw).hexdigest()}
    _write(EVIDENCE, evidence)
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    print(json.dumps(run(args.batch_size), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
