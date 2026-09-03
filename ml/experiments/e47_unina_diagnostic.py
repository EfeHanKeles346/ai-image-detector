"""Run the official UNINA StyleGAN2 detector as E47 post-final triage."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch
from PIL import Image

from experiments.e47_gan_recovery import (
    E46_FUSED,
    E46_FUSED_SHA256,
    E46_RESULT,
    E46_RESULT_SHA256,
    MANIFEST,
    MANIFEST_SHA256,
    _digest,
    _read_jsonl,
    _write,
    analyze,
)
from experiments.e47_univfd_diagnostic import validate_prefix
from pixelproof.project_paths import DATA_ROOT, ML_ROOT


ROOT = DATA_ROOT / "e47"
CHECKOUT = ML_ROOT / "external" / "GANimageDetection"
CHECKOUT_COMMIT = "543943cdf281df7417751e794109431d0975df88"
WEIGHTS = ROOT / "models" / "gandetection_resnet50nodown_stylegan2.pth"
WEIGHTS_SHA256 = "65467594eeb53945417c909390a3d872d55b6dbd819aa12cf01e4ced9c4d5a08"
LONG_SIDE_CAP = 512
SCORES = ROOT / "unina_stylegan2_scores.jsonl"
REPORT = ROOT / "unina_stylegan2_diagnostic.json"
EVIDENCE = ML_ROOT.parent / "evidence" / "e47_unina_diagnostic.json"


def _checkout_head() -> str:
    head = (CHECKOUT / ".git" / "HEAD").read_text().strip()
    if head.startswith("ref: "):
        ref = CHECKOUT / ".git" / head[5:]
        if ref.exists():
            return ref.read_text().strip()
        suffix = " " + head[5:]
        matches = [line.split()[0] for line in (CHECKOUT / ".git" / "packed-refs").read_text().splitlines()
                   if line.endswith(suffix)]
        if len(matches) != 1:
            raise ValueError("cannot resolve UNINA checkout HEAD")
        return matches[0]
    return head


def run(sync_every: int = 8, batch_size: int = 2) -> dict[str, Any]:
    if SCORES.exists() or REPORT.exists() or EVIDENCE.exists():
        raise FileExistsError("E47 UNINA diagnostic already complete; do not overwrite")
    identities = {
        "manifest_sha256": _digest(MANIFEST),
        "e46_result_sha256": _digest(E46_RESULT),
        "e46_fused_scores_sha256": _digest(E46_FUSED),
        "weights_sha256": _digest(WEIGHTS),
        "checkout_commit": _checkout_head(),
    }
    expected = {
        "manifest_sha256": MANIFEST_SHA256,
        "e46_result_sha256": E46_RESULT_SHA256,
        "e46_fused_scores_sha256": E46_FUSED_SHA256,
        "weights_sha256": WEIGHTS_SHA256,
        "checkout_commit": CHECKOUT_COMMIT,
    }
    if identities != expected:
        raise ValueError(f"E47 UNINA identity changed: {identities}")
    rows = json.loads(MANIFEST.read_text()).get("rows", [])
    if len(rows) != 2_000:
        raise ValueError("E47 UNINA diagnostic expects the consumed E46 manifest")

    partial = SCORES.with_suffix(SCORES.suffix + ".partial")
    if partial.exists() and partial.read_bytes() and not partial.read_bytes().endswith(b"\n"):
        raise ValueError("UNINA diagnostic partial line is truncated")
    scored = _read_jsonl(partial) if partial.exists() else []
    validate_prefix(scored, rows)
    sys.path.insert(0, str(CHECKOUT))
    from resnet50nodown import resnet50nodown

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = resnet50nodown(device, str(WEIGHTS))
    partial.parent.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode(), partial.open("ab") as stream:
        while len(scored) < len(rows):
            start = len(scored)
            first_size = (int(rows[start]["width"]), int(rows[start]["height"]))
            stop = start + 1
            while stop < min(start + batch_size, len(rows)):
                if (int(rows[stop]["width"]), int(rows[stop]["height"])) != first_size:
                    break
                stop += 1
            group = rows[start:stop]
            tensors = []
            for row in group:
                path = Path(row["path"])
                if _digest(path) != row["sha256"]:
                    raise ValueError(f"E47 UNINA payload changed: {row['record_id']}")
                with Image.open(path) as opened:
                    picture = opened.convert("RGB")
                    scale = min(1.0, LONG_SIDE_CAP / max(picture.size))
                    if scale < 1.0:
                        picture = picture.resize(
                            (round(picture.width * scale), round(picture.height * scale)),
                            Image.Resampling.LANCZOS,
                        )
                    tensors.append(model.transform(picture))
            values = model(torch.stack(tensors).to(device)).flatten().cpu().numpy()
            output = []
            for row, value in zip(group, values, strict=True):
                score = float(value)
                if not np.isfinite(score):
                    raise ValueError(f"non-finite UNINA logit: {row['record_id']}")
                output.append({"record_id": row["record_id"], "label": int(row["label"]),
                               "source": row["source"], "score": score, "status": "ok"})
            for item in output:
                stream.write((json.dumps(item, sort_keys=True) + "\n").encode())
            scored.extend(output)
            if len(scored) % sync_every == 0 or len(scored) == len(rows):
                stream.flush()
                os.fsync(stream.fileno())
            if len(scored) % 25 == 0 or len(scored) == len(rows):
                print(f"E47 UNINA diagnostic {len(scored)}/{len(rows)}", flush=True)

    validate_prefix(scored, rows)
    partial.replace(SCORES)
    raw = SCORES.read_bytes()
    diagnostic = analyze(scored, _read_jsonl(E46_FUSED))
    report = {
        "schema_version": 1,
        "state": "e47_unina_stylegan2_capped_diagnostic_complete",
        "boundary": "Post-final nonprofit research triage only; cannot repair E46 or fit E47.",
        "score_semantics": "raw fake-oriented logit after aspect-preserving 512px long-side cap",
        "long_side_cap": LONG_SIDE_CAP,
        "identities": identities,
        "counts": {"rows": len(scored), "real": 1_000, "ai": 1_000},
        "score_stream": {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()},
        **diagnostic,
        "next_action": "build new source-separated CAL/DEVELOPMENT with the selected specialist"
        if diagnostic["unlock"]["passed"] else "no frozen specialist passed; design new GAN training data",
    }
    report_raw = _write(REPORT, report)
    evidence = {**report, "report_bytes": len(report_raw),
                "report_sha256": hashlib.sha256(report_raw).hexdigest()}
    _write(EVIDENCE, evidence)
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sync-every", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args()
    print(json.dumps(run(args.sync_every, args.batch_size), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
