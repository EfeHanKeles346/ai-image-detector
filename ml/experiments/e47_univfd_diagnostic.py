"""Run the frozen official UnivFD arm as post-final E47 architecture triage."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import types
from typing import Any, Mapping, Sequence

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
from pixelproof.project_paths import DATA_ROOT, ML_ROOT


ROOT = DATA_ROOT / "e47"
CHECKOUT = ML_ROOT / "external" / "UniversalFakeDetect"
CHECKOUT_COMMIT = "030495aea3300a8b54c0ec37ec7fe1dd7e63c619"
BACKBONE = ROOT / "models" / "ViT-L-14.pt"
BACKBONE_SHA256 = "b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836"
HEAD = CHECKOUT / "pretrained_weights" / "fc_weights.pth"
HEAD_SHA256 = "477100745713bcc957beb2b40859536859b6483fd6301b3b9293151b194c7847"
SCORES = ROOT / "univfd_scores.jsonl"
REPORT = ROOT / "univfd_diagnostic.json"
EVIDENCE = ML_ROOT.parent / "evidence" / "e47_univfd_diagnostic.json"


def _git_commit(checkout: Path) -> str:
    head = (checkout / ".git" / "HEAD").read_text().strip()
    if not head.startswith("ref: "):
        return head
    ref = checkout / ".git" / head[5:]
    if ref.exists():
        return ref.read_text().strip()
    packed = (checkout / ".git" / "packed-refs").read_text().splitlines()
    suffix = " " + head[5:]
    matches = [line.split()[0] for line in packed if line.endswith(suffix)]
    if len(matches) != 1:
        raise ValueError("cannot resolve pinned UnivFD checkout HEAD")
    return matches[0]


def validate_prefix(
    scored: Sequence[Mapping[str, Any]], rows: Sequence[Mapping[str, Any]]
) -> None:
    if len(scored) > len(rows):
        raise ValueError("UnivFD diagnostic prefix exceeds manifest")
    for index, item in enumerate(scored):
        expected = rows[index]
        if (
            item.get("record_id") != expected.get("record_id")
            or int(item.get("label", -1)) != int(expected.get("label", -2))
            or item.get("source") != expected.get("source")
            or not np.isfinite(float(item.get("score", np.nan)))
        ):
            raise ValueError(f"UnivFD diagnostic prefix changed at row {index}")


def _load_official():
    from packaging import version

    sys.modules.setdefault(
        "pkg_resources",
        types.SimpleNamespace(packaging=types.SimpleNamespace(version=version)),
    )
    sys.modules.setdefault("ftfy", types.SimpleNamespace(fix_text=lambda value: value))
    sys.path.insert(0, str(CHECKOUT))
    from models.clip import clip

    model, preprocess = clip.load(str(BACKBONE), device="cpu")
    head = torch.nn.Linear(768, 1)
    head.load_state_dict(torch.load(HEAD, map_location="cpu", weights_only=False), strict=True)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    return model.to(device).eval(), head.to(device).eval(), preprocess, device


def run(batch_size: int = 16) -> dict[str, Any]:
    if SCORES.exists() or REPORT.exists() or EVIDENCE.exists():
        raise FileExistsError("E47 UnivFD diagnostic already complete; do not overwrite")
    identities = {
        "manifest_sha256": _digest(MANIFEST),
        "e46_result_sha256": _digest(E46_RESULT),
        "e46_fused_scores_sha256": _digest(E46_FUSED),
        "backbone_sha256": _digest(BACKBONE),
        "head_sha256": _digest(HEAD),
        "checkout_commit": _git_commit(CHECKOUT),
    }
    expected = {
        "manifest_sha256": MANIFEST_SHA256,
        "e46_result_sha256": E46_RESULT_SHA256,
        "e46_fused_scores_sha256": E46_FUSED_SHA256,
        "backbone_sha256": BACKBONE_SHA256,
        "head_sha256": HEAD_SHA256,
        "checkout_commit": CHECKOUT_COMMIT,
    }
    if identities != expected:
        raise ValueError(f"E47 UnivFD identity changed: {identities}")
    rows = json.loads(MANIFEST.read_text()).get("rows", [])
    if len(rows) != 2_000:
        raise ValueError("E47 UnivFD diagnostic expects the consumed 2,000-row E46 manifest")

    partial = SCORES.with_suffix(SCORES.suffix + ".partial")
    if partial.exists() and partial.read_bytes() and not partial.read_bytes().endswith(b"\n"):
        raise ValueError("UnivFD diagnostic partial line is truncated")
    scored = _read_jsonl(partial) if partial.exists() else []
    validate_prefix(scored, rows)
    model, head, preprocess, device = _load_official()

    with torch.inference_mode(), partial.open("ab") as stream:
        for start in range(len(scored), len(rows), batch_size):
            group = rows[start : start + batch_size]
            tensors = []
            for row in group:
                path = Path(row["path"])
                if _digest(path) != row["sha256"]:
                    raise ValueError(f"E47 UnivFD payload changed: {row['record_id']}")
                with Image.open(path) as opened:
                    tensors.append(preprocess(opened.convert("RGB")))
            batch = torch.stack(tensors).to(device)
            values = torch.sigmoid(head(model.encode_image(batch).float())).flatten().cpu().numpy()
            output = [
                {"record_id": row["record_id"], "label": int(row["label"]),
                 "source": row["source"], "score": float(value), "status": "ok"}
                for row, value in zip(group, values, strict=True)
            ]
            for item in output:
                stream.write((json.dumps(item, sort_keys=True) + "\n").encode())
            stream.flush()
            os.fsync(stream.fileno())
            scored.extend(output)
            print(f"E47 UnivFD diagnostic {len(scored)}/{len(rows)}", flush=True)

    validate_prefix(scored, rows)
    partial.replace(SCORES)
    raw = SCORES.read_bytes()
    diagnostic = analyze(scored, _read_jsonl(E46_FUSED))
    report = {
        "schema_version": 1,
        "state": "e47_univfd_diagnostic_complete",
        "boundary": "Post-final architecture triage only; cannot repair E46 or fit E47.",
        "identities": identities,
        "counts": {"rows": len(scored), "real": 1_000, "ai": 1_000},
        "score_stream": {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()},
        **diagnostic,
        "next_action": "build a new source-separated CAL/DEVELOPMENT pool" if diagnostic["unlock"]["passed"]
        else "reject UnivFD and evaluate the pinned UNINA compression-trained GAN specialist",
    }
    report_raw = _write(REPORT, report)
    evidence = {**report, "report_bytes": len(report_raw),
                "report_sha256": hashlib.sha256(report_raw).hexdigest()}
    _write(EVIDENCE, evidence)
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    print(json.dumps(run(args.batch_size), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
