"""Research-only folder scorer for the rejected-but-runnable E31 DINO candidate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image, ImageOps

from pixelproof.e31_candidate import CANDIDATE_SHA256, E31Candidate

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def image_files(root: Path) -> list[Path]:
    if root.is_file():
        return [root] if root.suffix.lower() in IMAGE_SUFFIXES else []
    return sorted(
        path for path in root.rglob("*")
        if path.is_file() and not path.name.startswith("._") and path.suffix.lower() in IMAGE_SUFFIXES
    )


def verdict(predicted_ai: bool) -> str:
    return "ai_signal_detected" if predicted_ai else "insufficient_evidence"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    args = parser.parse_args()
    files = image_files(args.input)
    if not files:
        parser.error("input contains no supported JPG/PNG/WEBP files")
    if args.output.exists():
        parser.error("output already exists; refusing to overwrite an evaluation")
    device = torch.device(
        "mps" if args.device == "auto" and torch.backends.mps.is_available() else
        "cpu" if args.device == "auto" else args.device
    )
    model = E31Candidate(device=device)
    root = args.input if args.input.is_dir() else args.input.parent
    rows = []
    for index, path in enumerate(files, 1):
        relative = str(path.relative_to(root))
        try:
            with Image.open(path) as opened:
                image = ImageOps.exif_transpose(opened).convert("RGB")
            result = model.score_image(image, f"folder:{relative}")
            row = {
                "path": relative,
                "status": "ok",
                "score": result.score,
                "threshold": result.threshold,
                "verdict": verdict(result.predicted_ai),
                "error": None,
            }
        except Exception as error:
            row = {
                "path": relative,
                "status": "error",
                "score": None,
                "threshold": model.threshold,
                "verdict": "insufficient_evidence",
                "error": f"{type(error).__name__}: {error}"[:300],
            }
        rows.append(row)
        print(f"{index}/{len(files)} {relative}: {row['verdict']}")
    payload = {
        "schema_version": 1,
        "detector": "E31 single DINOv2 research candidate",
        "candidate_sha256": CANDIDATE_SHA256,
        "status": "rejected_for_serving_after_E30_DEVELOPMENT",
        "warning": "This candidate measured 83.63% macro false positives on independent real DEVELOPMENT data. Scores are research diagnostics, not authenticity decisions.",
        "input": str(args.input),
        "accounting": {
            "files": len(rows),
            "succeeded": sum(row["status"] == "ok" for row in rows),
            "failed": sum(row["status"] != "ok" for row in rows),
            "ai_signals": sum(row["verdict"] == "ai_signal_detected" for row in rows),
        },
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".part")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
