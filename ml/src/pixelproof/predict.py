"""Command-line entry point with the same asymmetric contract as the API."""

from __future__ import annotations

import argparse
from pathlib import Path

from pixelproof.image_input import ImagePolicyError, decode_image, enough_evidence
from pixelproof.serve import METHODS, ModelRuntime


def official_label(result: dict) -> str:
    decision = result.get("decision")
    if not result.get("enough_evidence") or not decision:
        return "insufficient evidence"
    return "AI detected" if decision.get("label") == "ai" else "insufficient evidence"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("images", type=Path, nargs="+")
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    parser.add_argument("--method", choices=METHODS, default="auto")
    args = parser.parse_args()

    runtime = ModelRuntime(args.artifacts)
    if not runtime.ensure_loaded():
        parser.error(f"model runtime unavailable: {runtime.health()['load_errors']}")

    for path in args.images:
        try:
            raw = path.read_bytes()
            picture = decode_image(raw)
            result = runtime.predict(picture, len(raw), args.method)
            result["enough_evidence"] = enough_evidence(picture)
            if not result["enough_evidence"]:
                result["decision"] = None
        except (OSError, ImagePolicyError) as error:
            print(f"{path}: rejected ({error})")
            continue
        print(
            f"{path}: {official_label(result)} "
            f"(research_score={result['p_ai']:.3f}, method={result['method']})"
        )


if __name__ == "__main__":
    main()
